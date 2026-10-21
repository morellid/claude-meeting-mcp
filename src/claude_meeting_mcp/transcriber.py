"""Dual-backend transcription (MLX-Whisper / faster-whisper / remote) with speaker attribution."""

import logging
import os
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf

from . import _mlx
from .config import get_config, get_faster_model_id, get_mlx_model_id
from .schemas import Segment, Transcription
from .storage import TRANSCRIPTIONS_DIR, ensure_dirs

logger = logging.getLogger(__name__)

# Cached model instances (avoid reloading on every call)
_faster_model = None
_faster_model_name: str | None = None


def _get_backend(mode_override: str | None = None) -> str:
    """Detect best available transcription backend.

    Returns: 'mistral', 'remote', 'mlx', or 'faster'
    """
    config = get_config()
    mode = mode_override or config.transcription.mode

    if mode in ("remote", "mistral"):
        return mode

    # Local mode: prefer MLX on macOS Apple Silicon
    if sys.platform == "darwin":
        try:
            import mlx_whisper  # noqa: F401

            return "mlx"
        except ImportError:
            pass

    try:
        import faster_whisper  # noqa: F401

        return "faster"
    except ImportError:
        pass

    raise RuntimeError(
        "No whisper backend available. "
        "Install mlx-whisper (macOS Apple Silicon) or faster-whisper (pip install faster-whisper)."
    )


def _resolve_language(language: str | None) -> str | None:
    """Map the configured language to a Whisper language code, or None for auto-detect.

    "auto"/""/"none" (any case) mean: let Whisper detect the language per channel.
    Forcing the wrong language produces garbage AND is ~4x slower (temperature
    fallback re-decoding), so auto-detect is the safe default for bilingual use.
    """
    normalized = (language or "").strip()
    if not normalized or normalized.lower() in ("auto", "none"):
        return None
    return normalized


def _transcribe_mlx(
    audio: np.ndarray, samplerate: int, model: str | None = None, prompt: str | None = None
) -> list[dict]:
    """Transcribe audio with MLX-Whisper (macOS Apple Silicon)."""
    import mlx_whisper

    _mlx.cap_cache()

    config = get_config()
    model_id = get_mlx_model_id(model or config.transcription.model)
    language = _resolve_language(config.transcription.language)

    audio = audio.astype(np.float32, copy=False)
    try:
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=model_id,
            language=language,
            initial_prompt=prompt,
            word_timestamps=True,
            condition_on_previous_text=False,
            hallucination_silence_threshold=1.0,
            compression_ratio_threshold=2.4,
            no_speech_threshold=0.6,
        )
    finally:
        # This server outlives every transcription; drain the buffer cache here
        # or it accumulates for the life of the Claude session. See _mlx.
        _mlx.clear_cache()
    return result.get("segments", [])


def _transcribe_faster(
    audio: np.ndarray, samplerate: int, model: str | None = None, prompt: str | None = None
) -> list[dict]:
    """Transcribe audio with faster-whisper (CPU/CUDA)."""
    global _faster_model, _faster_model_name
    from faster_whisper import WhisperModel

    config = get_config()
    model_id = get_faster_model_id(model or config.transcription.model)
    language = _resolve_language(config.transcription.language)

    # Cache model singleton (reload if model changed)
    if _faster_model is None or _faster_model_name != model_id:
        _faster_model = WhisperModel(model_id, device="auto", compute_type="auto")
        _faster_model_name = model_id

    audio = audio.astype(np.float32, copy=False)
    segments_gen, info = _faster_model.transcribe(
        audio,
        language=language,
        initial_prompt=prompt,
        word_timestamps=False,
        vad_filter=True,
    )

    # Convert generator of Segment dataclass to list[dict]
    return [{"start": s.start, "end": s.end, "text": s.text} for s in segments_gen]


def _transcribe_remote_channel(wav_path: str, channel: str = "left") -> list[dict]:
    """Transcribe audio via remote OpenAI-compatible API.

    Args:
        wav_path: Path to a mono WAV file (single channel)
        channel: 'left' or 'right' for logging
    """
    config = get_config()
    url = config.transcription.remote.url
    api_key = os.environ.get(config.transcription.remote.api_key_env, "")

    if not url:
        raise RuntimeError("Remote transcription URL not configured. Set transcription.remote.url")

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    data = {
        "model": config.transcription.model,
        "response_format": "verbose_json",
    }
    # Omit "language" entirely for auto-detect; the OpenAI-compatible API
    # detects the language when the field is absent.
    language = _resolve_language(config.transcription.language)
    if language:
        data["language"] = language

    with open(wav_path, "rb") as f:
        response = httpx.post(
            url,
            headers=headers,
            files={"file": (f"audio_{channel}.wav", f, "audio/wav")},
            data=data,
            timeout=300.0,
        )

    response.raise_for_status()
    result = response.json()

    # OpenAI Whisper API returns segments in verbose_json format
    return [
        {"start": s["start"], "end": s["end"], "text": s["text"]}
        for s in result.get("segments", [])
    ]


MISTRAL_URL = "https://api.mistral.ai/v1/audio/transcriptions"
MISTRAL_MODEL = "voxtral-mini-latest"
MISTRAL_MAX_CONTEXT_BIAS = 100  # API limit on biasing terms


def _mistral_api_key() -> str:
    """MISTRAL_API_KEY from the environment, else from the macOS Keychain.

    The Claude app spawns MCP servers without a login shell, so an export in
    .zshrc never reaches this process; the Keychain does.
    """
    key = os.environ.get("MISTRAL_API_KEY", "").strip()
    if key:
        return key
    if sys.platform == "darwin":
        found = subprocess.run(
            ["security", "find-generic-password", "-s", "MISTRAL_API_KEY", "-w"],
            capture_output=True,
            text=True,
        )
        if found.returncode == 0 and found.stdout.strip():
            return found.stdout.strip()
    raise RuntimeError(
        "MISTRAL_API_KEY not found in the environment or the Keychain. Store it with: "
        'security add-generic-password -a "$USER" -s MISTRAL_API_KEY -w'
    )


def _mix_to_mono(
    left: np.ndarray, right: np.ndarray, left_silent: bool, right_silent: bool
) -> np.ndarray:
    """Mix the system and mic channels into one track for a diarizing provider.

    A silent channel is dropped rather than averaged in, which would only halve
    the level of the live one. Mixes into `left` in place: a two-hour channel is
    1.4 GB of float32, and the caller no longer needs the separate channels.
    """
    if left_silent:
        return right
    if right_silent:
        return left
    np.add(left, right, out=left)
    left *= 0.5
    return left


def _encode_for_upload(audio: np.ndarray, samplerate: int) -> str:
    """Write mono audio as 16 kHz Ogg/Opus to a temp file and return its path.

    A raw 48 kHz WAV is ~5.8 MB per minute; Opus at 16 kHz is a few hundred KB,
    which keeps a two-hour upload in seconds rather than minutes.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".ogg", delete=False)
    tmp.close()
    sf.write(
        tmp.name,
        _resample_to_16k(audio, samplerate),
        WHISPER_SAMPLE_RATE,
        format="OGG",
        subtype="OPUS",
    )
    return tmp.name


def _mistral_segments(result: dict) -> list[dict]:
    """Turn Voxtral's diarized segments into pipeline dicts with a speaker label.

    Voxtral ids ("speaker_3") carry no identity, so they are renumbered in order of
    first appearance: "Speaker 1" is whoever talks first. Mapping labels to names
    is left to the reader, who has the calendar and the context.
    """
    labels: dict[str, str] = {}
    segments: list[dict] = []
    for seg in result.get("segments", []):
        if not seg.get("text", "").strip():
            continue
        speaker_id = seg.get("speaker_id") or "unknown"
        if speaker_id not in labels:
            labels[speaker_id] = f"Speaker {len(labels) + 1}"
        segments.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "speaker": labels[speaker_id],
            }
        )
    return segments


def _transcribe_mistral(audio: np.ndarray, samplerate: int, names: list[str]) -> list[dict]:
    """Transcribe and diarize one mono track with Mistral Voxtral in a single request.

    Voxtral takes up to 3 hours per request, past the recording safety cap, so the
    file is never chunked. Known participant names go in as context_bias so they
    are spelled right, one word per term: the API rejects a term with whitespace.
    """
    path = _encode_for_upload(audio, samplerate)
    data: dict[str, str | list[str]] = {
        "model": MISTRAL_MODEL,
        "diarize": "true",
        "timestamp_granularities": "segment",
    }
    words: list[str] = []
    for word in (w for name in names for w in name.split()):
        if word.lower() not in {w.lower() for w in words}:
            words.append(word)
    if words:
        data["context_bias"] = words[:MISTRAL_MAX_CONTEXT_BIAS]
    language = _resolve_language(get_config().transcription.language)
    if language:
        data["language"] = language

    try:
        with open(path, "rb") as f:
            response = httpx.post(
                MISTRAL_URL,
                headers={"Authorization": f"Bearer {_mistral_api_key()}"},
                files={"file": (Path(path).name, f, "audio/ogg")},
                data=data,
                timeout=httpx.Timeout(900.0, connect=30.0),
            )
    finally:
        os.unlink(path)

    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        # The status line alone hides the body, which is where the API says what is wrong.
        raise httpx.HTTPStatusError(
            f"{e}\n{response.text}", request=e.request, response=e.response
        ) from e
    return _mistral_segments(response.json())


def _can_parallelize() -> bool:
    """Check if dual-channel transcription can be parallelized."""
    backend = _get_backend()
    if backend == "remote":
        return True
    if backend == "mlx":
        return False  # MLX monopolizes Metal GPU
    if backend == "faster":
        # CTranslate2 releases the GIL — safe to parallelize
        # Pre-load model if not yet loaded so first call benefits too
        if _faster_model is None:
            _ensure_faster_model()
        return True
    return False


def _ensure_faster_model() -> None:
    """Ensure faster-whisper model is loaded (for parallelization check)."""
    global _faster_model, _faster_model_name
    if _faster_model is not None:
        return
    from faster_whisper import WhisperModel

    config = get_config()
    model_id = get_faster_model_id(config.transcription.model)
    _faster_model = WhisperModel(model_id, device="auto", compute_type="auto")
    _faster_model_name = model_id
    logger.info("Faster-whisper model loaded: %s", model_id)


WHISPER_SAMPLE_RATE = 16000


def _resample_to_16k(audio: np.ndarray, samplerate: int) -> np.ndarray:
    """Resample audio to 16kHz (what Whisper expects internally).

    Polyphase (FIR), not scipy.signal.resample (FFT), for two reasons. pocketfft
    caches an FFT plan per distinct transform length and never evicts it; since
    every recording has a unique sample count that cache never hits and only
    grows - a day-old server held 12 GB across 89 dead plans. FFT resampling also
    assumes a periodic signal, so it wraps the end of a recording into its start.
    """
    if samplerate == WHISPER_SAMPLE_RATE:
        return audio
    from math import gcd

    from scipy.signal import resample_poly

    divisor = gcd(WHISPER_SAMPLE_RATE, samplerate)
    up, down = WHISPER_SAMPLE_RATE // divisor, samplerate // divisor
    logger.info(
        "Resampling %d Hz → %d Hz (%d → %d samples, %d/%d)",
        samplerate,
        WHISPER_SAMPLE_RATE,
        len(audio),
        int(len(audio) * WHISPER_SAMPLE_RATE / samplerate),
        up,
        down,
    )
    return resample_poly(audio, up, down).astype(np.float32)


# A channel whose RMS sits below this (~-60 dBFS) carries no usable speech, so we
# skip transcribing it rather than spend minutes on silence. Deliberately
# conservative: real speech sits far higher (~-25 dBFS), and the asymmetry favours
# caution - failing to skip only wastes time, whereas skipping a channel with real
# (quiet) speech would silently drop content. This catches a genuinely silent
# channel (e.g. a muted mic while recording a video); it does NOT catch a loud
# channel that merely lacks speech (noise/hum) - that would need VAD.
SILENCE_RMS_THRESHOLD = 1e-3


def _channel_is_silent(audio: np.ndarray, threshold: float = SILENCE_RMS_THRESHOLD) -> bool:
    """True if a channel's RMS energy is below the silence threshold."""
    if audio.size == 0:
        return True
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    return rms < threshold


def _build_name_prompt(*speaker_lists: str | None) -> str | None:
    """Build a Whisper initial_prompt that biases toward known participant names.

    Names are passed verbatim as a spelling hint so Whisper transcribes them
    correctly (e.g. "Davide", "Nadeem"). Returns None when no real names are available.
    """
    return ", ".join(_known_names(*speaker_lists)) or None


def _known_names(*speaker_lists: str | None) -> list[str]:
    """Real participant names from comma-separated lists, deduped case-insensitively.

    Placeholder labels ("Remote"/"Local") and blanks are ignored.
    """
    names: list[str] = []
    seen: set[str] = set()
    for raw in speaker_lists:
        if not raw:
            continue
        for name in raw.split(","):
            name = name.strip()
            if not name or name in ("Remote", "Local"):
                continue
            key = name.lower()
            if key not in seen:
                seen.add(key)
                names.append(name)
    return names


def transcribe_channel(
    audio: np.ndarray, samplerate: int, model: str | None = None, prompt: str | None = None
) -> list[dict]:
    """Transcribe a single audio channel using the configured backend."""
    # Resample to 16kHz — Whisper is trained on 16kHz and internal resampling
    # from 48kHz can produce artifacts/hallucinations
    audio = _resample_to_16k(audio, samplerate)
    backend = _get_backend()

    if backend == "mlx":
        return _transcribe_mlx(audio, WHISPER_SAMPLE_RATE, model, prompt)
    elif backend == "faster":
        return _transcribe_faster(audio, WHISPER_SAMPLE_RATE, model, prompt)
    else:
        raise RuntimeError("Remote backend should not call transcribe_channel directly")


def split_channels(wav_path: str) -> tuple[np.ndarray, np.ndarray, int]:
    """Split stereo WAV into left (system) and right (mic) channels."""
    data, samplerate = sf.read(wav_path, dtype="float32")
    if data.ndim == 1:
        return data, data.copy(), samplerate
    left = np.ascontiguousarray(data[:, 0])
    right = np.ascontiguousarray(data[:, 1])
    del data  # Free stereo buffer immediately (~1 GB for 3h recording)
    return left, right, samplerate


def merge_segments(
    left_segments: list[dict],
    right_segments: list[dict],
    left_speaker: str,
    right_speaker: str,
) -> list[Segment]:
    """Merge two channel transcriptions into a single timeline sorted by start time.

    If segments already have a "speaker" key (from diarization), it is used.
    Otherwise, left_speaker/right_speaker is applied as fallback.
    """
    segments = []
    for seg in left_segments:
        speaker = seg.get("speaker") or left_speaker
        segments.append(
            Segment(
                start=round(seg["start"], 2),
                end=round(seg["end"], 2),
                speaker=speaker,
                text=seg["text"].strip(),
            )
        )
    for seg in right_segments:
        speaker = seg.get("speaker") or right_speaker
        segments.append(
            Segment(
                start=round(seg["start"], 2),
                end=round(seg["end"], 2),
                speaker=speaker,
                text=seg["text"].strip(),
            )
        )
    segments.sort(key=lambda s: s.start)
    return segments


def _save_channel_temp(audio: np.ndarray, samplerate: int, suffix: str) -> str:
    """Save a channel as a temporary WAV file for remote transcription."""
    import tempfile

    tmp = tempfile.NamedTemporaryFile(suffix=f"_{suffix}.wav", delete=False)
    sf.write(tmp.name, audio, samplerate)
    tmp.close()
    return tmp.name


def _diarize_and_assign(
    audio: np.ndarray,
    samplerate: int,
    whisper_segments: list[dict],
    speaker_names: list[str],
    channel_prefix: str,
) -> list[dict]:
    """Run diarization on a channel and assign speakers to Whisper segments."""
    from .diarize import assign_speakers_to_segments, diarize_channel

    diar_segments = diarize_channel(audio, samplerate)
    return assign_speakers_to_segments(
        whisper_segments, diar_segments, speaker_names, channel_prefix
    )


def transcribe_meeting(
    wav_path: str,
    remote_speakers: str | None = None,
    local_speakers: str | None = None,
    model: str | None = None,
) -> Transcription:
    """Full transcription pipeline: split channels, transcribe each, merge.

    Args:
        wav_path: Path to the stereo WAV file
        remote_speakers: Comma-separated names for system audio (left channel)
        local_speakers: Comma-separated names for microphone (right channel)
        model: Whisper model override (default from config)
    """
    ensure_dirs()
    config = get_config()
    wav = Path(wav_path)

    remote_label = remote_speakers or "Remote"
    local_label = local_speakers or "Local"
    remote_names = [n.strip() for n in remote_label.split(",")]
    local_names = [n.strip() for n in local_label.split(",")]

    left_audio, right_audio, samplerate = split_channels(wav_path)
    duration = len(left_audio) / samplerate

    logger.info("Transcribing %s (%.1fs, %d Hz)", wav_path, duration, samplerate)
    t0 = time.monotonic()
    backend = _get_backend()

    # Skip channels that are silent - no point spending minutes transcribing silence.
    # On MLX (sequential) this halves the time for one-sided sessions, e.g. a muted
    # mic while recording a video.
    left_silent = _channel_is_silent(left_audio)
    right_silent = _channel_is_silent(right_audio)
    if left_silent:
        logger.info("System (left) channel is silent - skipping transcription")
    if right_silent:
        logger.info("Mic (right) channel is silent - skipping transcription")

    # Bias Whisper toward known participant names so it spells them right.
    # Both channels get the FULL name list on purpose: in conversation people
    # address each other by name across channels ("Thanks, Nadeem"), so a name
    # can be spoken on either side. Remote path keeps its own contract, untouched.
    name_prompt = _build_name_prompt(remote_speakers, local_speakers)
    if name_prompt:
        logger.info("Biasing transcription toward names: %s", name_prompt)

    if backend == "mistral":
        # Voxtral separates voices itself, so it gets one mono mix in one request.
        # Mic bleed makes the system/mic split mislabel speakers anyway; voice
        # diarization does not care which channel a voice leaked into.
        left_segments = _transcribe_mistral(
            _mix_to_mono(left_audio, right_audio, left_silent, right_silent),
            samplerate,
            _known_names(remote_speakers, local_speakers),
        )
        right_segments = []
    elif backend == "remote":
        left_segments, right_segments = [], []
        left_tmp = right_tmp = None
        try:
            with ThreadPoolExecutor(max_workers=2) as pool:
                fl = fr = None
                if not left_silent:
                    left_tmp = _save_channel_temp(left_audio, samplerate, "left")
                    fl = pool.submit(_transcribe_remote_channel, left_tmp, "left")
                if not right_silent:
                    right_tmp = _save_channel_temp(right_audio, samplerate, "right")
                    fr = pool.submit(_transcribe_remote_channel, right_tmp, "right")
                if fl is not None:
                    left_segments = fl.result()
                if fr is not None:
                    right_segments = fr.result()
        finally:
            if left_tmp is not None:
                os.unlink(left_tmp)
            if right_tmp is not None:
                os.unlink(right_tmp)
    elif _can_parallelize():
        logger.info("Parallel transcription (backend=%s)", backend)
        with ThreadPoolExecutor(max_workers=2) as pool:
            fl = None
            fr = None
            if not left_silent:
                fl = pool.submit(transcribe_channel, left_audio, samplerate, model, name_prompt)
            if not right_silent:
                fr = pool.submit(transcribe_channel, right_audio, samplerate, model, name_prompt)
            left_segments = fl.result() if fl is not None else []
            right_segments = fr.result() if fr is not None else []
    else:
        logger.info("Sequential transcription (backend=%s)", backend)
        left_segments = (
            [] if left_silent else transcribe_channel(left_audio, samplerate, model, name_prompt)
        )
        right_segments = (
            [] if right_silent else transcribe_channel(right_audio, samplerate, model, name_prompt)
        )

    # Apply diarization if enabled (Voxtral has already diarized)
    if config.diarization.enabled and backend != "mistral":
        left_segments = _diarize_and_assign(
            left_audio, samplerate, left_segments, remote_names, "remote"
        )
        right_segments = _diarize_and_assign(
            right_audio, samplerate, right_segments, local_names, "local"
        )
        segments = merge_segments(left_segments, right_segments, "", "")
    else:
        segments = merge_segments(left_segments, right_segments, remote_label, local_label)

    # Collect all unique speaker names
    all_speakers = sorted(set(s.speaker for s in segments))
    speakers_dict = {f"speaker_{i}": name for i, name in enumerate(all_speakers)}

    meeting_id = wav.stem
    transcription = Transcription(
        meeting_id=meeting_id,
        date=meeting_id[:10] if len(meeting_id) >= 10 else "",
        duration_seconds=round(duration, 1),
        speakers=speakers_dict,
        segments=segments,
    )

    # Save JSON atomically: write to a temp file then rename, so a reader (e.g. the
    # background-job disk fallback) never sees a half-written transcript if the
    # process is killed mid-write.
    output_path = TRANSCRIPTIONS_DIR / f"{meeting_id}.json"
    tmp_path = output_path.with_suffix(".json.tmp")
    tmp_path.write_text(transcription.to_json(), encoding="utf-8")
    os.replace(tmp_path, output_path)

    # Delete WAV after successful transcription — no longer needed
    try:
        wav.unlink()
        logger.info("Deleted WAV after transcription: %s", wav_path)
    except OSError as e:
        logger.warning("Could not delete WAV: %s", e)

    elapsed = time.monotonic() - t0
    logger.info(
        "Transcription complete: %d segments, %.1fs elapsed (backend=%s)",
        len(segments),
        elapsed,
        backend,
    )
    return transcription


def cli():
    """CLI entry point for manual transcription."""
    if len(sys.argv) < 2:
        print("Usage: transcribe <path_to_wav> [remote_speakers] [local_speakers] [model]")
        sys.exit(1)

    wav_path = sys.argv[1]
    remote = sys.argv[2] if len(sys.argv) > 2 else None
    local = sys.argv[3] if len(sys.argv) > 3 else None
    model = sys.argv[4] if len(sys.argv) > 4 else None

    print(f"Transcribing {wav_path} (backend: {_get_backend()})...")
    result = transcribe_meeting(wav_path, remote, local, model)
    print(f"Done. {len(result.segments)} segments, {result.duration_seconds}s")
    print(f"Saved to: transcriptions/{result.meeting_id}.json")
