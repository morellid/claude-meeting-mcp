"""Live transcript: a rough, running transcript of the recording in progress.

One worker process per recording (`python -m claude_meeting_mcp.live <wav>`), started
by whoever starts the capture (AutoRecorder or audio_record_start) and stopped before
the final transcription begins, so the live model and the batch run never hold memory
at the same time. The worker reads the WAV while audiocap writes it, transcribes each
channel separately with a small local Whisper, and appends JSON lines to
`<meeting_id>_live.jsonl`. Any Claude session reads that file through
audio_live_transcript; no session loads a model to do so.

The live transcript is for questions during the call only. Notes, tasks and the
meeting record still come from the final transcription.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from .config import get_mlx_model_id
from .live_translator import FileAudioSource
from .storage import RECORDINGS_DIR, TRANSCRIPTIONS_DIR
from .transcriber import _channel_is_silent, _resample_to_16k

logger = logging.getLogger(__name__)

# small: ~0.07x real time per channel on an M4 and ~1.1 GB footprint (model plus
# the 512 MB MLX cache cap, measured with footprint 2026-11-09), so it keeps
# up with two channels and leaves memory for the call app (measured 2026-11-09).
LIVE_MODEL = "small"
CHUNK_SECONDS = 10.0
# Until the language is known, chunks are 30 s: Whisper detects the language on a
# 30 s window, and on 10 s of call audio it picked Welsh and Russian (2026-11-09).
DETECT_SECONDS = 30.0
# audiocap writes system audio (the others) left and the mic (Davide) right.
SPEAKERS = ("Remote", "Davide")
# The worker exits on its own when the WAV stops growing this long: its parent
# (an MCP server, AutoRecorder) may have died without stopping it.
ORPHAN_SECONDS = 120.0
WAV_HEADER_BYTES = 44

# (16 kHz mono audio, language or None to detect) -> (segments, language used)
Transcribe = Callable[[np.ndarray, str | None], tuple[list[dict], str | None]]


def live_path(wav_path: str, transcripts_dir: Path = TRANSCRIPTIONS_DIR) -> Path:
    return transcripts_dir / f"{Path(wav_path).stem}_live.jsonl"


def cut_point(audio: np.ndarray, sr: int, search_seconds: float = 2.0) -> int:
    """Sample index of the quietest 100 ms in the last `search_seconds`.

    Ending a chunk in a pause instead of at a fixed length keeps words whole.
    """
    frame = sr // 10
    start = max(0, len(audio) - int(search_seconds * sr))
    tail = audio[start : start + ((len(audio) - start) // frame) * frame]
    if tail.size == 0:
        return len(audio)
    energy = np.square(tail.astype(np.float64)).reshape(-1, frame).mean(axis=1)
    return start + int(np.argmin(energy)) * frame + frame // 2


def whisper_small(audio: np.ndarray, language: str | None = None) -> tuple[list[dict], str | None]:
    """Transcribe 16 kHz mono audio with the small MLX Whisper.

    One greedy pass (temperature 0, no fallback): on real call audio the fallback
    re-decoded hallucination loops until a 10 s chunk took longer than 10 s and the
    worker fell behind the call (2026-11-09). `language` None = detect it on the
    first 30 s of the audio. Returns the segments and the language used.
    """
    import mlx_whisper

    from . import _mlx

    _mlx.cap_cache()
    try:
        result = mlx_whisper.transcribe(
            audio.astype(np.float32, copy=False),
            path_or_hf_repo=get_mlx_model_id(LIVE_MODEL),
            language=language,
            temperature=0.0,
            condition_on_previous_text=False,
            hallucination_silence_threshold=1.0,
        )
    finally:
        _mlx.clear_cache()
    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in result.get("segments", [])
        if s["text"].strip()
    ]
    return segments, result.get("language", language)


class LiveTranscriber:
    """Turns a growing stereo WAV into appended transcript lines, one chunk per step."""

    def __init__(
        self,
        wav_path: str,
        out_path: Path,
        transcribe: Transcribe = whisper_small,
        sample_rate: int = 48000,
        chunk_seconds: float = CHUNK_SECONDS,
        language: str | None = None,
    ) -> None:
        self._wav = wav_path
        self._out = out_path
        self._transcribe = transcribe
        self._chunk_seconds = chunk_seconds
        # None until detected on the first chunk with speech, then fixed for the call.
        self._language = language
        # sample_rate is only a fallback: the source switches to the header's rate.
        self._source = FileAudioSource(wav_path, sample_rate=sample_rate, channels=2)
        self._buf = np.zeros((0, 2), dtype=np.float32)
        self._pos = 0  # samples already transcribed, from the start of the recording
        self._ppid = os.getppid()
        self._seen = False  # the WAV has existed: from now on, missing means deleted

    def step(self) -> bool:
        """Transcribe one chunk if a full one is buffered. False when waiting for audio."""
        seconds = self._chunk_seconds if self._language else DETECT_SECONDS
        while len(self._buf) < seconds * self._source.get_sample_rate():
            frames = self._source.get_new_frames()
            if frames is None:
                return False
            self._seen = True
            self._buf = np.concatenate([self._buf, frames])

        sr = self._source.get_sample_rate()
        cut = cut_point(self._buf[: int(seconds * sr)].mean(axis=1), sr)
        chunk, self._buf = self._buf[:cut], self._buf[cut:]
        offset = self._pos / sr
        self._pos += cut

        rows = []
        for channel, speaker in enumerate(SPEAKERS):
            audio = chunk[:, channel]
            if _channel_is_silent(audio):
                continue
            segments, language = self._transcribe(_resample_to_16k(audio, sr), self._language)
            if self._language is None and language:
                self._language = language
                logger.info("Language detected: %s, kept for the rest of the call", language)
            for seg in segments:
                rows.append(
                    {
                        "start": round(offset + seg["start"], 2),
                        "end": round(offset + seg["end"], 2),
                        "speaker": speaker,
                        "text": seg["text"],
                    }
                )
        with open(self._out, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)
        return True

    def run(self, stop: threading.Event) -> None:
        # Create the file at once: with two calls back to back, the newest live
        # file must be the new meeting's from its first second, not the old one's.
        self._out.touch()
        started = time.time()
        while not stop.is_set():
            if os.getppid() != self._ppid:
                logger.warning("Parent process gone, exiting")
                return
            if self.step():
                continue
            try:
                idle = time.time() - Path(self._wav).stat().st_mtime
                self._seen = True
            except OSError:
                # Before audiocap has created the file, wait for it; once seen, a
                # missing WAV means the recording was discarded or transcribed.
                if self._seen or time.time() - started > ORPHAN_SECONDS:
                    return
                stop.wait(1.0)
                continue
            if idle > ORPHAN_SECONDS:
                logger.warning("WAV idle for %.0f s, exiting", idle)
                return
            stop.wait(1.0)


def _clock(seconds: float) -> str:
    s = int(seconds)
    h, m, s = s // 3600, s % 3600 // 60, s % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def read_live(
    meeting_id: str | None = None,
    last_minutes: float | None = None,
    transcripts_dir: Path = TRANSCRIPTIONS_DIR,
    recordings_dir: Path = RECORDINGS_DIR,
) -> dict:
    """The live transcript of `meeting_id`, or of the most recent recording.

    Both files can be deleted under us at any moment (the final transcription
    deletes the WAV, the 24 h cleanup the live file), so every read tolerates it.
    """
    missing = {"error": "No live transcript: it exists only during a recording (and a day after)."}
    try:
        if meeting_id:
            path = transcripts_dir / f"{meeting_id}_live.jsonl"
            files = [path] if path.exists() else []
        else:
            files = sorted(transcripts_dir.glob("*_live.jsonl"), key=lambda p: p.stat().st_mtime)
        if not files:
            return missing
        path = files[-1]
        text = path.read_text(encoding="utf-8")
    except OSError:
        return missing
    meeting_id = path.name.removesuffix("_live.jsonl")

    rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    rows.sort(key=lambda r: r["start"])
    transcribed_until = max((r["end"] for r in rows), default=0.0)
    if last_minutes is not None:
        rows = [r for r in rows if r["start"] >= transcribed_until - last_minutes * 60]

    result: dict = {
        "meeting_id": meeting_id,
        "transcript": "\n".join(
            f"[{_clock(r['start'])}] {r['speaker']}: {r['text']}" for r in rows
        ),
        "transcribed_until_seconds": round(transcribed_until),
        "recording_active": False,
        "note": "Rough live transcript (small local model): names and numbers may be wrong. "
        "'Remote' is the call audio, 'Davide' the mic. The mic also picks up people "
        "in the room with Davide, and without headphones it echoes Remote lines: "
        "a 'Davide' line that repeats a 'Remote' one a few seconds apart is echo.",
    }
    wav = recordings_dir / f"{meeting_id}.wav"
    try:
        stat = wav.stat()
    except OSError:
        return result  # no WAV: the recording is over and was transcribed (or discarded)
    header = FileAudioSource(str(wav))
    data_start = header._find_data_offset() or WAV_HEADER_BYTES  # None if it vanished: fine
    recorded = (stat.st_size - data_start) / (header.get_sample_rate() * 4)
    result["recording_active"] = time.time() - stat.st_mtime < 10
    result["recorded_seconds"] = round(recorded)
    result["lag_seconds"] = round(max(0.0, recorded - transcribed_until))
    return result


def spawn(wav_path: str) -> subprocess.Popen:
    """Start the worker for `wav_path`. stdout is not inherited: it is the MCP channel."""
    return subprocess.Popen(
        [sys.executable, "-m", "claude_meeting_mcp.live", wav_path],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
    )


def stop(proc: subprocess.Popen | None) -> None:
    """Stop the worker and wait until it has exited (and freed its model)."""
    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="Live transcript of a recording in progress")
    parser.add_argument("wav", help="The WAV audiocap is writing")
    parser.add_argument(
        "--language", help="Language of the call (e.g. en, it). Default: detect on the first 30 s"
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="meeting-live: %(message)s")
    logging.getLogger("claude_meeting_mcp.transcriber").setLevel(logging.WARNING)  # per-chunk noise

    # No SIGTERM handler on purpose: the default kills the process at once, even in
    # the middle of a chunk, so whoever stops it never waits on a transcription. The
    # only state is the append-only JSONL; at most the chunk in flight is lost.
    out = live_path(args.wav)
    out.parent.mkdir(parents=True, exist_ok=True)
    logger.info("%s -> %s", args.wav, out)
    LiveTranscriber(args.wav, out, language=args.language).run(threading.Event())


if __name__ == "__main__":
    main()
