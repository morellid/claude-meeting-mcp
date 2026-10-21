"""Tests for the Mistral Voxtral backend (one diarized request on a mono mix)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import numpy as np
import pytest
import soundfile as sf

from claude_meeting_mcp.config import Config, validate_config
from claude_meeting_mcp.transcriber import (
    _encode_for_upload,
    _get_backend,
    _mistral_api_key,
    _mistral_segments,
    _mix_to_mono,
    _transcribe_mistral,
    transcribe_meeting,
)

SR = 48000


def _noise(seconds: float, seed: int) -> np.ndarray:
    return (np.random.RandomState(seed).randn(int(SR * seconds)) * 0.05).astype(np.float32)


def test_mistral_is_a_valid_mode():
    config = Config()
    config.transcription.mode = "mistral"
    assert validate_config(config) == []


def test_get_backend_mistral():
    assert _get_backend(mode_override="mistral") == "mistral"


def test_mix_to_mono_averages_two_live_channels():
    left = np.full(10, 0.2, dtype=np.float32)
    right = np.full(10, 0.4, dtype=np.float32)
    assert np.allclose(_mix_to_mono(left, right, False, False), 0.3)


def test_mix_to_mono_keeps_only_the_live_channel_at_full_level():
    """Averaging with a silent channel would halve the level for nothing."""
    left = np.full(10, 0.2, dtype=np.float32)
    right = np.zeros(10, dtype=np.float32)
    assert np.allclose(_mix_to_mono(left, right, False, True), 0.2)
    assert np.allclose(_mix_to_mono(right, left, True, False), 0.2)


def test_encode_for_upload_is_small_16k_opus(tmp_path):
    """A raw 48 kHz WAV is ~5.8 MB/min; the upload must be a small 16 kHz Opus file."""
    path = _encode_for_upload(_noise(60, 1), SR)
    try:
        info = sf.info(path)
        assert info.format == "OGG" and info.subtype == "OPUS"
        assert info.samplerate == 16000 and info.channels == 1
        assert abs(info.duration - 60) < 0.5
        assert Path(path).stat().st_size < 600_000  # ~24 kbps, not 5.8 MB
    finally:
        Path(path).unlink()


def test_mistral_segments_labels_speakers_in_order_of_appearance():
    result = {
        "segments": [
            {"text": " Ciao.", "start": 0.0, "end": 1.0, "speaker_id": "speaker_3"},
            {"text": " Hello.", "start": 1.0, "end": 2.0, "speaker_id": "speaker_1"},
            {"text": " Di nuovo.", "start": 2.0, "end": 3.0, "speaker_id": "speaker_3"},
            {"text": " ", "start": 3.0, "end": 3.1, "speaker_id": "speaker_1"},
        ]
    }
    segments = _mistral_segments(result)
    assert [(s["speaker"], s["text"]) for s in segments] == [
        ("Speaker 1", " Ciao."),
        ("Speaker 2", " Hello."),
        ("Speaker 1", " Di nuovo."),
    ]


def test_mistral_api_key_prefers_env(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "from-env")
    assert _mistral_api_key() == "from-env"


def test_mistral_api_key_falls_back_to_keychain(monkeypatch):
    """MCP servers are spawned without the login shell, so .zshrc exports never reach them."""
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    done = MagicMock(returncode=0, stdout="from-keychain\n")
    with patch("claude_meeting_mcp.transcriber.subprocess.run", return_value=done):
        assert _mistral_api_key() == "from-keychain"


def test_mistral_api_key_missing_raises(monkeypatch):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    missing = MagicMock(returncode=44, stdout="")
    with patch("claude_meeting_mcp.transcriber.subprocess.run", return_value=missing):
        with pytest.raises(RuntimeError, match="MISTRAL_API_KEY"):
            _mistral_api_key()


def test_transcribe_meeting_mistral_sends_one_diarized_request(tmp_path, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    wav = tmp_path / "2026-10-20_12h00_meeting.wav"
    sf.write(wav, np.stack([_noise(2, 2), _noise(2, 3)], axis=1), SR)

    response = MagicMock()
    response.json.return_value = {
        "segments": [
            {"text": " Buongiorno.", "start": 0.0, "end": 1.0, "speaker_id": "speaker_1"},
            {"text": " Morning.", "start": 1.0, "end": 2.0, "speaker_id": "speaker_2"},
        ]
    }
    with (
        patch("claude_meeting_mcp.transcriber._get_backend", return_value="mistral"),
        patch("claude_meeting_mcp.transcriber.httpx.post", return_value=response) as post,
        patch("claude_meeting_mcp.transcriber.TRANSCRIPTIONS_DIR", tmp_path),
        patch("claude_meeting_mcp.transcriber.ensure_dirs"),
    ):
        result = transcribe_meeting(str(wav), "Alice, Remote", "Davide")

    assert post.call_count == 1
    kwargs = post.call_args.kwargs
    assert kwargs["headers"] == {"Authorization": "Bearer k"}
    fields = kwargs["data"]
    assert fields["model"] == "voxtral-mini-latest"
    assert fields["diarize"] == "true"
    assert fields["context_bias"] == ["Alice", "Davide"]  # placeholders dropped
    assert kwargs["files"]["file"][0].endswith(".ogg")

    assert [s.speaker for s in result.segments] == ["Speaker 1", "Speaker 2"]
    assert (tmp_path / "2026-10-20_12h00_meeting.json").exists()
    assert not wav.exists()  # deleted only after a successful transcript


def test_transcribe_meeting_mistral_failure_keeps_wav(tmp_path, monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    wav = tmp_path / "2026-10-20_12h00_meeting.wav"
    sf.write(wav, np.stack([_noise(1, 4), _noise(1, 5)], axis=1), SR)

    response = MagicMock()
    response.raise_for_status.side_effect = RuntimeError("503")
    with (
        patch("claude_meeting_mcp.transcriber._get_backend", return_value="mistral"),
        patch("claude_meeting_mcp.transcriber.httpx.post", return_value=response),
        patch("claude_meeting_mcp.transcriber.TRANSCRIPTIONS_DIR", tmp_path),
        patch("claude_meeting_mcp.transcriber.ensure_dirs"),
    ):
        with pytest.raises(RuntimeError, match="503"):
            transcribe_meeting(str(wav))

    assert wav.exists()


def test_transcribe_mistral_splits_full_names_into_single_word_bias_terms(monkeypatch):
    # The API rejects a context_bias term containing whitespace with a 400
    # (2026-10-21: "Benedek Szulyovszky" lost a 68-minute transcript).
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    response = MagicMock()
    response.json.return_value = {"segments": []}
    with patch("claude_meeting_mcp.transcriber.httpx.post", return_value=response) as post:
        _transcribe_mistral(_noise(1, 6), SR, ["Benedek Szulyovszky", "Davide Morelli", "davide"])

    assert post.call_args.kwargs["data"]["context_bias"] == [
        "Benedek",
        "Szulyovszky",
        "Davide",
        "Morelli",
    ]


def test_transcribe_mistral_error_carries_the_api_message(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    request = httpx.Request("POST", "https://api.mistral.ai/v1/audio/transcriptions")
    response = httpx.Response(400, request=request, text='{"message":"bad bias"}')
    with patch("claude_meeting_mcp.transcriber.httpx.post", return_value=response):
        with pytest.raises(httpx.HTTPStatusError, match="bad bias"):
            _transcribe_mistral(_noise(1, 7), SR, [])


def _ok(segments: list[dict]) -> MagicMock:
    response = MagicMock()
    response.json.return_value = {"segments": segments}
    return response


def test_transcribe_mistral_chunks_long_audio_and_keeps_chunk_speakers_apart(monkeypatch):
    # The plan allows 3600 audio-seconds per minute, so one request over an hour
    # is always a 429 (2026-10-21: a 68-minute meeting).
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    first = _ok([{"text": " A.", "start": 0.0, "end": 0.5, "speaker_id": "speaker_1"}])
    second = _ok([{"text": " B.", "start": 0.0, "end": 0.5, "speaker_id": "speaker_1"}])
    with (
        patch("claude_meeting_mcp.transcriber.MISTRAL_MAX_CHUNK_SECONDS", 1),
        patch("claude_meeting_mcp.transcriber.httpx.post", side_effect=[first, second]) as post,
    ):
        segments = _transcribe_mistral(_noise(1.5, 8), SR, [])

    assert post.call_count == 2
    assert [s["start"] for s in segments] == [0.0, 0.75]  # offset by the chunk start
    # Speaker ids restart in every chunk, so the same id is not the same voice.
    assert [s["speaker"] for s in segments] == ["Speaker 1", "Speaker 2"]


def test_transcribe_mistral_waits_and_retries_on_rate_limit(monkeypatch):
    monkeypatch.setenv("MISTRAL_API_KEY", "k")
    request = httpx.Request("POST", "https://api.mistral.ai/v1/audio/transcriptions")
    limited = httpx.Response(429, request=request, text="rate limited")
    ok = _ok([{"text": " Hi.", "start": 0.0, "end": 0.5, "speaker_id": "speaker_1"}])
    with (
        patch("claude_meeting_mcp.transcriber.httpx.post", side_effect=[limited, ok]) as post,
        patch("claude_meeting_mcp.transcriber.time.sleep") as sleep,
    ):
        segments = _transcribe_mistral(_noise(1, 9), SR, [])

    assert post.call_count == 2
    sleep.assert_called_once()
    assert [s["text"] for s in segments] == [" Hi."]
