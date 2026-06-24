"""Tests for transcriber module."""

from unittest.mock import patch

import numpy as np

from claude_meeting_mcp.schemas import Segment, Transcription
from claude_meeting_mcp.transcriber import (
    _get_backend,
    merge_segments,
    split_channels,
)


def test_segment_to_dict():
    seg = Segment(start=0.0, end=1.5, speaker="Bruno", text="Bonjour")
    d = seg.to_dict()
    assert d["speaker"] == "Bruno"
    assert d["start"] == 0.0


def test_transcription_json_roundtrip():
    t = Transcription(
        meeting_id="test",
        date="2026-04-14",
        duration_seconds=60.0,
        segments=[Segment(start=0.0, end=1.5, speaker="Bruno", text="Test")],
    )
    json_str = t.to_json()
    t2 = Transcription.from_json(json_str)
    assert t2.meeting_id == "test"
    assert len(t2.segments) == 1
    assert t2.segments[0].text == "Test"


def test_get_backend_returns_valid():
    backend = _get_backend()
    assert backend in ("mlx", "faster", "remote")


def test_get_backend_remote_override():
    backend = _get_backend(mode_override="remote")
    assert backend == "remote"


def test_merge_segments_ordering():
    left = [
        {"start": 0.0, "end": 2.0, "text": "Hello"},
        {"start": 5.0, "end": 7.0, "text": "How are you"},
    ]
    right = [
        {"start": 1.0, "end": 3.0, "text": "Hi"},
        {"start": 8.0, "end": 10.0, "text": "Good"},
    ]
    merged = merge_segments(left, right, "Alice", "Bob")
    assert len(merged) == 4
    assert merged[0].speaker == "Alice"
    assert merged[0].text == "Hello"
    assert merged[1].speaker == "Bob"
    assert merged[1].text == "Hi"
    assert merged[2].speaker == "Alice"
    assert merged[3].speaker == "Bob"
    # Verify sorted by start time
    for i in range(len(merged) - 1):
        assert merged[i].start <= merged[i + 1].start


def test_merge_segments_empty():
    merged = merge_segments([], [], "A", "B")
    assert merged == []


def test_merge_segments_single_channel():
    left = [{"start": 0.0, "end": 1.0, "text": "Solo"}]
    merged = merge_segments(left, [], "Speaker", "Nobody")
    assert len(merged) == 1
    assert merged[0].speaker == "Speaker"


def test_merge_segments_strips_whitespace():
    left = [{"start": 0.0, "end": 1.0, "text": "  padded text  "}]
    merged = merge_segments(left, [], "A", "B")
    assert merged[0].text == "padded text"


def test_split_channels_stereo(tmp_path):
    import soundfile as sf

    # Create a stereo WAV file
    samplerate = 44100
    duration = 1.0
    samples = int(samplerate * duration)
    left = np.sin(2 * np.pi * 440 * np.linspace(0, duration, samples))
    right = np.sin(2 * np.pi * 880 * np.linspace(0, duration, samples))
    stereo = np.column_stack([left, right])

    wav_path = str(tmp_path / "test_stereo.wav")
    sf.write(wav_path, stereo, samplerate)

    left_ch, r, sr = split_channels(wav_path)
    assert sr == samplerate
    assert len(left_ch) == samples
    assert len(r) == samples
    np.testing.assert_allclose(left_ch, left, atol=1e-4)
    np.testing.assert_allclose(r, right, atol=1e-4)


def test_split_channels_mono(tmp_path):
    import soundfile as sf

    samplerate = 44100
    mono = np.zeros(44100)
    wav_path = str(tmp_path / "test_mono.wav")
    sf.write(wav_path, mono, samplerate)

    left_ch, r, sr = split_channels(wav_path)
    assert sr == samplerate
    # Mono: both channels should be identical
    np.testing.assert_array_equal(left_ch, r)


@patch("claude_meeting_mcp.transcriber._get_backend", return_value="mlx")
def test_transcribe_channel_mlx_dispatch(mock_backend):
    """Verify that MLX backend is dispatched correctly."""
    with patch("claude_meeting_mcp.transcriber._transcribe_mlx") as mock_mlx:
        mock_mlx.return_value = [{"start": 0.0, "end": 1.0, "text": "test"}]
        from claude_meeting_mcp.transcriber import transcribe_channel

        result = transcribe_channel(np.zeros(16000, dtype=np.float32), 16000)
        mock_mlx.assert_called_once()
        assert len(result) == 1
        assert result[0]["text"] == "test"


@patch("claude_meeting_mcp.transcriber._get_backend", return_value="faster")
def test_transcribe_channel_faster_dispatch(mock_backend):
    """Verify that faster-whisper backend is dispatched correctly."""
    with patch("claude_meeting_mcp.transcriber._transcribe_faster") as mock_faster:
        mock_faster.return_value = [{"start": 0.0, "end": 2.0, "text": "test faster"}]
        from claude_meeting_mcp.transcriber import transcribe_channel

        result = transcribe_channel(np.zeros(16000, dtype=np.float32), 16000)
        mock_faster.assert_called_once()
        assert len(result) == 1
        assert result[0]["text"] == "test faster"


def test_live_backend_unavailable_for_remote(monkeypatch):
    import claude_meeting_mcp.transcriber as tr
    monkeypatch.setattr(tr, "_get_backend", lambda: "remote")
    ok, reason = tr.live_backend_available()
    assert ok is False
    assert "remote" in reason.lower()


def test_live_backend_available_for_mlx(monkeypatch):
    import claude_meeting_mcp.transcriber as tr
    monkeypatch.setattr(tr, "_get_backend", lambda: "mlx")
    ok, _ = tr.live_backend_available()
    assert ok is True


def test_live_backend_available_handles_missing_backend(monkeypatch):
    """If no whisper backend is installed, gating must report unavailable, not raise."""
    import claude_meeting_mcp.transcriber as tr

    def boom():
        raise RuntimeError("No whisper backend available.")

    monkeypatch.setattr(tr, "_get_backend", boom)
    ok, reason = tr.live_backend_available()
    assert ok is False
    assert "backend" in reason.lower()


def test_save_transcription_writes_json_and_deletes_wav(tmp_path, monkeypatch):
    import soundfile as sf

    import claude_meeting_mcp.transcriber as tr
    from claude_meeting_mcp.schemas import Segment, Transcription
    from claude_meeting_mcp.transcriber import save_transcription

    # Redirect the binding the function actually uses (imported by value).
    monkeypatch.setattr(tr, "TRANSCRIPTIONS_DIR", tmp_path)
    monkeypatch.setattr(tr, "ensure_dirs", lambda: None)

    wav = tmp_path / "2026-01-01-meeting.wav"
    sf.write(str(wav), np.zeros((10, 2), dtype="float32"), 48000)
    t = Transcription(
        meeting_id="2026-01-01-meeting", date="2026-01-01", duration_seconds=1.0,
        speakers={"left": "Remote", "right": "Local"},
        segments=[Segment(0.0, 1.0, "Remote", "hi")],
    )
    out = save_transcription(t, str(wav))
    assert out.exists()
    assert not wav.exists()  # WAV deleted only after the JSON is saved
