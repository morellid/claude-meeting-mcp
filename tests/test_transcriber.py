"""Tests for transcriber module."""

from unittest.mock import patch

import numpy as np

from claude_meeting_mcp.schemas import Segment, Transcription
from claude_meeting_mcp.transcriber import (
    _channel_is_silent,
    _get_backend,
    merge_segments,
    split_channels,
    transcribe_meeting,
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


def test_channel_is_silent():
    rng = np.random.RandomState(0)
    assert _channel_is_silent(np.zeros(16000, dtype=np.float32))
    assert _channel_is_silent(np.array([], dtype=np.float32))
    # Noise floor well below the threshold counts as silent.
    assert _channel_is_silent((rng.randn(16000) * 1e-4).astype(np.float32))
    # Speech-level energy (~-26 dBFS) is not silent.
    assert not _channel_is_silent((rng.randn(16000) * 0.05).astype(np.float32))


def test_transcribe_meeting_skips_silent_channel(tmp_path):
    """A silent channel is not sent to transcription; the other one still is."""
    sr = 16000
    rng = np.random.RandomState(1)
    left = (rng.randn(sr) * 0.05).astype(np.float32)  # speech-level
    right = np.zeros(sr, dtype=np.float32)  # silent
    calls = []

    def fake_transcribe_channel(audio, samplerate, *args, **kwargs):
        calls.append(len(audio))
        return [{"start": 0.0, "end": 1.0, "text": "hello"}]

    with (
        patch("claude_meeting_mcp.transcriber.split_channels", return_value=(left, right, sr)),
        patch("claude_meeting_mcp.transcriber._get_backend", return_value="mlx"),
        patch("claude_meeting_mcp.transcriber._can_parallelize", return_value=False),
        patch(
            "claude_meeting_mcp.transcriber.transcribe_channel",
            side_effect=fake_transcribe_channel,
        ),
        patch("claude_meeting_mcp.transcriber.TRANSCRIPTIONS_DIR", tmp_path),
        patch("claude_meeting_mcp.transcriber.ensure_dirs"),
    ):
        result = transcribe_meeting("/tmp/2026-01-01_meeting.wav", "Remote", "Local")

    # Only the non-silent (left) channel was transcribed.
    assert len(calls) == 1
    assert any(s.text == "hello" for s in result.segments)
