"""Tests for the LiveTranscriber orchestrator (no real audio / whisper)."""

from __future__ import annotations

import numpy as np

from claude_meeting_mcp.live_transcriber import LiveTranscriber


class FakeSource:
    """Yields scripted (left, right) chunks, then (None, None)."""

    def __init__(self, chunks, sr=48000):
        self._chunks = list(chunks)
        self._sr = sr

    def get_new_audio_both(self):
        if self._chunks:
            return self._chunks.pop(0)
        return None, None

    def get_sample_rate(self):
        return self._sr


def loud(n):
    return np.full(n, 0.3, dtype=np.float32)


def test_finalize_merges_channels_into_transcription(tmp_path, monkeypatch):
    sr = 48000
    src = FakeSource([(loud(sr), loud(sr))], sr=sr)
    lt = LiveTranscriber(source=src, output_path=str(tmp_path / "live.md"))

    # Fake the whisper call: left says "left line", right says "right line".
    def fake_transcribe(audio, channel):
        return [{"start": 0.0, "end": 1.0, "text": f"{channel} line"}]

    monkeypatch.setattr(lt, "_transcribe", fake_transcribe)

    lt._tick()  # ingest the one chunk
    t = lt.finalize(remote_speakers="Remote", local_speakers="Local")

    texts = sorted(s.text for s in t.segments)
    assert texts == ["left line", "right line"]
    speakers = {s.speaker for s in t.segments}
    assert speakers == {"Remote", "Local"}
    assert t.speakers == {"left": "Remote", "right": "Local"}


def test_finalize_empty_returns_none(tmp_path, monkeypatch):
    src = FakeSource([], sr=48000)
    lt = LiveTranscriber(source=src, output_path=str(tmp_path / "live.md"))
    monkeypatch.setattr(lt, "_transcribe", lambda audio, channel: [])
    assert lt.finalize("Remote", "Local") is None


def test_live_markdown_has_full_transcript(tmp_path, monkeypatch):
    sr = 48000
    src = FakeSource([(loud(sr), loud(sr))], sr=sr)
    lt = LiveTranscriber(source=src, output_path=str(tmp_path / "live.md"))
    monkeypatch.setattr(
        lt, "_transcribe",
        lambda audio, channel: [{"start": 0.0, "end": 1.0, "text": f"{channel} said hi"}],
    )
    lt._tick()
    lt._write_markdown()
    content = (tmp_path / "live.md").read_text()
    assert "said hi" in content
