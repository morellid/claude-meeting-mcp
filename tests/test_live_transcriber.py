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


def test_finalize_drains_audio_written_after_stop(tmp_path, monkeypatch):
    """finalize() must pick up audio written after the daemon stopped (the frames
    the capturer flushes during stop), otherwise the meeting's tail is truncated."""
    sr = 48000
    # Chunk 1 is ingested by the tick; chunk 2 is the post-stop flush the drain reads.
    src = FakeSource([(loud(sr), loud(sr)), (loud(sr), loud(sr))], sr=sr)
    lt = LiveTranscriber(source=src, output_path=str(tmp_path / "live.md"))

    def fake_transcribe(audio, channel):
        secs = round(len(audio) / sr, 1)
        return [{"start": 0.0, "end": secs, "text": f"{channel}:{secs}s"}]

    monkeypatch.setattr(lt, "_transcribe", fake_transcribe)

    lt._tick()  # ingests chunk 1 only (held, not committed)
    t = lt.finalize("Remote", "Local")
    assert t is not None
    # The drained tail makes each channel's flushed buffer span ~2s, not 1s.
    assert any("2.0s" in s.text for s in t.segments)


def test_finalize_transcribes_short_residual(tmp_path, monkeypatch):
    """A final utterance shorter than the per-tick 1s guard must still be flushed."""
    sr = 48000
    n = int(0.4 * sr)  # 0.4s total - below the old 1s finalize floor
    src = FakeSource([(loud(n), loud(n))], sr=sr)
    lt = LiveTranscriber(source=src, output_path=str(tmp_path / "live.md"))
    monkeypatch.setattr(
        lt, "_transcribe",
        lambda audio, channel: [{"start": 0.0, "end": 0.4, "text": f"{channel} short"}],
    )
    # No tick ran (meeting ended before the warmup); finalize is the only chance.
    t = lt.finalize("Remote", "Local")
    assert t is not None
    assert any("short" in s.text for s in t.segments)


def test_silent_cap_preserves_tentative(tmp_path):
    """Hitting the silence cap must commit uncommitted speech, not erase it."""
    from claude_meeting_mcp.streaming import Hyp

    sr = 48000
    lt = LiveTranscriber(source=FakeSource([], sr=sr), output_path=str(tmp_path / "live.md"))
    ch = lt._chan["left"]
    ch.tentative_segs = [Hyp(0.0, 1.0, "important words")]

    cap = int(ch.window_max_seconds * sr)
    lt._buf["left"] = np.zeros(cap + sr, dtype=np.float32)  # silence over the cap
    lt._total_samples = cap + sr
    lt._process_channel("left", frontier_abs=(cap + sr) / sr)

    assert [h.text for h in ch.committed] == ["important words"]
    assert ch.tentative_segs == []
    assert len(lt._buf["left"]) == 0


def test_finalize_falls_back_when_worker_still_running(tmp_path):
    """If the daemon could not be joined, finalize() must not race its buffers -
    it returns None so the caller does a clean batch pass instead."""
    from claude_meeting_mcp.streaming import Hyp

    sr = 48000
    lt = LiveTranscriber(source=FakeSource([], sr=sr), output_path=str(tmp_path / "live.md"))

    class AliveThread:
        def is_alive(self):
            return True

    lt._thread = AliveThread()
    lt._chan["left"].committed = [Hyp(0.0, 1.0, "x")]  # even with content already in
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
