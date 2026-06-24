# Live incremental transcription - Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the meeting transcript ready the instant recording stops, by accumulating a committed transcript during recording (dual-channel, append-only), with the existing batch path kept as fallback.

**Architecture:** A daemon thread (`LiveTranscriber`) reads the growing stereo WAV through `FileAudioSource`, runs Whisper each tick on each channel's uncommitted tail, and a pure `StreamingChannel` decides which segments are final (commit boundary). At `stop`, finalize from the accumulated `committed` segments (only the tail needs transcribing). A capability gate disables the live path for backends that cannot stream (remote, CPU-only), transparently falling back to today's record-then-batch behaviour.

**Tech Stack:** Python 3.11+, uv, MLX-Whisper / faster-whisper, numpy, soundfile, pytest. Spec: `docs/superpowers/specs/2026-06-24-live-transcription-design.md`.

---

## File structure

- Create `src/claude_meeting_mcp/streaming.py` - pure `StreamingChannel` commit-boundary logic. No audio, no whisper. The testable core.
- Create `src/claude_meeting_mcp/live_transcriber.py` - `LiveTranscriber` orchestrator: daemon thread, two `StreamingChannel`s, dual-channel source, live markdown, finalize.
- Modify `src/claude_meeting_mcp/live_translator.py` - add `get_new_audio_both()` to `FileAudioSource` (return both channels). Leave `LiveTranslator` untouched so the translation feature keeps working.
- Modify `src/claude_meeting_mcp/transcriber.py` - add `live_backend_available()` (capability gate) and `save_transcription()` (save JSON + delete WAV, shared by live finalize).
- Modify `src/claude_meeting_mcp/config.py` - extend `LiveTranslationConfig` (commit_hold_seconds, window_max_seconds, commit_policy), change live model default to `large-v3-turbo`, round-trip the new keys.
- Modify `src/claude_meeting_mcp/recorder.py` - choose LiveTranscriber vs LiveTranslator vs none at start; finalize at stop; capability gate.
- Modify `src/claude_meeting_mcp/server.py` - `audio_record_start(live=True, live_translate=None)`; `audio_stop_and_transcribe` finalize short-circuit; `audio_live_status` works for the transcriber.
- Create `tests/test_streaming.py`; create `tests/test_live_transcriber.py`; extend `tests/test_config.py`, `tests/test_recorder.py`, `tests/test_server.py` as needed.

Run all tests with: `uv run pytest -q`

---

## Task 1: Config - new live fields + model default

**Files:**
- Modify: `src/claude_meeting_mcp/config.py:68-73` (dataclass), `:132-141` (toml apply), `:213-217` (save), `:277-284` (update_config)
- Test: `tests/test_config.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config.py`:

```python
def test_live_config_new_fields_defaults():
    from claude_meeting_mcp.config import Config
    c = Config()
    assert c.live_translation.model == "large-v3-turbo"
    assert c.live_translation.commit_hold_seconds == 7.0
    assert c.live_translation.window_max_seconds == 30.0
    assert c.live_translation.commit_policy == "hold"


def test_live_config_toml_roundtrip(tmp_path, monkeypatch):
    import claude_meeting_mcp.config as cfg
    monkeypatch.setenv("CLAUDE_MEETING_CONFIG_DIR", str(tmp_path))
    c = cfg.Config()
    c.live_translation.commit_hold_seconds = 5.0
    c.live_translation.window_max_seconds = 20.0
    c.live_translation.commit_policy = "agreement"
    cfg.save_config(c)
    loaded = cfg.load_config()
    assert loaded.live_translation.commit_hold_seconds == 5.0
    assert loaded.live_translation.window_max_seconds == 20.0
    assert loaded.live_translation.commit_policy == "agreement"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py::test_live_config_new_fields_defaults tests/test_config.py::test_live_config_toml_roundtrip -v`
Expected: FAIL (AttributeError: no attribute 'commit_hold_seconds'; default model is 'medium').

- [ ] **Step 3: Edit the dataclass**

Replace `LiveTranslationConfig` (config.py:68-73):

```python
@dataclass
class LiveTranslationConfig:
    target_language: str = "en"  # "en" = Whisper translate, other = MCP Sampling
    model: str = "large-v3-turbo"  # same model as batch path; fast enough live on MLX
    chunk_seconds: float = 3.0  # how often to transcribe (latency)
    window_seconds: float = 15.0  # audio context for Whisper (quality)
    commit_hold_seconds: float = 7.0  # commit segments older than this behind the frontier
    window_max_seconds: float = 30.0  # force-commit cap when speech never pauses
    commit_policy: str = "hold"  # "hold" or "agreement" (LocalAgreement-2)
```

- [ ] **Step 4: Round-trip the new keys**

In `_apply_toml_to_config`, after the `window_seconds` block (config.py:140-141), add:

```python
        if "commit_hold_seconds" in lt:
            config.live_translation.commit_hold_seconds = float(lt["commit_hold_seconds"])
        if "window_max_seconds" in lt:
            config.live_translation.window_max_seconds = float(lt["window_max_seconds"])
        if "commit_policy" in lt:
            config.live_translation.commit_policy = str(lt["commit_policy"])
```

In `save_config`, replace the `[live_translation]` block lines (config.py:213-217) with:

```python
        "[live_translation]",
        f'target_language = "{config.live_translation.target_language}"',
        f'model = "{config.live_translation.model}"',
        f"chunk_seconds = {config.live_translation.chunk_seconds}",
        f"window_seconds = {config.live_translation.window_seconds}",
        f"commit_hold_seconds = {config.live_translation.commit_hold_seconds}",
        f"window_max_seconds = {config.live_translation.window_max_seconds}",
        f'commit_policy = "{config.live_translation.commit_policy}"',
```

In `update_config`, after the `window_seconds` branch (config.py:283-284), add:

```python
        elif section == "live_translation" and field_name == "commit_hold_seconds":
            config.live_translation.commit_hold_seconds = float(value)
        elif section == "live_translation" and field_name == "window_max_seconds":
            config.live_translation.window_max_seconds = float(value)
        elif section == "live_translation" and field_name == "commit_policy":
            config.live_translation.commit_policy = value
```

- [ ] **Step 5: Run tests**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (all config tests).

- [ ] **Step 6: Commit**

```bash
git add src/claude_meeting_mcp/config.py tests/test_config.py
git commit -m "feat(config): live commit-boundary settings + large-v3-turbo live default"
```

---

## Task 2: StreamingChannel - the pure commit-boundary core

**Files:**
- Create: `src/claude_meeting_mcp/streaming.py`
- Test: `tests/test_streaming.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_streaming.py`:

```python
"""Tests for the pure StreamingChannel commit-boundary logic."""

from __future__ import annotations

from claude_meeting_mcp.streaming import StreamingChannel


def seg(start, end, text):
    return {"start": start, "end": end, "text": text}


def test_holds_recent_segments_uncommitted():
    """A segment newer than commit_hold_seconds behind the frontier is not committed."""
    ch = StreamingChannel(commit_hold_seconds=7.0)
    # frontier at 5s; segment ends at 4s -> only 1s behind -> hold
    ch.update([seg(0.0, 4.0, "hello")], frontier_abs=5.0)
    assert ch.committed == []
    assert [h.text for h in ch.tentative_segs] == ["hello"]


def test_commits_aged_segment_with_absolute_times():
    """A segment older than the hold commits, with absolute timestamps preserved."""
    ch = StreamingChannel(commit_hold_seconds=7.0)
    # frontier at 12s; segment ends at 4s -> 8s behind -> commit
    ch.update([seg(0.0, 4.0, "hello")], frontier_abs=12.0)
    assert [(h.start, h.end, h.text) for h in ch.committed] == [(0.0, 4.0, "hello")]
    # buffer_start advanced to end of last committed segment
    assert ch.buffer_start_abs == 4.0


def test_second_tick_times_are_absolute_after_trim():
    """After a commit advances buffer_start, the next tick's relative times are offset."""
    ch = StreamingChannel(commit_hold_seconds=7.0)
    ch.update([seg(0.0, 4.0, "first")], frontier_abs=12.0)
    assert ch.buffer_start_abs == 4.0
    # next tick: buffer now starts at 4.0s; a segment at rel 0..3 is abs 4..7
    ch.update([seg(0.0, 3.0, "second")], frontier_abs=20.0)
    assert [(h.start, h.end, h.text) for h in ch.committed] == [
        (0.0, 4.0, "first"),
        (4.0, 7.0, "second"),
    ]


def test_force_commit_when_buffer_exceeds_window_max():
    """If nothing ages out but the buffer is too long, force-commit the oldest segment."""
    ch = StreamingChannel(commit_hold_seconds=7.0, window_max_seconds=30.0)
    # frontier 35s, buffer_start 0 -> buffer 35s > 30s. Segment ends at 33s
    # (only 2s behind frontier, would normally hold), force-commit fires.
    ch.update([seg(0.0, 33.0, "long talk"), seg(33.0, 34.5, "more")], frontier_abs=35.0)
    assert [h.text for h in ch.committed] == ["long talk"]


def test_agreement_policy_requires_two_matching_ticks():
    """Under 'agreement', a segment commits only if it matched the previous tick."""
    ch = StreamingChannel(commit_hold_seconds=2.0, commit_policy="agreement")
    # First sighting: aged enough but no prior agreement -> hold
    ch.update([seg(0.0, 4.0, "hello world")], frontier_abs=10.0)
    assert ch.committed == []
    # Second tick, identical segment -> agrees -> commit
    ch.update([seg(0.0, 4.0, "hello world")], frontier_abs=11.0)
    assert [h.text for h in ch.committed] == ["hello world"]


def test_flush_commits_everything():
    """flush() commits all remaining segments regardless of age (used at stop)."""
    ch = StreamingChannel(commit_hold_seconds=7.0)
    ch.flush([seg(0.0, 2.0, "tail one"), seg(2.0, 3.0, "tail two")])
    assert [h.text for h in ch.committed] == ["tail one", "tail two"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_streaming.py -v`
Expected: FAIL (ModuleNotFoundError: claude_meeting_mcp.streaming).

- [ ] **Step 3: Implement `streaming.py`**

```python
"""Pure commit-boundary logic for live incremental transcription.

A StreamingChannel turns a sequence of overlapping transcriptions (each of the
current uncommitted audio tail) into a growing, append-only list of committed
segments with absolute timestamps. It contains no audio and no Whisper calls -
the orchestrator feeds it the per-tick transcription and the audio frontier, and
it decides what is final. This makes the hardest logic trivially unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Hyp:
    """A hypothesis segment with absolute timestamps (seconds into the meeting)."""

    start: float
    end: float
    text: str


class StreamingChannel:
    def __init__(
        self,
        commit_hold_seconds: float = 7.0,
        window_max_seconds: float = 30.0,
        commit_policy: str = "hold",
    ) -> None:
        self.commit_hold_seconds = commit_hold_seconds
        self.window_max_seconds = window_max_seconds
        self.commit_policy = commit_policy
        self.buffer_start_abs: float = 0.0
        self.committed: list[Hyp] = []
        self.tentative_segs: list[Hyp] = []
        self._prev: list[Hyp] = []  # previous tick hypothesis (absolute)

    def _to_abs(self, rel_segments: list[dict]) -> list[Hyp]:
        out = []
        for s in rel_segments:
            text = s["text"].strip()
            if not text:
                continue
            out.append(Hyp(s["start"] + self.buffer_start_abs, s["end"] + self.buffer_start_abs, text))
        return out

    def _agrees(self, i: int, seg: Hyp) -> bool:
        return (
            i < len(self._prev)
            and abs(self._prev[i].start - seg.start) < 0.5
            and self._prev[i].text == seg.text
        )

    def _commit_count(self, abs_segs: list[Hyp], frontier_abs: float) -> int:
        n = 0
        for i, s in enumerate(abs_segs):
            aged = s.end <= frontier_abs - self.commit_hold_seconds
            ok = aged and (self._agrees(i, s) if self.commit_policy == "agreement" else True)
            if ok:
                n = i + 1
            else:
                break
        return n

    def update(self, rel_segments: list[dict], frontier_abs: float) -> float:
        """Process one tick. rel_segments are relative to buffer_start_abs.
        Returns the new buffer_start_abs so the caller can trim its audio buffer."""
        abs_segs = self._to_abs(rel_segments)
        commit_idx = self._commit_count(abs_segs, frontier_abs)

        # Force-commit cap: speech never paused, buffer too long -> commit oldest.
        buffer_len = frontier_abs - self.buffer_start_abs
        if commit_idx == 0 and abs_segs and buffer_len > self.window_max_seconds:
            commit_idx = 1

        if commit_idx > 0:
            newly = abs_segs[:commit_idx]
            self.committed.extend(newly)
            self.buffer_start_abs = newly[-1].end

        self.tentative_segs = abs_segs[commit_idx:]
        self._prev = abs_segs
        return self.buffer_start_abs

    def flush(self, rel_segments: list[dict]) -> None:
        """Commit all remaining segments unconditionally (called once at stop)."""
        abs_segs = self._to_abs(rel_segments)
        if abs_segs:
            self.committed.extend(abs_segs)
            self.buffer_start_abs = abs_segs[-1].end
        self.tentative_segs = []
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_streaming.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Commit**

```bash
git add src/claude_meeting_mcp/streaming.py tests/test_streaming.py
git commit -m "feat(streaming): pure StreamingChannel commit-boundary logic"
```

---

## Task 3: Capability gate + save helper in transcriber

**Files:**
- Modify: `src/claude_meeting_mcp/transcriber.py` (add two functions near the top-level helpers)
- Test: `tests/test_transcriber.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_transcriber.py`:

```python
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


def test_save_transcription_writes_json_and_deletes_wav(tmp_path):
    import soundfile as sf
    import numpy as np
    from claude_meeting_mcp.transcriber import save_transcription
    from claude_meeting_mcp.schemas import Segment, Transcription
    import claude_meeting_mcp.storage as storage

    storage.TRANSCRIPTIONS_DIR = tmp_path  # redirect output
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_transcriber.py -k "live_backend or save_transcription" -v`
Expected: FAIL (no attribute `live_backend_available` / `save_transcription`).

- [ ] **Step 3: Implement the helpers**

Add to `transcriber.py` (after `_can_parallelize`, near line 162):

```python
def live_backend_available() -> tuple[bool, str]:
    """Decide whether live incremental transcription can run on this machine.

    Live transcription runs Whisper on-device every few seconds, so it needs a
    fast LOCAL backend. Remote APIs and CPU-only setups fall back to the batch
    path. Returns (available, reason).
    """
    backend = _get_backend()
    if backend == "remote":
        return False, "live transcription needs a local backend; remote mode uses batch at stop"
    if backend == "mlx":
        return True, "mlx"
    if backend == "faster":
        try:
            import ctranslate2

            if ctranslate2.get_cuda_device_count() > 0:
                return True, "faster-whisper (CUDA)"
        except Exception:
            pass
        return False, "faster-whisper on CPU is too slow for real time; using batch at stop"
    return False, "no local whisper backend available"


def save_transcription(transcription, wav_path: str | None):
    """Save a Transcription JSON and delete the source WAV (only after the save).

    Shared by the batch path and the live finalize path so deletion ordering is
    identical: the WAV is removed only once a transcript is safely on disk.
    """
    ensure_dirs()
    output_path = TRANSCRIPTIONS_DIR / f"{transcription.meeting_id}.json"
    output_path.write_text(transcription.to_json(), encoding="utf-8")
    if wav_path:
        try:
            Path(wav_path).unlink()
            logger.info("Deleted WAV after transcription: %s", wav_path)
        except OSError as e:
            logger.warning("Could not delete WAV: %s", e)
    return output_path
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_transcriber.py -k "live_backend or save_transcription" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/claude_meeting_mcp/transcriber.py tests/test_transcriber.py
git commit -m "feat(transcriber): live capability gate + shared save_transcription helper"
```

---

## Task 4: FileAudioSource - read both channels

**Files:**
- Modify: `src/claude_meeting_mcp/live_translator.py:80-118` (add a method; do not change `get_new_audio`)
- Test: `tests/test_live_translator.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_live_translator.py`:

```python
def test_file_audio_source_reads_both_channels(tmp_path):
    """get_new_audio_both returns distinct left and right channels."""
    import soundfile as sf

    wav_path = str(tmp_path / "stereo.wav")
    sr = 48000
    n = sr  # 1 second
    left = np.full(n, 0.5, dtype=np.float32)
    right = np.full(n, -0.5, dtype=np.float32)
    sf.write(wav_path, np.column_stack([left, right]), sr, subtype="PCM_16")

    source = FileAudioSource(wav_path, sample_rate=sr, channels=2)
    l, r = source.get_new_audio_both()
    assert l is not None and r is not None
    assert len(l) == len(r) > 0
    assert l.mean() > 0.4 and r.mean() < -0.4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_live_translator.py::test_file_audio_source_reads_both_channels -v`
Expected: FAIL (no attribute `get_new_audio_both`).

- [ ] **Step 3: Add the method**

In `live_translator.py`, inside `FileAudioSource`, add after `get_new_audio` (after line 118):

```python
    def get_new_audio_both(self) -> tuple[np.ndarray, np.ndarray] | tuple[None, None]:
        """Return (left, right) new samples since last call, or (None, None).

        Reuses the same byte-offset bookkeeping as get_new_audio but keeps both
        channels. Mono input is duplicated so callers always get two channels.
        """
        if not os.path.exists(self._path):
            return None, None
        if self._data_offset is None:
            self._data_offset = self._find_data_offset()
            if self._data_offset is None:
                return None, None
            self._read_offset = 0
        try:
            file_size = os.path.getsize(self._path)
            available = file_size - self._data_offset - self._read_offset
            available = (available // self._frame_size) * self._frame_size
            if available <= 0:
                return None, None
            max_bytes = self._sample_rate * 5 * self._frame_size
            read_size = min(available, max_bytes)
            with open(self._path, "rb") as f:
                f.seek(self._data_offset + self._read_offset)
                raw = f.read(read_size)
            self._read_offset += len(raw)
            samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if self._channels > 1:
                frames = samples.reshape(-1, self._channels)
                return np.ascontiguousarray(frames[:, 0]), np.ascontiguousarray(frames[:, 1])
            return samples, samples.copy()
        except OSError:
            return None, None
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_live_translator.py -v`
Expected: PASS (existing tests + new one).

- [ ] **Step 5: Commit**

```bash
git add src/claude_meeting_mcp/live_translator.py tests/test_live_translator.py
git commit -m "feat(live): FileAudioSource.get_new_audio_both for dual-channel reads"
```

---

## Task 5: LiveTranscriber orchestrator

**Files:**
- Create: `src/claude_meeting_mcp/live_transcriber.py`
- Test: `tests/test_live_transcriber.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_live_transcriber.py`:

```python
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

    # Fake the whisper call: left says "remote line", right says "local line".
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_live_transcriber.py -v`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Implement `live_transcriber.py`**

```python
"""Live incremental dual-channel transcription during recording.

Reads the growing stereo WAV, transcribes each channel's uncommitted tail every
tick, commits finalized segments via StreamingChannel, and writes a live markdown
file holding the full running transcript. At stop, finalize() merges both
channels into a Transcription with no batch pass.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

from .config import get_config
from .schemas import Segment, Transcription
from .streaming import StreamingChannel

logger = logging.getLogger(__name__)

_SILENCE_RMS = 0.001


class LiveTranscriber:
    def __init__(self, source, output_path: str, remote_label: str = "Remote", local_label: str = "Local") -> None:
        cfg = get_config().live_translation
        self._source = source
        self._sr = source.get_sample_rate()
        self._output_path = Path(output_path)
        self._model = cfg.model
        self._chunk_seconds = cfg.chunk_seconds

        def _mk_channel() -> StreamingChannel:
            return StreamingChannel(
                commit_hold_seconds=cfg.commit_hold_seconds,
                window_max_seconds=cfg.window_max_seconds,
                commit_policy=cfg.commit_policy,
            )

        self._chan = {"left": _mk_channel(), "right": _mk_channel()}
        self._buf = {"left": np.empty(0, np.float32), "right": np.empty(0, np.float32)}
        self._buf_start_sample = {"left": 0, "right": 0}
        self._total_samples = 0

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0
        self._lock = threading.Lock()

    # --- lifecycle ---

    def start(self) -> None:
        self._start_time = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._write_markdown()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def _run(self) -> None:
        time.sleep(2.0)  # let audio start flowing
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            self._tick()
            self._write_markdown()
            elapsed = time.monotonic() - t0
            # Self-pacing: only sleep if the tick was faster than the cadence.
            if elapsed < self._chunk_seconds:
                self._stop_event.wait(timeout=self._chunk_seconds - elapsed)

    # --- core ---

    def _tick(self) -> None:
        left, right = self._source.get_new_audio_both()
        if left is not None and len(left) > 0:
            self._total_samples += len(left)
            self._buf["left"] = np.concatenate([self._buf["left"], left])
            self._buf["right"] = np.concatenate([self._buf["right"], right])
        frontier_abs = self._total_samples / self._sr
        for channel in ("left", "right"):
            self._process_channel(channel, frontier_abs)

    def _process_channel(self, channel: str, frontier_abs: float) -> None:
        buf = self._buf[channel]
        if len(buf) < self._sr:  # less than 1s of audio
            return
        rms = float(np.sqrt(np.mean(buf**2)))
        if rms < _SILENCE_RMS:
            # Idle channel: drop silence so the buffer does not grow unbounded.
            cap = int(self._chan[channel].window_max_seconds * self._sr)
            if len(buf) > cap:
                self._buf[channel] = np.empty(0, np.float32)
                self._buf_start_sample[channel] = self._total_samples
                self._chan[channel].buffer_start_abs = frontier_abs
            return
        rel_segments = self._transcribe(buf, channel)
        new_start_abs = self._chan[channel].update(rel_segments, frontier_abs)
        self._trim(channel, new_start_abs)

    def _trim(self, channel: str, new_start_abs: float) -> None:
        new_start_sample = round(new_start_abs * self._sr)
        drop = new_start_sample - self._buf_start_sample[channel]
        if drop > 0:
            self._buf[channel] = self._buf[channel][drop:]
            self._buf_start_sample[channel] = new_start_sample

    def _transcribe(self, audio: np.ndarray, channel: str) -> list[dict]:
        """Run Whisper on a channel buffer; return segments relative to buffer start."""
        from .config import get_config
        from .transcriber import _resample_to_16k, _transcribe_mlx, _transcribe_faster, _get_backend

        cfg = get_config()
        audio16 = _resample_to_16k(audio, self._sr)
        backend = _get_backend()
        try:
            if backend == "mlx":
                segs = _transcribe_mlx(audio16, 16000, self._model)
            else:
                segs = _transcribe_faster(audio16, 16000, self._model)
        except Exception as e:  # one bad tick must not kill the thread
            logger.error("Live transcribe error (%s): %s", channel, e)
            return []
        return [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segs]

    # --- finalize ---

    def finalize(self, remote_speakers: str, local_speakers: str) -> Transcription | None:
        """Flush remaining tails and merge both channels. None if nothing committed."""
        frontier_abs = self._total_samples / self._sr
        for channel in ("left", "right"):
            buf = self._buf[channel]
            if len(buf) >= self._sr:
                rms = float(np.sqrt(np.mean(buf**2)))
                if rms >= _SILENCE_RMS:
                    self._chan[channel].flush(self._transcribe(buf, channel))

        remote_label = (remote_speakers or "Remote").split(",")[0].strip()
        local_label = (local_speakers or "Local").split(",")[0].strip()
        labels = {"left": remote_label, "right": local_label}

        segments: list[Segment] = []
        for channel in ("left", "right"):
            for h in self._chan[channel].committed:
                segments.append(Segment(round(h.start, 2), round(h.end, 2), labels[channel], h.text))
        if not segments:
            return None
        segments.sort(key=lambda s: s.start)

        meeting_id = self._output_path.stem.replace("_live", "")
        return Transcription(
            meeting_id=meeting_id,
            date=meeting_id[:10] if len(meeting_id) >= 10 else "",
            duration_seconds=round(frontier_abs, 1),
            speakers={"left": remote_label, "right": local_label},
            segments=segments,
        )

    # --- status + markdown ---

    def get_status(self) -> dict:
        with self._lock:
            committed = self._chan["left"].committed + self._chan["right"].committed
            committed.sort(key=lambda h: h.start)
            return {
                "status": "stopped" if self._stop_event.is_set() else "transcribing",
                "elapsed_seconds": round(time.monotonic() - self._start_time, 1),
                "model": self._model,
                "committed_segments": len(committed),
                "latest_text": [h.text for h in committed[-5:]],
                "live_file": str(self._output_path),
            }

    def _write_markdown(self, final: bool = False) -> None:
        elapsed = time.monotonic() - self._start_time
        status = "Completed" if final else f"Recording ({int(elapsed // 60)}m {int(elapsed % 60):02d}s)"
        committed = self._chan["left"].committed + self._chan["right"].committed
        committed.sort(key=lambda h: h.start)
        lines = ["# Live Transcript", f"**Status**: {status}", "", "---", ""]
        for h in committed:
            lines.append(f"[{int(h.start // 60)}:{int(h.start % 60):02d}] {h.text}")
        content = "\n".join(lines) + "\n"
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._output_path.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.rename(self._output_path)
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_live_transcriber.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/claude_meeting_mcp/live_transcriber.py tests/test_live_transcriber.py
git commit -m "feat(live): LiveTranscriber dual-channel accumulator + finalize"
```

---

## Task 6: Wire LiveTranscriber into the recorder

**Files:**
- Modify: `src/claude_meeting_mcp/recorder.py` (start/stop/status + new finalize function)
- Test: `tests/test_recorder.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_recorder.py`:

```python
def test_start_recording_gates_live_off_for_remote(monkeypatch, tmp_path):
    import claude_meeting_mcp.recorder as rec

    class FakeCap:
        def is_available(self): return True
        def start(self, path): open(path, "wb").close()
        def stop(self): pass

    monkeypatch.setattr(rec, "get_capturer", lambda: FakeCap())
    monkeypatch.setattr(rec, "live_backend_available", lambda: (False, "remote"))
    result = rec.start_recording(live=True)
    assert result["status"] == "recording"
    assert result.get("live") is False
    assert "remote" in result.get("live_reason", "")
    rec.stop_recording()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_recorder.py::test_start_recording_gates_live_off_for_remote -v`
Expected: FAIL (start_recording has no `live` param / no gating).

- [ ] **Step 3: Edit `recorder.py`**

At the top, add import (recorder.py:8 area):

```python
from .transcriber import live_backend_available
```

Add a module global next to the others (recorder.py:19):

```python
_live_transcriber = None
```

Replace the signature and body of `start_recording` (recorder.py:28-100). Keep the existing capture-start block; replace the live section. New signature and live block:

```python
def start_recording(
    live: bool = True,
    live_translate: str | None = None,
    mcp_context=None,
    event_loop=None,
) -> dict:
    """Start recording system audio + microphone.

    Args:
        live: Accumulate a live transcript during recording (default on). Auto
            gated off when no fast local backend is available.
        live_translate: Optional target language for live TRANSLATION (legacy
            path via LiveTranslator). When set, takes precedence over `live`.
    """
    global _capturer, _current_file, _timeout_timer, _live_translator, _live_transcriber
    # ... (unchanged: lock, capturer availability, ensure_dirs, filename,
    #      capturer.start, _capturer, timeout timer, result dict) ...
```

Replace the `if live_translate:` block (recorder.py:67-97) with:

```python
        if live_translate:
            # Legacy live-translation path (unchanged behaviour).
            try:
                from .config import get_config
                from .live_translator import FileAudioSource, LiveTranslator

                config = get_config()
                meeting_id = filepath.stem
                live_file = str(TRANSCRIPTIONS_DIR / f"{meeting_id}_live.md")
                source = FileAudioSource(_current_file, sample_rate=config.recording.sample_rate)
                _live_translator = LiveTranslator(
                    source=source,
                    output_path=live_file,
                    target_language=live_translate,
                    model=config.live_translation.model,
                    chunk_seconds=config.live_translation.chunk_seconds,
                    window_seconds=config.live_translation.window_seconds,
                    mcp_context=mcp_context,
                    event_loop=event_loop,
                )
                _live_translator.start()
                result["live_translation"] = True
                result["live_file"] = live_file
            except Exception as e:
                logger.error("Failed to start live translation: %s", e)
                result["live_translation_error"] = str(e)
        elif live:
            available, reason = live_backend_available()
            if available:
                try:
                    from .config import get_config
                    from .live_transcriber import LiveTranscriber
                    from .live_translator import FileAudioSource

                    config = get_config()
                    meeting_id = filepath.stem
                    live_file = str(TRANSCRIPTIONS_DIR / f"{meeting_id}_live.md")
                    source = FileAudioSource(_current_file, sample_rate=config.recording.sample_rate)
                    _live_transcriber = LiveTranscriber(source=source, output_path=live_file)
                    _live_transcriber.start()
                    result["live"] = True
                    result["live_file"] = live_file
                except Exception as e:
                    logger.error("Failed to start live transcriber: %s", e)
                    result["live"] = False
                    result["live_error"] = str(e)
            else:
                result["live"] = False
                result["live_reason"] = reason

        logger.info("Recording started: %s", _current_file)
        return result
```

In `stop_recording` (recorder.py:103-137), after stopping the live translator block, also stop the transcriber. Replace the live-translator stop block with:

```python
        if _live_translator is not None:
            try:
                _live_translator.stop()
            except Exception as e:
                logger.error("Error stopping live translator: %s", e)
            _live_translator = None
        if _live_transcriber is not None:
            try:
                _live_transcriber.stop()  # stop thread only; do NOT finalize/delete here
            except Exception as e:
                logger.error("Error stopping live transcriber: %s", e)
            # keep the reference so stop_and_finalize() can use it
```

Add a new function after `stop_recording`:

```python
def finalize_live(remote_speakers: str, local_speakers: str):
    """Finalize the active live transcriber into a saved Transcription.

    Returns the Transcription, or None if there was no live transcriber or it
    produced nothing (caller then falls back to the batch path).
    """
    global _live_transcriber
    transcriber = _live_transcriber
    _live_transcriber = None
    if transcriber is None:
        return None
    return transcriber.finalize(remote_speakers, local_speakers)
```

Update `get_live_status` (recorder.py:140-144):

```python
def get_live_status() -> dict | None:
    if _live_transcriber is not None:
        return _live_transcriber.get_status()
    if _live_translator is not None:
        return _live_translator.get_status()
    return None
```

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_recorder.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/claude_meeting_mcp/recorder.py tests/test_recorder.py
git commit -m "feat(recorder): wire LiveTranscriber with capability gate; keep translator path"
```

---

## Task 7: Server tools - finalize short-circuit + live flag

**Files:**
- Modify: `src/claude_meeting_mcp/server.py:146-205` (record_start, record_stop, live_status), `:258-289` (stop_and_transcribe)
- Test: `tests/test_server.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_server.py`:

```python
def test_stop_and_transcribe_uses_live_finalize(monkeypatch):
    import claude_meeting_mcp.server as srv
    from claude_meeting_mcp.schemas import Segment, Transcription

    monkeypatch.setattr(srv, "stop_recording", lambda: {"status": "stopped", "file": "/x/2026-01-01-m.wav"})

    t = Transcription(
        meeting_id="2026-01-01-m", date="2026-01-01", duration_seconds=2.0,
        speakers={"left": "Remote", "right": "Local"},
        segments=[Segment(0.0, 1.0, "Remote", "hello")],
    )
    monkeypatch.setattr(srv, "finalize_live", lambda r, l: t)
    monkeypatch.setattr(srv, "save_transcription", lambda tr, wav: srv.TRANSCRIPTIONS_DIR / f"{tr.meeting_id}.json")

    called = {"batch": False}
    monkeypatch.setattr(srv, "transcribe_meeting", lambda *a, **k: called.__setitem__("batch", True) or t)

    result = srv.audio_stop_and_transcribe()
    assert called["batch"] is False  # live finalize used, batch NOT called
    assert result["meeting_id"] == "2026-01-01-m"
    assert result["segment_count"] == 1


def test_stop_and_transcribe_falls_back_to_batch_when_live_empty(monkeypatch):
    import claude_meeting_mcp.server as srv
    from claude_meeting_mcp.schemas import Segment, Transcription

    monkeypatch.setattr(srv, "stop_recording", lambda: {"status": "stopped", "file": "/x/2026-01-01-m.wav"})
    monkeypatch.setattr(srv, "finalize_live", lambda r, l: None)  # live produced nothing
    t = Transcription(meeting_id="2026-01-01-m", date="2026-01-01", duration_seconds=2.0,
                      segments=[Segment(0.0, 1.0, "Remote", "batch")])
    called = {"batch": False}

    def fake_batch(*a, **k):
        called["batch"] = True
        return t

    monkeypatch.setattr(srv, "transcribe_meeting", fake_batch)
    result = srv.audio_stop_and_transcribe()
    assert called["batch"] is True
    assert result["segment_count"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_server.py -k stop_and_transcribe -v`
Expected: FAIL (finalize_live / save_transcription not imported; batch always used).

- [ ] **Step 3: Edit `server.py`**

Update imports (server.py:16, 26):

```python
from .recorder import finalize_live, get_live_status, is_recording, start_recording, stop_recording
from .transcriber import _get_backend, save_transcription, transcribe_meeting
```

Replace `audio_record_start` param + call (server.py:147-176). Add a `live` param and keep `live_translate`:

```python
@mcp.tool()
def audio_record_start(
    ctx: Context,
    live: Annotated[
        bool,
        Field(description="Accumulate a live transcript during recording (default true). Auto-disabled on backends that cannot stream."),
    ] = True,
    live_translate: Annotated[
        str | None,
        Field(description="Optional: live TRANSLATION to this language code (e.g. 'fr'). Overrides live transcription."),
    ] = None,
) -> dict:
    """Use this when the user wants to record or extract audio from the computer.

    Stereo WAV: left = system audio, right = microphone. By default a live
    transcript accumulates during recording and is ready instantly at stop.
    """
    import asyncio

    loop = asyncio.get_event_loop()
    result = start_recording(live=live, live_translate=live_translate, mcp_context=ctx, event_loop=loop)
    if "error" not in result:
        result["next_step"] = "When done, call audio_stop_and_transcribe()"
        if result.get("live") or result.get("live_translation"):
            result["next_step"] += " | Check progress: audio_live_status()"
    return _enrich_result(result)
```

Replace the body of `audio_stop_and_transcribe` (server.py:284-289):

```python
    stop_result = stop_recording()
    if "error" in stop_result:
        return stop_result
    file_path = stop_result["file"]

    local = local_speakers or "Local"
    remote = remote_speakers or "Remote"

    # If a model override is requested, honour it via the batch path (the live
    # transcript was produced with the live-config model).
    if model is None:
        transcription = finalize_live(remote, local)
        if transcription is not None:
            save_transcription(transcription, file_path)
            return _enrich_result(
                {
                    "meeting_id": transcription.meeting_id,
                    "duration_seconds": transcription.duration_seconds,
                    "segment_count": len(transcription.segments),
                    "output_file": str(TRANSCRIPTIONS_DIR / f"{transcription.meeting_id}.json"),
                    "preview": [s.to_dict() for s in transcription.segments[:10]],
                    "next_step": f"Generate meeting minutes: audio_generate_pv('{transcription.meeting_id}')",
                }
            )
    # No live transcript (gated off, empty, or model override): batch path.
    return audio_transcribe(file_path, local, remote, model)
```

Update `audio_live_status` docstring/error text (server.py:194-204) to mention transcription, and `audio_record_stop` is unchanged (it calls `stop_recording`, which stops the thread without finalizing - correct).

- [ ] **Step 4: Run tests**

Run: `uv run pytest tests/test_server.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/claude_meeting_mcp/server.py tests/test_server.py
git commit -m "feat(server): live finalize short-circuit at stop; batch fallback + model override"
```

---

## Task 8: Full suite + changelog

**Files:**
- Modify: `CHANGELOG.md`
- Test: whole suite

- [ ] **Step 1: Run the full suite**

Run: `uv run pytest -q`
Expected: PASS (all tests, including the pre-existing ones).

- [ ] **Step 2: Add a CHANGELOG entry**

Prepend under the top heading of `CHANGELOG.md`:

```markdown
## Unreleased

- feat: live incremental transcription - the transcript is ready the instant you
  stop recording (dual-channel, append-only commit boundary). Auto-falls back to
  batch transcription on backends that cannot stream (remote, CPU-only).
```

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md
git commit -m "docs: changelog for live incremental transcription"
```

- [ ] **Step 4: Push the branch**

```bash
git push -u origin feat/live-transcription
```

---

## Self-review notes

- **Spec coverage:** dual-channel accumulator (Task 5), commit boundary hold+agreement+force-commit (Task 2), finalize at stop (Task 5/7), fallback ordering + WAV deletion only after save (Task 3/7), capability gating remote/CPU (Task 3/6), full live transcript in markdown (Task 5), config knobs (Task 1), translation path preserved (Task 6), edge cases: record_stop without finalize (Task 6 stop_recording), model override -> batch (Task 7). All covered.
- **Manual check after implementation:** record a real 2-3 minute meeting, confirm `audio_live_status()` shows growing committed segments, confirm `audio_stop_and_transcribe()` returns in under ~1s, and eyeball transcript quality vs the old batch output to decide whether Approach C (background re-pass) is ever needed.
