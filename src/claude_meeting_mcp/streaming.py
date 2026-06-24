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
            start = s["start"] + self.buffer_start_abs
            end = s["end"] + self.buffer_start_abs
            out.append(Hyp(start, end, text))
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
