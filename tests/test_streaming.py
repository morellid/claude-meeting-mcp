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
