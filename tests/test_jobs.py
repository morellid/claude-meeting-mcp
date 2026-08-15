"""Tests for the cross-process transcription job queue.

These cover the failure that motivated the queue: two processes transcribing the
same WAV at once, with neither able to see the other's work.
"""

import json
import subprocess
import sys

import pytest

from claude_meeting_mcp import jobs


@pytest.fixture(autouse=True)
def isolated_jobs_dir(tmp_path, monkeypatch):
    """Point the job registry at a scratch dir and drop any locks between tests."""
    monkeypatch.setenv("CLAUDE_MEETING_DATA_DIR", str(tmp_path))
    yield
    jobs._release_all_for_tests()


def test_first_enqueue_is_accepted_and_reports_an_empty_backlog():
    r = jobs.enqueue("m1", "/tmp/m1.wav")
    assert r.accepted
    assert r.reason is None
    # Nothing ahead of it: it is the one job in the system, not yet running.
    assert r.queue.queued == 1
    assert r.queue.active == 0
    assert r.queue.total == 1


def test_same_meeting_is_never_enqueued_twice():
    """The original bug: a second process starts work already in flight."""
    first = jobs.enqueue("m1", "/tmp/m1.wav")
    assert first.accepted

    second = jobs.enqueue("m1", "/tmp/m1.wav")
    assert not second.accepted
    assert second.reason == "already_active"
    assert second.queue.total == 1, "the duplicate must not create a second slot"


def test_queue_refuses_once_five_are_waiting():
    for i in range(5):
        assert jobs.enqueue(f"m{i}", f"/tmp/m{i}.wav").accepted

    sixth = jobs.enqueue("m5", "/tmp/m5.wav")
    assert not sixth.accepted
    assert sixth.reason == "queue_full"
    assert sixth.queue.queued == 5
    assert sixth.queue.total == 5


def test_running_job_does_not_count_against_the_waiting_cap():
    """Cap is on the backlog, so the job on the GPU never blocks a new one."""
    jobs.enqueue("running", "/tmp/running.wav")
    jobs.mark_running("running")

    for i in range(5):
        assert jobs.enqueue(f"m{i}", f"/tmp/m{i}.wav").accepted

    snap = jobs.snapshot()
    assert snap.active == 1
    assert snap.queued == 5
    assert not jobs.enqueue("overflow", "/tmp/overflow.wav").accepted


def test_release_frees_a_slot():
    for i in range(5):
        jobs.enqueue(f"m{i}", f"/tmp/m{i}.wav")
    assert not jobs.enqueue("blocked", "/tmp/blocked.wav").accepted

    jobs.release("m0")
    assert jobs.snapshot().queued == 4
    assert jobs.enqueue("now-fits", "/tmp/now-fits.wav").accepted


def test_claim_from_a_dead_process_is_reclaimed(tmp_path):
    """A killed owner must not wedge the queue - this is why liveness is an flock."""
    # A real child process takes a claim, then dies without releasing it.
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os;"
                f"os.environ['CLAUDE_MEETING_DATA_DIR']={str(tmp_path)!r};"
                "from claude_meeting_mcp import jobs;"
                "jobs.enqueue('orphan', '/tmp/orphan.wav');"
                "os._exit(0)"
            ),
        ],
        capture_output=True,
    )
    assert child.returncode == 0, child.stderr.decode()

    # The claim file is still on disk...
    assert (jobs._jobs_dir() / "orphan.json").exists()
    # ...but its owner is gone, so it must not be counted or block a retake.
    assert jobs.snapshot().total == 0
    assert jobs.enqueue("orphan", "/tmp/orphan.wav").accepted


def test_snapshot_distinguishes_active_from_waiting():
    jobs.enqueue("a", "/tmp/a.wav")
    jobs.enqueue("b", "/tmp/b.wav")
    jobs.mark_running("a")

    snap = jobs.snapshot()
    assert snap.active == 1
    assert snap.queued == 1
    assert snap.total == 2
    assert snap.to_dict() == {"active": 1, "queued": 1, "total": 2}


def test_snapshot_flags_the_inconsistent_state():
    """Davide's rule: a job in flight with an empty queue means something broke."""
    jobs.enqueue("a", "/tmp/a.wav")
    # Corrupt the registry the way a partial write or stray delete would.
    (jobs._jobs_dir() / "a.json").unlink()

    snap = jobs.snapshot()
    assert snap.inconsistent, "held a slot but the registry shows nothing"


def test_state_survives_across_processes():
    """Two MCP servers must see one queue, not two."""
    jobs.enqueue("mine", "/tmp/mine.wav")

    peer = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import os, json;"
                f"os.environ['CLAUDE_MEETING_DATA_DIR']={str(jobs._jobs_dir().parent)!r};"
                "from claude_meeting_mcp import jobs;"
                "print(json.dumps(jobs.snapshot().to_dict()))"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert peer.returncode == 0, peer.stderr
    assert json.loads(peer.stdout.strip())["total"] == 1
