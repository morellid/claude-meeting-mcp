"""Cross-process registry of in-flight transcriptions.

Why this exists: the job registry in ``server`` is a plain dict, so each MCP server
process could only see its own work. With one server per Claude session, a status
check on another process's job reported "no job found" - indistinguishable from a
dead run - and nothing stopped a second process transcribing a file already in
flight. Two whisper models then ground the same WAV concurrently and a 46-minute
recording took 2h40m instead of ~20 minutes.

The registry lives on disk so every process sees one queue. Liveness is an
``flock``, not a recorded pid: the kernel drops the lock when its holder dies, so a
crashed or killed owner frees its slot with no stale-claim sweeper to get wrong
(and no pid-reuse ambiguity to resolve).

Two locks with deliberately different jobs:

* one lock **per WAV**, for correctness - the same file is never transcribed twice;
* one **GPU lock** shared by all jobs, for scheduling - different files queue rather
  than thrash. MLX monopolises the Metal GPU, so overlapping transcriptions are
  strictly slower than running them in turn.

Recording is deliberately not gated on any of this. Capture runs in a separate
process and must always be allowed to start, however long the backlog is.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .storage import _get_data_dir

logger = logging.getLogger(__name__)

# Backlog cap. The job on the GPU does not count against it, so this bounds work
# that is waiting, not work in progress.
MAX_QUEUED = 5

QUEUED = "queued"
RUNNING = "running"

# meeting_id -> file descriptor of the claim lock this process holds. The fd must
# stay open for the job's lifetime: closing it releases the lock.
_held: dict[str, int] = {}


def _jobs_dir() -> Path:
    """Registry directory, resolved per call so tests can redirect the data dir."""
    d = _get_data_dir() / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _claim_path(meeting_id: str) -> Path:
    return _jobs_dir() / f"{meeting_id}.json"


def _lock_path(meeting_id: str) -> Path:
    # Separate from the claim file because the claim is rewritten on state change;
    # replacing a file swaps its inode and would silently drop the lock.
    return _jobs_dir() / f"{meeting_id}.lock"


def _is_live(meeting_id: str) -> bool:
    """True if some process still holds this claim.

    Probes with a second descriptor. ``flock`` treats descriptors independently even
    within one process, so this correctly reports our own claims as live without
    ever releasing them.
    """
    path = _lock_path(meeting_id)
    if not path.exists():
        return False
    fd = os.open(path, os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return True  # someone holds it
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _reap(meeting_id: str) -> None:
    """Remove the traces of a claim whose owner is gone."""
    for path in (_claim_path(meeting_id), _lock_path(meeting_id)):
        try:
            path.unlink()
        except OSError:
            pass


@dataclass(frozen=True)
class QueueState:
    active: int
    queued: int
    inconsistent: bool = False

    @property
    def total(self) -> int:
        return self.active + self.queued

    def to_dict(self) -> dict:
        return {"active": self.active, "queued": self.queued, "total": self.total}


@dataclass(frozen=True)
class EnqueueResult:
    accepted: bool
    queue: QueueState
    reason: str | None = None


def snapshot() -> QueueState:
    """Count live claims, reaping any whose owner died."""
    active = queued = 0
    for claim in sorted(_jobs_dir().glob("*.json")):
        meeting_id = claim.stem
        if not _is_live(meeting_id):
            logger.info("Reclaiming slot from dead owner: %s", meeting_id)
            _reap(meeting_id)
            continue
        try:
            state = json.loads(claim.read_text(encoding="utf-8")).get("state")
        except (ValueError, OSError):
            state = QUEUED  # unreadable claim still occupies a slot
        if state == RUNNING:
            active += 1
        else:
            queued += 1

    # Davide's check: if this process is holding slots the registry cannot account
    # for, the registry has been corrupted - surface it rather than report a lie.
    inconsistent = len(_held) > active + queued
    if inconsistent:
        logger.error(
            "Job registry inconsistent: holding %d claim(s) but only %d on disk",
            len(_held),
            active + queued,
        )
    return QueueState(active=active, queued=queued, inconsistent=inconsistent)


def _write_claim(meeting_id: str, file_path: str, state: str) -> None:
    _claim_path(meeting_id).write_text(
        json.dumps(
            {
                "meeting_id": meeting_id,
                "file": file_path,
                "state": state,
                "pid": os.getpid(),
                "since": time.time(),
            }
        ),
        encoding="utf-8",
    )


@contextmanager
def _registry_lock() -> Iterator[None]:
    """Serialise enqueue across processes so the cap cannot be raced past."""
    fd = os.open(_jobs_dir() / ".registry.lock", os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def enqueue(meeting_id: str, file_path: str) -> EnqueueResult:
    """Claim a slot for ``meeting_id``.

    Refuses if this file is already in flight anywhere, or if the backlog is full.
    """
    with _registry_lock():
        state = snapshot()

        if _is_live(meeting_id):
            return EnqueueResult(False, state, reason="already_active")

        if state.queued >= MAX_QUEUED:
            return EnqueueResult(False, state, reason="queue_full")

        _reap(meeting_id)  # clear anything left by a dead owner
        fd = os.open(_lock_path(meeting_id), os.O_RDWR | os.O_CREAT)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            return EnqueueResult(False, state, reason="already_active")

        _held[meeting_id] = fd
        _write_claim(meeting_id, file_path, QUEUED)
        return EnqueueResult(True, snapshot())


def state_of(meeting_id: str) -> str | None:
    """Current state of a claim held by *any* process, or None if there is none.

    This is what lets a status check see work owned by another MCP server instead
    of reporting "no job found" and inviting a duplicate run.
    """
    if not _is_live(meeting_id):
        return None
    try:
        return json.loads(_claim_path(meeting_id).read_text(encoding="utf-8")).get("state")
    except (ValueError, OSError):
        return QUEUED


def mark_running(meeting_id: str) -> None:
    """Promote a claim from waiting to on-the-GPU."""
    claim = _claim_path(meeting_id)
    try:
        data = json.loads(claim.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return
    data["state"] = RUNNING
    data["since"] = time.time()
    claim.write_text(json.dumps(data), encoding="utf-8")


def release(meeting_id: str) -> None:
    """Drop the claim and free its slot."""
    fd = _held.pop(meeting_id, None)
    if fd is not None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
    _reap(meeting_id)


@contextmanager
def gpu_slot() -> Iterator[None]:
    """Serialise the actual transcription across every process on this machine.

    Blocks until the GPU is free. Callers run in a background thread, so waiting
    here never delays a tool call - and never delays a recording.
    """
    fd = os.open(_jobs_dir() / ".gpu.lock", os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        os.close(fd)


def _release_all_for_tests() -> None:
    for meeting_id in list(_held):
        release(meeting_id)
