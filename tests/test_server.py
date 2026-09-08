"""Tests for server module."""

import sys
import threading
import time
from unittest.mock import MagicMock, patch


def test_audio_status_keys():
    """Verify audio_status returns all expected keys."""
    with (
        patch("claude_meeting_mcp.server.get_capturer") as mock_capturer,
        patch("claude_meeting_mcp.server._get_backend", return_value="mlx"),
        patch("claude_meeting_mcp.server.is_recording", return_value=False),
        patch("claude_meeting_mcp.server.list_recordings", return_value=[]),
    ):
        mock_cap = MagicMock()
        mock_cap.is_available.return_value = True
        mock_capturer.return_value = mock_cap

        from claude_meeting_mcp.server import audio_status

        result = audio_status()
        assert "platform" in result
        assert "audio_capture_available" in result
        assert "audio_capture_backend" in result
        assert "transcription_backend" in result
        assert "transcription_model" in result
        assert "currently_recording" in result
        assert "disk_free_gb" in result
        assert "last_recording" in result
        assert "capabilities" in result
        assert result["platform"] == sys.platform
        assert result["transcription_backend"] == "mlx"


def test_audio_status_capabilities():
    """Verify audio_status returns capabilities list."""
    with (
        patch("claude_meeting_mcp.server.get_capturer") as mock_capturer,
        patch("claude_meeting_mcp.server._get_backend", return_value="mlx"),
        patch("claude_meeting_mcp.server.is_recording", return_value=False),
        patch("claude_meeting_mcp.server.list_recordings", return_value=[]),
    ):
        mock_cap = MagicMock()
        mock_cap.is_available.return_value = True
        mock_capturer.return_value = mock_cap

        from claude_meeting_mcp.server import audio_status

        result = audio_status()
        assert len(result["capabilities"]) >= 3
        assert any("YouTube" in c for c in result["capabilities"])


def test_audio_configure_shows_menu():
    """Verify audio_configure without params returns config menu."""
    from claude_meeting_mcp.server import audio_configure

    result = audio_configure()
    assert "current_config" in result
    assert "available_settings" in result
    assert "wizard_hint" in result
    assert result["current_config"]["transcription"]["model"] == "large-v3-turbo"
    assert len(result["available_settings"]) >= 4


def test_audio_configure_valid_key():
    """Verify audio_configure updates config correctly."""
    from claude_meeting_mcp.server import audio_configure

    with patch("claude_meeting_mcp.server.update_config") as mock_update:
        from claude_meeting_mcp.config import Config

        mock_update.return_value = Config()
        result = audio_configure("transcription.model", "small")
        assert result["status"] == "updated"


def test_audio_configure_invalid_key():
    """Verify audio_configure returns error for invalid key."""
    from claude_meeting_mcp.server import audio_configure

    with patch(
        "claude_meeting_mcp.server.update_config",
        side_effect=ValueError("Unknown config key: bad.key"),
    ):
        result = audio_configure("bad.key", "value")
        assert "error" in result


def test_audio_configure_missing_value():
    """Verify audio_configure with key but no value returns error."""
    from claude_meeting_mcp.server import audio_configure

    result = audio_configure("transcription.model")
    assert "error" in result


def test_onboarding_injected_first_call():
    """Verify onboarding is injected into the first tool result."""
    import claude_meeting_mcp.server as srv

    srv._session_greeted = False
    result = srv._enrich_result({"status": "ok"})
    assert "onboarding" in result
    assert "capabilities" in result["onboarding"]


def test_onboarding_hint_after_first_call():
    """Verify hint (not full onboarding) on subsequent calls."""
    import claude_meeting_mcp.server as srv

    srv._session_greeted = True
    result = srv._enrich_result({"status": "ok"})
    assert "onboarding" not in result
    assert "hint" in result
    assert "audio_configure" in result["hint"]


def test_validate_meeting_id_valid():
    """Valid meeting IDs pass validation."""
    from claude_meeting_mcp.server import _validate_meeting_id

    assert _validate_meeting_id("2026-04-15_14h00_meeting") is None
    assert _validate_meeting_id("test-recording_123") is None


def test_validate_meeting_id_path_traversal():
    """Path traversal attempts are rejected."""
    from claude_meeting_mcp.server import _validate_meeting_id

    assert _validate_meeting_id("../../etc/passwd") is not None
    assert _validate_meeting_id("foo/bar") is not None
    assert _validate_meeting_id("foo\\bar") is not None
    assert _validate_meeting_id("") is not None


def _fake_transcription(meeting_id):
    from claude_meeting_mcp.schemas import Segment, Transcription

    return Transcription(
        meeting_id=meeting_id,
        date=meeting_id[:10],
        duration_seconds=12.3,
        segments=[Segment(start=0.0, end=1.0, speaker="Alice", text="hello")],
    )


def _poll_until_settled(meeting_id, timeout=5.0):
    """Poll audio_transcribe_status until the background job reaches a final state."""
    from claude_meeting_mcp.server import audio_transcribe_status

    deadline = time.monotonic() + timeout
    status = audio_transcribe_status(meeting_id)
    while status.get("status") in ("transcribing", "queued") and time.monotonic() < deadline:
        time.sleep(0.01)
        status = audio_transcribe_status(meeting_id)
    return status


def test_stop_and_transcribe_is_nonblocking_then_done():
    """stop_and_transcribe returns immediately; status reaches done with a preview."""
    meeting_id = "2026-01-02_test-nonblocking"
    with (
        patch(
            "claude_meeting_mcp.server.stop_recording",
            return_value={"file": f"/tmp/{meeting_id}.wav"},
        ),
        patch(
            "claude_meeting_mcp.server.transcribe_meeting",
            return_value=_fake_transcription(meeting_id),
        ) as mock_tm,
    ):
        from claude_meeting_mcp.server import audio_stop_and_transcribe

        result = audio_stop_and_transcribe(local_speakers="Alice", remote_speakers="Bob")
        # Returned right away, before transcription finished.
        assert result["status"] == "transcribing"
        assert result["meeting_id"] == meeting_id

        status = _poll_until_settled(meeting_id)
        assert status["status"] == "done"
        assert status["segment_count"] == 1
        assert status["preview"]
        # Names were forwarded to the transcription call.
        args, _ = mock_tm.call_args
        assert args[1] == "Bob" and args[2] == "Alice"


def test_transcribe_status_reports_error():
    """A failing background transcription surfaces as status 'error'."""
    meeting_id = "2026-01-03_test-error"
    with (
        patch(
            "claude_meeting_mcp.server.stop_recording",
            return_value={"file": f"/tmp/{meeting_id}.wav"},
        ),
        patch(
            "claude_meeting_mcp.server.transcribe_meeting",
            side_effect=RuntimeError("boom"),
        ),
    ):
        from claude_meeting_mcp.server import audio_stop_and_transcribe

        audio_stop_and_transcribe()
        status = _poll_until_settled(meeting_id)
        assert status["status"] == "error"
        assert "boom" in status["error"]


def test_transcribe_status_disk_fallback(tmp_path):
    """With no in-memory job, status falls back to an existing transcript on disk."""
    with patch("claude_meeting_mcp.server.TRANSCRIPTIONS_DIR", tmp_path):
        from claude_meeting_mcp.server import audio_transcribe_status

        # Unknown job, no file -> error.
        assert "error" in audio_transcribe_status("2099-01-01_unknown")

        # Unknown job, valid file present -> done.
        (tmp_path / "2099-01-01_ondisk.json").write_text("{}", encoding="utf-8")
        status = audio_transcribe_status("2099-01-01_ondisk")
        assert status["status"] == "done"

        # Unknown job, truncated/unparseable file -> error, not a false "done".
        (tmp_path / "2099-01-01_partial.json").write_text('{"segments": [', encoding="utf-8")
        status = audio_transcribe_status("2099-01-01_partial")
        assert status["status"] == "error"


def test_transcribe_existing_file_is_nonblocking(tmp_path):
    """audio_transcribe must not block: a long transcription would outlive the
    MCP idle timeout, so it starts a background job and returns immediately."""
    meeting_id = "2026-01-04_test-existing"
    wav = tmp_path / f"{meeting_id}.wav"
    wav.write_bytes(b"RIFF")
    release = threading.Event()

    def slow_transcribe(*args, **kwargs):
        release.wait(timeout=5.0)
        return _fake_transcription(meeting_id)

    with patch("claude_meeting_mcp.server.transcribe_meeting", side_effect=slow_transcribe):
        from claude_meeting_mcp.server import audio_transcribe

        result = audio_transcribe(file_path=str(wav), local_speakers="Alice", remote_speakers="Bob")
        # Returned while transcribe_meeting is still blocked.
        assert result["status"] == "transcribing"
        assert result["meeting_id"] == meeting_id

        release.set()
        status = _poll_until_settled(meeting_id)
        assert status["status"] == "done"
        assert status["segment_count"] == 1


def test_transcribe_existing_file_missing_fails_fast(tmp_path):
    """A bad path is reported by the call itself, not hidden behind a poll."""
    from claude_meeting_mcp.server import audio_transcribe

    result = audio_transcribe(file_path=str(tmp_path / "nope.wav"))
    assert "error" in result


# --- Queue behaviour: never transcribe the same file twice, never block recording ---


def test_same_file_is_not_transcribed_twice(tmp_path):
    """The bug this queue exists for: a second call must join, not duplicate."""
    meeting_id = "2026-01-05_test-dedupe"
    wav = tmp_path / f"{meeting_id}.wav"
    wav.write_bytes(b"RIFF")
    release = threading.Event()

    def slow_transcribe(*args, **kwargs):
        release.wait(timeout=5.0)
        return _fake_transcription(meeting_id)

    with patch(
        "claude_meeting_mcp.server.transcribe_meeting", side_effect=slow_transcribe
    ) as mock_tm:
        from claude_meeting_mcp.server import audio_transcribe

        first = audio_transcribe(file_path=str(wav))
        second = audio_transcribe(file_path=str(wav))

        assert first["status"] == "transcribing"
        assert "not starting a second run" in second["message"]
        assert second["queue"]["total"] == 1, "the duplicate must not take a slot"

        release.set()
        assert _poll_until_settled(meeting_id)["status"] == "done"
        assert mock_tm.call_count == 1, "transcribed twice - the whole point of the queue"


def test_queue_refuses_a_seventh_job(tmp_path):
    """Five waiting plus one running is the ceiling; the next is refused, not dropped."""
    release = threading.Event()

    def slow_transcribe(*args, **kwargs):
        release.wait(timeout=5.0)
        return _fake_transcription("x")

    with patch("claude_meeting_mcp.server.transcribe_meeting", side_effect=slow_transcribe):
        from claude_meeting_mcp import jobs
        from claude_meeting_mcp.server import audio_transcribe

        accepted = []
        for i in range(6):
            wav = tmp_path / f"2026-01-05_job{i}.wav"
            wav.write_bytes(b"RIFF")
            accepted.append(audio_transcribe(file_path=str(wav)))
            if i == 0:
                # Wait for the first job to take the GPU, so the remaining five are
                # unambiguously the backlog rather than racing it.
                deadline = time.monotonic() + 5.0
                while jobs.snapshot().active == 0 and time.monotonic() < deadline:
                    time.sleep(0.01)
                assert jobs.snapshot().active == 1

        overflow_wav = tmp_path / "2026-01-05_overflow.wav"
        overflow_wav.write_bytes(b"RIFF")
        overflow = audio_transcribe(file_path=str(overflow_wav))

        assert all("error" not in r for r in accepted)
        assert "queue is full" in overflow["error"]
        assert overflow["queue"]["queued"] == 5
        release.set()


def test_every_response_reports_the_queue(tmp_path):
    """Davide's rule: the queue length is visible on every call."""
    meeting_id = "2026-01-05_test-visible"
    wav = tmp_path / f"{meeting_id}.wav"
    wav.write_bytes(b"RIFF")

    with patch(
        "claude_meeting_mcp.server.transcribe_meeting",
        return_value=_fake_transcription(meeting_id),
    ):
        from claude_meeting_mcp.server import audio_transcribe, audio_transcribe_status

        assert "queue" in audio_transcribe(file_path=str(wav))
        _poll_until_settled(meeting_id)
        assert "queue" in audio_transcribe_status(meeting_id)
        assert "queue" in audio_transcribe_status("2026-01-05_no-such-meeting")


def test_recording_can_start_while_a_transcription_runs(tmp_path):
    """Recording must never be gated on the transcription backlog."""
    meeting_id = "2026-01-05_test-concurrent"
    wav = tmp_path / f"{meeting_id}.wav"
    wav.write_bytes(b"RIFF")
    release = threading.Event()

    def slow_transcribe(*args, **kwargs):
        release.wait(timeout=5.0)
        return _fake_transcription(meeting_id)

    with (
        patch("claude_meeting_mcp.server.transcribe_meeting", side_effect=slow_transcribe),
        patch(
            "claude_meeting_mcp.server.start_recording",
            return_value={"status": "recording", "file": "/tmp/new.wav"},
        ),
    ):
        from claude_meeting_mcp.server import audio_record_start, audio_transcribe

        assert audio_transcribe(file_path=str(wav))["status"] == "transcribing"

        started = audio_record_start(ctx=MagicMock())
        assert started["status"] == "recording", "transcription blocked a new recording"
        release.set()


def test_status_sees_a_job_owned_by_another_process(tmp_path):
    """A live claim from another server must not read as 'no job found'."""
    from claude_meeting_mcp import jobs
    from claude_meeting_mcp.server import audio_transcribe_status

    jobs.enqueue("2026-01-05_foreign", "/tmp/foreign.wav")
    status = audio_transcribe_status("2026-01-05_foreign")

    assert status["status"] == "queued"
    assert "another session" in status["message"]


def test_a_refused_auto_stop_transcription_is_not_lost(tmp_path):
    """Nobody awaits the cap's job, so a refusal has to be recorded to be findable."""
    meeting_id = "2026-01-05_capped"
    wav = tmp_path / f"{meeting_id}.wav"
    wav.write_bytes(b"RIFF")

    with patch(
        "claude_meeting_mcp.server._start_transcription_job",
        return_value={"error": "Transcription queue is full (5 waiting, max 5)."},
    ):
        from claude_meeting_mcp.server import _transcribe_auto_stopped

        _transcribe_auto_stopped(str(wav))

    from claude_meeting_mcp.server import audio_transcribe_status

    status = audio_transcribe_status(meeting_id)
    assert status["status"] == "error"
    assert "queue is full" in status["error"]
