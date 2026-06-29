"""Tests for server module."""

import sys
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
    """Poll audio_transcribe_status until the background job leaves 'transcribing'."""
    from claude_meeting_mcp.server import audio_transcribe_status

    deadline = time.monotonic() + timeout
    status = audio_transcribe_status(meeting_id)
    while status.get("status") == "transcribing" and time.monotonic() < deadline:
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

        # Unknown job, file present -> done.
        (tmp_path / "2099-01-01_ondisk.json").write_text("{}", encoding="utf-8")
        status = audio_transcribe_status("2099-01-01_ondisk")
        assert status["status"] == "done"
