"""Tests for recorder module."""

from unittest.mock import MagicMock, patch

from claude_meeting_mcp.config import Config
from claude_meeting_mcp.recorder import (
    _auto_stop,
    is_recording,
    set_auto_stop_handler,
    start_recording,
    stop_recording,
)


@patch("claude_meeting_mcp.recorder.get_capturer")
@patch("claude_meeting_mcp.recorder._capturer", None)
@patch("claude_meeting_mcp.recorder._current_file", None)
def test_start_stop_lifecycle(mock_get, tmp_path):
    mock_capturer = MagicMock()
    mock_capturer.is_available.return_value = True
    mock_get.return_value = mock_capturer

    result = start_recording()
    assert result["status"] == "recording"
    assert "file" in result

    # Reset internal state for stop
    import claude_meeting_mcp.recorder as rec

    rec._capturer = mock_capturer
    result = stop_recording()
    assert result["status"] == "stopped"
    assert not is_recording()


@patch("claude_meeting_mcp.recorder.get_capturer")
@patch("claude_meeting_mcp.recorder._capturer", "something")
@patch("claude_meeting_mcp.recorder._current_file", "/fake/path.wav")
def test_double_start_error(mock_get):
    result = start_recording()
    assert "error" in result
    assert "already in progress" in result["error"]


@patch("claude_meeting_mcp.recorder._capturer", None)
def test_stop_without_start():
    result = stop_recording()
    assert "error" in result
    assert "No recording" in result["error"]


@patch("claude_meeting_mcp.recorder.get_capturer")
@patch("claude_meeting_mcp.recorder._capturer", None)
@patch("claude_meeting_mcp.recorder._current_file", None)
def test_start_when_not_available(mock_get):
    mock_capturer = MagicMock()
    mock_capturer.is_available.return_value = False
    mock_get.return_value = mock_capturer

    result = start_recording()
    assert "error" in result
    assert "not available" in result["error"]


def _available_capturer() -> MagicMock:
    capturer = MagicMock()
    capturer.is_available.return_value = True
    return capturer


def _current_session() -> int:
    """The token the running recording's safety timer was armed with."""
    import claude_meeting_mcp.recorder as rec

    return rec._session


@patch("claude_meeting_mcp.recorder.get_capturer")
@patch("claude_meeting_mcp.recorder._capturer", None)
@patch("claude_meeting_mcp.recorder._current_file", None)
def test_safety_cap_is_armed_from_config(mock_get):
    """The cap is configurable, not a constant baked into the module."""
    mock_get.return_value = _available_capturer()
    config = Config()
    config.recording.max_seconds = 60

    with patch("claude_meeting_mcp.recorder.get_config", return_value=config):
        result = start_recording()

    import claude_meeting_mcp.recorder as rec

    assert result["max_seconds"] == 60
    assert rec._timeout_timer.interval == 60
    stop_recording()  # cancels the timer


@patch("claude_meeting_mcp.recorder.get_capturer")
@patch("claude_meeting_mcp.recorder._capturer", None)
@patch("claude_meeting_mcp.recorder._current_file", None)
def test_auto_stop_hands_the_recording_over(mock_get):
    """The cap must not orphan the WAV: stopping it is only half the job."""
    mock_get.return_value = _available_capturer()
    handled: list[str] = []

    started = start_recording()
    set_auto_stop_handler(handled.append)
    try:
        _auto_stop(_current_session())
    finally:
        set_auto_stop_handler(None)

    assert not is_recording()
    assert handled == [started["file"]]


@patch("claude_meeting_mcp.recorder.get_capturer")
@patch("claude_meeting_mcp.recorder._capturer", None)
@patch("claude_meeting_mcp.recorder._current_file", None)
def test_a_stale_cap_does_not_stop_the_next_recording(mock_get):
    """cancel() cannot call off a Timer already running: it must check identity."""
    mock_get.return_value = _available_capturer()
    handled: list[str] = []

    start_recording()
    stale = _current_session()
    stop_recording()
    start_recording()

    set_auto_stop_handler(handled.append)
    try:
        _auto_stop(stale)  # the cancelled timer fires late
    finally:
        set_auto_stop_handler(None)

    assert is_recording()  # the second meeting is untouched
    assert handled == []
    stop_recording()


@patch("claude_meeting_mcp.recorder.get_capturer")
@patch("claude_meeting_mcp.recorder._capturer", None)
@patch("claude_meeting_mcp.recorder._current_file", None)
def test_auto_stop_survives_a_failing_handler(mock_get):
    """_auto_stop runs on the timer thread - a raising handler must not escape it."""
    mock_get.return_value = _available_capturer()

    def explode(_file: str) -> None:
        raise RuntimeError("transcription queue is on fire")

    start_recording()
    set_auto_stop_handler(explode)
    try:
        _auto_stop(_current_session())
    finally:
        set_auto_stop_handler(None)

    assert not is_recording()


def test_get_capturer_returns_macos():
    """On macOS, get_capturer should return MacOSCapturer."""
    import sys

    if sys.platform == "darwin":
        from claude_meeting_mcp.capture import get_capturer
        from claude_meeting_mcp.capture._macos import MacOSCapturer

        capturer = get_capturer()
        assert isinstance(capturer, MacOSCapturer)


def _hold_device_lock(data_dir):
    """Take the shared recording lock the way the auto-recorder app does."""
    import fcntl
    import os

    data_dir.mkdir(parents=True, exist_ok=True)
    fd = os.open(data_dir / "recording.lock", os.O_CREAT | os.O_RDWR)
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    return fd


@patch("claude_meeting_mcp.recorder.get_capturer")
@patch("claude_meeting_mcp.recorder._capturer", None)
@patch("claude_meeting_mcp.recorder._current_file", None)
def test_start_refused_while_another_recorder_holds_the_lock(mock_get, tmp_path):
    """The auto-recorder and a Claude session must never capture the same call twice."""
    import os

    mock_get.return_value = MagicMock(**{"is_available.return_value": True})
    fd = _hold_device_lock(tmp_path)
    try:
        result = start_recording()
    finally:
        os.close(fd)
    assert "already recording" in result["error"]
    mock_get.return_value.start.assert_not_called()


@patch("claude_meeting_mcp.recorder.get_capturer")
@patch("claude_meeting_mcp.recorder._capturer", None)
@patch("claude_meeting_mcp.recorder._current_file", None)
def test_lock_held_while_recording_and_released_on_stop(mock_get, tmp_path):
    import fcntl
    import os

    mock_get.return_value = MagicMock(**{"is_available.return_value": True})
    assert start_recording()["status"] == "recording"
    try:
        _hold_device_lock(tmp_path)
        raise AssertionError("lock should be held during the recording")
    except BlockingIOError:
        pass
    stop_recording()
    fd = _hold_device_lock(tmp_path)  # free again
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)
