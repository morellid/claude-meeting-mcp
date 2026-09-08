"""Audio recording orchestration via platform-specific capture backends."""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable

from .capture import get_capturer
from .config import RecordingConfig, get_config
from .storage import RECORDINGS_DIR, TRANSCRIPTIONS_DIR, ensure_dirs, generate_filename

logger = logging.getLogger(__name__)

_lock = threading.RLock()  # Reentrant: _auto_stop calls stop_recording from Timer thread
_capturer = None
_current_file: str | None = None
_session: int = 0  # bumped on every start; identifies which recording a timer belongs to
_started_at: float | None = None
_timeout_timer: threading.Timer | None = None
_live_translator = None
_auto_stop_handler: Callable[[str], None] | None = None


def set_auto_stop_handler(handler: Callable[[str], None] | None) -> None:
    """Register what happens to the WAV when the safety cap fires.

    The cap used to stop the capture and do nothing else, which is the worst of
    both worlds: a session left running produced a multi-gigabyte recording, no
    transcript, and a later stop that reported "No recording in progress" with no
    hint that anything had been captured. With a handler registered, an
    auto-stopped meeting ends up in the same state as one the user stopped by
    hand. The recorder cannot import the server (the server imports it), so the
    dependency is inverted through this hook.
    """
    global _auto_stop_handler
    _auto_stop_handler = handler


def _auto_stop(session: int) -> None:
    """Called by the timeout timer to auto-stop a runaway recording.

    Takes the session it was armed for because ``cancel()`` cannot call off a
    Timer that has already begun executing: without the identity check, a
    callback firing just as the user stops meeting A would go on to stop meeting
    B, truncate it, and report it as safety-capped. The token is a counter rather
    than the file path because filenames are minute-resolution and two recordings
    in the same minute share one.
    """
    with _lock:
        if _capturer is None or session != _session:
            logger.info("Stale safety cap for session %d ignored", session)
            return
        logger.warning("Recording safety cap reached, auto-stopping %s", _current_file)
        result = stop_recording()

    # Hand over the file even when the capturer errored on the way out: a partial
    # WAV is still a meeting, and it is the only copy. Called outside the lock -
    # the handler queues work and must not hold up the next recording.
    file_path = result.get("file")
    handler = _auto_stop_handler
    if handler is None or file_path is None:
        return
    try:
        handler(file_path)
    except Exception:  # noqa: BLE001 - runs on the timer thread; never let it die silently
        logger.exception("Auto-stop handler failed for %s", file_path)


def start_recording(live_translate: str | None = None, mcp_context=None, event_loop=None) -> dict:
    """Start recording system audio + microphone.

    Args:
        live_translate: Target language for live translation (e.g., "en", "fr").
            None = no live translation.
        mcp_context: MCP Context for sampling-based translation (any lang).
        event_loop: Main asyncio event loop (for thread-safe MCP calls).
    """
    global _capturer, _current_file, _session, _started_at, _timeout_timer, _live_translator

    with _lock:
        if _capturer is not None:
            return {"error": "Recording already in progress", "file": _current_file}

        capturer = get_capturer()
        if not capturer.is_available():
            return {"error": "Audio capture not available on this platform. Check installation."}

        ensure_dirs()
        filename = generate_filename()
        filepath = RECORDINGS_DIR / filename
        _current_file = str(filepath)

        try:
            capturer.start(_current_file)
            _capturer = capturer
        except RuntimeError as e:
            logger.error("Failed to start recording: %s", e)
            _current_file = None
            return {"error": str(e)}

        # Start safety timeout timer
        max_seconds = get_config().recording.max_seconds
        if max_seconds <= 0:
            # A zero or negative cap would fire instantly and kill the meeting the
            # moment it starts. Refuse it here rather than trust the config file.
            logger.error(
                "Ignoring recording.max_seconds=%s; falling back to %ss",
                max_seconds,
                RecordingConfig.max_seconds,
            )
            max_seconds = RecordingConfig.max_seconds
        _session += 1
        _started_at = time.monotonic()
        _timeout_timer = threading.Timer(max_seconds, _auto_stop, args=(_session,))
        _timeout_timer.daemon = True
        _timeout_timer.start()

        result = {"status": "recording", "file": _current_file, "max_seconds": max_seconds}

        # Start live translation if requested
        if live_translate:
            try:
                from .live_translator import FileAudioSource, LiveTranslator

                config = get_config()
                meeting_id = filepath.stem
                live_file = str(TRANSCRIPTIONS_DIR / f"{meeting_id}_live.md")

                source = FileAudioSource(
                    _current_file,
                    sample_rate=config.recording.sample_rate,
                )
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
                logger.info("Live translation started: %s → %s", _current_file, live_file)
            except Exception as e:
                logger.error("Failed to start live translation: %s", e)
                result["live_translation_error"] = str(e)

        logger.info("Recording started: %s", _current_file)
        return result


def stop_recording() -> dict:
    """Stop current recording."""
    global _capturer, _current_file, _started_at, _timeout_timer, _live_translator

    with _lock:
        if _capturer is None:
            return {"error": "No recording in progress"}

        # Cancel timeout timer
        if _timeout_timer is not None:
            _timeout_timer.cancel()
            _timeout_timer = None

        # Stop live translator
        if _live_translator is not None:
            try:
                _live_translator.stop()
            except Exception as e:
                logger.error("Error stopping live translator: %s", e)
            _live_translator = None

        file_path = _current_file
        _started_at = None

        try:
            _capturer.stop()
        except RuntimeError as e:
            logger.error("Error stopping recording: %s", e)
            _capturer = None
            _current_file = None
            return {"error": str(e), "file": file_path}

        logger.info("Recording stopped: %s", file_path)
        _capturer = None
        _current_file = None
        return {"status": "stopped", "file": file_path}


def get_live_status() -> dict | None:
    """Get live translation status, or None if not active."""
    if _live_translator is not None:
        return _live_translator.get_status()
    return None


def is_recording() -> bool:
    return _capturer is not None


def recording_elapsed_seconds() -> float | None:
    """Seconds since the current recording started, or None if idle.

    Lets a caller report how long it has been recording and how much of the
    safety cap is left, without having to remember when it started - which a
    long-running assistant session cannot be relied on to do.
    """
    with _lock:
        if _capturer is None or _started_at is None:
            return None
        return time.monotonic() - _started_at
