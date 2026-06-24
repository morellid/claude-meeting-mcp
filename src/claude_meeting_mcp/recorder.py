"""Audio recording orchestration via platform-specific capture backends."""

from __future__ import annotations

import logging
import threading

from .capture import get_capturer
from .storage import RECORDINGS_DIR, TRANSCRIPTIONS_DIR, ensure_dirs, generate_filename
from .transcriber import live_backend_available

logger = logging.getLogger(__name__)

MAX_RECORDING_SECONDS = 4 * 3600  # 4 hours default safety limit

_lock = threading.RLock()  # Reentrant: _auto_stop calls stop_recording from Timer thread
_capturer = None
_current_file: str | None = None
_timeout_timer: threading.Timer | None = None
_live_translator = None
_live_transcriber = None


def _auto_stop() -> None:
    """Called by the timeout timer to auto-stop a runaway recording."""
    logger.warning("Recording timeout (%ds) reached, auto-stopping", MAX_RECORDING_SECONDS)
    stop_recording()


def start_recording(
    live: bool = True,
    live_translate: str | None = None,
    mcp_context=None,
    event_loop=None,
) -> dict:
    """Start recording system audio + microphone.

    Args:
        live: Accumulate a live transcript during recording (default on). Auto
            gated off when no fast local backend is available (remote, CPU-only).
        live_translate: Optional target language for live TRANSLATION (legacy
            path via LiveTranslator). When set, takes precedence over `live`.
        mcp_context: MCP Context for sampling-based translation (any lang).
        event_loop: Main asyncio event loop (for thread-safe MCP calls).
    """
    global _capturer, _current_file, _timeout_timer, _live_translator, _live_transcriber

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
        _timeout_timer = threading.Timer(MAX_RECORDING_SECONDS, _auto_stop)
        _timeout_timer.daemon = True
        _timeout_timer.start()

        result = {"status": "recording", "file": _current_file}

        # Live path: translation (legacy) takes precedence over transcription.
        if live_translate:
            try:
                from .config import get_config
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
                    source = FileAudioSource(
                        _current_file,
                        sample_rate=config.recording.sample_rate,
                    )
                    _live_transcriber = LiveTranscriber(source=source, output_path=live_file)
                    _live_transcriber.start()
                    result["live"] = True
                    result["live_file"] = live_file
                    logger.info("Live transcription started: %s → %s", _current_file, live_file)
                except Exception as e:
                    logger.error("Failed to start live transcriber: %s", e)
                    result["live"] = False
                    result["live_error"] = str(e)
            else:
                result["live"] = False
                result["live_reason"] = reason

        logger.info("Recording started: %s", _current_file)
        return result


def stop_recording() -> dict:
    """Stop current recording.

    Stops the live translator/transcriber daemon thread, but does NOT finalize
    the live transcript or delete the WAV - finalization happens in
    finalize_live() so a plain stop leaves the recording intact.
    """
    global _capturer, _current_file, _timeout_timer, _live_translator

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

        # Stop live transcriber thread only; keep the reference so finalize_live()
        # can drain the accumulated transcript afterwards.
        if _live_transcriber is not None:
            try:
                _live_transcriber.stop()
            except Exception as e:
                logger.error("Error stopping live transcriber: %s", e)

        file_path = _current_file

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


def finalize_live(remote_speakers: str, local_speakers: str):
    """Finalize the active live transcriber into a Transcription.

    Returns the Transcription, or None if there was no live transcriber or it
    produced nothing (the caller then falls back to the batch path). Clears the
    transcriber reference so it is finalized at most once.
    """
    global _live_transcriber

    with _lock:
        transcriber = _live_transcriber
        _live_transcriber = None
    if transcriber is None:
        return None
    return transcriber.finalize(remote_speakers, local_speakers)


def get_live_status() -> dict | None:
    """Get live transcription/translation status, or None if not active."""
    if _live_transcriber is not None:
        return _live_transcriber.get_status()
    if _live_translator is not None:
        return _live_translator.get_status()
    return None


def is_recording() -> bool:
    return _capturer is not None
