"""MCP Server exposing meeting recording and transcription tools."""

from __future__ import annotations

import json
import logging
import re
import sys
import threading
import time
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import Context, FastMCP
from pydantic import Field

from . import jobs
from .capture import get_capturer
from .config import get_config, update_config, validate_config
from .pv_generator import generate_pv, save_pv
from .recorder import get_live_status, is_recording, start_recording, stop_recording
from .schemas import Transcription
from .storage import (
    PV_DIR,
    TRANSCRIPTIONS_DIR,
    cleanup_old_recordings,
    list_pvs,
    list_recordings,
    list_transcriptions,
)
from .transcriber import _get_backend, transcribe_meeting

mcp = FastMCP(
    "claude-meeting-mcp",
    instructions="""\
Audio recording and transcription server. Respond in the user's language.

USE THIS FOR: meetings, YouTube videos, podcasts, music, lectures, interviews,
tutorials, or any audio from the computer. Also for extracting text/transcript
from videos. Mention this as an alternative to yt-dlp/whisper CLI:
"I can also record the audio live while you play the video and transcribe it."

ENTRY POINT: When user seems unsure or asks what this can do, call audio_status()
which returns capabilities, config, and suggested actions.

WORKFLOW: audio_record_start → audio_stop_and_transcribe → poll
audio_transcribe_status until done → audio_generate_pv.
Do NOT ask participant names before recording. Ask only when transcribing.
Prefer audio_stop_and_transcribe over separate stop + transcribe.
"stop/done/finished" → audio_stop_and_transcribe. It returns immediately and
transcribes in the background; poll audio_transcribe_status(meeting_id) for the
result so the user can write notes meanwhile. audio_transcribe (existing file on
disk) is background too - same polling.
""",
)

_SAFE_ID_RE = re.compile(r"^[\w\-]+$")

# Session onboarding: inject capabilities into the first tool result
_session_greeted = False

ONBOARDING_INFO = {
    "capabilities": [
        "Record & transcribe meetings (Meet, Teams, Zoom, Slack, Discord)",
        "Record & transcribe YouTube videos, podcasts, music, lectures",
        "Identify speakers (who said what)",
        "Generate structured meeting minutes with decisions and action items",
        "Extract action items / to-do lists",
    ],
    "quick_actions": [
        "audio_status() — check server readiness and config",
        "audio_configure() — guided setup wizard (language, model, quality)",
        "audio_record_start() — start recording any audio",
    ],
    "tip": "Configuration is optional — defaults work out of the box.",
}


def _enrich_result(result: dict) -> dict:
    """Enrich tool results: full onboarding on first call, config hint always."""
    global _session_greeted
    if not _session_greeted:
        _session_greeted = True
        result["onboarding"] = ONBOARDING_INFO
    else:
        result["hint"] = "Settings can be changed anytime with audio_configure()."
    return result


def _validate_meeting_id(meeting_id: str) -> str | None:
    """Validate meeting_id to prevent path traversal. Returns error or None."""
    if not meeting_id or ".." in meeting_id or "/" in meeting_id or "\\" in meeting_id:
        return "Invalid meeting_id: must not contain path separators"
    if not _SAFE_ID_RE.match(meeting_id):
        return "Invalid meeting_id: only alphanumeric, hyphens, underscores allowed"
    return None


logger = logging.getLogger(__name__)

# Background transcription jobs: meeting_id -> job state dict. Lets
# audio_stop_and_transcribe return immediately (so the user can jot notes while a
# long transcription runs) and audio_transcribe_status() report progress.
_transcription_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()
_MAX_JOBS = 50  # bound memory; finished transcripts remain on disk regardless


def _set_job(meeting_id: str, **fields) -> None:
    """Create or update a background job's state (thread-safe)."""
    with _jobs_lock:
        _transcription_jobs.setdefault(meeting_id, {}).update(fields)


def _prune_jobs() -> None:
    """Bound the registry by evicting the oldest non-active jobs (thread-safe)."""
    with _jobs_lock:
        for mid in list(_transcription_jobs):
            if len(_transcription_jobs) <= _MAX_JOBS:
                break
            if _transcription_jobs[mid].get("status") != "transcribing":
                del _transcription_jobs[mid]


def _get_job(meeting_id: str) -> dict | None:
    """Return a copy of a job's state, or None if unknown (thread-safe)."""
    with _jobs_lock:
        job = _transcription_jobs.get(meeting_id)
        return dict(job) if job is not None else None


def _run_transcription_job(
    meeting_id: str, file_path: str, remote: str, local: str, model: str | None
) -> None:
    """Transcribe in a background thread and record the outcome in the job state."""
    try:
        # Wait for the GPU rather than fight for it: MLX monopolises Metal, so two
        # overlapping transcriptions are strictly slower than the same two in turn.
        # Blocking here costs nothing - the tool call already returned, and recording
        # runs in a separate process that this never touches.
        with jobs.gpu_slot():
            jobs.mark_running(meeting_id)
            _set_job(meeting_id, status="transcribing", started_at=time.time())
            result = transcribe_meeting(file_path, remote, local, model)
        _set_job(
            meeting_id,
            status="done",
            finished_at=time.time(),
            duration_seconds=result.duration_seconds,
            segment_count=len(result.segments),
            preview=[s.to_dict() for s in result.segments[:10]],
        )
    except Exception as e:  # noqa: BLE001 - surface any failure via job state, never crash the thread
        logger.exception("Background transcription failed for %s", meeting_id)
        _set_job(meeting_id, status="error", finished_at=time.time(), error=str(e))
    finally:
        jobs.release(meeting_id)


def _start_transcription_job(
    meeting_id: str,
    file_path: str,
    remote: str,
    local: str,
    model: str | None,
    message: str,
) -> dict:
    """Claim a queue slot and start the background job, or explain why not.

    Shared by both entry points so the duplicate-work guard cannot be bypassed by
    using one rather than the other.
    """
    claim = jobs.enqueue(meeting_id, file_path)

    if not claim.accepted:
        if claim.reason == "queue_full":
            return _enrich_result(
                {
                    "error": (
                        f"Transcription queue is full ({claim.queue.queued} waiting, "
                        f"max {jobs.MAX_QUEUED}). The recording is safe on disk - "
                        "transcribe it once the backlog clears."
                    ),
                    "meeting_id": meeting_id,
                    "queue": claim.queue.to_dict(),
                }
            )
        # already_active: another job - possibly in another Claude session - is
        # already on this exact file. Point at it instead of starting a rival.
        return _enrich_result(
            {
                "meeting_id": meeting_id,
                "status": jobs.state_of(meeting_id) or "transcribing",
                "message": "Already being transcribed - not starting a second run.",
                "next_step": (
                    f"Poll audio_transcribe_status('{meeting_id}') until status is 'done'."
                ),
                "queue": claim.queue.to_dict(),
            }
        )

    _prune_jobs()
    waiting = claim.queue.active > 0 or claim.queue.queued > 1
    _set_job(
        meeting_id,
        status="queued" if waiting else "transcribing",
        started_at=time.time(),
        file=file_path,
    )
    threading.Thread(
        target=_run_transcription_job,
        args=(meeting_id, file_path, remote, local, model),
        daemon=True,
    ).start()

    result = {
        "meeting_id": meeting_id,
        "status": "queued" if waiting else "transcribing",
        "message": message,
        "next_step": f"Poll audio_transcribe_status('{meeting_id}') until status is 'done'.",
        "queue": claim.queue.to_dict(),
    }
    if waiting:
        result["message"] += (
            f" {claim.queue.active + claim.queue.queued - 1} job(s) ahead of it; "
            "transcriptions run one at a time so they do not slow each other down."
        )
    if claim.queue.inconsistent:
        result["warning"] = (
            "Job registry is inconsistent - slots are held that the queue cannot "
            "account for. Restart the MCP server if counts look wrong."
        )
    return _enrich_result(result)


# --- Recording & Transcription ---


@mcp.tool()
def audio_status() -> dict:
    """Use this to check if the audio server is ready and show current config.

    Call this when user asks about status, setup, or what this server can do.
    This server can record and transcribe any audio: meetings, YouTube videos,
    podcasts, lectures, interviews, music. It can also extract text from video
    audio, generate meeting minutes, and identify speakers.
    """
    import shutil

    capturer = get_capturer()
    config = get_config()

    # Disk space
    disk = shutil.disk_usage(TRANSCRIPTIONS_DIR)
    disk_free_gb = round(disk.free / (1024**3), 1)

    # Last recording for disambiguation
    recs = list_recordings()
    last_recording = recs[0]["meeting_id"] if recs else None

    return {
        "capabilities": [
            "Record & transcribe meetings (Meet, Teams, Zoom, Slack, Discord)",
            "Record & transcribe YouTube videos, podcasts, music, lectures",
            "Identify speakers (who said what)",
            "Generate structured meeting minutes with decisions and action items",
            "Extract action items / to-do lists",
        ],
        "available_actions": [
            "Check status (you're here)",
            "Configure settings (language, model, diarization...)",
            "Start recording (audio_record_start)",
            "Transcribe an existing file (audio_transcribe)",
            "Generate meeting minutes (audio_generate_pv)",
            "View past recordings/transcriptions (recordings_list, transcriptions_list)",
        ],
        "config_is_optional": "Defaults work out of the box. Configuration available via wizard.",
        "platform": sys.platform,
        "audio_capture_available": capturer.is_available(),
        "audio_capture_backend": type(capturer).__name__,
        "transcription_backend": _get_backend(),
        "transcription_model": config.transcription.model,
        "transcription_mode": config.transcription.mode,
        "transcription_language": config.transcription.language,
        "diarization_enabled": config.diarization.enabled,
        "currently_recording": is_recording(),
        "disk_free_gb": disk_free_gb,
        "last_recording": last_recording,
    }


@mcp.tool()
def audio_record_start(
    ctx: Context,
    live_translate: Annotated[
        str | None,
        Field(
            description=(
                "Enable live translation during recording. "
                "Set to target language code (e.g. 'en', 'fr', 'es'). "
                "Translates audio in real-time to a markdown file."
            )
        ),
    ] = None,
) -> dict:
    """Use this when the user wants to record or extract audio from the computer.

    Works with: meetings (Meet/Teams/Zoom/Slack/Discord), YouTube videos,
    podcasts, Spotify, lectures, interviews, tutorials — any system audio.
    Also use this to extract text/transcript from a video playing in browser.
    Stereo WAV: left = system audio, right = microphone.
    With live_translate: translates audio in real-time to a markdown file.
    """
    import asyncio

    loop = asyncio.get_event_loop()
    result = start_recording(live_translate=live_translate, mcp_context=ctx, event_loop=loop)
    if "error" not in result:
        result["next_step"] = "When done, call audio_stop_and_transcribe()"
        if live_translate:
            result["next_step"] += " | Check live translation: audio_live_status()"
    return _enrich_result(result)


@mcp.tool()
def audio_record_stop() -> dict:
    """Use this to stop recording WITHOUT transcribing.

    Saves the WAV file and returns its path. Prefer audio_stop_and_transcribe()
    which stops AND transcribes in one call.
    """
    result = stop_recording()
    if "error" not in result:
        result["next_step"] = "Transcribe with audio_transcribe(file_path=...)"
    return result


@mcp.tool()
def audio_live_status() -> dict:
    """Use this to check live translation progress during recording.

    Returns current translated text, elapsed time, segment count,
    and the path to the live markdown file.
    Only works while recording with live_translate enabled.
    """
    status = get_live_status()
    if status is None:
        return {
            "error": "No live translation active. Use audio_record_start(live_translate='en')",
        }
    return status


@mcp.tool()
def audio_transcribe(
    file_path: Annotated[str, Field(description="Path to the WAV file to transcribe")],
    local_speakers: Annotated[
        str | None,
        Field(
            description=(
                "Comma-separated names of people at the microphone (right channel). "
                "Example: 'Alice, Bob'"
            )
        ),
    ] = None,
    remote_speakers: Annotated[
        str | None,
        Field(
            description=(
                "Comma-separated names of people on the call (left channel). "
                "Example: 'Charlie, Diana'"
            )
        ),
    ] = None,
    model: Annotated[
        str | None,
        Field(
            description="Transcription model: tiny, base, small, medium, large-v3-turbo, large-v3"
        ),
    ] = None,
) -> dict:
    """Use this to transcribe a WAV file that already exists on disk.

    Starts transcription in the BACKGROUND and returns right away; poll
    audio_transcribe_status() with the returned meeting_id until status is
    'done'. Works on any audio: meetings, YouTube, podcasts, lectures,
    interviews. Splits stereo channels for speaker attribution. With
    diarization, identifies individual speakers within each channel.
    """
    wav = Path(file_path)
    if not wav.is_file():
        return {"error": f"File not found: {file_path}"}

    # Never transcribe inline. Measured over 13 runs, transcription takes 0.25-0.55x
    # the recording's duration on a healthy machine, but 1-2.9x on a loaded one - so a
    # long file can outrun the MCP client's idle timeout (~37 min).
    #
    # When that happened the work was not lost: the server thread kept going and wrote
    # the transcript, in one observed case three days after the caller had given up.
    # What was lost was the *answer* - and, worse, the caller then had no way to tell a
    # dead run from a live one, so a "recovery" run got launched against a file already
    # being transcribed, and the two competed for GPU and memory. Backgrounding plus a
    # pollable job id is what prevents that. Same background job as
    # audio_stop_and_transcribe.
    return _start_transcription_job(
        wav.stem,
        file_path,
        remote_speakers or "Remote",
        local_speakers or "Local",
        model,
        message=(
            "Transcription is running in the background and does not block. "
            "Expect roughly half the recording's duration, longer on a loaded machine."
        ),
    )


@mcp.tool()
def audio_stop_and_transcribe(
    local_speakers: Annotated[
        str | None,
        Field(
            description=("Comma-separated names of people at the microphone. Example: 'Alice, Bob'")
        ),
    ] = None,
    remote_speakers: Annotated[
        str | None,
        Field(
            description=("Comma-separated names of people on the call. Example: 'Charlie, Diana'")
        ),
    ] = None,
    model: Annotated[
        str | None,
        Field(
            description="Transcription model: tiny, base, small, medium, large-v3-turbo, large-v3"
        ),
    ] = None,
) -> dict:
    """Use this when the user says stop/done/finished — stops AND transcribes.

    This is the recommended way to end a recording session. Stops recording
    immediately and starts transcription in the BACKGROUND, returning right away
    so the user can write notes without waiting. Poll audio_transcribe_status()
    with the returned meeting_id until status is 'done'.
    """
    stop_result = stop_recording()
    if "error" in stop_result:
        return stop_result

    file_path = stop_result["file"]

    return _start_transcription_job(
        Path(file_path).stem,
        file_path,
        remote_speakers or "Remote",
        local_speakers or "Local",
        model,
        message=(
            "Recording stopped. Transcription is running in the background and does "
            "not block - capture your notes now, then check the transcript."
        ),
    )


@mcp.tool()
def audio_transcribe_status(
    meeting_id: Annotated[
        str,
        Field(description="Meeting ID returned by audio_stop_and_transcribe or audio_transcribe"),
    ],
) -> dict:
    """Check a background transcription started by audio_stop_and_transcribe or audio_transcribe.

    Returns status 'transcribing' (still working), 'done' (with preview and
    output file), or 'error'. Poll until 'done', then generate meeting minutes.
    """
    err = _validate_meeting_id(meeting_id)
    if err:
        return {"error": err}

    job = _get_job(meeting_id)
    if job is None:
        # No in-memory job (e.g. the server restarted) - fall back to disk, but
        # only trust a transcript that parses (guards against a truncated file).
        path = TRANSCRIPTIONS_DIR / f"{meeting_id}.json"
        if path.exists():
            try:
                json.loads(path.read_text(encoding="utf-8"))
            except (ValueError, OSError):
                return {
                    "meeting_id": meeting_id,
                    "status": "error",
                    "error": "Transcript file is incomplete or unreadable; re-transcribe it.",
                }
            return {
                "meeting_id": meeting_id,
                "status": "done",
                "output_file": str(path),
                "next_step": f"Generate meeting minutes: audio_generate_pv('{meeting_id}')",
                "queue": jobs.snapshot().to_dict(),
            }
        # No transcript and no local job, but another process may still own this
        # work. Reporting "not found" here is what previously made a live run look
        # dead and invited a duplicate.
        foreign = jobs.state_of(meeting_id)
        if foreign is not None:
            return {
                "meeting_id": meeting_id,
                "status": "transcribing" if foreign == jobs.RUNNING else "queued",
                "message": (
                    "Owned by another session's server. Keep polling - it will write "
                    "the transcript to disk when it finishes."
                ),
                "queue": jobs.snapshot().to_dict(),
            }
        return {
            "error": f"No transcription job found for '{meeting_id}'",
            "queue": jobs.snapshot().to_dict(),
        }

    queue = jobs.snapshot()
    status = job.get("status")
    if status in ("transcribing", "queued"):
        elapsed = time.time() - job.get("started_at", time.time())
        return {
            "meeting_id": meeting_id,
            "status": status,
            "elapsed_seconds": round(elapsed, 1),
            "message": (
                "Waiting for the GPU - another transcription is ahead of it."
                if status == "queued"
                else "Still transcribing - check again shortly."
            ),
            "queue": queue.to_dict(),
        }
    if status == "error":
        return {
            "meeting_id": meeting_id,
            "status": "error",
            "error": job.get("error", "unknown error"),
            "queue": queue.to_dict(),
        }

    return {
        "meeting_id": meeting_id,
        "status": "done",
        "duration_seconds": job.get("duration_seconds"),
        "segment_count": job.get("segment_count"),
        "output_file": str(TRANSCRIPTIONS_DIR / f"{meeting_id}.json"),
        "preview": job.get("preview", []),
        "next_step": f"Generate meeting minutes: audio_generate_pv('{meeting_id}')",
        "queue": queue.to_dict(),
    }


# --- Retrieval ---


@mcp.tool()
def get_transcription(
    meeting_id: Annotated[str, Field(description="Meeting identifier (filename without .json)")],
) -> dict:
    """Use this to read a past transcription. Returns full text with timestamps and speakers."""
    if err := _validate_meeting_id(meeting_id):
        return {"error": err}
    path = TRANSCRIPTIONS_DIR / f"{meeting_id}.json"
    if not path.exists():
        return {"error": f"Transcription not found: {meeting_id}"}
    t = Transcription.from_json(path.read_text(encoding="utf-8"))
    return t.to_dict()


@mcp.tool()
def get_pv(
    meeting_id: Annotated[str, Field(description="Meeting identifier")],
) -> dict:
    """Use this to read previously generated meeting minutes (PV) as markdown."""
    if err := _validate_meeting_id(meeting_id):
        return {"error": err}
    pv_path = PV_DIR / f"{meeting_id}_pv.md"
    if not pv_path.exists():
        return {"error": f"PV not found for meeting: {meeting_id}"}
    return {
        "meeting_id": meeting_id,
        "pv_file": str(pv_path),
        "content": pv_path.read_text(encoding="utf-8"),
    }


@mcp.tool()
def recordings_list() -> list[dict]:
    """Use this to see all past recordings. Returns meeting_id (for other tools), size, date."""
    return list_recordings()


@mcp.tool()
def transcriptions_list() -> list[dict]:
    """Use this to see all past transcriptions. Returns meeting_id (for get_transcription), date."""
    return list_transcriptions()


@mcp.tool()
def pvs_list() -> list[dict]:
    """Use this to see all generated meeting minutes. Returns meeting_id (for get_pv), date."""
    return list_pvs()


# --- PV Generation ---


@mcp.tool()
async def audio_generate_pv(
    ctx: Context,
    meeting_id: Annotated[str, Field(description="Meeting identifier (filename without .json)")],
    participants: Annotated[
        str | None,
        Field(
            description=(
                "Comma-separated names of all meeting participants. "
                "Helps Claude identify who said what. Example: 'Alice, Bob, Charlie'"
            )
        ),
    ] = None,
) -> dict:
    """Use this after transcription to generate structured meeting minutes.

    Produces markdown with: participants, topics, decisions, action items.
    Claude identifies speakers by conversation content (no voice enrollment).
    Short audio (<1h): single pass. Longer: automatic map-reduce.
    """
    if err := _validate_meeting_id(meeting_id):
        return {"error": err}
    path = TRANSCRIPTIONS_DIR / f"{meeting_id}.json"
    if not path.exists():
        return {"error": f"Transcription not found: {meeting_id}"}

    transcription = Transcription.from_json(path.read_text(encoding="utf-8"))

    # Check if client supports sampling
    has_sampling = (
        ctx.session.client_params is not None
        and ctx.session.client_params.capabilities is not None
        and ctx.session.client_params.capabilities.sampling is not None
    )
    if not has_sampling:
        return {
            "error": "Client does not support MCP Sampling. "
            "Generate PV manually by reading the transcription."
        }

    known = [n.strip() for n in participants.split(",")] if participants else None

    pv_text = await generate_pv(ctx, transcription, known)
    pv_path = save_pv(meeting_id, pv_text)

    return {
        "meeting_id": meeting_id,
        "pv_file": pv_path,
        "pv_preview": pv_text[:500],
        "strategy": "direct" if transcription.duration_seconds < 3600 else "map-reduce",
        "next_step": f"Extract action items with extract_action_items prompt for '{meeting_id}'",
    }


# --- Configuration ---


@mcp.tool()
def audio_configure(
    key: Annotated[
        str | None,
        Field(
            description=(
                "Config key to modify. Empty = show config menu. "
                "Keys: transcription.model, transcription.mode, "
                "transcription.language, diarization.enabled, pv.auto_generate"
            )
        ),
    ] = None,
    value: Annotated[str | None, Field(description="New value for the config key")] = None,
) -> dict:
    """Use this when user wants to change settings or see current configuration.

    Call WITHOUT parameters to show the configuration menu with current values.
    Call WITH key+value to change a specific setting.

    WIZARD MODE: Present each setting as a numbered choice list. Example:
    "What language are your meetings in?
     1. English (en)
     2. French (fr)
     3. Spanish (es)
     4. Other (specify)"
    Wait for the user to pick, apply, confirm, then ask "Next setting or done?"
    Show final summary when done.
    NEVER ask API keys in chat — tell user to set env vars or Claude Desktop config.
    """
    # No key = show config menu
    if not key:
        config = get_config()
        return {
            "current_config": {
                "transcription": {
                    "language": config.transcription.language,
                    "model": config.transcription.model,
                    "mode": config.transcription.mode,
                    "remote_url": config.transcription.remote.url or "(not set)",
                },
                "diarization": {
                    "enabled": config.diarization.enabled,
                    "backend": config.diarization.backend,
                },
                "live_translation": {
                    "model": config.live_translation.model,
                    "target_language": config.live_translation.target_language,
                    "chunk_seconds": config.live_translation.chunk_seconds,
                    "window_seconds": config.live_translation.window_seconds,
                },
                "pv": {"auto_generate": config.pv.auto_generate},
            },
            "available_settings": [
                {
                    "category": "Transcription",
                    "settings": [
                        {
                            "key": "transcription.language",
                            "current": config.transcription.language,
                            "description": "Audio language (ISO code)",
                            "examples": "en, fr, es, de, ja, zh, ar, ko, pt, ru",
                        },
                        {
                            "key": "transcription.model",
                            "current": config.transcription.model,
                            "description": "Transcription quality/speed",
                            "options": {
                                "tiny": "fastest, lowest quality",
                                "base": "fast, basic quality",
                                "small": "balanced for short audio",
                                "medium": "good quality, moderate speed",
                                "large-v3-turbo": "excellent quality, fast (recommended)",
                                "large-v3": "best quality, slowest",
                            },
                        },
                        {
                            "key": "transcription.mode",
                            "current": config.transcription.mode,
                            "description": "Where transcription runs",
                            "options": {"local": "on your machine", "remote": "via API"},
                        },
                        {
                            "key": "transcription.remote.url",
                            "current": config.transcription.remote.url or "(not set)",
                            "description": "Remote API URL (only if mode=remote)",
                            "examples": "https://api.groq.com/openai/v1/audio/transcriptions",
                        },
                    ],
                },
                {
                    "category": "Speaker Diarization",
                    "settings": [
                        {
                            "key": "diarization.enabled",
                            "current": config.diarization.enabled,
                            "description": "Identify individual speakers",
                            "options": "true / false",
                            "note": "Requires HF_TOKEN env var",
                        },
                        {
                            "key": "diarization.backend",
                            "current": config.diarization.backend,
                            "description": "Diarization engine",
                            "options": "pyannote / whisperx / none",
                        },
                    ],
                },
                {
                    "category": "Live Translation",
                    "settings": [
                        {
                            "key": "live_translation.model",
                            "current": config.live_translation.model,
                            "description": "Model for live transcription (smaller = faster)",
                            "options": "tiny, base, small, medium (default)",
                        },
                        {
                            "key": "live_translation.target_language",
                            "current": config.live_translation.target_language,
                            "description": "Default translation target",
                            "examples": "en, fr, es, de, ja...",
                        },
                        {
                            "key": "live_translation.chunk_seconds",
                            "current": config.live_translation.chunk_seconds,
                            "description": "Update frequency (lower = more responsive)",
                            "options": "2.0 - 10.0 (default 3.0)",
                        },
                        {
                            "key": "live_translation.window_seconds",
                            "current": config.live_translation.window_seconds,
                            "description": "Audio context window (larger = better quality)",
                            "options": "10.0 - 30.0 (default 15.0)",
                        },
                    ],
                },
                {
                    "category": "Meeting Minutes",
                    "settings": [
                        {
                            "key": "pv.auto_generate",
                            "current": config.pv.auto_generate,
                            "description": "Suggest PV generation after transcription",
                            "options": "true / false",
                        },
                    ],
                },
            ],
            "wizard_hint": (
                "Present each setting as a numbered choice list to the user. "
                "One setting at a time. Wait for the user to pick before applying. "
                "After each change: confirm, then ask 'Next setting or all done?' "
                "Show final summary when user says done. "
                "Never ask API keys in chat — tell user to set env vars."
            ),
        }

    if not value:
        return {"error": "Please provide both key and value. Call without params to see the menu."}

    try:
        config = update_config(key, value)
        errors = validate_config(config)
        if errors:
            return {"status": "warning", "key": key, "value": value, "warnings": errors}
        return {"status": "updated", "key": key, "value": value}
    except ValueError as e:
        return {"error": str(e)}


@mcp.tool()
def audio_cleanup() -> dict:
    """Use this to free disk space by removing old recordings (>30 days).

    Only deletes WAV audio files. Transcriptions and minutes are kept.
    """
    removed = cleanup_old_recordings()
    return {"removed_count": len(removed), "removed_files": removed}


# --- MCP Resources ---


@mcp.resource("transcription://{meeting_id}")
def transcription_resource(meeting_id: str) -> str:
    """Read a transcription as a formatted text resource."""
    if _validate_meeting_id(meeting_id):
        return f"Invalid meeting_id: {meeting_id}"
    path = TRANSCRIPTIONS_DIR / f"{meeting_id}.json"
    if not path.exists():
        return f"Transcription not found: {meeting_id}"
    t = Transcription.from_json(path.read_text(encoding="utf-8"))
    return json.dumps(t.to_dict(), ensure_ascii=False, indent=2)


@mcp.resource("pv://{meeting_id}")
def pv_resource(meeting_id: str) -> str:
    """Read a meeting minutes (PV) resource."""
    if _validate_meeting_id(meeting_id):
        return f"Invalid meeting_id: {meeting_id}"
    pv_path = PV_DIR / f"{meeting_id}_pv.md"
    if not pv_path.exists():
        return f"PV not found for meeting: {meeting_id}"
    return pv_path.read_text(encoding="utf-8")


# --- MCP Prompts ---


@mcp.prompt()
def regenerate_pv(meeting_id: str) -> str:
    """Regenerate meeting minutes with custom instructions.

    Use this when the user wants a different format, language, or focus.
    The user can add instructions after this prompt is loaded.
    """
    path = TRANSCRIPTIONS_DIR / f"{meeting_id}.json"
    if not path.exists():
        return f"Transcription not found: {meeting_id}"
    t = Transcription.from_json(path.read_text(encoding="utf-8"))
    transcript = json.dumps(t.to_dict(), ensure_ascii=False, indent=2)
    return (
        f"Here is the transcription of meeting {meeting_id}. "
        f"Generate structured meeting minutes in the user's language.\n\n"
        f"{transcript}"
    )


@mcp.prompt()
def extract_action_items(meeting_id: str) -> str:
    """Extract action items and tasks from a meeting.

    Returns a checklist of actions with responsible person and deadline.
    """
    path = TRANSCRIPTIONS_DIR / f"{meeting_id}.json"
    if not path.exists():
        return f"Transcription not found: {meeting_id}"
    t = Transcription.from_json(path.read_text(encoding="utf-8"))
    transcript = json.dumps(t.to_dict(), ensure_ascii=False, indent=2)
    return (
        f"Extract all action items from this meeting. "
        f"Format: - [ ] Action (responsible person, deadline if mentioned)\n"
        f"Respond in the user's language.\n\n{transcript}"
    )


def main():
    """Entry point for the MCP server."""
    # Auto-cleanup old recordings on startup
    removed = cleanup_old_recordings()
    if removed:
        import logging

        logging.getLogger(__name__).info("Auto-cleanup: removed %d old recordings", len(removed))

    mcp.run()


if __name__ == "__main__":
    main()
