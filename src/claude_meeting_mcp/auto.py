"""Finish step for the auto-recorder menu bar app (src/autorecorder).

The app records a call when a conferencing app opens the microphone, then runs
``meeting-auto finish <wav> <start> <end> <bundle_id>``. This step names the
meeting from the calendar, transcribes it, leaves a sidecar for the agent router
to file the note, and tells Davide on Telegram. Its last stdout line is a JSON
status the app shows in the menu bar; errors are classified so the menu can say
"Mistral credits exhausted" rather than "HTTP 402".
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import httpx

from .storage import TRANSCRIPTIONS_DIR
from .transcriber import MISTRAL_URL, _mistral_api_key, transcribe_meeting

logger = logging.getLogger(__name__)

SELF_NAME = "Davide Morelli"
TELEGRAM_HOOK = Path.home() / ".claude" / "hooks" / "telegram.sh"

# Calendars that are reminders, never meetings: "settimana Bianca" says where
# Bianca is that week, and its all-week events would out-overlap any real call.
IGNORED_CALENDARS = {"settimana Bianca"}

# Bundle id prefix -> what the Telegram message calls the app.
APP_LABELS = {
    "us.zoom.": "Zoom",
    "com.google.Chrome": "Chrome",
    "com.microsoft.teams": "Teams",
    "com.tinyspeck.slackmacgap": "Slack",
    "net.whatsapp.": "WhatsApp",
    "desktop.WhatsApp": "WhatsApp",
}


def _parse(value: str) -> datetime | None:
    """Timed agenda entries carry an offset; all-day ones are bare dates and are skipped."""
    if "T" not in value:
        return None
    return datetime.fromisoformat(value)


def pick_event(events: list[dict], start: datetime, end: datetime) -> dict | None:
    """The timed calendar event overlapping the recording the most, if any.

    Largest overlap rather than first match: the daily brief puts a 5-minute prep
    event right before each meeting, and a call that starts early overlaps it.
    """
    best, best_overlap = None, 0.0
    for event in events:
        if event.get("calendar") in IGNORED_CALENDARS:
            continue
        ev_start, ev_end = _parse(event.get("start", "")), _parse(event.get("end", ""))
        if ev_start is None or ev_end is None:
            continue
        overlap = (min(end, ev_end) - max(start, ev_start)).total_seconds()
        if overlap > best_overlap:
            best, best_overlap = event, overlap
    return best


def attendee_names(event: dict) -> list[str]:
    """Other people invited to the event: not Davide, not rooms, not those who declined."""
    names = []
    for a in event.get("attendees") or []:
        if a.get("self") or a.get("resource") or a.get("responseStatus") == "declined":
            continue
        names.append(a.get("displayName") or a.get("email", "").split("@")[0])
    return [n for n in names if n]


def _gws(*args: str) -> dict:
    out = subprocess.run(["gws", *args], capture_output=True, text=True, timeout=60, check=True)
    # gws prints a "Using keyring backend" line before the JSON on some setups.
    return json.loads(out.stdout[out.stdout.index("{") :])


def lookup_event(start: datetime, end: datetime) -> tuple[dict | None, list[str]]:
    """The calendar event this recording belongs to, and its attendees' names."""
    agenda = _gws("calendar", "+agenda", "--today", "--format", "json").get("events", [])
    event = pick_event(agenda, start, end)
    if event is None:
        return None, []
    params = {
        "calendarId": event["calendar"],
        "timeMin": event["start"],
        "timeMax": event["end"],
        "singleEvents": True,
    }
    try:
        items = _gws("calendar", "events", "list", "--params", json.dumps(params)).get("items", [])
    except (subprocess.SubprocessError, ValueError):
        # Some agenda entries carry a calendar's display name, not its id. Keep the
        # title; the names were only spelling hints.
        logger.warning(
            "No attendees for %r on calendar %r", event.get("summary"), event["calendar"]
        )
        return event, []
    full = next((i for i in items if i.get("summary") == event.get("summary")), None)
    return event, attendee_names(full) if full else []


def classify_error(exc: Exception) -> tuple[str, str]:
    """(kind, message) for the menu bar and Telegram."""
    if isinstance(exc, httpx.HTTPStatusError):
        code, body = exc.response.status_code, exc.response.text.lower()
        if code in (401, 403):
            return "mistral_auth", "Mistral rejected the API key. Check MISTRAL_API_KEY."
        money = ("credit", "quota", "billing", "payment", "insufficient")
        if code == 402 or (code == 429 and any(w in body for w in money)):
            return "mistral_credits", "Mistral credits exhausted or a billing problem."
        if code == 429:
            return "mistral_rate_limit", "Mistral rate limit hit. Retry in a few minutes."
        if code >= 500:
            return "mistral_unavailable", f"Mistral is unavailable (HTTP {code})."
        return "error", f"Mistral returned HTTP {code}."
    if isinstance(exc, httpx.TransportError):
        return "network", "No connection to Mistral."
    if "MISTRAL_API_KEY" in str(exc):
        return "mistral_key_missing", "No Mistral API key in the environment or the Keychain."
    return "error", f"{type(exc).__name__}: {exc}"


def notify(message: str) -> None:
    """Telegram via the existing hook, through an interactive zsh for its env vars.

    The app runs under launchd, which has no TELEGRAM_* variables; they live in
    .zshrc. Best effort: a missed Telegram must never fail the finish step.
    """
    try:
        subprocess.run(
            ["/bin/zsh", "-ic", f'"{TELEGRAM_HOOK}" "$1"', "zsh", message],
            capture_output=True,
            timeout=30,
        )
    except Exception:  # noqa: BLE001
        logger.exception("Telegram notification failed")


def _app_label(bundle_id: str) -> str:
    return next((v for k, v in APP_LABELS.items() if bundle_id.startswith(k)), bundle_id)


def _emit(result: dict) -> None:
    print(json.dumps(result), flush=True)


def finish(wav_path: str, start: datetime, end: datetime, bundle_id: str) -> int:
    """Name, transcribe and hand over one recording. Returns the process exit code."""
    try:
        event, names = lookup_event(start, end)
    except Exception:  # noqa: BLE001 - the calendar only names the meeting
        logger.exception("Calendar lookup failed")
        event, names = None, []
    title = event["summary"] if event else f"Unscheduled call ({_app_label(bundle_id)})"
    minutes = round((end - start).total_seconds() / 60)

    # The sidecar goes down BEFORE transcribing: transcribe_meeting deletes the WAV
    # as soon as the transcript is written, so a sidecar written afterwards could be
    # lost with nothing left to retry. The router only files a sidecar once its
    # transcript exists.
    meeting_id = Path(wav_path).stem
    sidecar = {
        "meeting_id": meeting_id,
        "event": title,
        "calendar": event["calendar"] if event else None,
        "attendees": names,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "app": bundle_id,
        "filed": False,
    }
    TRANSCRIPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    path = TRANSCRIPTIONS_DIR / f"{meeting_id}.auto.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(sidecar, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)

    try:
        transcribe_meeting(wav_path, ", ".join(names) or None, SELF_NAME)
    except Exception as exc:  # noqa: BLE001
        kind, message = classify_error(exc)
        logger.exception("Transcription failed for %s", wav_path)
        notify(
            f"Recorded {title} ({minutes} min) but transcription failed: {message} "
            "The recording is kept: use Retry in the menu bar."
        )
        _emit({"status": "error", "kind": kind, "message": message})
        return 1

    notify(f"Recorded {title} ({minutes} min), transcribed. The agent router files the note.")
    _emit({"status": "done", "meeting_id": meeting_id, "title": title})
    return 0


def check() -> int:
    """Preflight for the menu bar: is the Mistral key present and accepted?"""
    try:
        response = httpx.get(
            MISTRAL_URL.replace("/audio/transcriptions", "/models"),
            headers={"Authorization": f"Bearer {_mistral_api_key()}"},
            timeout=20.0,
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        kind, message = classify_error(exc)
        _emit({"status": "error", "kind": kind, "message": message})
        return 1
    _emit({"status": "ok"})
    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    args = sys.argv[1:]
    if args[:1] == ["check"]:
        sys.exit(check())
    if args[:1] == ["finish"] and len(args) == 5:
        _, wav, start, end, bundle_id = args
        sys.exit(finish(wav, datetime.fromisoformat(start), datetime.fromisoformat(end), bundle_id))
    print("Usage: meeting-auto check | finish <wav> <start-iso> <end-iso> <bundle-id>")
    sys.exit(2)
