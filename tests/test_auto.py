"""Tests for the auto-recorder finish step (calendar match, error classes, glue)."""

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import httpx
import pytest

from claude_meeting_mcp import auto

UTC = timezone.utc


def _dt(hhmm: str) -> datetime:
    h, m = hhmm.split(":")
    return datetime(2026, 10, 26, int(h), int(m), tzinfo=UTC)


AGENDA = [
    {"calendar": "me@gmail.com", "start": "2026-10-26", "end": "2026-10-27", "summary": "Home"},
    {
        "calendar": "davide@reset.clinic",
        "start": "2026-10-26T09:55:00+00:00",
        "end": "2026-10-26T10:00:00+00:00",
        "summary": "Prep: Weekly sync",
    },
    {
        "calendar": "davide@reset.clinic",
        "start": "2026-10-26T10:00:00+00:00",
        "end": "2026-10-26T10:30:00+00:00",
        "summary": "Weekly sync",
    },
    {
        "calendar": "me@gmail.com",
        "start": "2026-10-26T10:30:00+00:00",
        "end": "2026-10-26T11:00:00+00:00",
        "summary": "Dentist",
    },
]


def test_pick_event_takes_the_largest_overlap_and_skips_all_day():
    event = auto.pick_event(AGENDA, _dt("09:58"), _dt("10:35"))
    assert event is not None and event["summary"] == "Weekly sync"


def test_pick_event_none_for_an_unscheduled_call():
    assert auto.pick_event(AGENDA, _dt("14:00"), _dt("14:20")) is None


def test_attendee_names_excludes_self_and_declined_and_resources():
    event = {
        "attendees": [
            {"email": "me@x.com", "displayName": "Davide Morelli", "self": True},
            {"email": "oliver@reset.clinic", "displayName": "Oliver Clifton"},
            {"email": "ling@reset.clinic"},
            {"email": "no@reset.clinic", "displayName": "Nope", "responseStatus": "declined"},
            {"email": "room@resource.calendar.google.com", "resource": True},
        ]
    }
    assert auto.attendee_names(event) == ["Oliver Clifton", "ling"]


def _status_error(code: int, body: str = "") -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://api.mistral.ai/v1/audio/transcriptions")
    response = httpx.Response(code, request=request, text=body)
    return httpx.HTTPStatusError(f"{code}", request=request, response=response)


@pytest.mark.parametrize(
    ("exc", "kind"),
    [
        (_status_error(401), "mistral_auth"),
        (_status_error(402), "mistral_credits"),
        (_status_error(429, '{"message": "Insufficient credits"}'), "mistral_credits"),
        (_status_error(429, '{"message": "Rate limit exceeded"}'), "mistral_rate_limit"),
        (_status_error(503), "mistral_unavailable"),
        (httpx.ConnectError("no route"), "network"),
        (RuntimeError("MISTRAL_API_KEY not found in the environment"), "mistral_key_missing"),
        (ValueError("boom"), "error"),
    ],
)
def test_classify_error(exc: Exception, kind: str):
    assert auto.classify_error(exc)[0] == kind


def test_finish_success_writes_sidecar_and_reports_done(tmp_path, capsys):
    wav = tmp_path / "2026-10-26_10h00_meeting.wav"
    wav.write_bytes(b"")
    transcription = MagicMock(meeting_id="2026-10-26_10h00_meeting", segments=[1, 2])
    with (
        patch.object(auto, "TRANSCRIPTIONS_DIR", tmp_path),
        patch.object(auto, "lookup_event", return_value=(AGENDA[2], ["Oliver Clifton"])),
        patch.object(auto, "transcribe_meeting", return_value=transcription) as tm,
        patch.object(auto, "notify") as notify,
    ):
        code = auto.finish(str(wav), _dt("10:00"), _dt("10:32"), "us.zoom.xos")

    assert code == 0
    tm.assert_called_once_with(str(wav), "Oliver Clifton", "Davide Morelli")
    sidecar = json.loads((tmp_path / "2026-10-26_10h00_meeting.auto.json").read_text())
    assert sidecar["event"] == "Weekly sync"
    assert sidecar["attendees"] == ["Oliver Clifton"]
    assert sidecar["filed"] is False
    assert json.loads(capsys.readouterr().out.strip().splitlines()[-1])["status"] == "done"
    assert "Weekly sync" in notify.call_args.args[0]


def test_finish_failure_reports_the_error_kind_and_keeps_going(tmp_path, capsys):
    wav = tmp_path / "2026-10-26_14h00_meeting.wav"
    wav.write_bytes(b"")
    with (
        patch.object(auto, "TRANSCRIPTIONS_DIR", tmp_path),
        patch.object(auto, "lookup_event", return_value=(None, [])),
        patch.object(auto, "transcribe_meeting", side_effect=_status_error(402)),
        patch.object(auto, "notify") as notify,
    ):
        code = auto.finish(str(wav), _dt("14:00"), _dt("14:20"), "com.google.Chrome.helper")

    assert code == 1
    # The handoff is durable before transcription: Retry and the router still see it.
    assert (tmp_path / "2026-10-26_14h00_meeting.auto.json").exists()
    result = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert result == {"status": "error", "kind": "mistral_credits", "message": result["message"]}
    assert "credits" in result["message"].lower()
    assert "credits" in notify.call_args.args[0].lower()


def test_finish_survives_a_calendar_failure(tmp_path, capsys):
    """The calendar only names the meeting: gws being logged out must not lose it."""
    wav = tmp_path / "2026-10-26_14h00_meeting.wav"
    wav.write_bytes(b"")
    transcription = MagicMock(meeting_id="2026-10-26_14h00_meeting", segments=[])
    with (
        patch.object(auto, "TRANSCRIPTIONS_DIR", tmp_path),
        patch.object(auto, "lookup_event", side_effect=RuntimeError("gws auth expired")),
        patch.object(auto, "transcribe_meeting", return_value=transcription) as tm,
        patch.object(auto, "notify"),
    ):
        assert auto.finish(str(wav), _dt("14:00"), _dt("14:20"), "us.zoom.xos") == 0

    tm.assert_called_once_with(str(wav), None, "Davide Morelli")


def test_pick_event_ignores_reminder_calendars():
    """'settimana Bianca' marks where Bianca is that week: never a meeting."""
    agenda = [
        {
            "calendar": "settimana Bianca",
            "start": "2026-10-26T09:00:00+00:00",
            "end": "2026-10-26T18:00:00+00:00",
            "summary": "Bianca a Firenze",
        },
        {
            "calendar": "davide@reset.clinic",
            "start": "2026-10-26T10:00:00+00:00",
            "end": "2026-10-26T10:15:00+00:00",
            "summary": "Standup",
        },
    ]
    assert auto.pick_event(agenda, _dt("10:00"), _dt("10:40"))["summary"] == "Standup"
    assert auto.pick_event(agenda, _dt("14:00"), _dt("14:30")) is None
