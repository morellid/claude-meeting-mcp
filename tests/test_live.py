"""Tests for the live transcript: background worker, reader and lifecycle."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from claude_meeting_mcp import live

SR = 48000


def _tone(seconds: float) -> np.ndarray:
    t = np.arange(int(seconds * SR)) / SR
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def _write_stereo(path: Path, left: np.ndarray, right: np.ndarray) -> None:
    sf.write(str(path), np.column_stack([left, right]), SR, subtype="PCM_16")


def test_cut_point_picks_the_pause_near_the_end():
    # 10 s of sound with a 200 ms pause at 9.0 s: the chunk must end there,
    # not mid-word at 10.0 s.
    audio = _tone(10.0)
    audio[int(9.0 * SR) : int(9.2 * SR)] = 0.0
    cut = live.cut_point(audio, SR)
    assert 9.0 <= cut / SR <= 9.2


def test_worker_labels_channels_and_offsets_segments(tmp_path):
    # Left = system audio (the others), right = mic (Davide), as audiocap writes it.
    # Right is silent for the whole file, so only "Remote" lines may appear.
    left = _tone(25.0)
    left[int(9.0 * SR) : int(9.2 * SR)] = 0.0  # pause where chunk 1 ends
    right = np.zeros_like(left)
    wav = tmp_path / "2026-11-09_10h00_00_call.wav"
    _write_stereo(wav, left, right)

    calls: list[float] = []

    def fake_transcribe(audio: np.ndarray) -> list[dict]:
        calls.append(len(audio) / 16000)
        return [{"start": 2.0, "end": 3.0, "text": "ciao"}]

    out = tmp_path / "live.jsonl"
    worker = live.LiveTranscriber(str(wav), out, transcribe=fake_transcribe, sample_rate=SR)
    assert worker.step() is True  # chunk 1: 0 .. ~9.0 s
    assert worker.step() is True  # chunk 2: ~9.0 .. ~19 s
    lines = [json.loads(x) for x in out.read_text().splitlines()]

    assert [x["speaker"] for x in lines] == ["Remote", "Remote"]  # silent mic skipped
    assert lines[0]["start"] == 2.0
    assert 10.9 <= lines[1]["start"] <= 11.3  # 2.0 s into a chunk that starts at the pause
    assert len(calls) == 2  # the silent channel never reached the model


def test_worker_waits_for_a_full_chunk(tmp_path):
    wav = tmp_path / "short_call.wav"
    _write_stereo(wav, _tone(4.0), _tone(4.0))
    worker = live.LiveTranscriber(
        str(wav), tmp_path / "o.jsonl", transcribe=lambda a: [], sample_rate=SR
    )
    assert worker.step() is False


def test_read_live_interleaves_speakers_and_reports_lag(tmp_path):
    transcripts, recordings = tmp_path / "t", tmp_path / "r"
    transcripts.mkdir()
    recordings.mkdir()
    rows = [
        {"start": 70.0, "end": 72.0, "speaker": "Davide", "text": "e il budget?"},
        {"start": 5.0, "end": 8.0, "speaker": "Remote", "text": "partiamo"},
        {"start": 75.0, "end": 80.0, "speaker": "Remote", "text": "entro venerdì"},
    ]
    (transcripts / "m1_call_live.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    _write_stereo(recordings / "m1_call.wav", _tone(100.0), _tone(100.0))

    result = live.read_live(transcripts_dir=transcripts, recordings_dir=recordings)
    assert result["meeting_id"] == "m1_call"
    assert result["transcript"].splitlines() == [
        "[00:05] Remote: partiamo",
        "[01:10] Davide: e il budget?",
        "[01:15] Remote: entro venerdì",
    ]
    assert result["recording_active"] is True
    assert round(result["lag_seconds"]) == 20  # 100 s recorded, transcribed up to 80 s

    last = live.read_live(last_minutes=0.5, transcripts_dir=transcripts, recordings_dir=recordings)
    assert last["transcript"].splitlines() == [
        "[01:10] Davide: e il budget?",
        "[01:15] Remote: entro venerdì",
    ]


def test_read_live_without_any_live_file(tmp_path):
    result = live.read_live(transcripts_dir=tmp_path, recordings_dir=tmp_path)
    assert "error" in result


def test_stop_waits_for_the_worker_to_exit():
    # The batch transcription starts right after stop(); the live model must be out
    # of memory by then, so stop() returns only once the process has exited.
    proc = MagicMock()
    live.stop(proc)
    proc.terminate.assert_called_once()
    proc.wait.assert_called_once()


@patch("claude_meeting_mcp.recorder.get_capturer")
@patch("claude_meeting_mcp.recorder._capturer", None)
@patch("claude_meeting_mcp.recorder._current_file", None)
def test_recorder_starts_worker_and_stops_it_before_returning(mock_get):
    import claude_meeting_mcp.recorder as rec

    capturer = MagicMock()
    capturer.is_available.return_value = True
    mock_get.return_value = capturer
    order: list[str] = []
    proc = MagicMock()
    proc.wait.side_effect = lambda timeout=None: order.append("worker exited")
    capturer.stop.side_effect = lambda: order.append("capture stopped")

    with patch.object(live, "spawn", return_value=proc) as spawn:
        rec.start_recording()
        spawn.assert_called_once_with(rec._current_file)
        rec.stop_recording()

    assert order == ["worker exited", "capture stopped"]


def test_mcp_tool_rejects_path_like_ids():
    from claude_meeting_mcp.server import audio_live_transcript

    assert "error" in audio_live_transcript("../../etc/passwd")


def test_worker_waits_for_audiocap_to_create_the_wav(tmp_path):
    # audiocap creates the file only after setting up the tap, which takes longer
    # than starting the worker: a missing WAV at startup is not "recording over".
    import threading
    import time

    wav = tmp_path / "late_call.wav"
    out = tmp_path / "late.jsonl"
    worker = live.LiveTranscriber(
        str(wav),
        out,
        transcribe=lambda a: [{"start": 0.0, "end": 1.0, "text": "x"}],
        sample_rate=SR,
    )
    done = threading.Event()
    thread = threading.Thread(target=worker.run, args=(done,), daemon=True)
    thread.start()
    time.sleep(0.3)
    assert thread.is_alive()
    _write_stereo(wav, _tone(12.0), _tone(12.0))
    deadline = time.time() + 5
    while not (out.exists() and out.read_text()) and time.time() < deadline:
        time.sleep(0.1)
    done.set()
    thread.join(timeout=3)
    assert out.read_text()


def test_worker_and_reader_use_the_rate_in_the_wav_header(tmp_path):
    # audiocap writes at the output device's rate, which is not always 48 kHz.
    sr = 44100
    t = np.arange(int(25 * sr)) / sr
    left = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    left[int(9.0 * sr) : int(9.2 * sr)] = 0.0
    wav = tmp_path / "m2_call.wav"
    sf.write(str(wav), np.column_stack([left, np.zeros_like(left)]), sr, subtype="PCM_16")

    lengths: list[float] = []

    def fake(audio: np.ndarray) -> list[dict]:
        lengths.append(len(audio) / 16000)
        return [{"start": 2.0, "end": 3.0, "text": "x"}]

    out = tmp_path / "m2_call_live.jsonl"
    worker = live.LiveTranscriber(str(wav), out, transcribe=fake)  # default rate is 48 kHz
    assert worker.step() and worker.step()
    assert 9.0 <= lengths[0] <= 9.2  # the first chunk ends at the pause, in real seconds
    starts = [json.loads(x)["start"] for x in out.read_text().splitlines()]
    assert 10.9 <= starts[1] <= 11.3

    result = live.read_live(transcripts_dir=tmp_path, recordings_dir=tmp_path)
    assert result["recorded_seconds"] == 25


def test_worker_exits_when_its_parent_dies(tmp_path, monkeypatch):
    # An MCP server that dies leaves its children running: the worker must notice
    # it was re-parented and free its model, or it overlaps the next transcription.
    import os
    import threading

    wav = tmp_path / "o_call.wav"
    _write_stereo(wav, _tone(2.0), _tone(2.0))
    worker = live.LiveTranscriber(
        str(wav), tmp_path / "o.jsonl", transcribe=lambda a: [], sample_rate=SR
    )
    monkeypatch.setattr(os, "getppid", lambda: 1)
    thread = threading.Thread(target=worker.run, args=(threading.Event(),), daemon=True)
    thread.start()
    thread.join(timeout=3)
    assert not thread.is_alive()
