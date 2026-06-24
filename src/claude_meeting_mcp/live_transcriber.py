"""Live incremental dual-channel transcription during recording.

Reads the growing stereo WAV, transcribes each channel's uncommitted tail every
tick, commits finalized segments via StreamingChannel, and writes a live markdown
file holding the full running transcript (committed + current tentative tail). At
stop, finalize() merges both channels into a Transcription with no batch pass.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np

from .config import get_config
from .schemas import Segment, Transcription
from .streaming import Hyp, StreamingChannel

logger = logging.getLogger(__name__)

_SILENCE_RMS = 0.001
# At stop we transcribe whatever audio remains in a channel buffer exactly once,
# so we can afford a much smaller floor than the per-tick 1s guard - this keeps a
# short final utterance from being silently dropped.
_MIN_FINALIZE_SECONDS = 0.2


class LiveTranscriber:
    def __init__(
        self,
        source,
        output_path: str,
        remote_label: str = "Remote",
        local_label: str = "Local",
    ) -> None:
        cfg = get_config().live_translation
        self._source = source
        self._sr = source.get_sample_rate()
        self._output_path = Path(output_path)
        self._model = cfg.model
        self._chunk_seconds = cfg.chunk_seconds

        def _mk_channel() -> StreamingChannel:
            return StreamingChannel(
                commit_hold_seconds=cfg.commit_hold_seconds,
                window_max_seconds=cfg.window_max_seconds,
                commit_policy=cfg.commit_policy,
            )

        self._chan = {"left": _mk_channel(), "right": _mk_channel()}
        self._buf = {"left": np.empty(0, np.float32), "right": np.empty(0, np.float32)}
        self._buf_start_sample = {"left": 0, "right": 0}
        self._total_samples = 0

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0
        self._lock = threading.Lock()

    # --- lifecycle ---

    def start(self) -> None:
        self._start_time = time.monotonic()
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._write_markdown()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
            self._thread = None

    def _run(self) -> None:
        time.sleep(2.0)  # let audio start flowing
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                self._tick()
                self._write_markdown()
            except Exception as e:  # a bad tick must never kill the daemon
                logger.error("Live tick error: %s", e)
            elapsed = time.monotonic() - t0
            # Self-pacing: only sleep if the tick was faster than the cadence.
            if elapsed < self._chunk_seconds:
                self._stop_event.wait(timeout=self._chunk_seconds - elapsed)

    # --- core ---

    def _tick(self) -> None:
        # The daemon is the only writer of _buf / _total_samples, so these reads
        # and writes need no lock. The lock guards only the commit-state lists
        # that get_status() reads from another thread (see _process_channel).
        left, right = self._source.get_new_audio_both()
        if left is not None and len(left) > 0:
            self._total_samples += len(left)
            self._buf["left"] = np.concatenate([self._buf["left"], left])
            self._buf["right"] = np.concatenate([self._buf["right"], right])
        frontier_abs = self._total_samples / self._sr
        for channel in ("left", "right"):
            self._process_channel(channel, frontier_abs)

    def _process_channel(self, channel: str, frontier_abs: float) -> None:
        buf = self._buf[channel]
        if len(buf) < self._sr:  # less than 1s of audio
            return
        rms = float(np.sqrt(np.mean(buf**2)))
        if rms < _SILENCE_RMS:
            # Idle channel: drop silence so the buffer does not grow unbounded.
            cap = int(self._chan[channel].window_max_seconds * self._sr)
            if len(buf) > cap:
                with self._lock:
                    ch = self._chan[channel]
                    # Commit any uncommitted speech before discarding the buffer,
                    # otherwise a long trailing silence would erase it.
                    if ch.tentative_segs:
                        ch.committed.extend(ch.tentative_segs)
                        ch.tentative_segs = []
                    self._buf[channel] = np.empty(0, np.float32)
                    self._buf_start_sample[channel] = self._total_samples
                    ch.buffer_start_abs = frontier_abs
            return
        # _transcribe is slow (seconds) and must run WITHOUT the lock; only the
        # fast commit-state mutation below is locked against get_status().
        rel_segments = self._transcribe(buf, channel)
        with self._lock:
            new_start_abs = self._chan[channel].update(rel_segments, frontier_abs)
            self._trim_locked(channel, new_start_abs)

    def _trim_locked(self, channel: str, new_start_abs: float) -> None:
        """Drop committed audio from the front of the channel buffer.

        Caller must hold self._lock.
        """
        new_start_sample = round(new_start_abs * self._sr)
        drop = new_start_sample - self._buf_start_sample[channel]
        if drop > 0:
            self._buf[channel] = self._buf[channel][drop:]
            self._buf_start_sample[channel] = new_start_sample

    def _transcribe(self, audio: np.ndarray, channel: str) -> list[dict]:
        """Run Whisper on a channel buffer; return segments relative to buffer start."""
        from .transcriber import _get_backend, _resample_to_16k, _transcribe_faster, _transcribe_mlx

        audio16 = _resample_to_16k(audio, self._sr)
        backend = _get_backend()
        try:
            if backend == "mlx":
                segs = _transcribe_mlx(audio16, 16000, self._model)
            else:
                segs = _transcribe_faster(audio16, 16000, self._model)
        except Exception as e:  # one bad tick must not kill the thread
            logger.error("Live transcribe error (%s): %s", channel, e)
            return []
        return [{"start": s["start"], "end": s["end"], "text": s["text"]} for s in segs]

    # --- finalize ---

    def finalize(self, remote_speakers: str, local_speakers: str) -> Transcription | None:
        """Flush remaining tails and merge both channels. None if nothing committed."""
        # Drain any audio written after the daemon stopped - notably the frames the
        # capturer flushes during stop_recording - so the meeting's tail is not lost.
        left, right = self._source.get_new_audio_both()
        if left is not None and len(left) > 0:
            self._total_samples += len(left)
            self._buf["left"] = np.concatenate([self._buf["left"], left])
            self._buf["right"] = np.concatenate([self._buf["right"], right])

        frontier_abs = self._total_samples / self._sr
        min_tail = int(_MIN_FINALIZE_SECONDS * self._sr)
        for channel in ("left", "right"):
            buf = self._buf[channel]
            if len(buf) >= min_tail:
                rms = float(np.sqrt(np.mean(buf**2)))
                if rms >= _SILENCE_RMS:
                    self._chan[channel].flush(self._transcribe(buf, channel))

        remote_label = (remote_speakers or "Remote").split(",")[0].strip()
        local_label = (local_speakers or "Local").split(",")[0].strip()
        labels = {"left": remote_label, "right": local_label}

        segments: list[Segment] = []
        for channel in ("left", "right"):
            label = labels[channel]
            for h in self._chan[channel].committed:
                segments.append(Segment(round(h.start, 2), round(h.end, 2), label, h.text))
        if not segments:
            return None
        segments.sort(key=lambda s: s.start)

        meeting_id = self._output_path.stem.replace("_live", "")
        return Transcription(
            meeting_id=meeting_id,
            date=meeting_id[:10] if len(meeting_id) >= 10 else "",
            duration_seconds=round(frontier_abs, 1),
            speakers={"left": remote_label, "right": local_label},
            segments=segments,
        )

    # --- status + markdown ---

    def _running_transcript(self) -> list[Hyp]:
        """Committed + current tentative segments from both channels, time-ordered.

        This is what the live view shows: the latest words appear immediately as
        tentative, before they age into the committed (final) transcript.
        """
        hyps: list[Hyp] = []
        for channel in ("left", "right"):
            hyps.extend(self._chan[channel].committed)
            hyps.extend(self._chan[channel].tentative_segs)
        hyps.sort(key=lambda h: h.start)
        return hyps

    def get_status(self) -> dict:
        with self._lock:
            running = self._running_transcript()
            committed = self._chan["left"].committed + self._chan["right"].committed
            return {
                "status": "stopped" if self._stop_event.is_set() else "transcribing",
                "elapsed_seconds": round(time.monotonic() - self._start_time, 1),
                "model": self._model,
                "committed_segments": len(committed),
                "latest_text": [h.text for h in running[-5:]],
                "live_file": str(self._output_path),
            }

    def _write_markdown(self, final: bool = False) -> None:
        elapsed = time.monotonic() - self._start_time
        if final:
            status = "Completed"
        else:
            status = f"Recording ({int(elapsed // 60)}m {int(elapsed % 60):02d}s)"
        running = self._running_transcript()
        lines = ["# Live Transcript", f"**Status**: {status}", "", "---", ""]
        for h in running:
            lines.append(f"[{int(h.start // 60)}:{int(h.start % 60):02d}] {h.text}")
        content = "\n".join(lines) + "\n"
        self._output_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._output_path.with_suffix(".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.rename(self._output_path)
