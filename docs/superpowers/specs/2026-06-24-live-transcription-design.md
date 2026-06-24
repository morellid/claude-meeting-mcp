# Live incremental transcription - design

Date: 2026-06-24
Status: approved, ready for implementation plan
Branch: `feat/live-transcription`

## Problem

Today the recording flow is record-then-transcribe: the full meeting is captured,
and `transcribe_meeting()` runs only after `stop`. On the default config (local MLX
backend, `large-v3-turbo`, diarization off) the final pass splits the recording into
two channels (left = system/remote, right = mic/local) and transcribes them
**sequentially**, because MLX monopolizes the Metal GPU and `_can_parallelize()`
returns `False` for the `mlx` backend. A 1-hour meeting is therefore ~2 hours of
audio processed one channel after the other, which takes several minutes.

Consequence: by the time the transcript and follow-up questions are ready, the user
has already moved on and must context-switch back. The user wants the transcript to
be ready essentially the instant they say "stop", with mid-meeting querying as a
cheap bonus.

## Goals

- **No-wait (must-have):** the final transcript is ready in well under a second after
  `stop`, regardless of meeting length.
- **Mid-meeting querying (cheap bonus):** a live, growing transcript file Claude can
  read to answer questions during the meeting.
- Reuse the existing live infrastructure rather than building a parallel system.
- Keep the existing batch path as a fallback and for transcribing pre-existing files.

## Non-goals

- No daemon extraction, no menu-bar app, no CLI control surface. The recording engine
  stays embedded in the MCP server process; start/stop stays driven by Claude via MCP
  tools. (Considered and explicitly deferred - see "Deferred".)
- No diarization in the live loop. Channel separation (remote vs local) provides the
  speaker split.
- No background high-quality re-pass in v1 (Approach C). Kept in reserve; see
  "Deferred".

## Background: what already exists

`live_translator.py` already runs a daemon thread during recording that transcribes
every few seconds and writes a live markdown file. It is wired through
`recorder.start_recording(live_translate=...)` and torn down in `stop_recording()`.
It is not usable as a finished transcript because:

1. It keeps only a sliding window (default 15s) and **overwrites** its segments each
   cycle (`self._confirmed_segments = segments`) - there is no accumulated transcript,
   just a ticker of the last few seconds.
2. It reads **only the left/system channel** (`reshape(-1, channels)[:, 0]`) - the
   user's own mic is dropped.
3. It is framed as "translation", not "the transcript".

The stereo WAV written by the `audiocap` Swift binary already contains both channels;
the live code simply discards one. So the data we need is already on disk.

## Approach

**Approach A:** evolve the live path into an append-only, dual-channel transcriber
that *commits* finalized segments as the meeting proceeds. At `stop`, finalize from
what is already accumulated (only the last few seconds of tail need transcribing), so
there is no batch pass to wait for. The growing live file doubles as the mid-meeting
query surface.

Rejected alternatives:
- **B - just speed up the batch pass:** on MLX you cannot parallelize (Metal is
  single-tenant); the only real speedups are a smaller model (quality loss) or a
  remote API (privacy, network, cost), and it still leaves a wait that scales with
  meeting length. Does not deliver no-wait.
- **C - live now + silent background high-quality re-pass:** instant transcript plus a
  background `large-v3-turbo` pass that upgrades the saved JSON minutes later. Removes
  any quality compromise but adds two-transcript reconciliation and PV-timing
  complexity. Held in reserve; only build if the live transcript proves too rough in
  practice. Using `large-v3-turbo` live (see below) is expected to close most of the
  gap and make C unnecessary.

## Architecture and components

All units live inside the MCP server process, replacing/extending `live_translator.py`.

### 1. `DualChannelSource` (evolves `FileAudioSource`)

Reads the growing WAV via the existing byte-offset tracking, but returns
`(left_new, right_new)` instead of dropping the right channel. Same `data` chunk
discovery and 5s read cap it has today.

- Input: the WAV path, sample rate, channel count.
- Output per call: new samples for each channel since the last call, or `None`.

### 2. `StreamingChannel` (the core - one per channel)

Holds only the **uncommitted tail** of audio, not the whole meeting.

State:
- `buffer`: float32 audio since `buffer_start_abs`.
- `buffer_start_abs`: absolute seconds into the meeting of `buffer[0]`.
- `committed: list[Segment]`: finalized segments with **absolute** timestamps.
- `prev_hypothesis`: previous tick's segments (only used by the `agreement` policy).
- `tentative: str`: current uncommitted tail text (for live display).

Per tick:
1. Append new audio to `buffer`.
2. If `buffer` is below a minimum length or is silent (RMS gate), return early.
3. Transcribe `buffer` with Whisper. Whisper returns window-relative times; add
   `buffer_start_abs` to get absolute timestamps.
4. Decide the commit point (see "Commit boundary").
5. For committed segments: append to `committed`, set
   `buffer_start_abs = last_committed_end`, and **drop the committed audio from the
   buffer**. Finalized audio is never re-transcribed, so per-tick cost stays bounded
   regardless of meeting length (the whisper_streaming trim trick).
6. Remaining segments become `tentative`.
7. `prev_hypothesis = current segments`.

This unit is pure and unit-testable with synthetic segment sequences, no audio
required.

### 3. `LiveTranscriber` (orchestrator - replaces `LiveTranslator`)

Owns the daemon thread, a `DualChannelSource`, and two `StreamingChannel`s.

- Each tick: pull `(left_new, right_new)` from the source, feed each channel, RMS-gate
  silence so an idle channel costs zero GPU, write the live markdown. The live markdown
  holds the **full accumulated committed transcript plus the tentative tail** (not a
  window), so it is a complete running transcript Claude can read to answer questions
  mid-meeting, before `stop`.
- Self-pacing: if a tick takes longer than `chunk_seconds` (for example both channels
  busy on sequential MLX), skip the sleep and continue immediately to avoid backlog.
- `stop()`: stop the thread, transcribe each channel's remaining tail once, commit
  everything, merge both channels' `committed` by start time, build a `Transcription`,
  return it.

### 4. Server glue (`server.py`, `recorder.py`)

- `audio_record_start(live=...)`: starts a `LiveTranscriber` when a fast local backend
  is available (see "Compatibility and capability gating"). `live` defaults to **on**
  but transparently falls back to record-then-batch when it cannot run; pass
  `live=false` to force pure capture. `live_translate` kept as a back-compat alias.
- `audio_stop_and_transcribe(...)`: if a live transcriber is active, **finalize from
  it** (sub-second) instead of calling `transcribe_meeting`; otherwise the batch path,
  unchanged. Speaker names are still supplied here.
- `audio_live_status()`: already returns elapsed, segment count, latest text, and the
  live-file path; extend it to report committed-segment count. No new tool for "query
  the meeting" - Claude reads the live markdown file whose path is already in the
  status.

## Commit boundary (the core logic)

The only genuinely new decision: which segments are final (commit, never change) vs
tentative (may change next tick).

- **`hold` policy (ship first):** commit any segment whose end is more than
  `commit_hold_seconds` (~6-8s) behind the audio frontier. Deterministic, trivially
  unit-testable. Trades latency for stability.
- **`agreement` policy (upgrade later):** additionally require the segment to be
  unchanged from `prev_hypothesis` (LocalAgreement-2) before committing. Removes the
  "Whisper revised a word and the committed text was wrong" failure with less latency
  than a large hold. ~15 extra lines.
- **Force-commit cap:** if a channel has continuous speech with no pause, nothing ages
  out and `buffer` grows. When `buffer` exceeds `window_max_seconds` (~30), force-commit
  the oldest segment so per-tick compute and memory stay bounded.

`should_commit()` is the ~8-line heart of the feature; its policy defines the whole
feel and is the natural place for the author to make the call.

## Finalize at stop (the payoff)

1. Stop the thread; transcribe each channel's remaining tail once; commit everything.
2. Merge both channels' `committed` by start time. Channel gives the speaker label:
   left = remote names, right = local names, applied from the names passed to
   `audio_stop_and_transcribe` - exactly the path `merge_segments` already takes with
   diarization off.
3. Write the same `Transcription` JSON schema as today. Only the tail needed
   transcribing, so this completes in well under a second.

## Fallback and robustness (never lose a meeting)

Ordering matters:

- No live transcriber active (recording started with `live=false`): existing
  `transcribe_meeting`, untouched.
- Live transcriber active but produced **zero** segments or errored: fall back to
  `transcribe_meeting` on the WAV.
- Therefore: **finalize-from-live first, delete the WAV only after a non-empty result
  is saved.** Per-tick transcription errors stay caught-and-logged (as today) so one
  bad tick never kills the thread.

## Config

Extend the existing `[live_translation]` block (keep its keys working):

- `model`: default `large-v3-turbo` (same as the batch path). On Apple Silicon it
  transcribes a ~30s window in ~1-2s, so it keeps up, and using the same model is what
  closes the quality gap vs the batch pass. Auto-fall back to `medium` only if ticks
  consistently overrun.
- `chunk_seconds`: tick cadence (self-pacing if a tick overruns).
- `commit_hold_seconds`: ~6-8.
- `window_max_seconds`: ~30 (force-commit cap).
- `commit_policy`: `hold` to start, `agreement` later.
- Translation stays an optional layer on top (`target_language` != source), off by
  default for the no-wait transcript use case.
- Diarization stays off; channels give the speaker split.

`live` defaults to on for `audio_record_start`, subject to the capability gate below.
It never errors; it falls back to record-then-batch when a fast local backend is not
available.

## Compatibility and capability gating

`live` defaulting to on must never degrade or break a configuration that does not match
how we use it (local MLX on Apple Silicon). The live loop is intrinsically
local-and-fast: it runs Whisper on the device every few seconds. That is fine for MLX or
CUDA, but wrong for other setups. So "default on" means **on when a fast local backend is
available; otherwise transparently fall back to today's record-then-batch behaviour, with
no error.**

A `_live_available()` check decides at `audio_record_start`:

- `transcription.mode == "remote"`: live OFF. Per-tick calls to a remote API every few
  seconds, per channel, are impractical (latency, rate limits, cost), and the live path
  has no remote code path at all. Record normally; the existing remote batch pass runs at
  `stop`. The result reports `live=false` with a reason.
- `mode == "local"`, MLX present (macOS Apple Silicon): live ON. This is our case.
- `mode == "local"`, faster-whisper on CUDA: live ON.
- `mode == "local"`, faster-whisper on CPU only: live OFF by default (a large model on
  CPU will not keep up with real time). Allowed only via an explicit `live=true` override.
- No local whisper backend installed: live OFF (the batch path's own error handling is
  unchanged).

When live is gated off, `audio_record_start` still records and simply does not start the
transcriber; `audio_stop_and_transcribe` runs the batch path exactly as today. Nothing the
user relies on changes.

If faster-whisper is used live, the model must be cached across ticks (reuse the
transcriber's `_ensure_faster_model`), not reloaded each tick as the current live path
does.

Cross-platform note: all three capture backends already write a growing stereo WAV (macOS
audiocap; Linux and Windows flush every 500ms), so `DualChannelSource` works on every
platform the recorder supports - live is not macOS-only. It is simply less battle-tested on
Windows/Linux, so those stay behind the same capability gate and fall back cleanly.

## Edge cases and regression guards

- **`audio_record_stop` (stop without transcribe) with live active:** stop the live
  thread, but do NOT finalize or delete the WAV. A subsequent `audio_transcribe(file)` runs
  the batch path on the intact WAV. The live work is discarded; nothing breaks.
- **Explicit `model` override passed to `audio_stop_and_transcribe`:** the live transcript
  was produced with the live-config model, so honouring an override means preferring the
  batch path. If a model override is supplied, finalize via batch, not from live.
- **WAV deletion ordering:** only the finalize/transcribe step deletes the WAV, and only
  after a non-empty `Transcription` is saved. Stopping the live thread never deletes
  anything.
- **Schema parity:** finalize-from-live builds the exact same `Transcription` object
  (including the `speakers` dict) as the batch path, so `get_transcription`,
  `transcriptions_list`, and PV generation are unaffected.
- **Non-meeting captures (YouTube, podcast, music):** the mic/right channel is silent and
  RMS-gated, so only the system/left channel transcribes. Works; the only cost is local GPU
  use for the duration, the accepted trade-off of default-on.
- **`DualChannelSource` on a mono recording:** defensively duplicate the single channel (as
  `split_channels` already does), though tool-made recordings are always stereo.

## Quality mitigations

- Bring the batch path's anti-hallucination Whisper params into the live transcribe
  call: `no_speech_threshold`, `compression_ratio_threshold`,
  `condition_on_previous_text=False`. Today's live path only sets
  `hallucination_silence_threshold`.
- Per-channel RMS silence gate so an idle channel is skipped (no GPU, no hallucination
  on silence).
- Overlapping windows + commit boundary reduce boundary errors.

## Testing

Mirror the existing `tests/` layout.

- `test_streaming_channel.py` (new): the commit-boundary logic is pure. Feed scripted
  hypothesis sequences across ticks; assert what commits under `hold` vs `agreement`,
  absolute timestamps, buffer trim, and that the force-commit cap fires. Fast,
  deterministic, no audio. This is the core test.
- Extend `test_live_translator.py`: fake `LiveAudioSource` yielding scripted chunks +
  a fake transcribe fn; assert dual-channel merge, finalize yields a schema-valid
  `Transcription`, channel -> speaker labels, and the empty-result -> batch fallback
  path with the WAV deleted only after a non-empty save.

## Deferred (not in v1)

- **Approach C** (background high-quality re-pass): revisit only if the live transcript
  is too rough in real use.
- **Daemon extraction + menu-bar / CLI control:** considered. Recording is a singleton
  stateful resource (one audio tap, one WAV, one transcriber), so a Claude-independent
  one-click control surface would require pulling the engine into a detached process
  with both the menu-bar app and the MCP server as thin clients. Deferred in favor of
  keeping the engine embedded in the MCP server; start/stop stays via Claude.

## Open risks

- Streaming single-pass Whisper makes more boundary errors and is more prone to
  silence-hallucination than the offline pass. Mitigated by `large-v3-turbo`, the
  commit boundary, VAD gating, and anti-hallucination params; measured on a real
  meeting before deciding whether C is needed.
- Two busy channels on sequential MLX could push a tick past `chunk_seconds`; handled
  by self-pacing and the force-commit cap, but worth watching on long, fully
  overlapping conversations.
