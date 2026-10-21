# Capture-layer audit - 2026-07-29

**Status: proposal. Nothing implemented. Awaiting approval.**

> ## UPDATE (same day, after the transcript completed) - READ THIS FIRST
>
> The recording measured below was **not** an orphaned capture of a quiet room, as this document originally assumed. It was a **real 55-minute CompanionAI clinical product review**. The finished transcript changes the priority ordering of everything that follows, and adds a finding more severe than any below.
>
> ### Finding 0 (SEVERE) - on speakers the mic channel is ~94% speaker bleed, and the channel split produces confident WRONG attribution
>
> Measured on the completed transcript (`2026-07-29_12h36_meeting.json`, 1245 segments):
>
> | Metric | Value |
> |---|---|
> | Speech time on the `Davide Morelli` (mic) channel | 2972 s of a 3310 s recording |
> | ...of which **simultaneous with remote speech** | **2786 s = 93.8%** |
> | Segments textually duplicated across both channels (>=80% similarity) | 34-36% |
> | Median time offset of those duplicates | **+/- 0.04 s** (i.e. the same instant) |
>
> The 34% duplicate figure is a **lower bound**. Inspecting the non-matching segments shows similarity scores clustered at **0.47-0.58** - the same speech transcribed slightly differently in each channel because the noise floors differ. The true bleed fraction is the 93.8% temporal overlap.
>
> Content confirms it. Segments attributed to `Davide Morelli` include *"My internet dropped out as you asked that question, Rochelle"* and *"we've done the 77 test user interactions"* - other people's words, labelled as his.
>
> **The failure is not that the mic is missing. The mic works.** He was on speakers, the mic captured the room, and the AGC in Finding 1 amplified that bleed to parity with his own voice. The `Local` / `Remote` split - the entire speaker-attribution mechanism the pipeline rests on - is therefore not *degraded*, it is **inverted**: it emits confident, wrong labels that no downstream reader can detect.
>
> This is worse than a hallucination loop. A hallucination is visibly garbage. This is fluent, plausible, correctly-transcribed text attributed to the wrong person, filed into a knowledge base, in a clinical-safety context.
>
> **Mitigating factor observed:** the `/record-meeting` skill's step-5 inference (calendar attendees + context) largely repaired the damage in the filed note `work/reset/2026-07-29-companionai-product-review.md`. Only one claim there is attributed to Davide personally. So the *labels* are broken but the *skill* is compensating - which means this has been silently failing for some time without producing obviously wrong notes.
>
> ### What this does to the findings below
>
> - **Finding 1 (AGC) is confirmed and is a direct amplifier of Finding 0.** `micGain` applies up to +26 dB whenever Davide is quiet - which, at 93.8% overlap, is most of the meeting. It boosts the remote bleed to full transcribable level. Still worth fixing, still a net deletion.
> - **Finding 2 (3-channel interleave) is NOT confirmed and did not fire here.** The mic produced intelligible speech for 55 minutes, so the channel-count bug cannot have been active. It remains a real latent defect worth fixing, but the claim in the original text that it is "the likely no-mic-on-speakers fix" is **withdrawn** - the speakers problem is acoustic bleed, not interleaving.
> - **Findings 3, 4, 5 are unchanged.**
>
> ### Revised verdict: is this worth working on?
>
> **The transcription pipeline is fine. Capture is what is broken, and the single highest-value fix costs nothing:**
>
> 1. **Wear earbuds.** Removes ~94% of the problem at the source, today, with zero code. This was already the standing conclusion; the measurement now quantifies it. It is not a preference, it is the difference between working and not working.
> 2. **Delete the AGC chain** (Finding 1). One hour, a net deletion, removes the amplifier.
> 3. **Surface the swallowed mic error** (Finding 4). Trivial.
>
> That is the whole job **if earbuds are acceptable**. If speakers must work, the only real fix is **offline acoustic echo cancellation**: you already record a clean reference (the system tap) and the complete file, so an NLMS adaptive filter subtracting the delayed system channel from the mic is tractable in post - far easier than real-time AEC, and it is the one thing no commercial local recorder does either. Budget ~150-300 lines plus tests, and validate against exactly the metric above (temporal overlap should fall from 94% to conversational levels).
>
> A **cheap safety net worth adding regardless**: cross-channel dedup at transcript level - drop any `Local` segment matching a `Remote` segment within +/- 1 s at >80% similarity. ~30 lines. It would have removed 209 misattributed segments from today's transcript. It cannot recover Davide's real words when he speaks over bleed, so it is a guard, not a fix.
>
> **Do not trust `Local` / `Remote` labels from any speakers-mode recording made before these fixes land.**

---

# REVISED DESIGN (2026-07-29, after requirements clarified)

## Requirements, as stated

1. **On speakers:** mic-only is acceptable - the mic captures both Davide and the remote party. Bleed is a *feature* in this mode, not a defect.
2. **On earbuds:** must also work - the mic hears only Davide, so the remote side must come from the system tap.
3. **Device switching mid-meeting must work.**

**Consequence: offline AEC is dropped.** It was proposed to remove bleed from the mic in speakers mode. Requirement 1 says the bleed is wanted. That deletes the single most expensive item in this document (~150-300 lines of DSP). Nothing else in the plan depended on it.

## The key design decision: no mode detection

The obvious implementation is to detect the output device (speakers vs headphones) and switch policy. **Do not do this.** It needs a device-type query, a mode variable, a transition state machine, and it fails silently whenever the detection is wrong.

**Cross-channel dedup subsumes it.** After transcribing both channels, drop any `Local` segment that duplicates a `Remote` segment at the same timestamp. Then:

| Situation | Bleed present? | Dedup behaviour | Result |
|---|---|---|---|
| Earbuds | No | Nothing matches - no-op | Clean `Local`/`Remote` split, unchanged |
| Speakers | Heavy | Bleed segments dropped from `Local` | Clean split **restored** |
| Switch mid-meeting | Changes at the switch | Per-segment, adapts instantly | No transition logic needed at all |

One mechanism, three requirements, **zero device detection and zero state**. It self-adapts because it keys off the acoustic evidence itself rather than off a guess about the hardware.

**This also beats the stated requirement 1.** Mic-only in speakers mode would give one undifferentiated channel with no attribution. Dedup instead keeps the *tap* as a clean, high-quality remote channel and leaves the mic carrying only Davide - so speakers mode gains speaker separation it does not have today, at no extra cost.

Tuning: gate on segment length (>= 4 words) so short backchannels ("yeah", "okay") are never dropped. Today's transcript has `Yeah.` 6x on remote and 5x on local; those survive the gate and cause negligible duplication. Match window +/- 1 s, similarity >= 0.80. Applied to today's data this removes 209 misattributed segments.

## What actually has to be built

The tap already satisfies requirement 2's hard part and all of requirement 3's output side - `AudioTapManager.swift:15` creates a **global** process tap and `:42-52` builds an aggregate device containing **only** the tap, no sub-devices. It is device-independent by construction. Verified.

**So the only unmet hard requirement is mic device-follow.** Everything below is scoped to that plus the two confirmed defects.

### 1. Mic device-follow + format-follow (requirement 3 - currently broken)

`StereoRecorder.swift:72-74` resolves device ID, sample rate and `resampleRatio` **once** in `init()` and treats them as constant. Put on earbuds mid-meeting and the mic IOProc is left bound to a device that is no longer default: your own voice is lost for the remainder of the meeting.

- Make `micDeviceID` / `micSampleRate` / `resampleRatio` mutable, guarded by the writer lock.
- Register `AudioObjectAddPropertyListenerBlock` on `kAudioHardwarePropertyDefaultInputDevice` (system object) **and** on `kAudioDevicePropertyStreamFormat` for the current mic device.
- On either firing, rebuild on `writerQueue`: stop + destroy the mic IOProc, re-resolve device and full format (rate **and** channel count), resize and clear the mic ring, reset `resamplePhase`, recreate and start.
- The ~100-500 ms rebuild gap needs no special handling: the drain loop already writes `micVal = 0.0` on ring underrun (`:280`, `:284`), so channel time-alignment is preserved for free.

### 2. Honour the mic channel count (now a prerequisite, not an optional fix)

Finding 2 below was previously "a latent defect". Requirement 3 promotes it to **blocking**: a device-follow that re-resolves the device but keeps assuming mono will corrupt the mic the moment you switch to any multichannel input. The deinterleave must land with the listener, in the same change.

### 3. Delete the AGC chain (Finding 1)

Unchanged, and still a net deletion. In speakers mode it amplifies bleed to parity with Davide's voice, which is what defeats dedup's ability to tell them apart by level; in both modes it lifts the noise floor 26 dB and disables the -60 dBFS silence gate.

### 4. Surface the swallowed mic-start error (Finding 4)

Unchanged. With a device-follow listener added, a *rebuild* failure needs the same treatment - it must not silently downgrade to system-only.

## Practical note worth more than any of the code

When AirPods become the default **input**, macOS drops them to a low-rate voice path (16-24 kHz) and mic quality collapses. Setting output to AirPods while leaving input on the MacBook mic avoids that entirely - and, as a side effect, means an output switch never touches the input device at all. Worth trying before writing any of the above, because it may make requirement 3 moot in the common case.

## Revised effort

| Item | Size | Blocking? |
|---|---|---|
| Mic device-follow + format-follow listeners | ~60-80 lines + tests | **Yes** (requirement 3) |
| Deinterleave mic channels | ~15 lines | **Yes** (prerequisite for the above) |
| Delete AGC chain | net **-40** lines | No, but highest value/effort ratio |
| Surface mic-start / rebuild failure | ~10 lines | No |
| Cross-channel dedup | ~30 lines, Python side | No, but it is what makes speakers mode correct |
| ~~Offline AEC~~ | ~~150-300 lines~~ | **Dropped** - requirement 1 wants the bleed |

### Validation

- Unit: synthetic 3-channel interleaved buffer -> only channel 0 reaches the right WAV channel; -60 dBFS noise in -> ~-60 dBFS out, not -37; simulated default-device change mid-stream -> mic resumes with correct rate and no misalignment.
- Real call, the only test that matters: start on speakers, **switch to earbuds mid-meeting**, switch back. Then check the metric that exposed this whole problem - `Local`/`Remote` temporal overlap should read conversational (<20%) after dedup in both halves, versus **93.8%** today.

## Provenance

Verified this is the code actually in production before auditing:

| Check | Result |
|---|---|
| MCP registration | `claude-meeting-mcp: uv --directory /Users/davidemorelli/personal/claude-meeting-mcp run claude-meeting-mcp` - **Connected** |
| Branch / HEAD | `main` @ `29fc08b` ("Merge feat/faster-meetings"), 6 commits ahead of `origin/main`, **working tree clean** |
| Running capture process | `/Users/davidemorelli/personal/claude-meeting-mcp/src/audiocap/.build/release/audiocap` |
| Binary freshness | built Jun 17 17:41:47, sources last modified Jun 17 17:39:37 - **binary matches source** |
| Fork status | `feat/live-transcription` exists but is **not** checked out; production is on `main` (no live daemon), consistent with the 2026-06-29 decision |

So: yes, you are running this repo, this branch, this binary. The audit below is against `src/audiocap/Sources/AudioCap/StereoRecorder.swift` as it exists on `main`.

## Evidence base

Measured against a **real 49-minute recording** that was in progress during the audit (`2026-07-29_12h36_meeting.wav`, 567 MB). The WAV header was unfinalized (`nframes=0` until `stop()` closes the `AVAudioFile`), so raw PCM was read past the 44-byte header.

50 ms block-RMS distribution over a 60 s window mid-file:

| Channel | min | p5 | p50 | p95 | max | p5..p95 range |
|---|---|---|---|---|---|---|
| **L (system tap)** | -85.0 | **-70.2** | -24.0 | -17.0 | -12.9 | **53.2 dB** |
| **R (microphone)** | -62.6 | **-37.5** | -23.2 | -17.4 | -13.0 | **20.1 dB** |

L/R sample correlation over 10 s: **0.019** (no meaningful acoustic bleed between channels in this recording).

**Read this table carefully - it is the whole finding.** The system channel is healthy: it reaches -85 dBFS in quiet blocks and spans 53 dB, exactly what real content looks like. The microphone channel bottoms out at **-37.5 dBFS** and spans only **20 dB**. The mic noise floor is sitting roughly **30 dB higher than the system channel's**, and the dynamic range is less than half.

That is not a room-tone difference. That is the capture chain.

---

## Finding 1 (CRITICAL) - the mic AGC lifts the noise floor by up to +26 dB

**Where:** `StereoRecorder.swift:294-306`

```swift
private let targetRMS: Float = 0.1       // ~-20dB
private let maxGain: Float = 20.0
...
let micGain = micRMS > 0.001 ? min(targetRMS / micRMS, maxGain) : 1.0
```

This is an automatic gain control with **no noise gate and no floor**. During a quiet passage `micRMS` collapses, so the gain saturates at the `maxGain` cap of 20x = **+26 dB**, and room tone gets amplified straight into the recording.

The arithmetic matches the measurement exactly: observed mic minimum **-62.6 dBFS**, plus 26 dB, lands at **-36.6 dBFS** - against a measured p5 of **-37.5 dBFS**. The `compressSample` stage (`:308-312`, threshold 0.15, ratio 4:1) then squashes the loud end, which is where the 20 dB range comes from.

### Why this is the root cause of a whole family of downstream bugs

1. **The silent-channel skip can never fire on the mic.** `transcriber.py:230` sets `SILENCE_RMS_THRESHOLD = 1e-3`, i.e. **-60 dBFS**. The mic channel's floor is -37.5 dBFS = 0.013 linear - **13x above the gate**. The optimisation added in `4d4f48b` is structurally dead on the exact channel that needs it.

2. **Whisper hallucination loops.** The documented `"Computer"` x200 and `"she says"` x100 collapses are Whisper being handed *amplified room tone* that no longer looks like silence. The tuning response so far has been to fight this downstream - `hallucination_silence_threshold=1.0`, `no_speech_threshold=0.6`, `compression_ratio_threshold=2.4` (`transcriber.py:88-90`), and eventually a silero-VAD gate on the fork. **All of that is treating a symptom manufactured in the capture layer.**

3. **This is the open "loud-no-speech mic channel" item.** The 2026-06-29 note flags a mic channel at "~-2.4 dBFS, ~67% active - loud garbage, neither speech nor silence" and guesses "Bug #4 lineage / AEC / device handling". It is neither AEC nor device handling. It is `micGain`.

4. **It plausibly explains "no mic when on speakers" without any AEC involved.** On speakers the mic hears the remote party loudly, so `micRMS` rises, so `micGain` **drops** - and your own quieter voice is pushed down with it, then 4:1 compressed. The AGC is a single-channel gain riding a signal dominated by speaker bleed. With earbuds there is no bleed, `micRMS` tracks only your voice, and the gain behaves. Same code, opposite outcome, entirely explained by the AGC.

### Proposed fix

**Record flat. Delete the entire processing chain from the capture path** (`:294-318`: normalize, compress, limit, plus the `systemRMS`/`micRMS`/`sysEnvelope`/`micEnvelope`/`compressSample` machinery).

Rationale, in order:
- Whisper performs its own log-mel feature normalisation. Pre-normalising gains nothing and **destroys the silence information** the pipeline depends on.
- A recorder's job is to capture what happened. Gain staging is a mastering decision, and it does not belong upstream of an irreversible write.
- Every downstream anti-hallucination workaround exists to undo this stage.

If a level guard is still wanted, the only defensible one is a **hard limiter at -0.5 dBFS to prevent clipping** (keep `:314-318`, drop everything above it). That is non-destructive; AGC is not.

This is the highest-value change in the document and it is a **net deletion**.

---

## Finding 2 (CRITICAL) - the mic IOProc assumes mono and does not enforce it

**Where:** `StereoRecorder.swift:63`, `:138-142`, `:233-238`

```swift
let micCapacity = Int(sampleRate) * 1 * 5 * MemoryLayout<Float>.size  // :63 - sized for mono
...
// Mic may be multi-channel but we only take first channel's worth   // :141 - the comment
ring.writeBytes(data, count: Int(buf.mDataByteSize))                 // :142 - but the code takes ALL of it
...
let micSamples = micRead / micBytesPerSample                         // :238 - read back as mono Float32
```

The comment states the intent; the code does not implement it. Every interleaved channel is written into a ring buffer that the reader treats as mono.

**Verified on your machine right now:** the built-in mic reports **1 channel / 48 kHz / 4 bytes per frame**, which is why this works most of the time.

**But your own 2026-06-16 finding documents the failure trigger:** when Chrome/Meet opens the mic for WebRTC it forces voice-processing mode and the device goes to the **3-channel / 48 kHz** format, and Core Audio shares a device's nominal format across all clients. When that happens, three interleaved channels are read as consecutive mono samples: the mic track becomes garbage running at 3x rate.

This maps onto the reported symptom precisely. On speakers the input device is the built-in mic, which is the device Meet reconfigures. With earbuds the input device is a different device, which may keep its 1-channel format. **Same code, different device, different outcome** - which is exactly what "breaks on speakers, works on earbuds" looks like.

Note this is a *separate* mechanism from Finding 1 and they compound.

### Proposed fix

1. Query `kAudioDevicePropertyStreamFormat` on the mic device at start; store `micChannels` and `micBytesPerFrame`.
2. In the IOProc, **deinterleave: write only channel 0** into the ring (stride by `micChannels`), or write frames and stride on read. Never write raw `mDataByteSize` again.
3. Size `micCapacity` from the real channel count.
4. Also handle `mBufferList` correctly - `:113` and `:140` both read only `mBuffers` (the first buffer). Correct for interleaved, silently wrong for non-interleaved multi-buffer devices.

---

## Finding 3 (HIGH) - nothing observes device or format changes

**Where:** `StereoRecorder.swift:72-74`, in `init()`

```swift
self.micDeviceID = try StereoRecorder.getDefaultInputDevice()
self.micSampleRate = StereoRecorder.getDeviceSampleRate(micDeviceID) ?? sampleRate
self.resampleRatio = micSampleRate / sampleRate
```

Device identity, sample rate and resample ratio are resolved **once at construction** and assumed constant for the entire meeting. There is no listener anywhere in the file for:

- `kAudioHardwarePropertyDefaultInputDevice` - you put earbuds on mid-call
- `kAudioDevicePropertyStreamFormat` - Meet reconfigures the mic mid-call

These are failure modes 2 and 3 from the requirements list, confirmed by absence rather than by bug. The device-follow listener is already flagged as an unimplemented ~40-line gap in the 2026-06-17 note; the format listener is not yet recorded anywhere.

### Proposed fix

Register `AudioObjectAddPropertyListenerBlock` for both properties. On either firing: stop and destroy the mic IOProc, re-resolve device + format, rebuild the ring buffer, restart. The system tap is unaffected in both cases (a global process tap is device-independent by construction), so the remote side keeps recording throughout the rebuild - only your own channel has a sub-second gap.

---

## Finding 4 (HIGH) - mic start failure is silently swallowed

**Where:** `StereoRecorder.swift:153-160`

```swift
status = AudioDeviceStart(micDeviceID, mProcID)
if status != noErr {
    fputs("audiocap: Mic start failed (\(status)), recording system audio only\n", stderr)
    AudioDeviceDestroyIOProcID(micDeviceID, mProcID)
    self.micProcID = nil
}
```

The recording continues, system-audio-only, and the only trace is a line on stderr that nobody reads. **This is the "sometimes fails randomly" report.** It is not random - it is a discarded `OSStatus`.

### Proposed fix

Surface it. The mic channel is half the product, so a failure to start it must reach the user before the meeting, not after. Either fail the whole `start()`, or propagate a `mic_active: false` field through the MCP `audio_record_start` response so the skill can warn immediately. My preference is the latter - degraded recording beats no recording, but silent degradation beats nothing.

While here: add the channel count to the existing startup log at `:75-79`, so the Finding 2 trigger is visible in the logs the moment it happens.

---

## Finding 5 (MEDIUM) - mic/system clock drift

**Where:** `StereoRecorder.swift:233-237`

```swift
let wantMicSamples = Int(ceil(Double(frameCount) * resampleRatio)) + 2  // +2 for interpolation margin
```

Two extra samples are read from the ring every chunk as interpolation margin, but `resamplePhase` only consumes `frameCount * resampleRatio` of them. The surplus is **not returned to the ring** - it is discarded. The mic therefore drains faster than it fills, accumulating skew against the system channel over the length of the meeting, and eventually underrunning into `micVal = 0.0` (`:280`, `:284`).

Additionally, the linear interpolator at `:270-285` is a low-quality resampler; for speech feeding an ASR model it is acceptable, but it is worth noting that on a device already running at 48 kHz (as measured) `resampleRatio == 1.0` and the whole path is unnecessary work.

### Proposed fix

Peek rather than consume the margin: read the extra samples without advancing the ring's read pointer, or carry the unconsumed tail into the next chunk. Add a fast path that bypasses interpolation entirely when `resampleRatio == 1.0`.

---

## Finding 6 (LOW, operational) - orphaned recording

A recording was running for **49 minutes / 567 MB** during this audit with no meeting in progress. `recorder.py:13` does set `MAX_RECORDING_SECONDS = 4 * 3600`, so there is a backstop - but 4 hours at ~11.6 MB/min is roughly **2.8 GB** before it fires.

Worth considering: a shorter default cap, or a "still recording" reminder surfaced through the MCP after N minutes.

---

## Recommended order of work

Findings 1 and 2 are independent root causes that compound; both are needed.

1. **Finding 1 - delete the AGC/compressor chain.** Largest impact, and it is a deletion. Expected effect: mic floor drops back to true room tone (~-63 dBFS), the `SILENCE_RMS_THRESHOLD` skip starts working, and the hallucination pressure that motivated the VAD gate on the fork largely disappears.
2. **Finding 4 - stop swallowing the mic error.** Trivial, and it makes every subsequent test legible.
3. **Finding 2 - deinterleave the mic.** The likely "no mic on speakers" fix.
4. **Finding 3 - add both listeners.** Closes failure modes 2 and 3.
5. **Finding 5 - drift.** Cheap once the above are in.

### Validation

- **Regression tests first** (TDD, per repo convention): feed a synthetic 3-channel interleaved buffer and assert only channel 0 reaches the right WAV channel; feed a -60 dBFS noise buffer and assert the output floor is still ~-60 dBFS, not -37.
- **The probe that settles Finding 2 empirically:** join a Meet call on speakers and query `kAudioDevicePropertyStreamFormat` on the built-in mic. If it reports 3 channels, Finding 2 is confirmed as the speakers-case trigger. A working probe already exists from this audit and can be folded into the repo as a diagnostic subcommand.
- **Real-call validation** after the fixes: on speakers, then mid-call earbud switch, checking that your own voice survives both.
- **Codex adversarial pass** before merging, per the global rule for multi-file changes.

## What this does NOT fix

If the probe reports the built-in mic at **1 channel during a live Meet call** and your voice is still absent, then Finding 2 is not the speakers trigger and the AEC hypothesis survives on its own merits. In that case no local capture strategy helps, and the honest options are the ones in the notes-repo market survey: platform-side bot capture, or a hardware loopback rig with the mic on an interface input the conferencing app never touches.

Finding 1 is worth doing regardless of how that probe lands.
