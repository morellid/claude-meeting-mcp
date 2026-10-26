import AudioToolbox
import AVFoundation
import CoreAudio
import Foundation

/// Records stereo WAV: Left = system audio (tap), Right = microphone.
/// Uses two independent IOProcs with separate ring buffers.
final class StereoRecorder {
    // System audio (tap via aggregate device)
    private let tapDeviceID: AudioObjectID
    private let systemRing: RingBuffer
    private var systemProcID: AudioDeviceIOProcID?

    // Microphone (default input device). Rebuilt whenever the default input, its
    // sample rate or its channel layout changes: a call app switching the route
    // (AirPods into call mode, voice processing) used to leave us reading a dead
    // stream for the rest of the meeting. All mic state is touched on writerQueue only.
    private var micDeviceID: AudioObjectID = kAudioObjectUnknown
    private var micSink: MicSink
    private var micProcID: AudioDeviceIOProcID?
    private var micListener: AudioObjectPropertyListenerBlock?
    private var defaultInputListener: AudioObjectPropertyListenerBlock?
    private var pendingRebuild: DispatchWorkItem?

    // Output
    private let outputPath: String
    private var audioFile: AVAudioFile?
    private let writerQueue = DispatchQueue(label: "audiocap.writer", qos: .userInitiated)
    private var writerTimer: DispatchSourceTimer?

    // Format
    private let sampleRate: Double
    private let systemChannels: Int
    private var micSampleRate: Double       // native mic rate (may differ from system)
    private var resampleRatio: Double       // micSampleRate / sampleRate
    private var resamplePhase: Double = 0.0 // fractional position in mic buffer

    // Audio processing chain: normalize → compress → limit
    // Normalization: running RMS per channel, gain to target level
    private var systemRMS: Float = 0.0
    private var micRMS: Float = 0.0
    private let rmsSmoothing: Float = 0.05
    private let maxGain: Float = 20.0
    private let targetRMS: Float = 0.1       // ~-20dB

    // Compressor: reduces dynamic range for consistent speech levels
    private let compThreshold: Float = 0.15  // ~-16dB — compress above this
    private let compRatio: Float = 4.0       // 4:1 ratio
    private var sysEnvelope: Float = 0.0     // envelope follower state
    private var micEnvelope: Float = 0.0
    private let attackCoeff: Float           // computed from sample rate
    private let releaseCoeff: Float

    // Limiter: hard ceiling to prevent clipping
    private let limiterCeiling: Float = 0.95 // ~-0.5dB

    // Level watch, on RAW samples (before the gain chain). Prints a level line every
    // minute, and SILENT when neither channel has passed -55 dBFS for two minutes,
    // which the AutoRecorder app turns into a notification. A stalled tap (no
    // frames at all) counts as silence too.
    private static let audibleRMS: Float = 0.0018      // ~-55 dBFS
    private static let silentAfter: UInt64 = 120       // seconds
    private var lastAudible = DispatchTime.now().uptimeNanoseconds
    private var lastSystemFrames = DispatchTime.now().uptimeNanoseconds
    private var reportedSilent = false
    private var reportedStall = false
    private var windowStart = DispatchTime.now().uptimeNanoseconds
    private var windowSysMax: Float = 0
    private var windowMicMax: Float = 0

    init(
        aggregateID: AudioObjectID,
        outputPath: String,
        tapFormat: AudioStreamBasicDescription
    ) throws {
        self.tapDeviceID = aggregateID
        self.outputPath = outputPath
        self.sampleRate = tapFormat.mSampleRate
        self.systemChannels = Int(tapFormat.mChannelsPerFrame)

        // Ring buffers: ~5 seconds each
        let sysCapacity = Int(sampleRate) * systemChannels * 5 * MemoryLayout<Float>.size
        self.systemRing = RingBuffer(capacity: sysCapacity)

        // Compressor envelope coefficients (attack 5ms, release 100ms)
        self.attackCoeff = expf(-1.0 / (Float(sampleRate) * 0.005))
        self.releaseCoeff = expf(-1.0 / (Float(sampleRate) * 0.100))

        // The mic is resolved in startMic(); until then, a placeholder.
        self.micSampleRate = sampleRate
        self.resampleRatio = 1.0
        self.micSink = MicSink(seconds: 5, sampleRate: sampleRate)
        fputs("audiocap: System: \(sampleRate) Hz, \(systemChannels) ch\n", stderr)

        // Create WAV: stereo 16-bit at tap sample rate
        let settings: [String: Any] = [
            AVFormatIDKey: kAudioFormatLinearPCM,
            AVSampleRateKey: sampleRate,
            AVNumberOfChannelsKey: 2,
            AVLinearPCMBitDepthKey: 16,
            AVLinearPCMIsFloatKey: false,
            AVLinearPCMIsBigEndianKey: false,
            AVLinearPCMIsNonInterleaved: false,
        ]
        do {
            self.audioFile = try AVAudioFile(
                forWriting: URL(fileURLWithPath: outputPath),
                settings: settings,
                commonFormat: .pcmFormatFloat32,
                interleaved: false
            )
        } catch {
            throw AudioCapError.fileCreationFailed("\(outputPath): \(error)")
        }
    }

    func start() throws {
        // --- System audio IOProc (on aggregate device) ---
        let sysClientData = Unmanaged.passUnretained(self.systemRing).toOpaque()
        var sysProcID: AudioDeviceIOProcID?
        var status = AudioDeviceCreateIOProcID(
            tapDeviceID,
            { (_, _, inInputData, _, _, _, inClientData) -> OSStatus in
                guard let cd = inClientData else { return noErr }
                let ring = Unmanaged<RingBuffer>.fromOpaque(cd).takeUnretainedValue()
                let buf = inInputData.pointee.mBuffers
                guard let data = buf.mData, buf.mDataByteSize > 0 else { return noErr }
                ring.writeBytes(data, count: Int(buf.mDataByteSize))
                return noErr
            },
            sysClientData,
            &sysProcID
        )
        guard status == noErr, let sysProcID else {
            throw AudioCapError.ioProcFailed(status)
        }
        self.systemProcID = sysProcID

        status = AudioDeviceStart(tapDeviceID, sysProcID)
        guard status == noErr else {
            throw AudioCapError.deviceStartFailed(status)
        }
        fputs("audiocap: System IOProc started\n", stderr)

        // --- Microphone IOProc (on default input device), and follow changes ---
        writerQueue.sync { startMic() }
        let listener: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            self?.scheduleMicRebuild("default input device changed")
        }
        var address = StereoRecorder.address(kAudioHardwarePropertyDefaultInputDevice)
        AudioObjectAddPropertyListenerBlock(
            AudioObjectID(kAudioObjectSystemObject), &address, writerQueue, listener)
        defaultInputListener = listener

        startWriterTimer()
        fputs("audiocap: Recording started\n", stderr)
    }

    func stop() {
        writerTimer?.cancel()
        writerTimer = nil

        if let procID = systemProcID {
            AudioDeviceStop(tapDeviceID, procID)
            AudioDeviceDestroyIOProcID(tapDeviceID, procID)
            systemProcID = nil
        }
        if let listener = defaultInputListener {
            var address = StereoRecorder.address(kAudioHardwarePropertyDefaultInputDevice)
            AudioObjectRemovePropertyListenerBlock(
                AudioObjectID(kAudioObjectSystemObject), &address, writerQueue, listener)
            defaultInputListener = nil
        }

        // Final drain
        writerQueue.sync {
            self.pendingRebuild?.cancel()
            self.stopMic()
            self.drainBuffers()
        }

        audioFile = nil
        fputs("audiocap: Recording stopped\n", stderr)
    }

    // MARK: - Writer

    private func startWriterTimer() {
        let timer = DispatchSource.makeTimerSource(queue: writerQueue)
        timer.schedule(deadline: .now(), repeating: .milliseconds(10))
        timer.setEventHandler { [weak self] in
            self?.drainBuffers()
            self?.checkLevels()
        }
        timer.resume()
        writerTimer = timer
    }

    private func drainBuffers() {
        guard let audioFile else { return }

        let float32Size = MemoryLayout<Float32>.stride
        let sysBytesPerFrame = systemChannels * float32Size
        let micBytesPerSample = float32Size  // mono

        // Read system audio
        let sysAvailable = systemRing.availableBytesToRead
        guard sysAvailable > 0 else { return }
        lastSystemFrames = DispatchTime.now().uptimeNanoseconds
        let micRing = micSink.ring

        let maxFrames = 4096
        let maxSysBytes = maxFrames * sysBytesPerFrame
        let maxMicBytes = maxFrames * micBytesPerSample

        let sysBuffer = UnsafeMutableRawPointer.allocate(byteCount: maxSysBytes, alignment: 16)
        let micBuffer = UnsafeMutableRawPointer.allocate(byteCount: maxMicBytes, alignment: 16)
        defer {
            sysBuffer.deallocate()
            micBuffer.deallocate()
        }

        var sysRemaining = sysAvailable
        while sysRemaining > 0 {
            let toReadSys = min(sysRemaining, maxSysBytes)
            let alignedSys = (toReadSys / sysBytesPerFrame) * sysBytesPerFrame
            guard alignedSys > 0 else { break }

            let sysRead = systemRing.readBytes(sysBuffer, count: alignedSys)
            guard sysRead > 0 else { break }

            let frameCount = sysRead / sysBytesPerFrame

            // Read mic data: we need frameCount * resampleRatio mic samples
            let wantMicSamples = Int(ceil(Double(frameCount) * resampleRatio)) + 2  // +2 for interpolation margin
            let wantMicBytes = wantMicSamples * micBytesPerSample
            let actualMicBytes = min(wantMicBytes, maxMicBytes)
            let micRead = micRing.readBytes(micBuffer, count: actualMicBytes)
            let micSamples = micRead / micBytesPerSample

            let sysPtr = sysBuffer.assumingMemoryBound(to: Float32.self)
            let micPtr = micBuffer.assumingMemoryBound(to: Float32.self)

            // Write stereo: L = system (mono mixdown), R = mic (resampled)
            guard let pcmBuffer = AVAudioPCMBuffer(
                pcmFormat: audioFile.processingFormat,
                frameCapacity: AVAudioFrameCount(frameCount)
            ) else { break }

            pcmBuffer.frameLength = AVAudioFrameCount(frameCount)
            guard let channelData = pcmBuffer.floatChannelData else { break }

            let leftCh = channelData[0]
            let rightCh = channelData[1]

            // === Step 1: Extract raw samples + resample mic ===
            var sysSum: Float = 0.0
            var micSum: Float = 0.0

            for f in 0..<frameCount {
                // Left: system audio
                let sysVal: Float
                if systemChannels >= 2 {
                    sysVal = (sysPtr[f * systemChannels] + sysPtr[f * systemChannels + 1]) * 0.5
                } else {
                    sysVal = sysPtr[f]
                }
                leftCh[f] = sysVal
                sysSum += sysVal * sysVal

                // Right: mic with linear interpolation resampling
                let micVal: Float
                if micSamples > 0 {
                    let idx = Int(resamplePhase)
                    let frac = Float(resamplePhase - Double(idx))
                    if idx + 1 < micSamples {
                        micVal = micPtr[idx] * (1.0 - frac) + micPtr[idx + 1] * frac
                    } else if idx < micSamples {
                        micVal = micPtr[idx]
                    } else {
                        micVal = 0.0
                    }
                    resamplePhase += resampleRatio
                } else {
                    micVal = 0.0
                }
                rightCh[f] = micVal
                micSum += micVal * micVal
            }

            // Keep fractional phase, subtract consumed integer samples
            let consumed = Int(resamplePhase)
            resamplePhase -= Double(consumed)

            // === Step 2: Normalize — bring both channels to target RMS ===
            let chunkSysRMS = sqrtf(sysSum / Float(max(frameCount, 1)))
            let chunkMicRMS = sqrtf(micSum / Float(max(frameCount, 1)))
            windowSysMax = max(windowSysMax, chunkSysRMS)
            windowMicMax = max(windowMicMax, chunkMicRMS)
            if max(chunkSysRMS, chunkMicRMS) > StereoRecorder.audibleRMS {
                lastAudible = DispatchTime.now().uptimeNanoseconds
            }
            systemRMS += (chunkSysRMS - systemRMS) * rmsSmoothing
            micRMS += (chunkMicRMS - micRMS) * rmsSmoothing

            let sysGain = systemRMS > 0.001 ? min(targetRMS / systemRMS, maxGain) : 1.0
            let micGain = micRMS > 0.001 ? min(targetRMS / micRMS, maxGain) : 1.0

            for f in 0..<frameCount {
                leftCh[f] *= sysGain
                rightCh[f] *= micGain
            }

            // === Step 3: Compress — reduce dynamic range (4:1 above threshold) ===
            for f in 0..<frameCount {
                leftCh[f] = compressSample(leftCh[f], envelope: &sysEnvelope)
                rightCh[f] = compressSample(rightCh[f], envelope: &micEnvelope)
            }

            // === Step 4: Limit — hard ceiling at -0.5dB ===
            for f in 0..<frameCount {
                leftCh[f] = max(-limiterCeiling, min(limiterCeiling, leftCh[f]))
                rightCh[f] = max(-limiterCeiling, min(limiterCeiling, rightCh[f]))
            }

            do {
                try audioFile.write(from: pcmBuffer)
            } catch {
                fputs("audiocap: Write error: \(error)\n", stderr)
                break
            }

            sysRemaining -= sysRead
        }
    }

    // MARK: - Audio Processing

    /// Soft-knee compressor: 4:1 ratio above threshold, with envelope follower.
    private func compressSample(_ sample: Float, envelope: inout Float) -> Float {
        let absSample = abs(sample)

        // Envelope follower (fast attack, slow release)
        if absSample > envelope {
            envelope = attackCoeff * envelope + (1.0 - attackCoeff) * absSample
        } else {
            envelope = releaseCoeff * envelope + (1.0 - releaseCoeff) * absSample
        }

        // Compute gain reduction
        if envelope > compThreshold {
            let overDB = 20.0 * log10f(envelope / compThreshold)
            let reducedDB = overDB / compRatio
            let targetLevel = compThreshold * powf(10.0, reducedDB / 20.0)
            let gain = targetLevel / envelope
            return sample * gain
        }

        return sample
    }

    // MARK: - Mic lifecycle (writerQueue only)

    private func startMic() {
        do {
            micDeviceID = try StereoRecorder.getDefaultInputDevice()
        } catch {
            fputs("audiocap: MIC LOST: no default input device\n", stderr)
            return
        }
        micSampleRate = StereoRecorder.getDeviceSampleRate(micDeviceID) ?? sampleRate
        resampleRatio = micSampleRate / sampleRate  // e.g. 44100/48000 = 0.91875
        resamplePhase = 0
        micSink = MicSink(seconds: 5, sampleRate: micSampleRate)
        fputs("audiocap: Mic device: \(micDeviceID), \(micSampleRate) Hz\n", stderr)

        var procID: AudioDeviceIOProcID?
        var status = AudioDeviceCreateIOProcID(
            micDeviceID,
            { (_, _, inInputData, _, _, _, inClientData) -> OSStatus in
                guard let cd = inClientData else { return noErr }
                Unmanaged<MicSink>.fromOpaque(cd).takeUnretainedValue().write(inInputData)
                return noErr
            },
            Unmanaged.passUnretained(micSink).toOpaque(),
            &procID
        )
        guard status == noErr, let procID else {
            fputs("audiocap: MIC LOST: IOProc failed (\(status))\n", stderr)
            return
        }
        status = AudioDeviceStart(micDeviceID, procID)
        guard status == noErr else {
            fputs("audiocap: MIC LOST: start failed (\(status)), system audio only\n", stderr)
            AudioDeviceDestroyIOProcID(micDeviceID, procID)
            return
        }
        micProcID = procID
        fputs("audiocap: Mic IOProc started\n", stderr)

        // Follow this device's format: voice processing or a Bluetooth call mode
        // changes rate or channel layout without changing the default device.
        let listener: AudioObjectPropertyListenerBlock = { [weak self] _, _ in
            self?.scheduleMicRebuild("mic format changed")
        }
        for (selector, scope) in StereoRecorder.micWatched {
            var address = StereoRecorder.address(selector, scope)
            AudioObjectAddPropertyListenerBlock(micDeviceID, &address, writerQueue, listener)
        }
        micListener = listener
    }

    private func stopMic() {
        if let listener = micListener {
            for (selector, scope) in StereoRecorder.micWatched {
                var address = StereoRecorder.address(selector, scope)
                AudioObjectRemovePropertyListenerBlock(micDeviceID, &address, writerQueue, listener)
            }
            micListener = nil
        }
        if let procID = micProcID {
            AudioDeviceStop(micDeviceID, procID)
            AudioDeviceDestroyIOProcID(micDeviceID, procID)
            micProcID = nil
        }
    }

    /// Several properties fire together on one route change; rebuild once.
    private func scheduleMicRebuild(_ reason: String) {
        pendingRebuild?.cancel()
        let work = DispatchWorkItem { [weak self] in
            guard let self, self.writerTimer != nil else { return }
            fputs("audiocap: \(reason), rebuilding mic\n", stderr)
            self.stopMic()
            self.startMic()
        }
        pendingRebuild = work
        writerQueue.asyncAfter(deadline: .now() + .milliseconds(300), execute: work)
    }

    private static let micWatched: [(AudioObjectPropertySelector, AudioObjectPropertyScope)] = [
        (kAudioDevicePropertyNominalSampleRate, kAudioObjectPropertyScopeGlobal),
        (kAudioDevicePropertyStreamConfiguration, kAudioDevicePropertyScopeInput),
        (kAudioDevicePropertyDeviceIsAlive, kAudioObjectPropertyScopeGlobal),
    ]

    // MARK: - Level watch (writerQueue only)

    private func checkLevels() {
        let now = DispatchTime.now().uptimeNanoseconds
        let second: UInt64 = 1_000_000_000

        if now - lastSystemFrames > 5 * second, !reportedStall {
            fputs("audiocap: STALLED system tap delivered nothing for 5 s\n", stderr)
            reportedStall = true
        } else if now - lastSystemFrames <= 5 * second {
            reportedStall = false
        }

        let silentFor = (now - lastAudible) / second
        if silentFor >= StereoRecorder.silentAfter, !reportedSilent {
            fputs("audiocap: SILENT both channels below -55 dBFS for \(silentFor) s\n", stderr)
            reportedSilent = true
        } else if silentFor < StereoRecorder.silentAfter, reportedSilent {
            fputs("audiocap: AUDIO back\n", stderr)
            reportedSilent = false
        }

        if now - windowStart >= 60 * second {
            fputs(String(format: "audiocap: levels sys=%.1f mic=%.1f dBFS (loudest chunk, last minute, raw)\n",
                         StereoRecorder.dbfs(windowSysMax), StereoRecorder.dbfs(windowMicMax)), stderr)
            windowStart = now
            windowSysMax = 0
            windowMicMax = 0
        }
    }

    private static func dbfs(_ rms: Float) -> Float { rms > 0 ? 20 * log10f(rms) : -120 }

    // MARK: - Helpers

    private static func address(
        _ selector: AudioObjectPropertySelector,
        _ scope: AudioObjectPropertyScope = kAudioObjectPropertyScopeGlobal
    ) -> AudioObjectPropertyAddress {
        AudioObjectPropertyAddress(mSelector: selector, mScope: scope, mElement: kAudioObjectPropertyElementMain)
    }

    private static func getDefaultInputDevice() throws -> AudioObjectID {
        var deviceID: AudioObjectID = kAudioObjectUnknown
        var size = UInt32(MemoryLayout<AudioObjectID>.size)
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyDefaultInputDevice,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        let status = AudioObjectGetPropertyData(
            AudioObjectID(kAudioObjectSystemObject), &address, 0, nil, &size, &deviceID
        )
        guard status == noErr, deviceID != kAudioObjectUnknown else {
            throw AudioCapError.noInputDevice
        }
        return deviceID
    }

    private static func getDeviceSampleRate(_ deviceID: AudioObjectID) -> Double? {
        var rate: Float64 = 0
        var size = UInt32(MemoryLayout<Float64>.size)
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioDevicePropertyNominalSampleRate,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain
        )
        let status = AudioObjectGetPropertyData(deviceID, &address, 0, nil, &size, &rate)
        return status == noErr ? rate : nil
    }
}

/// What the mic IOProc writes into; replaced with the IOProc on every mic rebuild.
final class MicSink {
    let ring: RingBuffer
    private let scratch: UnsafeMutablePointer<Float32>
    private let scratchFrames = 16384

    init(seconds: Int, sampleRate: Double) {
        ring = RingBuffer(capacity: Int(sampleRate) * seconds * MemoryLayout<Float32>.size)
        scratch = .allocate(capacity: scratchFrames)
    }

    deinit { scratch.deallocate() }

    /// Keeps channel 0 only. A multichannel input interleaves its channels, and
    /// writing them all into a mono ring plays the mic at N times its rate.
    func write(_ list: UnsafePointer<AudioBufferList>) {
        let buf = list.pointee.mBuffers
        guard let data = buf.mData, buf.mDataByteSize > 0 else { return }
        let channels = Int(max(buf.mNumberChannels, 1))
        if channels == 1 {
            ring.writeBytes(data, count: Int(buf.mDataByteSize))
            return
        }
        let frames = min(Int(buf.mDataByteSize) / (MemoryLayout<Float32>.size * channels), scratchFrames)
        let source = data.assumingMemoryBound(to: Float32.self)
        for f in 0..<frames { scratch[f] = source[f * channels] }
        ring.writeBytes(scratch, count: frames * MemoryLayout<Float32>.size)
    }
}
