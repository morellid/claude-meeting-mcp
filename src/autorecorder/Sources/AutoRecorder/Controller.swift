import Foundation

/// Paths shared with the Python side (claude_meeting_mcp.storage, the default data dir).
enum Paths {
    static let home = FileManager.default.homeDirectoryForCurrentUser
    static let repo = URL(
        fileURLWithPath: ProcessInfo.processInfo.environment["MEETING_REPO"]
            ?? home.appendingPathComponent("personal/claude-meeting-mcp").path)
    static let audiocap = repo.appendingPathComponent("src/audiocap/.build/release/audiocap")
    static let uv = home.appendingPathComponent(".local/bin/uv")
    static let data = home.appendingPathComponent("Library/Application Support/claude-meeting-mcp")
    static let recordings = data.appendingPathComponent("recordings")
    static let transcriptions = data.appendingPathComponent("transcriptions")
    static let lock = data.appendingPathComponent("recording.lock")
    static let log = home.appendingPathComponent("Library/Logs/AutoRecorder.log")
}

func log(_ message: String) {
    let line = "\(ISO8601DateFormatter().string(from: Date())) \(message)\n"
    if let handle = try? FileHandle(forWritingTo: Paths.log) {
        handle.seekToEndOfFile()
        handle.write(Data(line.utf8))
        try? handle.close()
    } else {
        try? line.write(to: Paths.log, atomically: true, encoding: .utf8)
    }
}

/// The machine-wide recording lock, the same flock the MCP recorder takes, so a
/// Claude session and this app never capture one call twice.
final class DeviceLock {
    private var fd: Int32 = -1

    func acquire() -> Bool {
        try? FileManager.default.createDirectory(at: Paths.data, withIntermediateDirectories: true)
        let handle = open(Paths.lock.path, O_CREAT | O_RDWR, 0o644)
        guard handle >= 0 else { return false }
        if flock(handle, LOCK_EX | LOCK_NB) != 0 {
            close(handle)
            return false
        }
        fd = handle
        return true
    }

    func release() {
        if fd >= 0 {
            close(fd)
            fd = -1
        }
    }

    /// True when someone else (a Claude session) is recording right now.
    func heldElsewhere() -> Bool {
        if fd >= 0 { return false }
        if acquire() {
            release()
            return false
        }
        return true
    }
}

struct Recording {
    let file: URL
    let start: Date
    let app: String
    let process: Process
    let manual: Bool
    var lastSeen: Date
}

struct Failure {
    let file: URL
    let start: Date
    let end: Date
    let app: String
    let kind: String
    let message: String
}

/// Watches for calls and drives one recording at a time. Main thread only.
final class Controller {
    // A call app must hold the mic this long before recording starts (mic tests,
    // a quick voice note in a browser), and release it this long before it stops
    // (a dropped connection, a rejoin).
    static let startAfter: TimeInterval = 8
    static let stopAfter: TimeInterval = 90
    static let maxLength: TimeInterval = 2 * 3600
    static let minLength: TimeInterval = 60

    private(set) var recording: Recording?
    private(set) var recordingElsewhere = false
    private(set) var transcribing: [String] = []
    private(set) var failures: [Failure] = []
    private(set) var mistralProblem: String?
    private(set) var lastDone: String?
    var paused: Bool {
        get { UserDefaults.standard.bool(forKey: "paused") }
        set { UserDefaults.standard.set(newValue, forKey: "paused") }
    }

    var onChange: () -> Void = {}

    private let lock = DeviceLock()
    private var candidateSince: Date?
    /// After a manual stop the call is still on: do not restart until the mic is free.
    private var suppressUntilMicFree = false

    func tick() {
        let app = MicWatcher.callAppUsingMic()
        if app == nil { suppressUntilMicFree = false }
        let now = Date()

        if var current = recording {
            if !current.process.isRunning {
                captureDied(current)
            } else {
                if app != nil { current.lastSeen = now }
                recording = current
                let idle = now.timeIntervalSince(current.lastSeen)
                if !current.manual && idle > Controller.stopAfter {
                    stop(reason: "call ended")
                } else if now.timeIntervalSince(current.start) > Controller.maxLength {
                    stop(reason: "2h cap")
                }
            }
        } else {
            recordingElsewhere = lock.heldElsewhere()
            if let app, !paused, !suppressUntilMicFree, !recordingElsewhere {
                if let since = candidateSince {
                    if now.timeIntervalSince(since) >= Controller.startAfter {
                        candidateSince = nil
                        start(app: app, manual: false)
                    }
                } else {
                    candidateSince = now
                }
            } else {
                candidateSince = nil
            }
        }
        onChange()
    }

    func start(app: String, manual: Bool) {
        guard recording == nil else { return }
        guard lock.acquire() else {
            recordingElsewhere = true
            onChange()
            return
        }
        try? FileManager.default.createDirectory(at: Paths.recordings, withIntermediateDirectories: true)
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH'h'mm_ss"
        let file = Paths.recordings.appendingPathComponent("\(formatter.string(from: Date()))_call.wav")

        let process = Process()
        process.executableURL = Paths.audiocap
        process.arguments = ["--output", file.path]
        let logHandle = try? FileHandle(forWritingTo: Paths.log)
        logHandle?.seekToEndOfFile()
        process.standardError = logHandle
        process.standardOutput = logHandle
        do {
            try process.run()
        } catch {
            lock.release()
            fail(Failure(file: file, start: Date(), end: Date(), app: app, kind: "capture",
                         message: "Could not start audiocap: \(error.localizedDescription)"))
            suppressUntilMicFree = true
            return
        }
        recording = Recording(file: file, start: Date(), app: app, process: process,
                              manual: manual, lastSeen: Date())
        log("recording started (\(manual ? "manual" : app)): \(file.lastPathComponent)")
        onChange()
    }

    func stop(reason: String, manualStop: Bool = false) {
        guard let current = recording else { return }
        recording = nil
        if manualStop { suppressUntilMicFree = true }
        current.process.interrupt()  // SIGINT: audiocap finalises the WAV header
        let end = Date()
        log("recording stopped (\(reason)): \(current.file.lastPathComponent)")
        DispatchQueue.global().async { [self] in
            current.process.waitUntilExit()
            DispatchQueue.main.async { [self] in
                lock.release()
                finishRecording(current, end: end)
            }
        }
        onChange()
    }

    /// Synchronous stop for quit/logout: the WAV must be complete before we exit.
    func stopForExit() {
        guard let current = recording else { return }
        recording = nil
        current.process.interrupt()
        current.process.waitUntilExit()
        lock.release()
        finishRecording(current, end: Date())
    }

    private func captureDied(_ current: Recording) {
        recording = nil
        lock.release()
        suppressUntilMicFree = true
        let end = Date()
        let size = (try? current.file.resourceValues(forKeys: [.fileSizeKey]).fileSize) ?? 0
        if size > 0 && end.timeIntervalSince(current.start) >= Controller.minLength {
            finishRecording(current, end: end)
        }
        fail(Failure(file: current.file, start: current.start, end: end, app: current.app,
                     kind: "capture",
                     message: "audiocap exited (code \(current.process.terminationStatus)). "
                         + "Check microphone and system audio permissions; see the log."))
    }

    private func finishRecording(_ current: Recording, end: Date) {
        if end.timeIntervalSince(current.start) < Controller.minLength {
            try? FileManager.default.removeItem(at: current.file)
            log("discarded recording under \(Int(Controller.minLength)) s")
            onChange()
            return
        }
        transcribe(file: current.file, start: current.start, end: end, app: current.app)
    }

    private func transcribe(file: URL, start: Date, end: Date, app: String) {
        let iso = ISO8601DateFormatter()
        let name = file.lastPathComponent
        transcribing.append(name)
        onChange()
        runPython(["finish", file.path, iso.string(from: start), iso.string(from: end), app]) {
            [self] result in
            transcribing.removeAll { $0 == name }
            if result["status"] as? String == "done" {
                lastDone = "\(result["title"] as? String ?? name) at "
                    + DateFormatter.localizedString(from: Date(), dateStyle: .none, timeStyle: .short)
                failures.removeAll { $0.file == file }
                mistralProblem = nil
            } else {
                let kind = result["kind"] as? String ?? "error"
                let message = result["message"] as? String ?? "Transcription failed; see the log."
                fail(Failure(file: file, start: start, end: end, app: app, kind: kind, message: message))
            }
            onChange()
        }
    }

    private func fail(_ failure: Failure) {
        failures.removeAll { $0.file == failure.file }
        failures.append(failure)
        if failure.kind.hasPrefix("mistral") { mistralProblem = failure.message }
        log("failure [\(failure.kind)]: \(failure.message)")
        onChange()
    }

    func retryFailures() {
        let pending = failures.filter { FileManager.default.fileExists(atPath: $0.file.path) }
        failures.removeAll()
        for failure in pending {
            transcribe(file: failure.file, start: failure.start, end: failure.end, app: failure.app)
        }
        onChange()
    }

    func checkMistral() {
        runPython(["check"]) { [self] result in
            mistralProblem = result["status"] as? String == "ok"
                ? nil : (result["message"] as? String ?? "Mistral check failed; see the log.")
            onChange()
        }
    }

    /// Runs `meeting-auto <args>` from the repo and hands its last stdout line, parsed.
    private func runPython(_ args: [String], completion: @escaping ([String: Any]) -> Void) {
        let process = Process()
        process.executableURL = Paths.uv
        process.arguments = ["--directory", Paths.repo.path, "run", "meeting-auto"] + args
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = "/opt/homebrew/bin:\(Paths.home.path)/.local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
        process.environment = env
        let stdout = Pipe()
        process.standardOutput = stdout
        let logHandle = try? FileHandle(forWritingTo: Paths.log)
        logHandle?.seekToEndOfFile()
        process.standardError = logHandle
        process.terminationHandler = { _ in
            let output = String(decoding: stdout.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
            let last = output.split(separator: "\n").last.map(String.init) ?? ""
            let parsed = (try? JSONSerialization.jsonObject(with: Data(last.utf8))) as? [String: Any]
            DispatchQueue.main.async {
                completion(parsed ?? ["status": "error", "kind": "error",
                                      "message": "meeting-auto \(args[0]) gave no result; see the log."])
            }
        }
        do {
            try process.run()
        } catch {
            completion(["status": "error", "kind": "error",
                        "message": "Could not run uv: \(error.localizedDescription)"])
        }
    }
}
