import AppKit
import UserNotifications

// =============================================================================
// AutoRecorder - menu bar app that records every call automatically
//
// A call is on when a conferencing app (Zoom, Meet in a browser, Teams, Slack)
// holds the microphone. The app then runs audiocap until the mic has been free
// for 30 s, and hands the WAV to `meeting-auto finish` (Python), which names it
// from the calendar, transcribes it with Mistral and leaves it for the agent
// router to file. The menu bar icon shows the state; errors offer a retry.
// =============================================================================

final class AppDelegate: NSObject, NSApplicationDelegate, NSMenuDelegate,
    UNUserNotificationCenterDelegate
{
    private let controller = Controller()
    private var statusItem: NSStatusItem!
    private var timers: [Timer] = []
    private var signalSources: [DispatchSourceSignal] = []

    func applicationDidFinishLaunching(_ notification: Notification) {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)
        let menu = NSMenu()
        menu.delegate = self
        statusItem.menu = menu
        controller.onChange = { [weak self] in self?.updateIcon() }
        controller.onEvent = { [weak self] title, body in self?.notify(title, body) }
        UNUserNotificationCenter.current().delegate = self
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound]) { granted, _ in
            log("notifications \(granted ? "allowed" : "not allowed")")
        }

        timers.append(Timer.scheduledTimer(withTimeInterval: 3, repeats: true) { [weak self] _ in
            self?.controller.tick()
        })
        timers.append(Timer.scheduledTimer(withTimeInterval: 6 * 3600, repeats: true) { [weak self] _ in
            self?.controller.checkMistral()
        })
        controller.checkMistral()

        // launchd sends SIGTERM at logout: finish the WAV instead of dying mid-call.
        for sig in [SIGTERM, SIGINT] {
            signal(sig, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
            source.setEventHandler { NSApp.terminate(nil) }
            source.resume()
            signalSources.append(source)
        }
        log("AutoRecorder started")
        controller.tick()
    }

    func applicationWillTerminate(_ notification: Notification) {
        controller.stopForExit()
        log("AutoRecorder quit")
    }

    // MARK: - Notifications

    private func notify(_ title: String, _ body: String) {
        let content = UNMutableNotificationContent()
        content.title = title
        content.body = body
        let request = UNNotificationRequest(identifier: UUID().uuidString, content: content, trigger: nil)
        UNUserNotificationCenter.current().add(request)
    }

    /// Show banners even though an accessory app is never "in the foreground".
    func userNotificationCenter(
        _ center: UNUserNotificationCenter, willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        completionHandler([.banner, .list])
    }

    // MARK: - Icon

    private func updateIcon() {
        let (symbol, color, tip): (String, NSColor?, String) = {
            if controller.recording != nil {
                return ("record.circle.fill", .systemRed, "Recording")
            }
            if controller.recordingElsewhere {
                return ("record.circle", .systemRed, "Recording from a Claude session")
            }
            if !controller.failures.isEmpty || controller.mistralProblem != nil {
                return ("exclamationmark.triangle.fill", .systemOrange, "Needs attention")
            }
            if !controller.transcribing.isEmpty {
                return ("waveform", nil, "Transcribing")
            }
            if controller.paused {
                return ("mic.slash", nil, "Auto-recording paused")
            }
            return ("mic", nil, "Waiting for a call")
        }()
        let image = NSImage(systemSymbolName: symbol, accessibilityDescription: tip)
        image?.isTemplate = color == nil
        statusItem.button?.image = image
        statusItem.button?.contentTintColor = color
        statusItem.button?.toolTip = "AutoRecorder: \(tip)"
    }

    // MARK: - Menu

    func menuNeedsUpdate(_ menu: NSMenu) {
        menu.removeAllItems()

        func info(_ text: String) {
            let item = NSMenuItem(title: text, action: nil, keyEquivalent: "")
            item.isEnabled = false
            menu.addItem(item)
        }
        func action(_ title: String, _ selector: Selector, checked: Bool = false) {
            let item = NSMenuItem(title: title, action: selector, keyEquivalent: "")
            item.target = self
            item.state = checked ? .on : .off
            menu.addItem(item)
        }

        if let rec = controller.recording {
            let minutes = Int(Date().timeIntervalSince(rec.start) / 60)
            info("Recording \(Controller.describe(rec.app)) - \(minutes) min")
            if let seconds = controller.stoppingIn {
                info("Call ended - stopping in \(seconds) s")
            }
            if let since = controller.silentSince {
                info("Silent since \(DateFormatter.localizedString(from: since, dateStyle: .none, timeStyle: .short))")
            }
        } else if controller.recordingElsewhere {
            info("Recording from a Claude session")
        } else if controller.paused {
            info("Auto-recording paused")
        } else {
            info("Waiting for a call")
        }
        for name in controller.transcribing {
            info("Transcribing \(name)")
        }
        if let problem = controller.mistralProblem {
            info("Mistral: \(problem)")
        }
        for failure in controller.failures {
            info("Failed: \(failure.file.lastPathComponent) - \(failure.message)")
        }
        if let done = controller.lastDone {
            info("Last transcribed: \(done)")
        }

        menu.addItem(.separator())
        if controller.recording != nil {
            action("Stop Recording", #selector(stopRecording))
        } else if !controller.recordingElsewhere {
            action("Start Recording Now", #selector(startRecording))
        }
        action("Pause Auto-Recording", #selector(togglePause), checked: controller.paused)
        if !controller.failures.isEmpty {
            action("Retry Failed Transcriptions (\(controller.failures.count))", #selector(retry))
        }
        action("Check Mistral Now", #selector(checkMistral))
        menu.addItem(.separator())
        action("Open Transcripts Folder", #selector(openTranscripts))
        action("Open Log", #selector(openLog))
        action("Quit AutoRecorder", #selector(quit))
    }

    @objc private func stopRecording() { controller.stop(reason: "stopped by hand", manualStop: true) }
    @objc private func startRecording() { controller.start(app: "manual", manual: true) }
    @objc private func togglePause() {
        controller.paused.toggle()
        updateIcon()
    }
    @objc private func retry() { controller.retryFailures() }
    @objc private func checkMistral() { controller.checkMistral() }
    @objc private func openTranscripts() { NSWorkspace.shared.open(Paths.transcriptions) }
    @objc private func openLog() { NSWorkspace.shared.open(Paths.log) }
    @objc private func quit() { NSApp.terminate(nil) }
}

let app = NSApplication.shared
let delegate = AppDelegate()
app.delegate = delegate
app.setActivationPolicy(.accessory)
app.run()
