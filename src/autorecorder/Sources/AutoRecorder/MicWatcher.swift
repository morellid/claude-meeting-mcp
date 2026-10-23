import CoreAudio
import Foundation

/// Which call app, if any, currently has the microphone open.
///
/// Core Audio lists every process with an audio session and whether it is
/// running input (macOS 14.4+). No permission is needed to read it. Only call
/// apps count: Siri (com.apple.CoreSpeech) holds the mic all day, and personal
/// call apps (FaceTime, WhatsApp) are deliberately absent so they never record.
enum MicWatcher {
    /// Bundle id prefixes of apps whose mic use means "a meeting is on".
    /// Browsers capture through helper processes (com.google.Chrome.helper).
    static let callApps = [
        "us.zoom.",
        "com.google.Chrome",
        "com.microsoft.teams",
        "com.microsoft.edgemac",
        "com.tinyspeck.slackmacgap",
        "com.brave.Browser",
        "company.thebrowser.",
        "org.mozilla.firefox",
        "com.apple.Safari",
        "com.apple.WebKit.GPU",
        "Cisco-Systems.Spark",
    ]

    static func callAppUsingMic() -> String? {
        for process in processObjects() {
            guard property(process, kAudioProcessPropertyIsRunningInput, UInt32(0)) == 1 else {
                continue
            }
            let bundle = bundleID(process)
            if callApps.contains(where: { bundle.hasPrefix($0) }) {
                return bundle
            }
        }
        return nil
    }

    private static func processObjects() -> [AudioObjectID] {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioHardwarePropertyProcessObjectList,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        let system = AudioObjectID(kAudioObjectSystemObject)
        var size: UInt32 = 0
        guard AudioObjectGetPropertyDataSize(system, &address, 0, nil, &size) == noErr else {
            return []
        }
        var ids = [AudioObjectID](repeating: 0, count: Int(size) / MemoryLayout<AudioObjectID>.size)
        guard AudioObjectGetPropertyData(system, &address, 0, nil, &size, &ids) == noErr else {
            return []
        }
        return ids
    }

    private static func property<T>(
        _ object: AudioObjectID, _ selector: AudioObjectPropertySelector, _ initial: T
    ) -> T? {
        var address = AudioObjectPropertyAddress(
            mSelector: selector,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var value = initial
        var size = UInt32(MemoryLayout<T>.size)
        let status = withUnsafeMutablePointer(to: &value) {
            AudioObjectGetPropertyData(object, &address, 0, nil, &size, $0)
        }
        return status == noErr ? value : nil
    }

    private static func bundleID(_ object: AudioObjectID) -> String {
        var address = AudioObjectPropertyAddress(
            mSelector: kAudioProcessPropertyBundleID,
            mScope: kAudioObjectPropertyScopeGlobal,
            mElement: kAudioObjectPropertyElementMain)
        var value: Unmanaged<CFString>?
        var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        guard AudioObjectGetPropertyData(object, &address, 0, nil, &size, &value) == noErr,
            let string = value?.takeRetainedValue()
        else { return "" }
        return string as String
    }
}
