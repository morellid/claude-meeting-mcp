// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "AutoRecorder",
    platforms: [.macOS("14.4")],
    targets: [
        .executableTarget(name: "AutoRecorder", path: "Sources/AutoRecorder"),
    ]
)
