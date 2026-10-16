// swift-tools-version:5.9
import PackageDescription

let package = Package(
    name: "transcribe-audiotap",
    platforms: [.macOS("14.2")],
    targets: [
        .executableTarget(name: "transcribe-audiotap", path: "Sources/transcribe-audiotap")
    ]
)
