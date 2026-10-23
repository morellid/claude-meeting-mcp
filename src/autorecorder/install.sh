#!/bin/bash
# Build AutoRecorder.app into ~/Applications and start it at login via launchd.
# Re-run after any change. macOS asks again for microphone and system audio
# access after each rebuild, because an ad-hoc signature changes with the binary.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
APP="$HOME/Applications/AutoRecorder.app"
LABEL="com.davidemorelli.autorecorder"
AGENT="$HOME/Library/LaunchAgents/$LABEL.plist"

swift build -c release --package-path "$REPO/src/audiocap"
swift build -c release --package-path "$HERE"

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS"
cp "$HERE/.build/release/AutoRecorder" "$APP/Contents/MacOS/AutoRecorder"
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleIdentifier</key><string>$LABEL</string>
  <key>CFBundleName</key><string>AutoRecorder</string>
  <key>CFBundleExecutable</key><string>AutoRecorder</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleShortVersionString</key><string>1.0</string>
  <key>LSMinimumSystemVersion</key><string>14.4</string>
  <key>LSUIElement</key><true/>
  <key>NSMicrophoneUsageDescription</key><string>AutoRecorder records your side of calls so they can be transcribed.</string>
  <key>NSAudioCaptureUsageDescription</key><string>AutoRecorder records the other participants of calls so they can be transcribed.</string>
</dict>
</plist>
PLIST
codesign --force --sign - --identifier "$LABEL" "$APP"

mkdir -p "$(dirname "$AGENT")"
cat > "$AGENT" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$LABEL</string>
  <key>ProgramArguments</key><array><string>$APP/Contents/MacOS/AutoRecorder</string></array>
  <key>EnvironmentVariables</key><dict><key>MEETING_REPO</key><string>$REPO</string></dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ProcessType</key><string>Interactive</string>
</dict>
</plist>
PLIST
launchctl bootstrap "gui/$(id -u)" "$AGENT"
echo "AutoRecorder installed and running: $APP"
