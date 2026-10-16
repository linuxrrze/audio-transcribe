# transcribe-audiotap (macOS 14.2+)

Captures system audio with a Core Audio **process tap** and writes raw 16 kHz mono
float32 little-endian PCM to stdout - the format `transcribe.audio.command.CommandSource`
reads.

```sh
./build.sh
.build/apple/Products/Release/transcribe-audiotap --rate 16000            # all audio
.build/apple/Products/Release/transcribe-audiotap --rate 16000 --pid 1234 # one app
```

- The first run triggers the **"System Audio Recording"** permission prompt; it is granted
  to the responsible app (Terminal, or the packaged transcribe app, whose Info.plist
  carries `NSAudioCaptureUsageDescription`).
- `--pid` limits capture to one process (e.g. Zoom/Teams) - the basis for meeting
  sources.
- Older macOS: use a virtual device (BlackHole) as microphone input instead.

Status: written against the macOS 14.2 `CATapDescription` /
`AudioHardwareCreateProcessTap` APIs but **not yet compiled or tested** (developed on
Linux); the M6 CI builds it on a macOS runner.
