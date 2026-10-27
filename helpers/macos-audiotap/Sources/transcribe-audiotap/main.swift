// transcribe-audiotap: system audio -> 16 kHz mono float32 PCM on stdout.
// Uses Core Audio process taps (macOS 14.2+): a private tap is wrapped in a private
// aggregate device whose IOProc receives the tapped audio.
import AVFoundation
import CoreAudio
import Foundation

struct Options {
    var rate: Double = 16000
    var pid: pid_t? = nil
    var list = false
}

func parseOptions() -> Options {
    var opts = Options()
    var args = CommandLine.arguments.dropFirst().makeIterator()
    while let arg = args.next() {
        switch arg {
        case "--rate": opts.rate = Double(args.next() ?? "") ?? opts.rate
        case "--pid": opts.pid = pid_t(args.next() ?? "")
        case "--list": opts.list = true
        case "-h", "--help":
            print("usage: transcribe-audiotap [--rate HZ] [--pid PID] | --list")
            exit(0)
        default:
            fail("unknown argument \(arg)")
        }
    }
    return opts
}

func fail(_ message: String, _ status: OSStatus = 0) -> Never {
    FileHandle.standardError.write("transcribe-audiotap: \(message)\(status != 0 ? " (OSStatus \(status))" : "")\n".data(using: .utf8)!)
    exit(1)
}

func check(_ status: OSStatus, _ what: String) {
    if status != noErr { fail(what, status) }
}

func globalAddress(_ selector: AudioObjectPropertySelector) -> AudioObjectPropertyAddress {
    AudioObjectPropertyAddress(mSelector: selector, mScope: kAudioObjectPropertyScopeGlobal,
                               mElement: kAudioObjectPropertyElementMain)
}

/// Read an object-ID property (e.g. the default output device).
func getObjectID(_ object: AudioObjectID,
                 _ selector: AudioObjectPropertySelector) -> AudioObjectID {
    var address = globalAddress(selector)
    var value = AudioObjectID(kAudioObjectUnknown)
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    check(AudioObjectGetPropertyData(object, &address, 0, nil, &size, &value),
          "AudioObjectGetPropertyData(\(selector))")
    return value
}

/// Read a stream-format property (e.g. the tap's format).
func getStreamFormat(_ object: AudioObjectID,
                     _ selector: AudioObjectPropertySelector) -> AudioStreamBasicDescription {
    var address = globalAddress(selector)
    var value = AudioStreamBasicDescription()
    var size = UInt32(MemoryLayout<AudioStreamBasicDescription>.size)
    check(AudioObjectGetPropertyData(object, &address, 0, nil, &size, &value),
          "AudioObjectGetPropertyData(\(selector))")
    return value
}

/// Read a CFString property. Core Audio returns a +1 (copied) reference, so the
/// result is taken as retained via Unmanaged instead of being written into a CFString
/// variable through a raw pointer (which bypasses Swift reference counting).
func getStringProperty(_ object: AudioObjectID,
                       _ selector: AudioObjectPropertySelector) -> String {
    var address = globalAddress(selector)
    var value: Unmanaged<CFString>?
    var size = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
    check(AudioObjectGetPropertyData(object, &address, 0, nil, &size, &value),
          "AudioObjectGetPropertyData(\(selector))")
    guard let value else { fail("property \(selector) is empty") }
    return value.takeRetainedValue() as String
}

/// Translate a Unix pid into a Core Audio process object.
func processObject(for pid: pid_t) -> AudioObjectID {
    var address = AudioObjectPropertyAddress(
        mSelector: kAudioHardwarePropertyTranslatePIDToProcessObject,
        mScope: kAudioObjectPropertyScopeGlobal, mElement: kAudioObjectPropertyElementMain)
    var qualifier = pid
    var object = AudioObjectID(kAudioObjectUnknown)
    var size = UInt32(MemoryLayout<AudioObjectID>.size)
    check(AudioObjectGetPropertyData(AudioObjectID(kAudioObjectSystemObject), &address,
                                     UInt32(MemoryLayout<pid_t>.size), &qualifier, &size,
                                     &object), "translate pid \(pid)")
    if object == kAudioObjectUnknown { fail("process \(pid) has no audio object") }
    return object
}

/// --list: one JSON line per process known to Core Audio:
/// {"pid": 123, "bundle": "us.zoom.xos", "output": true} ("output": playing right now).
func listProcesses() -> Never {
    let system = AudioObjectID(kAudioObjectSystemObject)
    var address = globalAddress(kAudioHardwarePropertyProcessObjectList)
    var size: UInt32 = 0
    check(AudioObjectGetPropertyDataSize(system, &address, 0, nil, &size),
          "AudioObjectGetPropertyDataSize(process list)")
    var objects = [AudioObjectID](repeating: kAudioObjectUnknown,
                                  count: Int(size) / MemoryLayout<AudioObjectID>.size)
    check(AudioObjectGetPropertyData(system, &address, 0, nil, &size, &objects),
          "AudioObjectGetPropertyData(process list)")
    for object in objects {
        var pidAddress = globalAddress(kAudioProcessPropertyPID)
        var pid: pid_t = -1
        var pidSize = UInt32(MemoryLayout<pid_t>.size)
        guard AudioObjectGetPropertyData(object, &pidAddress, 0, nil, &pidSize, &pid) == noErr,
              pid > 0 else { continue }
        var runningAddress = globalAddress(kAudioProcessPropertyIsRunningOutput)
        var running: UInt32 = 0
        var runningSize = UInt32(MemoryLayout<UInt32>.size)
        _ = AudioObjectGetPropertyData(object, &runningAddress, 0, nil, &runningSize, &running)
        var bundleAddress = globalAddress(kAudioProcessPropertyBundleID)
        var bundle: Unmanaged<CFString>?
        var bundleSize = UInt32(MemoryLayout<Unmanaged<CFString>?>.size)
        var bundleID = ""
        if AudioObjectGetPropertyData(object, &bundleAddress, 0, nil, &bundleSize,
                                      &bundle) == noErr, let bundle {
            bundleID = bundle.takeRetainedValue() as String
        }
        let row: [String: Any] = ["pid": Int(pid), "bundle": bundleID, "output": running != 0]
        if let data = try? JSONSerialization.data(withJSONObject: row) {
            FileHandle.standardOutput.write(data + "\n".data(using: .utf8)!)
        }
    }
    exit(0)
}

let opts = parseOptions()
if opts.list { listProcesses() }

// 1. Tap: all system audio, or one process mixed down to stereo.
let tapDescription: CATapDescription
if let pid = opts.pid {
    tapDescription = CATapDescription(stereoMixdownOfProcesses: [processObject(for: pid)])
} else {
    tapDescription = CATapDescription(stereoGlobalTapButExcludeProcesses: [])
}
tapDescription.uuid = UUID()
tapDescription.isPrivate = true
tapDescription.muteBehavior = .unmuted
var tapID = AudioObjectID(kAudioObjectUnknown)
check(AudioHardwareCreateProcessTap(tapDescription, &tapID), "AudioHardwareCreateProcessTap")

// 2. Private aggregate device containing the tap (clocked by the default output device).
let outputDevice = getObjectID(AudioObjectID(kAudioObjectSystemObject),
                               kAudioHardwarePropertyDefaultOutputDevice)
let outputUID = getStringProperty(outputDevice, kAudioDevicePropertyDeviceUID)
let aggregate: [String: Any] = [
    kAudioAggregateDeviceNameKey: "transcribe-audiotap",
    kAudioAggregateDeviceUIDKey: UUID().uuidString,
    kAudioAggregateDeviceMainSubDeviceKey: outputUID,
    kAudioAggregateDeviceIsPrivateKey: true,
    kAudioAggregateDeviceIsStackedKey: false,
    kAudioAggregateDeviceTapAutoStartKey: true,
    kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outputUID]],
    kAudioAggregateDeviceTapListKey: [[
        kAudioSubTapDriftCompensationKey: true,
        kAudioSubTapUIDKey: tapDescription.uuid.uuidString,
    ]],
]
var aggregateID = AudioObjectID(kAudioObjectUnknown)
check(AudioHardwareCreateAggregateDevice(aggregate as CFDictionary, &aggregateID),
      "AudioHardwareCreateAggregateDevice")

// 3. Converter: tap format -> mono float32 at the requested rate.
var tapFormat = getStreamFormat(tapID, kAudioTapPropertyFormat)
guard let inFormat = AVAudioFormat(streamDescription: &tapFormat),
      let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: opts.rate,
                                    channels: 1, interleaved: false),
      let converter = AVAudioConverter(from: inFormat, to: outFormat)
else { fail("unsupported tap format \(tapFormat)") }
let stdout = FileHandle.standardOutput

// 4. IOProc: convert each buffer and write it out.
var procID: AudioDeviceIOProcID?
let queue = DispatchQueue(label: "transcribe-audiotap")
check(AudioDeviceCreateIOProcIDWithBlock(&procID, aggregateID, queue) {
    _, inputData, _, _, _ in
    guard let input = AVAudioPCMBuffer(pcmFormat: inFormat, bufferListNoCopy: inputData,
                                       deallocator: nil) else { return }
    let capacity = AVAudioFrameCount(Double(input.frameLength) * opts.rate
                                     / inFormat.sampleRate) + 32
    guard let output = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: capacity)
    else { return }
    var consumed = false
    var error: NSError?
    converter.convert(to: output, error: &error) { _, status in
        if consumed { status.pointee = .noDataNow; return nil }
        consumed = true
        status.pointee = .haveData
        return input
    }
    if error != nil || output.frameLength == 0 { return }
    let samples = output.floatChannelData![0]
    stdout.write(Data(bytes: samples, count: Int(output.frameLength) * 4))
}, "AudioDeviceCreateIOProcIDWithBlock")

func shutdown() {
    if let procID {
        AudioDeviceStop(aggregateID, procID)
        AudioDeviceDestroyIOProcID(aggregateID, procID)
    }
    AudioHardwareDestroyAggregateDevice(aggregateID)
    AudioHardwareDestroyProcessTap(tapID)
}

signal(SIGPIPE, SIG_IGN)  // reader went away: write fails, we exit on SIGTERM/EOF
for sig in [SIGINT, SIGTERM] {
    signal(sig, SIG_IGN)
    let source = DispatchSource.makeSignalSource(signal: sig, queue: .main)
    source.setEventHandler { shutdown(); exit(0) }
    source.resume()
    _ = Unmanaged.passRetained(source)  // keep alive for the process lifetime
}

check(AudioDeviceStart(aggregateID, procID), "AudioDeviceStart")
dispatchMain()
