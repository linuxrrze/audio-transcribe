// transcribe-audiotap: system audio -> 16 kHz mono float32 PCM on stdout.
// Uses Core Audio process taps (macOS 14.2+): a private tap is wrapped in a private
// aggregate device whose IOProc receives the tapped audio.
import AVFoundation
import CoreAudio
import Foundation

struct Options {
    var rate: Double = 16000
    var pid: pid_t? = nil
}

func parseOptions() -> Options {
    var opts = Options()
    var args = CommandLine.arguments.dropFirst().makeIterator()
    while let arg = args.next() {
        switch arg {
        case "--rate": opts.rate = Double(args.next() ?? "") ?? opts.rate
        case "--pid": opts.pid = pid_t(args.next() ?? "")
        case "-h", "--help":
            print("usage: transcribe-audiotap [--rate HZ] [--pid PID]")
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

func getProperty<T>(_ object: AudioObjectID, _ selector: AudioObjectPropertySelector,
                    _ initial: T) -> T {
    var address = AudioObjectPropertyAddress(mSelector: selector,
                                             mScope: kAudioObjectPropertyScopeGlobal,
                                             mElement: kAudioObjectPropertyElementMain)
    var value = initial
    var size = UInt32(MemoryLayout<T>.size)
    check(AudioObjectGetPropertyData(object, &address, 0, nil, &size, &value),
          "AudioObjectGetPropertyData(\(selector))")
    return value
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

let opts = parseOptions()

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
let outputDevice: AudioDeviceID = getProperty(AudioObjectID(kAudioObjectSystemObject),
                                              kAudioHardwarePropertyDefaultOutputDevice,
                                              AudioDeviceID(0))
let outputUID: CFString = getProperty(outputDevice, kAudioDevicePropertyDeviceUID,
                                      "" as CFString)
let aggregate: [String: Any] = [
    kAudioAggregateDeviceNameKey: "transcribe-audiotap",
    kAudioAggregateDeviceUIDKey: UUID().uuidString,
    kAudioAggregateDeviceMainSubDeviceKey: outputUID as String,
    kAudioAggregateDeviceIsPrivateKey: true,
    kAudioAggregateDeviceIsStackedKey: false,
    kAudioAggregateDeviceTapAutoStartKey: true,
    kAudioAggregateDeviceSubDeviceListKey: [[kAudioSubDeviceUIDKey: outputUID as String]],
    kAudioAggregateDeviceTapListKey: [[
        kAudioSubTapDriftCompensationKey: true,
        kAudioSubTapUIDKey: tapDescription.uuid.uuidString,
    ]],
]
var aggregateID = AudioObjectID(kAudioObjectUnknown)
check(AudioHardwareCreateAggregateDevice(aggregate as CFDictionary, &aggregateID),
      "AudioHardwareCreateAggregateDevice")

// 3. Converter: tap format -> mono float32 at the requested rate.
var tapFormat: AudioStreamBasicDescription = getProperty(tapID, kAudioTapPropertyFormat,
                                                         AudioStreamBasicDescription())
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
