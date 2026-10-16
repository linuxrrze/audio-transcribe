#!/bin/sh
# Build the macOS system-audio helper (requires Xcode 15.1+ command line tools).
set -eu
cd "$(dirname "$0")"
swift build -c release --arch arm64 --arch x86_64
echo "Built: $(swift build -c release --arch arm64 --arch x86_64 --show-bin-path)/transcribe-audiotap"
