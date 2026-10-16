#!/bin/sh
# Build whisper.cpp (pinned release) twice: CPU-only and Vulkan, for benchmarking
#   python -m transcribe.bench.whispercpp --binary cpu=... --binary vulkan=...
#
# Needs: cmake, a C++17 compiler, and for Vulkan the loader + headers + glslc
#   Debian/Ubuntu: apt install cmake g++ libvulkan-dev glslc mesa-vulkan-drivers
# Usage: scripts/build_whispercpp.sh [DEST]   (default: ~/.cache/transcribe/whisper.cpp)
# Extra CMake options (e.g. non-standard Vulkan locations) via CMAKE_ARGS="..."
set -eu
VERSION=v1.9.5
DEST=${1:-$HOME/.cache/transcribe/whisper.cpp}
SRC=$DEST/src-$VERSION
JOBS=$(nproc 2>/dev/null || echo 4)
mkdir -p "$DEST"
if [ ! -d "$SRC" ]; then
  curl -fsSL "https://github.com/ggml-org/whisper.cpp/archive/refs/tags/$VERSION.tar.gz" \
    | tar xz -C "$DEST"
  mv "$DEST/whisper.cpp-${VERSION#v}" "$SRC"
fi
cmake -S "$SRC" -B "$DEST/build-cpu" -DCMAKE_BUILD_TYPE=Release -DWHISPER_BUILD_TESTS=OFF ${CMAKE_ARGS:-}
cmake --build "$DEST/build-cpu" -j "$JOBS" --target whisper-cli whisper-bench
if [ "${SKIP_VULKAN:-0}" != 1 ]; then
  cmake -S "$SRC" -B "$DEST/build-vulkan" -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON \
    -DWHISPER_BUILD_TESTS=OFF ${CMAKE_ARGS:-}
  cmake --build "$DEST/build-vulkan" -j "$JOBS" --target whisper-cli whisper-bench
fi
echo "CPU:    $DEST/build-cpu/bin/whisper-cli (+ whisper-bench)"
[ "${SKIP_VULKAN:-0}" = 1 ] || echo "Vulkan: $DEST/build-vulkan/bin/whisper-cli (+ whisper-bench)"
