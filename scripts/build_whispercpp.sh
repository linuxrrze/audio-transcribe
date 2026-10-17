#!/bin/sh
# Build whisper.cpp (pinned release) twice: CPU-only and GPU (Vulkan; Metal on macOS).
#   whisper-server: used by the whisper.cpp ASR engine ([asr] engine = "whisper.cpp"),
#     which prefers the GPU build over build-cpu
#   whisper-cli, whisper-bench: python -m transcribe.bench.whispercpp --binary cpu=... ...
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
# macOS: Metal instead of Vulkan (and Metal off in the CPU build, where it is the default)
if [ "$(uname -s)" = Darwin ]; then
  GPU=metal; GPU_FLAGS="-DGGML_METAL=ON"; CPU_FLAGS="-DGGML_METAL=OFF"
else
  GPU=vulkan; GPU_FLAGS="-DGGML_VULKAN=ON"; CPU_FLAGS=""
fi
TARGETS="whisper-cli whisper-server whisper-bench"
cmake -S "$SRC" -B "$DEST/build-cpu" -DCMAKE_BUILD_TYPE=Release -DWHISPER_BUILD_TESTS=OFF \
  $CPU_FLAGS ${CMAKE_ARGS:-}
cmake --build "$DEST/build-cpu" -j "$JOBS" --target $TARGETS
if [ "${SKIP_GPU:-${SKIP_VULKAN:-0}}" != 1 ]; then
  cmake -S "$SRC" -B "$DEST/build-$GPU" -DCMAKE_BUILD_TYPE=Release $GPU_FLAGS \
    -DWHISPER_BUILD_TESTS=OFF ${CMAKE_ARGS:-}
  cmake --build "$DEST/build-$GPU" -j "$JOBS" --target $TARGETS
fi
echo "CPU:    $DEST/build-cpu/bin/whisper-server (+ whisper-cli, whisper-bench)"
[ "${SKIP_GPU:-${SKIP_VULKAN:-0}}" = 1 ] || \
  echo "$GPU: $DEST/build-$GPU/bin/whisper-server (+ whisper-cli, whisper-bench)"
