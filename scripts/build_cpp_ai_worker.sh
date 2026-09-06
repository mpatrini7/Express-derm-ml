#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_DIR="$ROOT_DIR/cpp/ai_worker"
BUILD_DIR="${EXPRESS_DERM_CPP_WORKER_BUILD_DIR:-$SOURCE_DIR/build}"

if [[ "$(uname -s)" != "Linux" ]]; then
  printf 'The TensorRT C++ worker must be built on a compatible Linux system.\n' >&2
  printf 'Detected: %s %s\n' "$(uname -s)" "$(uname -m)" >&2
  exit 1
fi

for command_name in cmake c++ pkg-config; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    printf 'Missing build command: %s\n' "$command_name" >&2
    exit 1
  fi
done

for required_file in \
  /usr/include/NvInfer.h \
  /usr/include/aarch64-linux-gnu/NvInfer.h \
  /usr/include/x86_64-linux-gnu/NvInfer.h; do
  if [[ -f "$required_file" ]]; then
    tensorrt_header_found=true
    break
  fi
done
if [[ "${tensorrt_header_found:-false}" != true ]]; then
  printf 'NvInfer.h was not found. Install the TensorRT C++ development packages.\n' >&2
  exit 1
fi

if ! pkg-config --exists opencv4; then
  printf 'OpenCV C++ development files were not found (pkg-config opencv4).\n' >&2
  exit 1
fi
if [[ ! -f /usr/include/nlohmann/json.hpp ]]; then
  printf 'nlohmann/json.hpp was not found. Install nlohmann-json3-dev.\n' >&2
  exit 1
fi

cmake -S "$SOURCE_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" --parallel "${EXPRESS_DERM_CPP_BUILD_JOBS:-2}"

WORKER_BINARY="$BUILD_DIR/express-derm-ai-worker"
if [[ ! -x "$WORKER_BINARY" ]]; then
  printf 'Worker build completed without an executable: %s\n' "$WORKER_BINARY" >&2
  exit 1
fi

printf 'Built persistent TensorRT C++ worker: %s\n' "$WORKER_BINARY"
printf 'The worker is not authorized for use until target parity and soak tests pass.\n'
