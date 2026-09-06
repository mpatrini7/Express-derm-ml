#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 MODEL.onnx MODEL.engine [DYNAMIC_IMAGE_SIZE]"
  exit 2
fi

ONNX_PATH="$1"
ENGINE_PATH="$2"
DYNAMIC_IMAGE_SIZE="${3:-}"

if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "ONNX model not found: ${ONNX_PATH}"
  exit 1
fi

if [[ -e "${ENGINE_PATH}" ]]; then
  echo "TensorRT engine already exists and will not be overwritten: ${ENGINE_PATH}"
  exit 1
fi

ENGINE_DIR="$(dirname -- "${ENGINE_PATH}")"
ENGINE_NAME="$(basename -- "${ENGINE_PATH}")"
if [[ ! -d "${ENGINE_DIR}" ]]; then
  echo "TensorRT engine destination directory does not exist: ${ENGINE_DIR}"
  exit 1
fi

command -v trtexec >/dev/null 2>&1 || {
  echo "trtexec not found. Use an installed TensorRT environment."
  exit 1
}
command -v sha256sum >/dev/null 2>&1 || {
  echo "sha256sum not found. Install GNU coreutils before building the engine."
  exit 1
}

TEMP_ENGINE="${ENGINE_DIR}/.${ENGINE_NAME}.building.$$"
if [[ -e "${TEMP_ENGINE}" ]]; then
  echo "Temporary engine path already exists: ${TEMP_ENGINE}"
  exit 1
fi
cleanup() {
  rm -f "${TEMP_ENGINE}"
}
trap cleanup EXIT

TRTEXEC_ARGUMENTS=(
  --onnx="${ONNX_PATH}"
  --saveEngine="${TEMP_ENGINE}"
  --fp16
  --profilingVerbosity=detailed
)
if [[ -n "${DYNAMIC_IMAGE_SIZE}" ]]; then
  if [[ ! "${DYNAMIC_IMAGE_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Dynamic image size must be a positive integer: ${DYNAMIC_IMAGE_SIZE}"
    exit 2
  fi
  TRTEXEC_ARGUMENTS+=(
    --minShapes=image:1x3x"${DYNAMIC_IMAGE_SIZE}"x"${DYNAMIC_IMAGE_SIZE}"
    --optShapes=image:1x3x"${DYNAMIC_IMAGE_SIZE}"x"${DYNAMIC_IMAGE_SIZE}"
    --maxShapes=image:1x3x"${DYNAMIC_IMAGE_SIZE}"x"${DYNAMIC_IMAGE_SIZE}"
  )
fi

trtexec "${TRTEXEC_ARGUMENTS[@]}"

if [[ ! -s "${TEMP_ENGINE}" ]]; then
  echo "TensorRT engine was not created or is empty: ${TEMP_ENGINE}"
  exit 1
fi

ln "${TEMP_ENGINE}" "${ENGINE_PATH}"
rm -f "${TEMP_ENGINE}"
trap - EXIT

echo "Built TensorRT engine: ${ENGINE_PATH}"
sha256sum "${ENGINE_PATH}"
echo "Benchmark the engine and compare outputs with ONNX before enabling it."
