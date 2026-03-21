#!/usr/bin/env bash
# Convert YuNet + SFace ONNX models to TensorRT FP16 engines.
# Run inside the witness container where trtexec is available.
#
# Usage: bash scripts/convert_face_trt.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ONNX_DIR="$PROJECT_ROOT/models/opencv"
TRT_DIR="$PROJECT_ROOT/models/trt"

TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
if [ ! -x "$TRTEXEC" ]; then
    echo "[convert] trtexec not found at $TRTEXEC"
    echo "[convert] Set TRTEXEC env var or run inside the container."
    exit 1
fi

mkdir -p "$TRT_DIR"

# --- SFace (face recognition) ---
# Fixed input: 1x3x112x112
SFACE_ONNX="$ONNX_DIR/face_recognition_sface_2021dec.onnx"
SFACE_ENGINE="$TRT_DIR/sface_fp16.engine"

if [ -f "$SFACE_ENGINE" ]; then
    echo "[convert] SFace engine already exists: $SFACE_ENGINE (skipping)"
else
    echo "[convert] Converting SFace ONNX -> TRT FP16..."
    "$TRTEXEC" \
        --onnx="$SFACE_ONNX" \
        --saveEngine="$SFACE_ENGINE" \
        --fp16 \
        2>&1 | tail -5
    echo "[convert] SFace engine saved: $SFACE_ENGINE"
fi

# --- YuNet (face detection) ---
# Dynamic input height/width. We fix to 640x480 for our webcam pipeline.
YUNET_ONNX="$ONNX_DIR/face_detection_yunet_2023mar.onnx"
YUNET_ENGINE="$TRT_DIR/yunet_640x480_fp16.engine"

if [ -f "$YUNET_ENGINE" ]; then
    echo "[convert] YuNet engine already exists: $YUNET_ENGINE (skipping)"
else
    echo "[convert] Converting YuNet ONNX -> TRT FP16 (640x480)..."
    "$TRTEXEC" \
        --onnx="$YUNET_ONNX" \
        --saveEngine="$YUNET_ENGINE" \
        --fp16 \
        --minShapes=input:1x3x480x640 \
        --optShapes=input:1x3x480x640 \
        --maxShapes=input:1x3x480x640 \
        2>&1 | tail -5
    echo "[convert] YuNet engine saved: $YUNET_ENGINE"
fi

echo "[convert] Done. Engines in $TRT_DIR/"
ls -lh "$TRT_DIR/"
