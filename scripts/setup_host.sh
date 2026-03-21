#!/usr/bin/env bash
# Setup Jetson Orin Nano host for Jarvis hackathon pipeline (no Docker needed).
# JetPack 6.2, Ubuntu 22.04, CUDA 12.6, TensorRT 10.3
#
# Usage: bash scripts/setup_host.sh
#
# Run steps selectively with:
#   bash scripts/setup_host.sh --step apt
#   bash scripts/setup_host.sh --step venv
#   bash scripts/setup_host.sh --step trt_pose
#   bash scripts/setup_host.sh --step node
#   bash scripts/setup_host.sh --step trt_engines
#   bash scripts/setup_host.sh --step models

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_ROOT/venv"

STEP="${1:-all}"

# ─── Colors ───
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[setup]${NC} $*"; }
warn()  { echo -e "${YELLOW}[setup]${NC} $*"; }
fail()  { echo -e "${RED}[setup]${NC} $*"; exit 1; }

# ─── Step 1: System packages ───
setup_apt() {
    info "Installing system packages..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        python3-pip python3-venv python3-dev \
        libportaudio2 portaudio19-dev libsndfile1 \
        xdotool \
        libopenblas-dev libjpeg-dev zlib1g-dev \
        cmake build-essential
    info "System packages done."
}

# ─── Step 2: Python venv + dependencies ───
setup_venv() {
    if [ ! -f "$VENV_DIR/bin/activate" ]; then
        info "Creating Python venv..."
        rm -rf "$VENV_DIR"
        python3 -m venv "$VENV_DIR" --system-site-packages
    fi
    source "$VENV_DIR/bin/activate"

    info "Installing Python packages..."

    # Core packages
    pip install --no-cache-dir -q \
        numpy scipy \
        opencv-python-headless \
        sounddevice \
        num2words \
        requests \
        openai \
        scikit-learn \
        piper-tts

    # faster-whisper (local ASR)
    pip install --no-cache-dir -q faster-whisper

    # PyTorch — use NVIDIA's Jetson wheel if not already present
    if ! python3 -c "import torch" 2>/dev/null; then
        info "Installing PyTorch from NVIDIA Jetson index..."
        pip install --no-cache-dir \
            --extra-index-url https://developer.download.nvidia.com/compute/redist/jp/v60 \
            torch torchvision torchaudio
    else
        info "PyTorch already installed: $(python3 -c 'import torch; print(torch.__version__)')"
    fi

    # llama-cpp-python with CUDA (optional, for local LLM fallback)
    if ! python3 -c "import llama_cpp" 2>/dev/null; then
        info "Building llama-cpp-python with CUDA..."
        CUDACXX=/usr/local/cuda/bin/nvcc \
        CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=87" \
        pip install --no-cache-dir llama-cpp-python || \
            warn "llama-cpp-python build failed (OK if using --llm-backend openai)"
    fi

    # transformers + accelerate (for SmolVLM, optional)
    pip install --no-cache-dir -q transformers accelerate huggingface-hub

    info "Python venv ready at $VENV_DIR"
    info "Activate with: source $VENV_DIR/bin/activate"
}

# ─── Step 3: trt_pose + torch2trt ───
setup_trt_pose() {
    source "$VENV_DIR/bin/activate"

    if python3 -c "import torch2trt" 2>/dev/null; then
        info "torch2trt already installed."
    else
        info "Building torch2trt..."
        TMPDIR=$(mktemp -d)
        git clone --depth 1 https://github.com/NVIDIA-AI-IOT/torch2trt.git "$TMPDIR/torch2trt"
        cd "$TMPDIR/torch2trt"
        CUDACXX=/usr/local/cuda/bin/nvcc CMAKE_CUDA_ARCHITECTURES=87 \
            python3 setup.py install
        cd "$PROJECT_ROOT"
        rm -rf "$TMPDIR/torch2trt"
        info "torch2trt installed."
    fi

    if python3 -c "import trt_pose" 2>/dev/null; then
        info "trt_pose already installed."
    else
        info "Building trt_pose..."
        TMPDIR=$(mktemp -d)
        git clone --depth 1 https://github.com/NVIDIA-AI-IOT/trt_pose.git "$TMPDIR/trt_pose"
        cd "$TMPDIR/trt_pose"
        python3 setup.py install
        cd "$PROJECT_ROOT"
        rm -rf "$TMPDIR/trt_pose"
        info "trt_pose installed."
    fi

    # trt_pose_hand is just Python, no build needed
    if python3 -c "import trt_pose_hand" 2>/dev/null; then
        info "trt_pose_hand already installed."
    else
        info "Installing trt_pose_hand..."
        pip install --no-cache-dir git+https://github.com/NVIDIA-AI-IOT/trt_pose_hand.git || \
            warn "trt_pose_hand install failed — gesture recognition will not work"
    fi
}

# ─── Step 4: Node.js 22 + OpenClaw ───
setup_node() {
    if command -v node &>/dev/null && [[ "$(node -v)" == v22* ]]; then
        info "Node.js $(node -v) already installed."
    else
        info "Installing Node.js 22..."
        curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
        sudo apt-get install -y -qq nodejs
        info "Node.js $(node -v) installed."
    fi

    if command -v openclaw &>/dev/null; then
        info "OpenClaw already installed."
    else
        info "Installing OpenClaw..."
        sudo npm install -g openclaw || \
            warn "OpenClaw install failed — install manually: sudo npm install -g openclaw"
    fi

    info "Node/OpenClaw done."
    info "Run 'openclaw onboard --install-daemon' to configure (needs API key at hackathon)."
}

# ─── Step 5: Convert face ONNX -> TensorRT engines ───
setup_trt_engines() {
    TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
    if [ ! -x "$TRTEXEC" ]; then
        # Try common JetPack paths
        for p in /usr/src/tensorrt/bin/trtexec /usr/bin/trtexec; do
            if [ -x "$p" ]; then
                TRTEXEC="$p"
                break
            fi
        done
    fi

    if [ ! -x "$TRTEXEC" ]; then
        warn "trtexec not found. Skipping TRT engine conversion."
        warn "Face detection will fall back to OpenCV (slower but works)."
        return
    fi

    info "Converting face models to TensorRT..."
    bash "$SCRIPT_DIR/convert_face_trt.sh"
}

# ─── Step 6: Download trt_pose_hand model weights ───
setup_models() {
    MODEL_DIR="$PROJECT_ROOT/models/trt_pose"
    mkdir -p "$MODEL_DIR"

    HAND_POSE_JSON="$MODEL_DIR/hand_pose.json"
    HAND_POSE_MODEL="$MODEL_DIR/hand_pose_resnet18_att_244_244.pth"

    if [ ! -f "$HAND_POSE_JSON" ]; then
        info "Downloading hand_pose.json..."
        curl -fsSL -o "$HAND_POSE_JSON" \
            "https://raw.githubusercontent.com/NVIDIA-AI-IOT/trt_pose_hand/master/preprocess/hand_pose.json" || \
            warn "Failed to download hand_pose.json"
    fi

    if [ ! -f "$HAND_POSE_MODEL" ]; then
        info "Downloading hand pose model weights (~30MB)..."
        # The model is hosted on NVIDIA's trt_pose_hand releases
        curl -fsSL -o "$HAND_POSE_MODEL" \
            "https://github.com/NVIDIA-AI-IOT/trt_pose_hand/releases/download/v0.0.1/hand_pose_resnet18_att_244_244.pth" || \
            warn "Failed to download hand pose model. Download manually from trt_pose_hand GitHub releases."
    fi

    if [ -f "$HAND_POSE_JSON" ] && [ -f "$HAND_POSE_MODEL" ]; then
        info "Hand pose model files ready in $MODEL_DIR"
    fi
}

# ─── Main ───
info "Jarvis hackathon setup — Jetson Orin Nano"
info "Project: $PROJECT_ROOT"

case "$STEP" in
    --step)
        shift
        case "$1" in
            apt)         setup_apt ;;
            venv)        setup_venv ;;
            trt_pose)    setup_trt_pose ;;
            node)        setup_node ;;
            trt_engines) setup_trt_engines ;;
            models)      setup_models ;;
            *)           fail "Unknown step: $1. Options: apt, venv, trt_pose, node, trt_engines, models" ;;
        esac
        ;;
    all)
        setup_apt
        setup_venv
        setup_trt_pose
        setup_node
        setup_trt_engines
        setup_models
        ;;
    *)
        fail "Usage: $0 [all | --step <apt|venv|trt_pose|node|trt_engines|models>]"
        ;;
esac

echo ""
info "========================================="
info "  Setup complete!"
info "========================================="
info ""
info "Activate venv:  source venv/bin/activate"
info ""
info "Test components:"
info "  python3 pipeline/openclaw_bridge.py        # command classification"
info "  python3 pipeline/actions.py                # action mapper"
info "  python3 pipeline/brain.py --llm-backend openai  # needs OPENAI_API_KEY"
info "  python3 pipeline/faces_trt.py              # face detection"
info "  python3 scripts/test_gestures.py           # live gesture viewer"
info "  python3 scripts/test_actions.py            # gesture -> desktop actions"
info ""
info "Full pipeline:"
info "  python3 main.py --llm-backend openai --asr-backend cloud --no-vlm"
