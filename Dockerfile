FROM dustynv/pytorch:2.7-r36.4.0

# Fix the dead Jetson pip index that blocks installs
ENV PIP_INDEX_URL=https://pypi.org/simple
ENV PIP_EXTRA_INDEX_URL=https://pypi.ngc.nvidia.com

# System dependencies
RUN apt-get update -qq && apt-get install -y -qq \
    libportaudio2 portaudio19-dev libsndfile1 \
    xdotool \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
RUN pip install --no-cache-dir \
    faster-whisper transformers accelerate \
    sounddevice scipy num2words \
    opencv-python-headless piper-tts \
    openai requests \
    'numpy<2'

# Build llama-cpp-python with CUDA SM87 (Jetson Orin)
RUN CUDACXX=/usr/local/cuda/bin/nvcc \
    CMAKE_ARGS="-DGGML_CUDA=on -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CUDA_ARCHITECTURES=87" \
    pip install --no-cache-dir llama-cpp-python

# Hand pose estimation (NVIDIA TensorRT) — trt_pose + trt_pose_hand
RUN pip install --no-cache-dir traitlets cython tqdm scikit-learn pycocotools

RUN CUDACXX=/usr/local/cuda/bin/nvcc \
    CMAKE_CUDA_ARCHITECTURES=87 \
    pip install --no-cache-dir torch2trt

RUN pip install --no-cache-dir \
    git+https://github.com/NVIDIA-AI-IOT/trt_pose.git \
    git+https://github.com/NVIDIA-AI-IOT/trt_pose_hand.git

WORKDIR /workspace
