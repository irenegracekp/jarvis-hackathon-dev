#!/bin/bash
# Launch The Witness container with GPU, camera, mic, and audio
sudo docker run -it --rm \
  --runtime nvidia \
  --network host \
  --privileged \
  -v /home/orin/hacky:/workspace \
  -v /home/orin/.cache:/root/.cache \
  -v /dev:/dev \
  -e PULSE_SERVER=unix:/run/user/1000/pulse/native \
  -v /run/user/1000/pulse:/run/user/1000/pulse \
  -w /workspace \
  witness:latest \
  bash
