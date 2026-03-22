# Jarvis - Reachy Mini Hackathon Robot

Jarvis is an AI-powered personal assistant built for Reachy Mini on Jetson Orin Nano.

The idea is simple: instead of a chatbot on a screen, Jarvis lives in a robot body. It can see people, talk to them, remember them, react with motion and emotion, and help with real tasks on a connected desktop through OpenClaw.

That means Jarvis can do things like:

- play music or videos
- open websites and apps
- search the web
- book flights or navigate travel sites
- click, type, scroll, and control desktop workflows
- greet people, hold short conversations, and remember facts about them
- react with head motion, antenna movement, emotions, and smart lights

The runtime persona currently used by the robot is **EVA**: short-spoken, curious, efficient, and expressive through movement. Jarvis is the project/system; EVA is the character the robot speaks as.

## What The System Is

Jarvis is a multi-modal assistant stack that combines:

- speech input
- LLM reasoning
- text-to-speech output
- camera perception
- face recognition and person memory
- hand gesture recognition
- desktop control through OpenClaw
- Reachy Mini robot control
- optional smart light integration
- optional Agora cloud voice mode

It is designed to run primarily on **Jetson Orin Nano** with **Reachy Mini** attached.

## Platform

- **Hardware:** Jetson Orin Nano, Reachy Mini robot, ZED 2 stereo camera, Hollyland wireless mic, Bluetooth speaker
- **OS:** JetPack 6 (L4T R36.4.3), CUDA 12.6, TensorRT 10.3
- **Python env:** Conda `trt_pose` env has torch 2.5 (CUDA), torchvision 0.19, OpenCV 4.8, ultralytics
- **Run with:** `/home/orin/miniconda3/envs/trt_pose/bin/python`
- **Docker alternative:** [`run_docker.sh`](./run_docker.sh) runs `witness:latest` with NVIDIA runtime, host network, and device access

When running outside Docker on the Jetson host:

```bash
export LD_LIBRARY_PATH="/home/orin/miniconda3/envs/trt_pose/lib/python3.10/site-packages/nvidia/cusparselt/lib:${LD_LIBRARY_PATH}"
```

## What Jarvis Can Do

### Conversation and presence

- Detect when people arrive or leave
- Greet known and unknown people
- Hold short conversations
- Keep a simple memory per recognized face
- Use scene context from the camera to ground responses

### Desktop assistance

Jarvis routes practical computer requests to OpenClaw. This is what makes it a real assistant instead of only a conversational demo.

Examples:

- "Open YouTube"
- "Play some music"
- "Search for flights to San Francisco"
- "Open Gmail"
- "Scroll down"
- "Type this into the search bar"

Under the hood, Jarvis decides whether speech is:

- normal conversation
- or a desktop command that should be sent to OpenClaw

The bridge for that is [`pipeline/openclaw_bridge.py`](./pipeline/openclaw_bridge.py).

### Robot embodiment

Jarvis can express itself through:

- head movement
- body yaw
- antenna movement
- prerecorded emotion motions
- optional smart lights

This lets the robot communicate with physical behavior, not just speech.

## Persona

The current system prompt gives the robot an **EVA-like** personality:

- sleek, efficient, and curious
- warm but initially reserved
- expressive through movement
- direct and short-spoken
- action-oriented rather than verbose

The intended interaction style is:

- 1 to 2 sentences max
- sometimes a single short line
- frequent use of physical motion or emotion instead of extra words

So the architecture is "Jarvis", while the user-facing character is "EVA".

## Architecture Overview

```text
                    main.py (orchestrator)
                    State machine: IDLE -> AMBIENT -> ENGAGED
                    4 threads: listen, brain, output, gesture
                         |
    +--------------------+------------------------+
    |                    |                        |
 INPUT               COGNITION                 OUTPUT
 -----               ---------                 ------
 listen.py            brain.py                  speak.py
  (mic->ASR)           (LLM orchestration)       (Piper TTS)
 vision.py            memory.py                 reachy_bridge.py
  (camera+VLM)         (person facts store)      (head/body/antenna)
 faces.py             openclaw_bridge.py        hue.py
  (YuNet+SFace)        (desktop commands)        (Philips Hue lights)
 gestures.py          llm_proxy.py              actions.py
  (hand pose)          (Agora LLM proxy)         (gesture->keyboard)
 person_tracker.py                              dashboard.py
  (YOLOv8-pose+ReID)                              (MJPEG web UI)
```

## How It Works

The main entrypoint is [`main.py`](./main.py). It runs 4 daemon threads coordinated through a shared `State` object and queues.

### State machine

- **IDLE**: no faces detected
- **AMBIENT**: faces are present, the robot observes and can greet arrivals
- **ENGAGED**: one person is the active conversation target

### Threads

1. **listen_loop**
   Mic -> ASR -> `audio_queue`
2. **brain_loop**
   Face events, scene context, memory context, LLM calls, OpenClaw routing
3. **output_loop**
   TTS, robot actions, light changes
4. **gesture_loop**
   Hand pose -> gesture classification -> desktop actions

### Two main operating modes

1. **Local mode**
   Everything runs on-device: listening, reasoning, speaking, gestures, dashboard.

2. **Agora mode**
   Browser handles RTC voice, while Jetson still runs robot control, tool dispatch, and agent infrastructure.

## Core Modules

### [`pipeline/listen.py`](./pipeline/listen.py)

- Audio input and ASR
- Captures from the mic with `sounddevice`
- Supports local `faster-whisper` or cloud Whisper API

### [`pipeline/brain.py`](./pipeline/brain.py)

- Central LLM orchestration
- Builds prompts from speech, memory, and scene context
- Supports local GGUF, OpenAI, and OpenRouter backends
- Produces structured responses like speech + emotion + head direction

### [`pipeline/vision.py`](./pipeline/vision.py)

- Camera input and scene analysis
- Handles ZED stereo cropping
- Produces face events and optional scene description

### [`pipeline/faces.py`](./pipeline/faces.py)

- Face detection and identification
- Uses YuNet + SFace
- Persists enrolled identities

### [`pipeline/person_tracker.py`](./pipeline/person_tracker.py)

- Person tracking with YOLOv8 pose + re-identification
- Detects hand raise events
- Used for follow / targeting workflows

### [`pipeline/gestures.py`](./pipeline/gestures.py)

- Hand gesture recognition
- Supports TRT pose backend and MediaPipe fallback
- Produces gesture labels plus motion data

### [`pipeline/actions.py`](./pipeline/actions.py)

- Gesture-to-desktop mapping
- Converts recognized gestures into `xdotool` actions

### [`pipeline/openclaw_bridge.py`](./pipeline/openclaw_bridge.py)

- Decides whether speech is conversation or a computer command
- Sends tool-style desktop tasks to OpenClaw

It currently shells out like this:

```bash
openclaw agent --agent main --message "<command>" --json
```

### [`pipeline/reachy_bridge.py`](./pipeline/reachy_bridge.py)

- Direct Reachy Mini hardware control
- Head movement
- body yaw
- antenna movement
- emotion playback
- audio-reactive wobble during TTS

### [`pipeline/hue.py`](./pipeline/hue.py)

- Philips Hue emotion-to-color mapping
- Optional light reactions for greetings and conversation mood

### [`pipeline/agora_web_server.py`](./pipeline/agora_web_server.py)

- FastAPI server for Agora browser mode
- Serves the Agora web client
- Receives agent datastream messages
- Dispatches robot actions

### [`pipeline/mcp_server.py`](./pipeline/mcp_server.py)

- FastMCP tool server for Agora mode
- Exposes desktop-command execution to the cloud agent

## Running On Jetson

This repo is meant to be run primarily on Jetson.

### Host setup

The most practical setup path already in the repo is:

```bash
bash scripts/setup_host.sh
```

That script installs and configures:

- apt packages
- Python venv
- audio / OpenCV / whisper dependencies
- optional CUDA build of `llama-cpp-python`
- TensorRT pose dependencies
- Node.js 22
- OpenClaw
- face TRT engines
- hand pose model files

You can also run setup incrementally:

```bash
bash scripts/setup_host.sh --step apt
bash scripts/setup_host.sh --step venv
bash scripts/setup_host.sh --step trt_pose
bash scripts/setup_host.sh --step node
bash scripts/setup_host.sh --step trt_engines
bash scripts/setup_host.sh --step models
```

After setup:

```bash
source venv/bin/activate
```

### Local mode

This is the main "Jarvis on the robot" path.

Example:

```bash
python main.py --llm-backend openai --asr-backend cloud --no-vlm
```

Safer non-hardware-heavy test:

```bash
python main.py --llm-backend openrouter --no-robot
```

Useful flags:

- `--no-vlm`
- `--no-tts`
- `--no-robot`
- `--no-listen`
- `--no-gestures`
- `--no-openclaw`
- `--agora`
- `--llm-backend local|openai|openrouter`
- `--asr-backend local|cloud`
- `--vlm-backend transformers|llama_cpp`
- `--ambient-interval`
- `--silence-timeout`
- `--departure-buffer`

### Agora mode

Run:

```bash
python main.py --agora --no-vlm
```

Or use the startup wrapper:

```bash
bash start.sh
```

`start.sh` does three things:

1. starts the Reachy daemon
2. starts a Cloudflare tunnel for the MCP endpoint
3. launches the Agora pipeline

## Environment Variables

```bash
# LLM backends
OPENAI_API_KEY
OPENROUTER_API_KEY

# Agora
AGORA_APP_ID
AGORA_CUSTOMER_ID
AGORA_CUSTOMER_SECRET
AGORA_CHANNEL
AGORA_RTC_TOKEN

# MCP / OpenClaw
MCP_PUBLIC_URL

# Philips Hue
HUE_BRIDGE_IP
HUE_LIGHT_IDS

# TTS (Agora mode)
MINIMAX_GROUP_ID
MINIMAX_API_KEY
```

`main.py` also loads a local `.env` file if present.

## Common Commands

```bash
# Run full system in local mode
python main.py --llm-backend openrouter --no-robot

# Run with Agora cloud voice
python main.py --agora --no-robot

# Person follow demo
python scripts/person_follow.py --no-robot --debug

# Dance demo
python scripts/dance_to_music.py --no-robot

# Enroll a face
python scripts/enroll_face.py --photo face.jpg --name "Alice" --face-id alice --facts "Engineer"

# Test gestures
python scripts/test_gestures.py
```

## Repo Layout

- [`main.py`](./main.py): orchestrator
- [`pipeline/`](./pipeline): core runtime code
- [`scripts/`](./scripts): demos, tests, setup helpers
- [`config/`](./config): prompts and ambient lines
- [`models/`](./models): models, TRT engines, recorded emotion motions
- [`static/agora/`](./static/agora): browser client for Agora mode
- [`CLAUDE.md`](./CLAUDE.md): detailed engineering notes
- [`INTEGRATION_PLAN.md`](./INTEGRATION_PLAN.md): roadmap for deeper subsystem integration

## Notes

- This repo is optimized for Jetson/Linux rather than macOS development.
- OpenClaw is a key dependency if you want Jarvis to do useful desktop work.
- Agora mode is additive, not the whole system. The Jetson-side robot/tool orchestration still matters.
- [`CLAUDE.md`](./CLAUDE.md) contains the most detailed subsystem notes.
- [`INTEGRATION_PLAN.md`](./INTEGRATION_PLAN.md) is a roadmap, not a guarantee that all planned architecture is already merged.
