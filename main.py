"""Main orchestrator -- worker threads + state machine coordination

State machine:
  IDLE     → no faces detected
  AMBIENT  → face(s) present, robot observing, greets arrivals
  ENGAGED  → active conversation locked to one person

Threads:
  1. listen_loop    — Mic → Whisper → text queue
  2. brain_loop     — Face tracking + speech routing + LLM
  3. output_loop    — Response → TTS + Robot
  4. gesture_loop   — Hand gestures → desktop actions (independent)
"""

import argparse
import json
import queue
import re
import threading
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env file if present (no extra dependency needed)
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _, _val = _line.partition("=")
                _key, _val = _key.strip(), _val.strip()
                if _key and _val:
                    os.environ.setdefault(_key, _val)


def parse_args():
    parser = argparse.ArgumentParser(description="Jarvis - Reachy Mini Hackathon")
    parser.add_argument("--no-vlm", action="store_true", help="Skip VLM (vision-language model)")
    parser.add_argument("--no-tts", action="store_true", help="Skip TTS (print instead of speak)")
    parser.add_argument("--no-robot", action="store_true", default=True, help="Skip robot commands (default)")
    parser.add_argument("--no-listen", action="store_true", help="Skip mic input (for testing)")
    parser.add_argument("--no-gestures", action="store_true", help="Skip gesture recognition")
    parser.add_argument("--3d-viewer", action="store_true", dest="viewer_3d",
                        help="Launch arch-viewer-full with hand gesture control (swipe/pinch)")
    parser.add_argument("--no-openclaw", action="store_true", help="Skip OpenClaw routing")
    parser.add_argument("--lights-backend", default="none", choices=["none", "hue", "lifx", "govee"],
                        help="Smart light backend: none, hue, lifx, or govee (default: none)")
    parser.add_argument("--agora", action="store_true", help="Use Agora Conversational AI (cloud ASR+LLM+TTS)")
    parser.add_argument("--agora-channel", default=None, help="Agora channel name (default: from AGORA_CHANNEL env)")
    parser.add_argument("--proxy-port", type=int, default=8001, help="MCP tool server port (default: 8001)")
    parser.add_argument("--ambient-only", action="store_true", help="Only ambient mode")
    parser.add_argument("--engaged-only", action="store_true", help="Only engaged mode (test conversation)")
    parser.add_argument("--vlm-backend", default="transformers", choices=["transformers", "llama_cpp"],
                        help="VLM backend")
    parser.add_argument("--llm-model", default=None, help="LLM GGUF filename in models/")
    parser.add_argument("--llm-backend", default="local", choices=["local", "openai", "openrouter"],
                        help="LLM backend: local (Gemma GGUF), openai (GPT-4o-mini), or openrouter (Claude Sonnet)")
    parser.add_argument("--asr-backend", default="local", choices=["local", "cloud"],
                        help="ASR backend: local (faster-whisper) or cloud (OpenAI Whisper API)")
    parser.add_argument("--gpu-layers", type=int, default=-1, help="Number of LLM layers on GPU (-1=all, 0=CPU)")
    parser.add_argument("--ambient-interval", type=float, default=30.0,
                        help="Seconds between ambient lines (default 30)")
    parser.add_argument("--silence-timeout", type=float, default=30.0,
                        help="Seconds of silence before returning to ambient (default 30)")
    parser.add_argument("--departure-buffer", type=float, default=10.0,
                        help="Seconds face must be gone before counting as departed (default 10)")
    return parser.parse_args()


LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

_interaction_count = 0

def _log_interaction(frame, speech, scene_desc, response, latency=None):
    """Save frame + context for each engaged interaction."""
    global _interaction_count
    _interaction_count += 1
    ts = time.strftime("%Y%m%d_%H%M%S")
    prefix = f"{ts}_{_interaction_count:03d}"

    if frame is not None:
        import cv2
        cv2.imwrite(os.path.join(LOG_DIR, f"{prefix}.jpg"), frame)

    entry = {
        "timestamp": ts,
        "speech": speech,
        "scene_description": scene_desc,
        "response": response,
        "latency": latency,
    }
    with open(os.path.join(LOG_DIR, f"{prefix}.json"), "w") as f:
        json.dump(entry, f, indent=2)


GREET_COOLDOWN = 300  # 5 minutes

# Goodbye patterns
_BYE_PATTERN = re.compile(
    r'\b(bye|goodbye|see you|gotta go|have to go|i.m leaving|later|peace out|take care)\b',
    re.IGNORECASE
)


class State:
    def __init__(self, departure_buffer=5.0):
        self.mode = "idle"  # "idle", "ambient", "engaged"
        self.last_speech_time = 0.0
        self.audio_queue = queue.Queue()
        self.response_queue = queue.Queue()
        self.running = True
        self.current_face_id = None
        self._greet_times = {}
        self._last_seen = {}  # face_id -> last time face was detected
        self.departure_buffer = departure_buffer
        self.tts_speaking = False  # True while TTS is outputting audio

    def should_greet(self, face_id):
        last = self._greet_times.get(face_id, 0)
        return (time.time() - last) > GREET_COOLDOWN

    def record_greet(self, face_id):
        self._greet_times[face_id] = time.time()

    def update_face_seen(self, face_id):
        self._last_seen[face_id] = time.time()

    def is_face_departed(self, face_id):
        """Face must be gone for departure_buffer seconds."""
        last = self._last_seen.get(face_id, 0)
        return (time.time() - last) > self.departure_buffer

    def is_goodbye(self, text):
        return bool(_BYE_PATTERN.search(text))


def listen_loop(state, args):
    """Thread 1: Mic -> Whisper -> audio_queue"""
    if args.no_listen or args.ambient_only:
        print("[main] Listen loop disabled.")
        return

    from pipeline.listen import ListenPipeline
    listener = ListenPipeline(asr_backend=args.asr_backend)
    listener.start()
    print("[main] Listen loop running.")

    while state.running:
        try:
            text, ts = listener.text_queue.get(timeout=1)
            # Ignore speech while TTS is playing (prevents feedback loop)
            if state.tts_speaking:
                print(f"[listen] Ignored (TTS playing): {text}")
                continue
            state.audio_queue.put((text, ts))
            state.last_speech_time = ts
        except queue.Empty:
            continue

    listener.stop()


def gesture_loop(state, args):
    """Thread 4: Hand gestures -> desktop actions (independent of conversation)"""
    if args.no_gestures:
        print("[main] Gesture loop disabled.")
        return

    try:
        from pipeline.gestures import GestureRecognizer
        from pipeline.actions import ActionMapper
    except ImportError as e:
        print(f"[main] Gesture loop disabled (missing dep): {e}")
        return

    import cv2

    gr = GestureRecognizer()
    if not gr.load_model():
        print("[main] Gesture model failed to load. Gesture loop disabled.")
        return

    am = ActionMapper()
    print("[main] Gesture loop running.")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[main] Gesture loop: cannot open camera.")
        return

    while state.running:
        try:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            # ZED stereo: left half
            if frame.shape[1] > 2000:
                frame = frame[:, :frame.shape[1] // 2]

            result = gr.process_frame(frame)
            gesture = result["gesture"]
            motion = result["motion"]
            hand_pos = result["hand_position"]

            action = am.execute_gesture(gesture, hand_pos, motion)
            if action:
                print(f"[gesture] {gesture} -> {action}")

        except Exception as e:
            print(f"[main] Gesture loop error: {e}")
            time.sleep(1)

    cap.release()


def viewer_3d_loop(state, args):
    """Launch the 3D arch-viewer with gesture control (WebSocket server + HTTP + browser)."""
    import subprocess
    import asyncio as _asyncio

    project_root = os.path.dirname(os.path.abspath(__file__))
    viewer_dir = os.path.join(project_root, "arch-viewer-full")
    server_script = os.path.join(project_root, "gesture_ws_server.py")

    if not os.path.isdir(viewer_dir):
        print("[3d-viewer] arch-viewer-full/ not found. Skipping.")
        return

    # 1. Start HTTP server for the viewer (port 8090 to avoid clash with dashboard on 8080)
    http_port = 8090
    http_proc = subprocess.Popen(
        ["python3", "-m", "http.server", str(http_port)],
        cwd=viewer_dir,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    print(f"[3d-viewer] HTTP server on port {http_port} (PID {http_proc.pid})")

    # 2. Start the gesture WebSocket server
    ws_proc = subprocess.Popen(
        ["python3", server_script],
        cwd=project_root,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    print(f"[3d-viewer] Gesture WS server on port 8765 (PID {ws_proc.pid})")

    # 3. Wait a moment for servers to start, then open browser
    time.sleep(3)
    display = os.environ.get("DISPLAY", ":0")
    try:
        subprocess.Popen(
            ["chromium", f"http://localhost:{http_port}"],
            env={**os.environ, "DISPLAY": display},
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        print(f"[3d-viewer] Opened Chromium → http://localhost:{http_port}")
    except FileNotFoundError:
        print(f"[3d-viewer] Chromium not found. Open http://localhost:{http_port} manually.")

    # 4. Stream gesture server logs while running
    while state.running:
        line = ws_proc.stdout.readline()
        if line:
            text = line.decode("utf-8", errors="replace").rstrip()
            if "[GESTURE]" in text or "[WS]" in text or "[ERROR]" in text:
                print(f"[3d-viewer] {text}")
        if ws_proc.poll() is not None:
            print("[3d-viewer] Gesture server exited unexpectedly, restarting...")
            time.sleep(2)
            ws_proc = subprocess.Popen(
                ["python3", server_script],
                cwd=project_root,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            )

    # Cleanup
    ws_proc.terminate()
    http_proc.terminate()
    print("[3d-viewer] Stopped.")


def brain_loop(state, args, dashboard=None):
    """Thread 2: Orchestrate idle/ambient/engaged modes with face recognition + memory + OpenClaw"""
    from pipeline.brain import BrainPipeline
    from pipeline.vision import VisionPipeline
    from pipeline.memory import MemoryStore

    memory = MemoryStore()

    brain = BrainPipeline(
        model_name=args.llm_model,
        n_gpu_layers=args.gpu_layers,
        memory_store=memory,
        llm_backend=args.llm_backend,
    )
    vision = VisionPipeline(
        use_vlm=not args.no_vlm,
        vlm_backend=args.vlm_backend,
    )

    # OpenClaw bridge (optional)
    openclaw = None
    if not args.no_openclaw:
        try:
            from pipeline.openclaw_bridge import OpenClawBridge
            openclaw = OpenClawBridge()
            if openclaw.is_available():
                print("[main] OpenClaw bridge connected.")
            else:
                print("[main] OpenClaw not running. Voice commands will go to brain only.")
        except ImportError:
            print("[main] OpenClaw bridge not available.")

    if not args.no_vlm:
        vision.open_camera()

    last_ambient_time = 0.0
    last_face_event_time = 0.0
    face_event_interval = 3.0  # Check faces every 3s in ambient mode
    last_dashboard_frame_time = 0.0
    dashboard_frame_interval = 0.15
    arrival_confirmations = {}  # face_id -> first_seen_time (must be seen 3s before greeting)
    ARRIVAL_CONFIRM_SEC = 3.0  # Face must be present for 3s before we greet
    print("[main] Brain loop running.")

    def _dash_event(etype, text):
        if dashboard:
            dashboard.state.add_event(etype, text)

    def _dash_update_frame(frame, faces=None):
        if dashboard and frame is not None:
            dashboard.state.update_frame(frame, faces, memory)

    def _dash_faces_present(faces):
        if not dashboard:
            return
        face_info = []
        for f in faces:
            fid = f["face_id"]
            person = memory.get_person(fid)
            info = {"face_id": fid, "is_known": f.get("is_known", False)}
            if person:
                info["name"] = person["name"]
                info["facts"] = person.get("facts", [])
                info["times_seen"] = person.get("times_seen", 1)
            face_info.append(info)
        dashboard.state.update_state(faces_present=face_info)

    def _handle_speech(text, ts):
        """Route speech: OpenClaw command or brain conversation."""
        t_start = time.time()

        # Check for goodbye
        if state.is_goodbye(text):
            print(f"[main] Goodbye detected: '{text}'")
            face_id = state.current_face_id or "default"
            person = memory.get_person(face_id)
            name = person["name"] if person else "friend"
            response = {"speech": f"See you later, {name}! It was nice chatting.",
                        "emotion": "calm", "head_direction": "nod", "antenna_state": "wiggle"}
            state.response_queue.put(response)
            _dash_event("speech", f"Goodbye: {name}")

            # Return to ambient
            state.mode = "ambient"
            brain.clear_conversation(face_id)
            state.current_face_id = None
            state.lights.set_state("ambient")
            if dashboard:
                dashboard.state.update_state(mode="ambient", current_face=None)
            return

        # Check if this is an OpenClaw command
        if openclaw and openclaw.is_available() and openclaw.is_agent_command(text):
            print(f"[main] OpenClaw command: '{text}'")
            _dash_event("openclaw", f"Command: \"{text}\"")
            result = openclaw.send_command(text)
            print(f"[main] OpenClaw result: {result}")
            _dash_event("openclaw", f"Result: {result}")

            # Tell the brain about it so the robot can comment
            face_id = state.current_face_id or "default"
            memory_context = memory.get_context_string(face_id)
            response = brain.engage(
                f"I just executed a computer command for the user: '{text}'. Result: {result}. "
                f"Comment on it briefly.",
                "", face_id=face_id, memory_context=memory_context)
            state.response_queue.put(response)
            return

        # Regular conversation
        state.mode = "engaged"
        state.last_speech_time = ts
        print(f"[main] ENGAGED: '{text}'")
        state.lights.set_state("engaged")
        if dashboard:
            dashboard.state.update_state(mode="engaged", last_speech=text)
            _dash_event("speech", f"Heard: \"{text}\"")

        scene_desc = ""
        frame = None
        t_vlm = 0.0
        face_id = state.current_face_id or "default"

        if not args.no_vlm:
            t0 = time.time()
            frame = vision.grab_frame()
            if frame is not None:
                events = vision.get_face_events(frame)
                faces = events["present"]
                _dash_update_frame(frame, faces)
                _dash_faces_present(faces)

                # Update seen timestamps
                for f in faces:
                    if f["face_id"]:
                        state.update_face_seen(f["face_id"])

                if faces:
                    speaker = vision.face_recognizer.get_closest_to_center(faces, frame.shape[1])
                    if speaker and speaker["face_id"]:
                        face_id = speaker["face_id"]
                        state.current_face_id = face_id
                        if dashboard:
                            person = memory.get_person(face_id)
                            dashboard.state.update_state(
                                current_face=person["name"] if person else face_id)
                        if not speaker["is_known"]:
                            vision.face_recognizer.enroll(face_id, speaker["embedding"])

                scene_desc = vision.get_scene_description(frame)
                print(f"[main] Scene: {scene_desc}")
            t_vlm = time.time() - t0

        memory_context = memory.get_context_string(face_id)
        if memory_context:
            print(f"[main] Memory: {memory_context}")

        t0 = time.time()
        response = brain.engage(text, scene_desc, face_id=face_id, memory_context=memory_context)
        t_llm = time.time() - t0
        state.response_queue.put(response)
        _dash_event("speech", f"Said: \"{response.get('speech', '')}\"")

        # Save memory from LLM response
        save_mem = response.get("save_memory")
        if save_mem:
            _dash_event("memory", f"Saved: {save_mem}")
            if memory.get_person(face_id) is None:
                if save_mem.lower().startswith("name is "):
                    name = save_mem[8:].strip()
                    memory.create_person(face_id, name)
                else:
                    memory.create_person(face_id, face_id, facts=[save_mem])
            else:
                if save_mem.lower().startswith("name is "):
                    name = save_mem[8:].strip()
                    memory.set_name(face_id, name)
                else:
                    memory.add_fact(face_id, save_mem)

        t_total = time.time() - t_start
        latency = {"total": round(t_total, 2), "vlm": round(t_vlm, 2), "llm": round(t_llm, 2)}
        print(f"[latency] total={t_total:.1f}s  vlm={t_vlm:.1f}s  llm={t_llm:.1f}s")
        _log_interaction(frame, text, scene_desc, response, latency)

    while state.running:
        try:
            # Check for incoming speech
            try:
                text, ts = state.audio_queue.get(timeout=0.5)
                if args.ambient_only:
                    continue
                _handle_speech(text, ts)
                continue
            except queue.Empty:
                pass

            # Check if engaged person left or silence timeout
            if state.mode == "engaged":
                silence = time.time() - state.last_speech_time
                face_gone = (state.current_face_id and
                             state.is_face_departed(state.current_face_id))

                if face_gone:
                    print(f"[main] Engaged face departed -> ambient mode")
                    state.mode = "ambient"
                    brain.clear_conversation(state.current_face_id)
                    state.current_face_id = None
                    state.lights.set_state("ambient")
                    if dashboard:
                        dashboard.state.update_state(mode="ambient", current_face=None)
                        _dash_event("system", "Speaker left -> ambient mode")
                elif silence > args.silence_timeout:
                    print("[main] Silence timeout -> ambient mode")
                    state.mode = "ambient"
                    brain.clear_all_conversations()
                    state.current_face_id = None
                    state.lights.set_state("ambient")
                    if dashboard:
                        dashboard.state.update_state(mode="ambient", current_face=None)
                        _dash_event("system", "Silence timeout -> ambient mode")

            # Ambient / idle mode: watch for faces
            if state.mode in ("ambient", "idle") and not args.engaged_only:
                now = time.time()

                if not args.no_vlm and (now - last_face_event_time > face_event_interval):
                    frame = vision.grab_frame()
                    if frame is not None:
                        events = vision.get_face_events(frame)
                        last_face_event_time = now

                        _dash_update_frame(frame, events["present"])
                        _dash_faces_present(events["present"])

                        # Update seen timestamps for all present faces
                        for f in events["present"]:
                            if f["face_id"]:
                                state.update_face_seen(f["face_id"])

                        # Transition idle -> ambient when faces appear
                        if events["present"] and state.mode == "idle":
                            state.mode = "ambient"
                            state.lights.set_state("ambient")
                            if dashboard:
                                dashboard.state.update_state(mode="ambient")
                                _dash_event("system", "Faces detected -> ambient mode")

                        # Transition ambient -> idle when no faces for a while
                        if not events["present"] and state.mode == "ambient":
                            # Check if ALL tracked faces are departed
                            all_gone = all(state.is_face_departed(fid)
                                          for fid in vision._previous_faces)
                            if all_gone and not vision._previous_faces:
                                state.mode = "idle"
                                state.lights.set_state("idle")
                                if dashboard:
                                    dashboard.state.update_state(mode="idle")

                        # Handle arrivals — require face to be present for ARRIVAL_CONFIRM_SEC
                        for face in events["arrivals"]:
                            fid = face["face_id"]
                            if fid not in arrival_confirmations:
                                arrival_confirmations[fid] = now
                                # Enroll unknown faces immediately for tracking
                                if not face["is_known"]:
                                    vision.face_recognizer.enroll(fid, face["embedding"])

                        # Check confirmed arrivals (present long enough to greet)
                        for fid in list(arrival_confirmations.keys()):
                            # Is this face still present?
                            present_ids = {f["face_id"] for f in events["present"]}
                            if fid not in present_ids:
                                # Face left before confirmation
                                del arrival_confirmations[fid]
                                continue

                            elapsed = now - arrival_confirmations[fid]
                            if elapsed >= ARRIVAL_CONFIRM_SEC and state.should_greet(fid):
                                person = memory.get_person(fid)
                                if person:
                                    memory.record_seen(fid)
                                    print(f"[main] Known face confirmed: {person['name']} ({fid})")
                                    _dash_event("arrive", f"Known: {person['name']}")
                                    response = brain.greet(fid, person,
                                                           vision.get_scene_description(frame))
                                else:
                                    print(f"[main] Unknown face confirmed: {fid}")
                                    _dash_event("arrive", f"Unknown face: {fid}")
                                    response = brain.greet(fid, None,
                                                           vision.get_scene_description(frame))

                                state.response_queue.put(response)
                                state.record_greet(fid)
                                state.lights.flash(response.get("emotion", "excited"))
                                del arrival_confirmations[fid]

                        # Log departures (only if buffered)
                        for fid in events["departures"]:
                            arrival_confirmations.pop(fid, None)
                            if state.is_face_departed(fid):
                                person = memory.get_person(fid)
                                name = person["name"] if person else fid
                                print(f"[main] Face departed: {name}")
                                _dash_event("depart", f"Left: {name}")

                # Dashboard frame updates
                elif not args.no_vlm and dashboard and (now - last_dashboard_frame_time > dashboard_frame_interval):
                    frame = vision.grab_frame()
                    if frame is not None:
                        faces = vision.identify_faces(frame)
                        _dash_update_frame(frame, faces)
                        last_dashboard_frame_time = now

                # Periodic ambient lines (only if faces present)
                if state.mode == "ambient" and now - last_ambient_time > args.ambient_interval:
                    face_count = len(vision._previous_faces) if not args.no_vlm else 0
                    if face_count > 0 or args.no_vlm:
                        response = brain.ambient_react(f"{face_count} faces detected")
                        state.response_queue.put(response)
                        last_ambient_time = now

            time.sleep(0.1)

        except Exception as e:
            print(f"[main] Brain loop error: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(1)

    vision.release()


def output_loop(state, args):
    """Thread 3: Response -> TTS + Robot actions + Lights"""
    if args.no_tts:
        from pipeline.speak import DummySpeakPipeline
        speaker = DummySpeakPipeline()
    else:
        from pipeline.speak import SpeakPipeline
        speaker = SpeakPipeline()

    from pipeline.robot import RobotController
    robot = RobotController(real_robot=not args.no_robot)

    # Smart light integration (hue / lifx / govee / none)
    from pipeline.lights import create_light_backend
    lights = create_light_backend(args.lights_backend)
    state.lights = lights  # Expose to brain_loop for state transitions

    speaker.start()
    print("[main] Output loop running.")

    while state.running:
        try:
            response = state.response_queue.get(timeout=1)
            robot.execute_response(response)

            # Update lights with emotion
            emotion = response.get("emotion", "")
            if emotion:
                lights.set_emotion(emotion)

            speech = response.get("speech", "")
            if speech:
                state.tts_speaking = True
                speaker.say_blocking(speech)
                state.tts_speaking = False

            print(f"[main] Output: [{emotion}] {speech}")

        except queue.Empty:
            continue
        except Exception as e:
            state.tts_speaking = False
            print(f"[main] Output error: {e}")

    lights.off()
    speaker.stop()


def start_agora_web_server(port=8080):
    """Start the Agora web voice server in a background thread."""
    from pipeline.agora_web_server import run_server
    t = threading.Thread(
        target=run_server,
        kwargs={"port": port, "open_browser": True},
        daemon=True,
        name="agora_web",
    )
    t.start()
    print(f"[main] Agora voice server started on port {port}.")
    print(f"[main] Open http://localhost:{port} in Chromium to connect.")
    return t


def start_mcp_server(port=8000):
    """Start the MCP tool server in a background thread."""
    from pipeline.mcp_server import run_server
    t = threading.Thread(
        target=run_server,
        kwargs={"port": port},
        daemon=True,
        name="mcp_server",
    )
    t.start()
    print(f"[main] MCP tool server started on port {port}.")
    return t


def start_agora_agent(args):
    """Call Agora REST API to start the Conversational AI agent."""
    import base64
    import requests as req

    app_id = os.environ.get("AGORA_APP_ID", "")
    customer_id = os.environ.get("AGORA_CUSTOMER_ID", "")
    customer_secret = os.environ.get("AGORA_CUSTOMER_SECRET", "")
    channel = args.agora_channel or os.environ.get("AGORA_CHANNEL", "reachy_conversation")
    token = os.environ.get("AGORA_RTC_TOKEN", "")
    mcp_url = os.environ.get("MCP_PUBLIC_URL", f"http://localhost:{args.proxy_port}")

    if not all([app_id, customer_id, customer_secret]):
        print("[main] Missing AGORA_APP_ID, AGORA_CUSTOMER_ID, or AGORA_CUSTOMER_SECRET.")
        return None

    auth = base64.b64encode(f"{customer_id}:{customer_secret}".encode()).decode()

    minimax_group_id = os.environ.get("MINIMAX_GROUP_ID", "")
    minimax_api_key = os.environ.get("MINIMAX_API_KEY", "")

    # Load system prompt for the LLM
    _cfg_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config")
    _prompt_path = os.path.join(_cfg_dir, "system_prompt_agora.txt")
    if not os.path.exists(_prompt_path):
        _prompt_path = os.path.join(_cfg_dir, "system_prompt.txt")
    with open(_prompt_path) as _f:
        system_prompt = _f.read().strip()

    payload = {
        "name": f"jarvis-{int(time.time())}",
        "preset": "openai_gpt_4_1_mini,minimax_speech_2_6_turbo",
        "properties": {
            "channel": channel,
            "token": token,
            "agent_rtc_uid": "1000",
            "remote_rtc_uids": ["12345"],
            "enable_string_uid": False,
            "idle_timeout": 120,
            "advanced_features": {
                "enable_tools": True,
            },
            "asr": {
                "language": "en-US",
            },
            "llm": {
                "system_messages": [
                    {"role": "system", "content": system_prompt}
                ],
                "greeting_message": "Hey there! I'm The Witness. What can I do for you?",
                "max_history": 10,
                "mcp_servers": [
                    {
                        "name": "openclaw",
                        "endpoint": f"{mcp_url}/mcp",
                        "transport": "streamable_http",
                        "allowed_tools": ["execute_desktop_command"],
                        "timeout_ms": 30000,
                    },
                ],
            },
            "tts": {
                "vendor": "minimax",
                "params": {
                    "group_id": minimax_group_id,
                    "api_key": minimax_api_key,
                    "voice_setting": {
                        "voice_id": "English_Strong-WilledBoy",
                    },
                    "audio_setting": {
                        "sample_rate": 44100,
                    },
                },
            },
        },
    }

    try:
        resp = req.post(
            f"https://api.agora.io/api/conversational-ai-agent/v2/projects/{app_id}/join",
            headers={
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        data = resp.json()
        agent_id = data.get("agent_id")
        status = data.get("status")
        print(f"[main] Agora agent started: id={agent_id}, status={status}")
        if not agent_id:
            print(f"[main] Agora agent response: {data}")
        return agent_id
    except Exception as e:
        print(f"[main] Failed to start Agora agent: {e}")
        return None


def stop_agora_agent(agent_id):
    """Call Agora REST API to stop the agent."""
    import base64
    import requests as req

    app_id = os.environ.get("AGORA_APP_ID", "")
    customer_id = os.environ.get("AGORA_CUSTOMER_ID", "")
    customer_secret = os.environ.get("AGORA_CUSTOMER_SECRET", "")

    if not all([app_id, customer_id, customer_secret, agent_id]):
        return

    auth = base64.b64encode(f"{customer_id}:{customer_secret}".encode()).decode()

    try:
        resp = req.post(
            f"https://api.agora.io/api/conversational-ai-agent/v2/projects/{app_id}/agents/{agent_id}/leave",
            headers={
                "Authorization": f"Basic {auth}",
                "Content-Type": "application/json",
            },
            timeout=10,
        )
        print(f"[main] Agora agent stopped (status={resp.status_code}).")
    except Exception as e:
        print(f"[main] Failed to stop Agora agent: {e}")


def keyboard_input_loop(state, args):
    """Optional: type text instead of speaking (for testing without mic)."""
    if not args.no_listen and not args.ambient_only:
        return
    if args.ambient_only:
        return

    print("[main] Keyboard input mode. Type to talk, 'quit' to exit.")
    while state.running:
        try:
            text = input("> ")
            if text.lower() in ("quit", "exit", "q"):
                state.running = False
                break
            if text.strip():
                state.audio_queue.put((text.strip(), time.time()))
        except (EOFError, KeyboardInterrupt):
            state.running = False
            break


def main():
    args = parse_args()
    state = State(departure_buffer=args.departure_buffer)

    print("=" * 50)
    print("  JARVIS - Reachy Mini Hackathon")
    print("=" * 50)
    print(f"  VLM: {'OFF' if args.no_vlm else args.vlm_backend}")
    print(f"  LLM: {args.llm_backend}")
    print(f"  ASR: {'AGORA' if args.agora else args.asr_backend}")
    print(f"  TTS: {'AGORA' if args.agora else 'OFF' if args.no_tts else 'ON'}")
    print(f"  Robot: {'OFF' if args.no_robot else 'ON'}")
    print(f"  Listen: {'AGORA' if args.agora else 'OFF' if args.no_listen else 'ON'}")
    print(f"  Gestures: {'OFF' if args.no_gestures else 'ON'}")
    print(f"  OpenClaw: {'OFF' if args.no_openclaw else 'ON'}")
    print(f"  Lights: {args.lights_backend}")
    print(f"  3D Viewer: {'ON' if args.viewer_3d else 'OFF'}")
    if args.agora:
        print(f"  Agora: ON (voice on :8080, MCP on :{args.proxy_port})")
    print(f"  Mode: {'ambient-only' if args.ambient_only else 'engaged-only' if args.engaged_only else 'full'}")
    if not args.agora:
        print(f"  Dashboard: http://localhost:8080")
    print("=" * 50)

    threads = []

    if args.agora:
        # Agora mode: web server (browser handles RTC) + optional MCP for OpenClaw

        # Start MCP tool server if tunnel URL is configured
        mcp_url = os.environ.get("MCP_PUBLIC_URL", "").strip()
        if mcp_url:
            mcp_thread = start_mcp_server(port=args.proxy_port)
            threads.append(mcp_thread)
            time.sleep(1)
            print(f"[main] MCP tool server ready. Agora will call {mcp_url}/mcp")
        else:
            print("[main] MCP_PUBLIC_URL not set. OpenClaw tool calling disabled.")
            print("[main] Set MCP_PUBLIC_URL in .env and run a tunnel (cloudflared/ngrok) to enable.")

        # Start Agora web voice server (serves browser UI + manages agent)
        # This replaces: listen_loop, output_loop TTS, and the old start_agora_agent
        agora_thread = start_agora_web_server(port=8080)
        threads.append(agora_thread)

        # Brain loop still runs for face detection + ambient behavior
        dashboard = None  # Dashboard port used by Agora web server
        t_brain = threading.Thread(target=brain_loop, args=(state, args, dashboard), daemon=True, name="brain")
        t_brain.start()
        threads.append(t_brain)

        # Gestures
        t_gestures = threading.Thread(target=gesture_loop, args=(state, args), daemon=True, name="gestures")
        t_gestures.start()
        threads.append(t_gestures)

    else:
        # Original mode: local listen + brain + output + gestures
        from pipeline.dashboard import DashboardServer
        dashboard = DashboardServer(port=8080)
        dashboard.start()

        t1 = threading.Thread(target=listen_loop, args=(state, args), daemon=True, name="listen")
        t1.start()
        threads.append(t1)

        t2 = threading.Thread(target=brain_loop, args=(state, args, dashboard), daemon=True, name="brain")
        t2.start()
        threads.append(t2)

        t3 = threading.Thread(target=output_loop, args=(state, args), daemon=True, name="output")
        t3.start()
        threads.append(t3)

        t4 = threading.Thread(target=gesture_loop, args=(state, args), daemon=True, name="gestures")
        t4.start()
        threads.append(t4)

    # 3D Viewer (gesture-controlled arch viewer)
    if args.viewer_3d:
        t_viewer = threading.Thread(target=viewer_3d_loop, args=(state, args), daemon=True, name="3d-viewer")
        t_viewer.start()
        threads.append(t_viewer)

    # Main thread: keyboard input or just wait
    try:
        if not args.agora and args.no_listen and not args.ambient_only:
            keyboard_input_loop(state, args)
        else:
            while state.running:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[main] Shutting down...")
        state.running = False

    for t in threads:
        t.join(timeout=2)

    print("[main] Done.")


if __name__ == "__main__":
    main()
