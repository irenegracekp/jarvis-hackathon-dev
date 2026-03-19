"""Main orchestrator -- three worker threads + coordination"""

import argparse
import queue
import threading
import time
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def parse_args():
    parser = argparse.ArgumentParser(description="The Witness - Reachy Mini Hackathon")
    parser.add_argument("--no-vlm", action="store_true", help="Skip VLM (vision-language model)")
    parser.add_argument("--no-tts", action="store_true", help="Skip TTS (print instead of speak)")
    parser.add_argument("--no-robot", action="store_true", default=True, help="Skip robot commands (default)")
    parser.add_argument("--no-listen", action="store_true", help="Skip mic input (for testing)")
    parser.add_argument("--ambient-only", action="store_true", help="Only ambient mode")
    parser.add_argument("--engaged-only", action="store_true", help="Only engaged mode (test conversation)")
    parser.add_argument("--vlm-backend", default="transformers", choices=["transformers", "llama_cpp"],
                        help="VLM backend")
    parser.add_argument("--llm-model", default=None, help="LLM GGUF filename in models/")
    parser.add_argument("--gpu-layers", type=int, default=0, help="Number of LLM layers on GPU (default 0=CPU)")
    parser.add_argument("--ambient-interval", type=float, default=30.0,
                        help="Seconds between ambient lines (default 30)")
    parser.add_argument("--silence-timeout", type=float, default=30.0,
                        help="Seconds of silence before returning to ambient (default 30)")
    return parser.parse_args()


# Shared state
class State:
    def __init__(self):
        self.mode = "ambient"  # "ambient" or "engaged"
        self.last_speech_time = 0.0
        self.audio_queue = queue.Queue()     # (text, timestamp) from listen
        self.response_queue = queue.Queue()  # response dicts from brain
        self.running = True


def listen_loop(state, args):
    """Thread 1: Mic -> Whisper -> audio_queue"""
    if args.no_listen or args.ambient_only:
        print("[main] Listen loop disabled.")
        return

    from pipeline.listen import ListenPipeline
    listener = ListenPipeline()
    listener.start()
    print("[main] Listen loop running.")

    while state.running:
        try:
            text, ts = listener.text_queue.get(timeout=1)
            state.audio_queue.put((text, ts))
            state.last_speech_time = ts
        except queue.Empty:
            continue

    listener.stop()


def brain_loop(state, args):
    """Thread 2: Orchestrate ambient/engaged modes"""
    from pipeline.brain import BrainPipeline
    from pipeline.vision import VisionPipeline

    brain = BrainPipeline(model_name=args.llm_model, n_gpu_layers=args.gpu_layers)
    vision = VisionPipeline(
        use_vlm=not args.no_vlm,
        vlm_backend=args.vlm_backend,
    )

    if not args.no_vlm:
        vision.open_camera()

    last_ambient_time = 0.0
    last_face_count = 0
    print("[main] Brain loop running.")

    while state.running:
        try:
            # Check for incoming speech
            try:
                text, ts = state.audio_queue.get(timeout=0.5)
                if args.ambient_only:
                    continue

                # Switch to engaged mode
                state.mode = "engaged"
                state.last_speech_time = ts
                print(f"[main] ENGAGED: '{text}'")

                # Get scene description
                scene_desc = ""
                if not args.no_vlm:
                    frame = vision.grab_frame()
                    if frame is not None:
                        scene_desc = vision.get_scene_description(frame)
                        print(f"[main] Scene: {scene_desc}")

                # Get LLM response
                response = brain.engage(text, scene_desc)
                state.response_queue.put(response)
                continue

            except queue.Empty:
                pass

            # Check if we should return to ambient
            if state.mode == "engaged":
                silence = time.time() - state.last_speech_time
                if silence > args.silence_timeout:
                    print("[main] Silence timeout -> ambient mode")
                    state.mode = "ambient"
                    brain.clear_all_conversations()

            # Ambient mode: periodic ambient lines
            if state.mode == "ambient" and not args.engaged_only:
                now = time.time()
                if now - last_ambient_time > args.ambient_interval:
                    # Optionally check faces
                    face_count = 0
                    scene_desc = ""
                    if not args.no_vlm:
                        frame = vision.grab_frame()
                        if frame is not None:
                            faces = vision.detect_faces(frame)
                            face_count = len(faces)
                            scene_desc = f"{face_count} faces detected"

                    # Only speak if faces are present (or no camera)
                    if face_count > 0 or args.no_vlm:
                        response = brain.ambient_react(scene_desc)
                        state.response_queue.put(response)
                        last_ambient_time = now

                    last_face_count = face_count

            time.sleep(0.1)

        except Exception as e:
            print(f"[main] Brain loop error: {e}")
            time.sleep(1)

    vision.release()


def output_loop(state, args):
    """Thread 3: Response -> TTS + Robot actions"""
    if args.no_tts:
        from pipeline.speak import DummySpeakPipeline
        speaker = DummySpeakPipeline()
    else:
        from pipeline.speak import SpeakPipeline
        speaker = SpeakPipeline()

    from pipeline.robot import RobotController
    robot = RobotController(real_robot=not args.no_robot)

    speaker.start()
    print("[main] Output loop running.")

    while state.running:
        try:
            response = state.response_queue.get(timeout=1)

            # Execute robot actions
            robot.execute_response(response)

            # Speak
            speech = response.get("speech", "")
            if speech:
                speaker.say(speech)

            emotion = response.get("emotion", "")
            print(f"[main] Output: [{emotion}] {speech}")

        except queue.Empty:
            continue
        except Exception as e:
            print(f"[main] Output error: {e}")

    speaker.stop()


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
    state = State()

    print("=" * 50)
    print("  THE WITNESS - Reachy Mini Hackathon Project")
    print("=" * 50)
    print(f"  VLM: {'OFF' if args.no_vlm else args.vlm_backend}")
    print(f"  TTS: {'OFF' if args.no_tts else 'ON'}")
    print(f"  Robot: {'OFF' if args.no_robot else 'ON'}")
    print(f"  Listen: {'OFF' if args.no_listen else 'ON'}")
    print(f"  Mode: {'ambient-only' if args.ambient_only else 'engaged-only' if args.engaged_only else 'full'}")
    print("=" * 50)

    threads = []

    # Start listen thread
    t1 = threading.Thread(target=listen_loop, args=(state, args), daemon=True, name="listen")
    t1.start()
    threads.append(t1)

    # Start brain thread
    t2 = threading.Thread(target=brain_loop, args=(state, args), daemon=True, name="brain")
    t2.start()
    threads.append(t2)

    # Start output thread
    t3 = threading.Thread(target=output_loop, args=(state, args), daemon=True, name="output")
    t3.start()
    threads.append(t3)

    # Main thread: keyboard input or just wait
    try:
        if args.no_listen and not args.ambient_only:
            keyboard_input_loop(state, args)
        else:
            while state.running:
                time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[main] Shutting down...")
        state.running = False

    # Brief wait for threads to finish
    for t in threads:
        t.join(timeout=2)

    print("[main] Done.")


if __name__ == "__main__":
    main()
