"""Camera -> VLM scene description"""

import cv2
import numpy as np
import threading
import time

# ZED camera is side-by-side stereo — use left half only
ZED_STEREO = True

# OpenCV DNN face detector paths (ships with opencv)
FACE_PROTO = cv2.data.haarcascades  # fallback to haar if DNN not available


class VisionPipeline:
    def __init__(self, camera_index=0, use_vlm=True, vlm_backend="transformers"):
        """
        vlm_backend: "transformers" (SmolVLM2) or "llama_cpp" (multimodal GGUF)
        """
        self.camera_index = camera_index
        self.use_vlm = use_vlm
        self.vlm_backend = vlm_backend

        self._cap = None
        self._vlm_model = None
        self._vlm_processor = None
        self._face_cascade = None
        self._lock = threading.Lock()

    def open_camera(self):
        if self._cap is None or not self._cap.isOpened():
            self._cap = cv2.VideoCapture(self.camera_index)
            if not self._cap.isOpened():
                print(f"[vision] WARNING: Cannot open camera {self.camera_index}")
                return False
            print(f"[vision] Camera {self.camera_index} opened.")
        return True

    def grab_frame(self):
        """Grab a frame from the camera. Returns left-eye frame (or None)."""
        if self._cap is None or not self._cap.isOpened():
            if not self.open_camera():
                return None
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None
        # ZED stereo: take left half
        if ZED_STEREO and frame.shape[1] > 2000:
            frame = frame[:, : frame.shape[1] // 2]
        return frame

    def _load_face_detector(self):
        if self._face_cascade is None:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._face_cascade = cv2.CascadeClassifier(cascade_path)
            print("[vision] Face detector loaded (Haar cascade).")

    def detect_faces(self, frame):
        """Detect faces in frame. Returns list of (x, y, w, h) tuples."""
        self._load_face_detector()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        if len(faces) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]

    def _load_vlm(self):
        if self._vlm_model is not None:
            return True

        if self.vlm_backend == "transformers":
            return self._load_vlm_transformers()
        elif self.vlm_backend == "llama_cpp":
            return self._load_vlm_llamacpp()
        return False

    def _load_vlm_transformers(self):
        """Load SmolVLM2-2.2B-Instruct via transformers."""
        try:
            import torch
            from transformers import AutoProcessor, AutoModelForImageTextToText

            model_name = "HuggingFaceTB/SmolVLM2-256M-Video-Instruct"
            print(f"[vision] Loading VLM: {model_name}...")
            self._vlm_processor = AutoProcessor.from_pretrained(model_name)
            self._vlm_model = AutoModelForImageTextToText.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="cuda"
            )
            print("[vision] SmolVLM2 loaded.")
            return True
        except Exception as e:
            print(f"[vision] Failed to load SmolVLM2: {e}")
            print("[vision] Falling back to no-VLM mode.")
            self.use_vlm = False
            return False

    def _load_vlm_llamacpp(self):
        """Load multimodal GGUF via llama-cpp-python (Option A fallback)."""
        # Placeholder for hackathon — would need a multimodal GGUF model
        print("[vision] llama_cpp VLM backend not yet configured.")
        self.use_vlm = False
        return False

    def get_scene_description(self, frame):
        """Run VLM on a frame to get a text description of the scene."""
        if not self.use_vlm:
            return self._simple_scene_description(frame)

        with self._lock:
            if not self._load_vlm():
                return self._simple_scene_description(frame)

            try:
                return self._run_vlm_transformers(frame)
            except Exception as e:
                print(f"[vision] VLM inference error: {e}")
                return self._simple_scene_description(frame)

    def _run_vlm_transformers(self, frame):
        """Run SmolVLM2 inference on a frame."""
        import torch
        from PIL import Image

        # Convert BGR to RGB PIL image
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Resize to save memory
        h, w = rgb.shape[:2]
        if w > 640:
            scale = 640 / w
            rgb = cv2.resize(rgb, (640, int(h * scale)))
        image = Image.fromarray(rgb)

        # Use chat template format for SmolVLM2
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "Briefly describe the scene. Focus on people, their expressions, and what they're doing. One or two sentences max."}
                ]
            }
        ]
        prompt = self._vlm_processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self._vlm_processor(text=prompt, images=[image], return_tensors="pt")
        inputs = {k: v.to("cuda") if hasattr(v, "to") else v for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = self._vlm_model.generate(**inputs, max_new_tokens=80)

        # Decode only new tokens
        input_len = inputs["input_ids"].shape[1]
        result = self._vlm_processor.decode(output_ids[0][input_len:], skip_special_tokens=True)
        return result.strip()

    def _simple_scene_description(self, frame):
        """Fallback: describe scene using face detection only."""
        faces = self.detect_faces(frame)
        n = len(faces)
        if n == 0:
            return "The scene appears empty, no people visible."
        elif n == 1:
            x, y, w, h = faces[0]
            cx = x + w // 2
            fw = frame.shape[1]
            pos = "left" if cx < fw // 3 else ("right" if cx > 2 * fw // 3 else "center")
            return f"One person visible, positioned {pos} of frame."
        else:
            return f"{n} people visible in the scene."

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None


if __name__ == "__main__":
    vp = VisionPipeline(use_vlm=False)
    if vp.open_camera():
        frame = vp.grab_frame()
        if frame is not None:
            faces = vp.detect_faces(frame)
            print(f"Faces detected: {len(faces)}")
            desc = vp.get_scene_description(frame)
            print(f"Scene: {desc}")
        vp.release()
    else:
        print("No camera available (OK for pre-hackathon testing)")
