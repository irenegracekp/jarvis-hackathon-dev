"""Hand gesture recognition using trt_pose_hand (TensorRT) with MediaPipe fallback.

Primary: trt_pose ResNet18 hand pose estimation + SVM gesture classifier (Jetson-optimized).
Fallback: MediaPipe Hands + rule-based classifier (CPU, works everywhere).

Detects hand keypoints and classifies gestures: fist, pan, stop, fine, peace, no_hand.
"""

import json
import os
import pickle
import time

import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MODELS_DIR = os.path.join(_PROJECT_ROOT, "models", "trt_pose")

HAND_POSE_JSON = os.path.join(_MODELS_DIR, "hand_pose.json")
TRT_MODEL_PATH = os.path.join(_MODELS_DIR, "hand_pose_resnet18_baseline_att_224x224_A_trt.pth")
PYTORCH_MODEL_PATH = os.path.join(_MODELS_DIR, "hand_pose_resnet18_baseline_att_224x224_A.pth")
SVM_MODEL_PATH = os.path.join(_MODELS_DIR, "svmmodel.sav")

INPUT_SIZE = 224

GESTURE_CLASSES = ["fist", "pan", "stop", "fine", "peace", "no_hand", "none"]


class _TRTPoseBackend:
    """trt_pose_hand backend: ResNet18 keypoints + SVM classifier."""

    def __init__(self):
        self._model = None
        self._parse_objects = None
        self._topology = None
        self._num_parts = 0
        self._num_links = 0
        self._svm = None
        self._mean = None
        self._std = None

    def load(self):
        """Load trt_pose model and SVM classifier. Returns True on success."""
        try:
            import torch
            import trt_pose.coco
            import trt_pose.models
            from torch2trt import TRTModule
            from trt_pose.parse_objects import ParseObjects
        except ImportError as e:
            print(f"[gestures] trt_pose not available: {e}")
            return False

        if not os.path.exists(HAND_POSE_JSON):
            print(f"[gestures] Missing hand_pose.json at {HAND_POSE_JSON}")
            return False

        # Load topology
        with open(HAND_POSE_JSON, "r") as f:
            hand_pose = json.load(f)

        self._topology = trt_pose.coco.coco_category_to_topology(hand_pose)
        self._num_parts = len(hand_pose["keypoints"])
        self._num_links = len(hand_pose["skeleton"])

        # Load TRT-optimized model (preferred) or convert from PyTorch weights
        if os.path.exists(TRT_MODEL_PATH):
            print("[gestures] Loading TRT hand pose model...")
            self._model = TRTModule()
            self._model.load_state_dict(torch.load(TRT_MODEL_PATH))
        elif os.path.exists(PYTORCH_MODEL_PATH):
            print("[gestures] TRT model not found, converting from PyTorch weights...")
            self._model = self._convert_to_trt(torch, trt_pose)
            if self._model is None:
                return False
        else:
            print(f"[gestures] No hand pose model found. Download weights to {PYTORCH_MODEL_PATH}")
            print("[gestures] Run: python scripts/setup_hand_pose.py")
            return False

        # Parse objects (keypoint extraction from heatmaps)
        self._parse_objects = ParseObjects(
            self._topology, cmap_threshold=0.15, link_threshold=0.15
        )

        # ImageNet normalization constants
        self._mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        self._std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

        # Load SVM gesture classifier
        if os.path.exists(SVM_MODEL_PATH):
            with open(SVM_MODEL_PATH, "rb") as f:
                self._svm = pickle.load(f)
            print("[gestures] SVM gesture classifier loaded.")
        else:
            print(f"[gestures] No SVM model at {SVM_MODEL_PATH}, using rule-based fallback.")

        print("[gestures] trt_pose hand model loaded.")
        return True

    def _convert_to_trt(self, torch, trt_pose):
        """Convert PyTorch hand pose model to TRT engine."""
        from torch2trt import torch2trt, TRTModule

        print("[gestures] Loading PyTorch model...")
        model = trt_pose.models.resnet18_baseline_att(
            self._num_parts, 2 * self._num_links
        ).cuda().eval()
        model.load_state_dict(torch.load(PYTORCH_MODEL_PATH))

        print("[gestures] Converting to TensorRT (this may take a few minutes)...")
        data = torch.zeros((1, 3, INPUT_SIZE, INPUT_SIZE)).cuda()
        model_trt = torch2trt(model, [data], fp16_mode=True, max_workspace_size=1 << 25)

        # Save for next time
        torch.save(model_trt.state_dict(), TRT_MODEL_PATH)
        print(f"[gestures] TRT model saved to {TRT_MODEL_PATH}")

        return model_trt

    def _preprocess(self, frame):
        """Preprocess frame for trt_pose: resize, normalize, NCHW tensor."""
        import torch
        import torchvision.transforms.functional as F
        from PIL import Image

        resized = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        tensor = F.to_tensor(img).cuda()
        tensor.sub_(self._mean[:, None, None]).div_(self._std[:, None, None])
        return tensor[None, ...]

    def detect_hands(self, frame):
        """Run hand keypoint detection. Returns list of hand dicts with keypoints."""
        import torch

        data = self._preprocess(frame)
        cmap, paf = self._model(data)
        cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
        counts, objects, peaks = self._parse_objects(cmap, paf)

        h, w = frame.shape[:2]
        hands = []

        count = int(counts[0])
        for i in range(count):
            obj = objects[0][i]
            keypoints = {}
            kp_list = []

            for j in range(self._num_parts):
                k = int(obj[j])
                if k >= 0:
                    peak = peaks[0][j][k]
                    x = float(peak[1]) * w
                    y = float(peak[0]) * h
                    keypoints[j] = (x, y)
                    kp_list.append((x, y))

            if len(keypoints) < 5:
                continue

            center_x = np.mean([p[0] for p in kp_list])
            center_y = np.mean([p[1] for p in kp_list])
            hands.append({
                "keypoints": keypoints,
                "center": (center_x, center_y),
                "num_keypoints": len(keypoints),
            })

        return hands

    def classify_gesture(self, keypoints):
        """Classify gesture from keypoints using SVM or rule-based fallback."""
        if not keypoints or len(keypoints) < 5:
            return "no_hand"

        if self._svm is not None:
            return self._classify_svm(keypoints)
        return _classify_rule_based(keypoints)

    def _classify_svm(self, keypoints):
        """SVM-based gesture classification using pairwise joint distances.

        The trt_pose_hand SVM expects 441 features: pairwise Euclidean distances
        between all 21 hand joints (21 x 21 matrix flattened).
        Classes: 1=fist, 2=pan, 3=stop, 4=fine, 5=peace, 6=no_hand.
        """
        import math

        # Build 21-joint list: (x, y) for each joint, (0, 0) if missing
        joints = []
        for j in range(21):
            if j in keypoints:
                joints.append(keypoints[j])
            else:
                joints.append((0, 0))

        # Compute pairwise distances (441 features)
        features = []
        for i in joints:
            for j in joints:
                dist = math.sqrt((i[0] - j[0]) ** 2 + (i[1] - j[1]) ** 2)
                features.append(dist)

        try:
            prediction = self._svm.predict([features])
            label = int(prediction[0])
            # Map SVM class labels to gesture names
            svm_labels = {1: "fist", 2: "pan", 3: "stop", 4: "fine", 5: "peace", 6: "no_hand"}
            return svm_labels.get(label, "none")
        except Exception as e:
            print(f"[gestures] SVM prediction error: {e}")
            return _classify_rule_based(keypoints)


class _MediaPipeBackend:
    """MediaPipe Hands fallback backend."""

    def __init__(self, max_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        self._max_hands = max_hands
        self._min_det = min_detection_confidence
        self._min_track = min_tracking_confidence
        self._hands = None

    def load(self):
        """Load MediaPipe Hands model."""
        try:
            import mediapipe as mp
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self._max_hands,
                min_detection_confidence=self._min_det,
                min_tracking_confidence=self._min_track,
            )
            print("[gestures] MediaPipe Hands loaded (fallback).")
            return True
        except ImportError:
            print("[gestures] mediapipe not installed. Run: pip install mediapipe")
            return False
        except Exception as e:
            print(f"[gestures] Failed to load MediaPipe: {e}")
            return False

    def detect_hands(self, frame):
        """Detect hand keypoints in frame. Returns list of hand dicts."""
        if self._hands is None:
            return []

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._hands.process(rgb)

        if not results.multi_hand_landmarks:
            return []

        hands = []
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints = {}
            kp_list = []
            for idx, lm in enumerate(hand_landmarks.landmark):
                x = lm.x * w
                y = lm.y * h
                keypoints[idx] = (x, y)
                kp_list.append((x, y))

            center_x = np.mean([p[0] for p in kp_list])
            center_y = np.mean([p[1] for p in kp_list])
            hands.append({
                "keypoints": keypoints,
                "center": (center_x, center_y),
                "num_keypoints": len(keypoints),
            })

        return hands

    def classify_gesture(self, keypoints):
        """Classify hand gesture from keypoints using rules."""
        return _classify_rule_based(keypoints)


def _classify_rule_based(keypoints):
    """Rule-based gesture classification from hand keypoints.

    Keypoint indices (both MediaPipe and trt_pose use the same 21-point layout):
    0: wrist
    1-4: thumb (CMC, MCP, IP, tip)
    5-8: index finger (MCP, PIP, DIP, tip)
    9-12: middle finger
    13-16: ring finger
    17-20: pinky
    """
    if not keypoints or len(keypoints) < 10:
        return "no_hand"

    def is_extended(tip_idx, pip_idx):
        if tip_idx not in keypoints or pip_idx not in keypoints:
            return False
        return keypoints[tip_idx][1] < keypoints[pip_idx][1]

    # Thumb: compare x position (left/right hand aware)
    thumb_ext = False
    if 4 in keypoints and 3 in keypoints and 0 in keypoints:
        wrist_x = keypoints[0][0]
        thumb_tip_dist = abs(keypoints[4][0] - wrist_x)
        thumb_ip_dist = abs(keypoints[3][0] - wrist_x)
        thumb_ext = thumb_tip_dist > thumb_ip_dist

    index_ext = is_extended(8, 6)
    middle_ext = is_extended(12, 10)
    ring_ext = is_extended(16, 14)
    pinky_ext = is_extended(20, 18)

    extended_count = sum([index_ext, middle_ext, ring_ext, pinky_ext])

    # "stop" — all fingers extended (open palm)
    if extended_count >= 4:
        return "stop"

    # "peace" — index + middle extended, others curled
    if index_ext and middle_ext and not ring_ext and not pinky_ext:
        return "peace"

    # "fist" — no fingers extended
    if extended_count == 0:
        return "fist"

    # "fine" (OK sign) — thumb tip near index tip, other fingers extended
    if 4 in keypoints and 8 in keypoints:
        thumb_tip = np.array(keypoints[4])
        index_tip = np.array(keypoints[8])
        dist = np.linalg.norm(thumb_tip - index_tip)
        if dist < 40 and (middle_ext or ring_ext):
            return "fine"

    # "pan" — index extended only (pointing)
    if index_ext and not middle_ext and not ring_ext and not pinky_ext:
        return "pan"

    return "none"


class GestureRecognizer:
    """Detect hand keypoints and classify gestures.

    Tries trt_pose_hand (TensorRT) first, falls back to MediaPipe.
    """

    def __init__(self, max_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5):
        self._max_hands = max_hands
        self._min_det = min_detection_confidence
        self._min_track = min_tracking_confidence
        self._backend = None
        self._backend_name = None

        # Motion tracking
        self._prev_hand_center = None
        self._prev_time = None

    def load_model(self):
        """Load gesture model. Tries trt_pose first, then MediaPipe."""
        if self._backend is not None:
            return True

        # Try trt_pose_hand first
        trt_backend = _TRTPoseBackend()
        if trt_backend.load():
            self._backend = trt_backend
            self._backend_name = "trt_pose"
            return True

        # Fall back to MediaPipe
        print("[gestures] Falling back to MediaPipe...")
        mp_backend = _MediaPipeBackend(
            max_hands=self._max_hands,
            min_detection_confidence=self._min_det,
            min_tracking_confidence=self._min_track,
        )
        if mp_backend.load():
            self._backend = mp_backend
            self._backend_name = "mediapipe"
            return True

        return False

    @property
    def backend_name(self):
        return self._backend_name

    def detect_hands(self, frame):
        """Detect hand keypoints in frame. Returns list of hand dicts."""
        if self._backend is None:
            return []
        return self._backend.detect_hands(frame)

    def classify_gesture(self, keypoints):
        """Classify hand gesture from keypoints. Returns gesture name string."""
        if self._backend is None:
            return "no_hand"
        return self._backend.classify_gesture(keypoints)

    def get_motion(self, hand_center):
        """Track hand motion between frames. Returns motion dict or None."""
        now = time.time()
        motion = None

        if self._prev_hand_center is not None and self._prev_time is not None:
            dt = now - self._prev_time
            if 0 < dt < 0.5:
                dx = hand_center[0] - self._prev_hand_center[0]
                dy = hand_center[1] - self._prev_hand_center[1]
                speed = np.sqrt(dx**2 + dy**2) / dt

                if speed > 50:
                    if abs(dx) > abs(dy):
                        direction = "right" if dx > 0 else "left"
                    else:
                        direction = "down" if dy > 0 else "up"

                    motion = {
                        "direction": direction,
                        "speed": speed,
                        "dx": dx,
                        "dy": dy,
                    }

        self._prev_hand_center = hand_center
        self._prev_time = now
        return motion

    def process_frame(self, frame):
        """Full pipeline: detect hand, classify gesture, track motion."""
        hands = self.detect_hands(frame)

        if not hands:
            self._prev_hand_center = None
            return {
                "gesture": "no_hand",
                "hand_position": None,
                "motion": None,
                "keypoints": {},
            }

        hand = max(hands, key=lambda h: h["num_keypoints"])
        gesture = self.classify_gesture(hand["keypoints"])
        motion = self.get_motion(hand["center"])

        return {
            "gesture": gesture,
            "hand_position": hand["center"],
            "motion": motion,
            "keypoints": hand["keypoints"],
        }


if __name__ == "__main__":
    gr = GestureRecognizer()
    if not gr.load_model():
        print("No gesture backend available.")
        print("  For TRT: run scripts/setup_hand_pose.py inside Docker")
        print("  For MediaPipe: pip install mediapipe")
    else:
        print(f"Gesture backend: {gr.backend_name}")
        print("Run scripts/test_gestures.py for live test.")
