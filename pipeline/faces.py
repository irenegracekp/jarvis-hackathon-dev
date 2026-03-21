"""Face detection (YuNet) + recognition (SFace) via OpenCV DNN."""

import json
import os
import threading

import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_PROJECT_ROOT, "models", "opencv")

DEFAULT_DET_MODEL = os.path.join(MODEL_DIR, "face_detection_yunet_2023mar.onnx")
DEFAULT_REC_MODEL = os.path.join(MODEL_DIR, "face_recognition_sface_2021dec.onnx")
DEFAULT_EMBEDDINGS = os.path.join(_PROJECT_ROOT, "data", "known_faces", "embeddings.json")

COSINE_THRESHOLD = 0.30  # Lowered from 0.363 default for webcam robustness


class FaceRecognizer:
    def __init__(self, det_model_path=DEFAULT_DET_MODEL, rec_model_path=DEFAULT_REC_MODEL,
                 embeddings_path=DEFAULT_EMBEDDINGS, cosine_threshold=COSINE_THRESHOLD):
        self.det_model_path = det_model_path
        self.rec_model_path = rec_model_path
        self.embeddings_path = embeddings_path
        self.cosine_threshold = cosine_threshold

        self._detector = None
        self._recognizer = None
        self._known = {}  # face_id -> embedding (list of floats)
        self._unknown_counter = 0
        self._lock = threading.Lock()
        self._last_input_size = (0, 0)

    def load_models(self):
        if self._detector is not None:
            return True
        try:
            if not os.path.exists(self.det_model_path):
                print(f"[faces] Detection model not found: {self.det_model_path}")
                return False
            if not os.path.exists(self.rec_model_path):
                print(f"[faces] Recognition model not found: {self.rec_model_path}")
                return False

            self._detector = cv2.FaceDetectorYN.create(
                self.det_model_path, "", (320, 320), 0.7, 0.3, 5000
            )
            self._recognizer = cv2.FaceRecognizerSF.create(self.rec_model_path, "")
            print("[faces] YuNet + SFace models loaded.")
            self.load_known_embeddings()
            return True
        except Exception as e:
            print(f"[faces] Failed to load models: {e}")
            return False

    def load_known_embeddings(self):
        if os.path.exists(self.embeddings_path):
            try:
                with open(self.embeddings_path, "r") as f:
                    self._known = json.load(f)
                print(f"[faces] Loaded {len(self._known)} known face embeddings.")
            except (json.JSONDecodeError, IOError) as e:
                print(f"[faces] Failed to load embeddings: {e}")
                self._known = {}
        else:
            self._known = {}

    def save_known_embeddings(self):
        os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
        with open(self.embeddings_path, "w") as f:
            json.dump(self._known, f, indent=2)

    def detect_and_identify(self, frame):
        """Detect faces and identify them. Returns list of dicts with face_id, bbox, confidence, is_known."""
        with self._lock:
            if not self.load_models():
                return []

            h, w = frame.shape[:2]
            if (w, h) != self._last_input_size:
                self._detector.setInputSize((w, h))
                self._last_input_size = (w, h)

            _, raw_faces = self._detector.detect(frame)
            if raw_faces is None:
                return []

            results = []
            for face_raw in raw_faces:
                bbox = face_raw[:4].astype(int).tolist()  # x, y, w, h
                confidence = float(face_raw[14]) if face_raw.shape[0] > 14 else float(face_raw[-1])

                # Align and extract embedding
                aligned = self._recognizer.alignCrop(frame, face_raw)
                embedding = self._recognizer.feature(aligned)
                embedding_flat = embedding.flatten().tolist()

                # Match against known faces
                face_id, is_known = self._match_embedding(embedding)

                results.append({
                    "face_id": face_id,  # None if unknown
                    "bbox": bbox,
                    "confidence": confidence,
                    "is_known": is_known,
                    "embedding": embedding_flat,
                })

            return results

    def _match_embedding(self, embedding):
        """Match embedding against known faces. Returns (face_id, is_known).
        Returns (None, False) for unknown faces — caller handles ID assignment."""
        best_id = None
        best_score = -1.0

        for fid, known_emb in self._known.items():
            known_arr = np.array(known_emb, dtype=np.float32).reshape(1, -1)
            score = self._recognizer.match(embedding, known_arr, cv2.FaceRecognizerSF_FR_COSINE)
            if score > best_score:
                best_score = score
                best_id = fid

        if best_score >= self.cosine_threshold and best_id is not None:
            return best_id, True

        return None, False

    def next_unknown_id(self):
        """Generate next unique unknown face ID."""
        self._unknown_counter += 1
        return f"unknown_{self._unknown_counter}"

    def enroll(self, face_id, embedding):
        """Add a new known face embedding."""
        with self._lock:
            if isinstance(embedding, np.ndarray):
                embedding = embedding.flatten().tolist()
            self._known[face_id] = embedding
            self.save_known_embeddings()
            print(f"[faces] Enrolled face: {face_id}")

    def get_closest_to_center(self, faces, frame_width):
        """Return the face closest to horizontal center of the frame."""
        if not faces:
            return None
        center_x = frame_width / 2
        best = None
        best_dist = float("inf")
        for face in faces:
            bx, by, bw, bh = face["bbox"]
            face_cx = bx + bw / 2
            dist = abs(face_cx - center_x)
            if dist < best_dist:
                best_dist = dist
                best = face
        return best


if __name__ == "__main__":
    fr = FaceRecognizer()
    if fr.load_models():
        # Test with test_frame.jpg if available
        test_path = os.path.join(_PROJECT_ROOT, "test_frame.jpg")
        if os.path.exists(test_path):
            frame = cv2.imread(test_path)
            faces = fr.detect_and_identify(frame)
            print(f"Detected {len(faces)} faces:")
            for f in faces:
                print(f"  {f['face_id']}: bbox={f['bbox']}, conf={f['confidence']:.2f}, known={f['is_known']}")
        else:
            print("No test_frame.jpg found, but models loaded OK.")
    else:
        print("Failed to load models.")
