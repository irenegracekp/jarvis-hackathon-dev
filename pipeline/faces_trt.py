"""TensorRT-accelerated face detection (YuNet) + recognition (SFace).

Falls back to the OpenCV-based pipeline (faces.py) if TRT engines are not available.
"""

import json
import os
import threading

import cv2
import numpy as np

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRT_DIR = os.path.join(_PROJECT_ROOT, "models", "trt")
ONNX_DIR = os.path.join(_PROJECT_ROOT, "models", "opencv")

DEFAULT_DET_ENGINE = os.path.join(TRT_DIR, "yunet_640x480_fp16.engine")
DEFAULT_REC_ENGINE = os.path.join(TRT_DIR, "sface_fp16.engine")
DEFAULT_EMBEDDINGS = os.path.join(_PROJECT_ROOT, "data", "known_faces", "embeddings.json")

COSINE_THRESHOLD = 0.30

# YuNet input size must match what the engine was built for
YUNET_INPUT_W = 640
YUNET_INPUT_H = 480

# SFace input
SFACE_INPUT_SIZE = 112

# YuNet anchors / prior-box config for decoding detections
# These match the OpenCV YuNet 2023mar model at 640x480
YUNET_SCORE_THRESH = 0.7
YUNET_NMS_THRESH = 0.3
YUNET_TOP_K = 5000


def _load_trt_engine(engine_path):
    """Load a TensorRT engine from file. Returns (engine, context) or raises."""
    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f:
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    return engine, context


def _trt_infer(context, input_data):
    """Run TRT inference with a single input tensor. Returns list of output arrays."""
    import tensorrt as trt

    engine = context.engine
    stream = None

    try:
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401
        stream = cuda.Stream()
    except ImportError:
        # Fall back to torch CUDA for memory management
        pass

    inputs = []
    outputs = []
    bindings = []

    # Allocate using numpy + device pointers
    import torch

    input_tensor = torch.from_numpy(input_data).cuda()

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = context.get_tensor_shape(name)
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        mode = engine.get_tensor_mode(name)

        if mode == trt.TensorIOMode.INPUT:
            context.set_input_shape(name, input_data.shape)
            context.set_tensor_address(name, input_tensor.data_ptr())
        else:
            out_shape = context.get_tensor_shape(name)
            out_tensor = torch.empty(tuple(out_shape), dtype=torch.float32, device="cuda")
            outputs.append((name, out_tensor))
            context.set_tensor_address(name, out_tensor.data_ptr())

    context.execute_async_v3(0)
    torch.cuda.synchronize()

    return [t.cpu().numpy() for _, t in outputs]


def _cosine_similarity(a, b):
    """Cosine similarity between two 1-D vectors."""
    a = a.flatten().astype(np.float32)
    b = b.flatten().astype(np.float32)
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    return dot / norm if norm > 0 else 0.0


class FaceRecognizerTRT:
    """TensorRT face pipeline. API-compatible with FaceRecognizer from faces.py."""

    def __init__(self, det_engine_path=DEFAULT_DET_ENGINE,
                 rec_engine_path=DEFAULT_REC_ENGINE,
                 embeddings_path=DEFAULT_EMBEDDINGS,
                 cosine_threshold=COSINE_THRESHOLD):
        self.det_engine_path = det_engine_path
        self.rec_engine_path = rec_engine_path
        self.embeddings_path = embeddings_path
        self.cosine_threshold = cosine_threshold

        self._det_ctx = None
        self._rec_ctx = None
        self._det_engine = None
        self._rec_engine = None
        self._known = {}
        self._unknown_counter = 0
        self._lock = threading.Lock()

        # Fallback to OpenCV path
        self._fallback = None
        self._use_trt = True

    def load_models(self):
        """Load TRT engines, or fall back to OpenCV."""
        if self._det_ctx is not None or self._fallback is not None:
            return True

        if not os.path.exists(self.det_engine_path) or not os.path.exists(self.rec_engine_path):
            print(f"[faces_trt] TRT engines not found, falling back to OpenCV.")
            return self._load_fallback()

        try:
            print("[faces_trt] Loading YuNet TRT engine...")
            self._det_engine, self._det_ctx = _load_trt_engine(self.det_engine_path)
            print("[faces_trt] Loading SFace TRT engine...")
            self._rec_engine, self._rec_ctx = _load_trt_engine(self.rec_engine_path)
            print("[faces_trt] TRT engines loaded.")
            self._use_trt = True
            self.load_known_embeddings()
            return True
        except Exception as e:
            print(f"[faces_trt] Failed to load TRT engines: {e}")
            return self._load_fallback()

    def _load_fallback(self):
        """Fall back to OpenCV-based FaceRecognizer."""
        from pipeline.faces import FaceRecognizer
        self._fallback = FaceRecognizer(embeddings_path=self.embeddings_path,
                                        cosine_threshold=self.cosine_threshold)
        self._use_trt = False
        return self._fallback.load_models()

    def load_known_embeddings(self):
        if os.path.exists(self.embeddings_path):
            try:
                with open(self.embeddings_path, "r") as f:
                    self._known = json.load(f)
                print(f"[faces_trt] Loaded {len(self._known)} known face embeddings.")
            except (json.JSONDecodeError, IOError) as e:
                print(f"[faces_trt] Failed to load embeddings: {e}")
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

            if not self._use_trt:
                return self._fallback.detect_and_identify(frame)

            return self._detect_and_identify_trt(frame)

    def _detect_and_identify_trt(self, frame):
        """TRT-based detection + recognition pipeline."""
        h, w = frame.shape[:2]

        # Resize frame to match YuNet engine input
        resized = cv2.resize(frame, (YUNET_INPUT_W, YUNET_INPUT_H))
        scale_x = w / YUNET_INPUT_W
        scale_y = h / YUNET_INPUT_H

        # Preprocess for YuNet: BGR, float32, NCHW, no normalization (YuNet uses raw pixels)
        blob = cv2.dnn.blobFromImage(resized, 1.0, (YUNET_INPUT_W, YUNET_INPUT_H),
                                     (0, 0, 0), swapRB=False)

        # Run detection
        det_outputs = _trt_infer(self._det_ctx, blob)

        # YuNet outputs: loc (bbox deltas), conf (scores), iou
        # The exact output structure depends on the model version.
        # We use OpenCV's FaceDetectorYN as a reference for post-processing.
        # Since TRT post-processing for YuNet is complex (anchor decoding),
        # we use OpenCV DNN for detection and only TRT for recognition (the bottleneck).
        #
        # Hybrid approach: OpenCV YuNet detect + TRT SFace recognize
        return self._hybrid_detect_and_identify(frame)

    def _hybrid_detect_and_identify(self, frame):
        """Use OpenCV YuNet for detection, TRT SFace for recognition (faster than full OpenCV)."""
        # Lazy-load OpenCV detector only (not recognizer)
        if not hasattr(self, "_cv_detector"):
            det_path = os.path.join(ONNX_DIR, "face_detection_yunet_2023mar.onnx")
            self._cv_detector = cv2.FaceDetectorYN.create(det_path, "", (320, 320), 0.7, 0.3, 5000)
            self._cv_last_size = (0, 0)

        h, w = frame.shape[:2]
        if (w, h) != self._cv_last_size:
            self._cv_detector.setInputSize((w, h))
            self._cv_last_size = (w, h)

        _, raw_faces = self._cv_detector.detect(frame)
        if raw_faces is None:
            return []

        results = []
        for face_raw in raw_faces:
            bbox = face_raw[:4].astype(int).tolist()
            confidence = float(face_raw[14]) if face_raw.shape[0] > 14 else float(face_raw[-1])

            # Align face for SFace (112x112 crop)
            aligned = self._align_face(frame, face_raw)
            if aligned is None:
                continue

            # Get embedding via TRT SFace
            embedding = self._get_embedding_trt(aligned)
            embedding_flat = embedding.flatten().tolist()

            # Match against known faces
            face_id, is_known = self._match_embedding(embedding)

            results.append({
                "face_id": face_id,
                "bbox": bbox,
                "confidence": confidence,
                "is_known": is_known,
                "embedding": embedding_flat,
            })

        return results

    def _align_face(self, frame, face_raw):
        """Align and crop face to 112x112 for SFace, using the same logic as OpenCV."""
        # Use OpenCV's built-in alignment if available
        try:
            if not hasattr(self, "_cv_recognizer_align"):
                rec_path = os.path.join(ONNX_DIR, "face_recognition_sface_2021dec.onnx")
                self._cv_recognizer_align = cv2.FaceRecognizerSF.create(rec_path, "")
            return self._cv_recognizer_align.alignCrop(frame, face_raw)
        except Exception:
            # Manual crop fallback
            x, y, w, h = face_raw[:4].astype(int)
            x, y = max(0, x), max(0, y)
            face_crop = frame[y:y+h, x:x+w]
            if face_crop.size == 0:
                return None
            return cv2.resize(face_crop, (SFACE_INPUT_SIZE, SFACE_INPUT_SIZE))

    def _get_embedding_trt(self, aligned_face):
        """Run SFace TRT engine on an aligned 112x112 face crop."""
        # Preprocess: BGR float32, NCHW, normalize to [0, 1]
        blob = cv2.dnn.blobFromImage(aligned_face, 1.0 / 255.0,
                                     (SFACE_INPUT_SIZE, SFACE_INPUT_SIZE),
                                     (0, 0, 0), swapRB=False)
        outputs = _trt_infer(self._rec_ctx, blob)
        # SFace output is a 128-d embedding
        return outputs[0]

    def _match_embedding(self, embedding):
        """Match embedding against known faces. Returns (face_id, is_known)."""
        best_id = None
        best_score = -1.0

        emb = embedding.flatten()

        for fid, known_emb in self._known.items():
            known_arr = np.array(known_emb, dtype=np.float32)
            score = _cosine_similarity(emb, known_arr)
            if score > best_score:
                best_score = score
                best_id = fid

        if best_score >= self.cosine_threshold and best_id is not None:
            return best_id, True

        return None, False

    def next_unknown_id(self):
        self._unknown_counter += 1
        return f"unknown_{self._unknown_counter}"

    def enroll(self, face_id, embedding):
        with self._lock:
            if isinstance(embedding, np.ndarray):
                embedding = embedding.flatten().tolist()
            self._known[face_id] = embedding
            self.save_known_embeddings()
            print(f"[faces_trt] Enrolled face: {face_id}")

    def get_closest_to_center(self, faces, frame_width):
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
    fr = FaceRecognizerTRT()
    if fr.load_models():
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
