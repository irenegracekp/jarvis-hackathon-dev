"""Hollyland mic -> Whisper ASR -> transcribed text

Supports two backends:
  - "local" : faster-whisper running locally (default)
  - "cloud" : OpenAI Whisper API — requires OPENAI_API_KEY env var
"""

import io
import os
import queue
import threading
import time
import numpy as np

WHISPER_RATE = 16000  # Whisper expects 16kHz
CAPTURE_RATE = 48000  # Hollyland mic native rate
CHANNELS = 1
BLOCK_DURATION = 0.5  # seconds per audio block
ENERGY_THRESHOLD = 0.01  # RMS threshold for voice activity
SILENCE_TIMEOUT = 0.8  # seconds of silence before finalizing utterance
MAX_UTTERANCE_SEC = 15  # max seconds per utterance


def _find_hollyland_device():
    """Find Hollyland wireless mic device index."""
    import sounddevice as sd
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        name = d["name"].lower()
        if d["max_input_channels"] > 0 and (
            "wireless" in name or "hollyland" in name or "shenzhen" in name
        ):
            print(f"[listen] Found Hollyland mic: {d['name']} (index {i})")
            return i
    # Fall back to default input
    default = sd.default.device[0]
    if default is not None and default >= 0:
        print(f"[listen] Hollyland not found, using default input: {devices[default]['name']}")
        return default
    # Last resort: first input device
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            print(f"[listen] Using first available input: {d['name']} (index {i})")
            return i
    raise RuntimeError("No audio input device found")


class ListenPipeline:
    def __init__(self, model_size="base", device="auto", compute_type="auto",
                 energy_threshold=ENERGY_THRESHOLD, asr_backend="local"):
        self.model_size = model_size
        self.asr_backend = asr_backend
        # CTranslate2 pip wheel may lack CUDA on aarch64; auto-detect
        if device == "auto":
            try:
                import ctranslate2
                if "cuda" in ctranslate2.get_supported_compute_types("cuda"):
                    device, compute_type = "cuda", "float16"
                else:
                    device, compute_type = "cpu", "int8"
            except Exception:
                device, compute_type = "cpu", "int8"
            print(f"[listen] ASR device: {device} ({compute_type})")
        self.asr_device = device
        self.compute_type = compute_type
        self.energy_threshold = energy_threshold

        self.text_queue = queue.Queue()  # output: (text, timestamp)
        self._running = False
        self._thread = None
        self._model = None
        self._openai_client = None

    def _load_model(self):
        if self.asr_backend == "cloud":
            if self._openai_client is None:
                try:
                    from openai import OpenAI
                    api_key = os.environ.get("OPENAI_API_KEY")
                    if not api_key:
                        print("[listen] OPENAI_API_KEY not set, falling back to local ASR.")
                        self.asr_backend = "local"
                    else:
                        self._openai_client = OpenAI(api_key=api_key)
                        print("[listen] Using OpenAI Whisper API for ASR.")
                        return
                except ImportError:
                    print("[listen] openai package not installed, falling back to local ASR.")
                    self.asr_backend = "local"
            else:
                return

        if self._model is None:
            from faster_whisper import WhisperModel
            print(f"[listen] Loading faster-whisper '{self.model_size}' on {self.asr_device}...")
            self._model = WhisperModel(
                self.model_size, device=self.asr_device, compute_type=self.compute_type
            )
            print("[listen] Whisper model loaded.")

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _resample(self, audio_48k: np.ndarray) -> np.ndarray:
        """Resample from 48kHz to 16kHz for Whisper."""
        import scipy.signal
        num_samples = int(len(audio_48k) * WHISPER_RATE / CAPTURE_RATE)
        return scipy.signal.resample(audio_48k, num_samples).astype(np.float32)

    def _run(self):
        import sounddevice as sd

        self._load_model()
        device_idx = _find_hollyland_device()
        block_size = int(CAPTURE_RATE * BLOCK_DURATION)

        audio_buffer = []
        silence_start = None
        is_speaking = False

        def callback(indata, frames, time_info, status):
            nonlocal audio_buffer, silence_start, is_speaking
            if status:
                print(f"[listen] sounddevice status: {status}")
            chunk = indata[:, 0].copy()
            rms = np.sqrt(np.mean(chunk ** 2))

            if rms > self.energy_threshold:
                audio_buffer.append(chunk)
                silence_start = None
                is_speaking = True
            elif is_speaking:
                audio_buffer.append(chunk)
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > SILENCE_TIMEOUT:
                    # Utterance complete — resample and send for transcription
                    full_audio = np.concatenate(audio_buffer)
                    max_samples = int(MAX_UTTERANCE_SEC * CAPTURE_RATE)
                    if len(full_audio) > max_samples:
                        full_audio = full_audio[:max_samples]
                    resampled = self._resample(full_audio)
                    self._transcribe(resampled)
                    audio_buffer = []
                    silence_start = None
                    is_speaking = False

        try:
            with sd.InputStream(
                samplerate=CAPTURE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=block_size,
                device=device_idx,
                callback=callback,
            ):
                print("[listen] Listening...")
                while self._running:
                    time.sleep(0.1)
        except Exception as e:
            print(f"[listen] Audio stream error: {e}")

    def _transcribe(self, audio: np.ndarray):
        if self.asr_backend == "cloud" and self._openai_client is not None:
            self._transcribe_cloud(audio)
        else:
            self._transcribe_local(audio)

    def _transcribe_local(self, audio: np.ndarray):
        try:
            segments, info = self._model.transcribe(
                audio, beam_size=3, language="en", vad_filter=True
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            if text and len(text) > 1:
                print(f"[listen] Heard: {text}")
                self.text_queue.put((text, time.time()))
        except Exception as e:
            print(f"[listen] Transcription error: {e}")

    def _transcribe_cloud(self, audio: np.ndarray):
        try:
            import wave

            # Convert float32 audio to WAV bytes for the API
            audio_int16 = (audio * 32767).astype(np.int16)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(WHISPER_RATE)
                wf.writeframes(audio_int16.tobytes())
            buf.seek(0)
            buf.name = "audio.wav"

            response = self._openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=buf,
                language="en",
            )
            text = response.text.strip()
            if text and len(text) > 1:
                print(f"[listen] Heard (cloud): {text}")
                self.text_queue.put((text, time.time()))
        except Exception as e:
            print(f"[listen] Cloud transcription error: {e}")


if __name__ == "__main__":
    lp = ListenPipeline()
    lp.start()
    try:
        while True:
            try:
                text, ts = lp.text_queue.get(timeout=1)
                print(f">> [{ts:.1f}] {text}")
            except queue.Empty:
                pass
    except KeyboardInterrupt:
        lp.stop()
