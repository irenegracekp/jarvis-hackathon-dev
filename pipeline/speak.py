"""TTS output -- text to speech via Piper"""

import io
import os
import subprocess
import threading
import queue
import wave
import time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPER_VOICE = os.path.join(_PROJECT_ROOT, "models", "piper", "en_US-amy-medium.onnx")


class SpeakPipeline:
    def __init__(self, voice_path=None, output_device=None):
        self.voice_path = voice_path or PIPER_VOICE
        self.output_device = output_device

        self._queue = queue.Queue()
        self._running = False
        self._thread = None

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

    def say(self, text):
        """Queue text for TTS. Non-blocking."""
        if text and text.strip():
            self._queue.put(text.strip())

    def say_blocking(self, text):
        """Synthesize and play immediately (blocking)."""
        if text and text.strip():
            self._synthesize_and_play(text.strip())

    def _run(self):
        print("[speak] TTS worker started.")
        while self._running:
            try:
                text = self._queue.get(timeout=0.5)
                self._synthesize_and_play(text)
            except queue.Empty:
                continue

    def _synthesize_and_play(self, text):
        """Use piper CLI to synthesize and aplay to play."""
        if not os.path.exists(self.voice_path):
            print(f"[speak] Voice model not found: {self.voice_path}")
            print(f"[speak] Would say: {text}")
            return

        try:
            # Piper outputs raw WAV to stdout
            # Pipe: echo text | piper --model voice.onnx --output_raw | aplay -r 22050 -f S16_LE
            cmd = (
                f'echo "{self._escape_text(text)}" | '
                f'piper --model {self.voice_path} --output_raw | '
                f'aplay -r 22050 -f S16_LE -t raw -c 1 -q'
            )
            result = subprocess.run(
                cmd, shell=True, capture_output=True, timeout=30
            )
            if result.returncode != 0:
                stderr = result.stderr.decode(errors="replace")
                if stderr.strip():
                    print(f"[speak] TTS error: {stderr[:200]}")
        except subprocess.TimeoutExpired:
            print(f"[speak] TTS timed out for: {text[:50]}...")
        except Exception as e:
            print(f"[speak] TTS error: {e}")

    @staticmethod
    def _escape_text(text):
        """Escape text for shell echo."""
        return text.replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")


class DummySpeakPipeline:
    """No-TTS fallback: just prints."""

    def start(self):
        print("[speak] Dummy TTS (print-only mode).")

    def stop(self):
        pass

    def say(self, text):
        if text:
            print(f"[SPEAK] {text}")

    def say_blocking(self, text):
        if text:
            print(f"[SPEAK] {text}")


if __name__ == "__main__":
    sp = SpeakPipeline()
    sp.start()
    sp.say("Hello! I am The Witness. I see everything.")
    time.sleep(5)
    sp.stop()
