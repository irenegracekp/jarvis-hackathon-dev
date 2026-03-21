"""Agora RTC client -- bridges local mic/speaker to an Agora channel.

Captures audio from the Hollyland mic via sounddevice, pushes PCM to
the Agora channel. Receives TTS audio from the Agora Conversational AI
agent and plays it on the Bluetooth speaker via paplay.
"""

import os
import subprocess
import threading
import time

import numpy as np

CAPTURE_RATE = 48000   # Hollyland mic native rate
AGORA_RATE = 16000     # Agora expects 16kHz
CHANNELS = 1
BLOCK_DURATION_MS = 20  # 20ms blocks for Agora (must align to 1ms boundaries)


def _find_hollyland_device():
    """Find Hollyland wireless mic device index (same logic as listen.py)."""
    import sounddevice as sd
    devices = sd.query_devices()
    for i, d in enumerate(devices):
        name = d["name"].lower()
        if d["max_input_channels"] > 0 and (
            "wireless" in name or "hollyland" in name or "shenzhen" in name
        ):
            print(f"[agora_rtc] Found Hollyland mic: {d['name']} (index {i})")
            return i
    default = sd.default.device[0]
    if default is not None and default >= 0:
        print(f"[agora_rtc] Using default input: {devices[default]['name']}")
        return default
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            print(f"[agora_rtc] Using first input: {d['name']} (index {i})")
            return i
    raise RuntimeError("No audio input device found")


class AgoraRTCClient:
    def __init__(self, app_id, on_tts_state_change=None):
        """
        Args:
            app_id: Agora App ID
            on_tts_state_change: optional callback(speaking: bool)
        """
        self.app_id = app_id
        self.on_tts_state_change = on_tts_state_change
        self._running = False
        self._connected = False
        self._mic_thread = None
        self._connection = None
        self._service = None
        self._paplay_proc = None
        self._last_audio_time = 0
        self._tts_speaking = False
        self._tts_monitor_thread = None

    @property
    def is_connected(self):
        return self._connected

    @property
    def tts_speaking(self):
        return self._tts_speaking

    def join(self, channel_name, token="", user_id="12345"):
        """Join an Agora RTC channel and start audio I/O."""
        from agora.rtc.agora_service import AgoraService, AgoraServiceConfig
        from agora.rtc.agora_base import (
            RTCConnConfig, RtcConnectionPublishConfig,
            AudioSubscriptionOptions, AudioScenarioType,
        )
        from agora.rtc.rtc_connection_observer import IRTCConnectionObserver
        from agora.rtc.audio_frame_observer import IAudioFrameObserver

        # Initialize Agora service
        config = AgoraServiceConfig()
        config.appid = self.app_id
        config.enable_audio_processor = 1
        config.enable_audio_device = 0
        config.enable_video = 0
        config.log_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs", "agora.log"
        )

        self._service = AgoraService()
        self._service.initialize(config)

        # Connection config: subscribe to audio from the AI agent
        conn_config = RTCConnConfig()
        conn_config.auto_subscribe_audio = 1
        conn_config.auto_subscribe_video = 0
        conn_config.audio_subs_options = AudioSubscriptionOptions(
            pcm_data_only=1,
            bytes_per_sample=2,
            number_of_channels=1,
            sample_rate_hz=AGORA_RATE,
        )

        # Publish config: send mic audio as PCM
        pub_config = RtcConnectionPublishConfig()
        pub_config.is_publish_audio = True
        pub_config.is_publish_video = False

        self._connection = self._service.create_rtc_connection(conn_config, pub_config)

        # Connection observer
        client = self

        class ConnObserver(IRTCConnectionObserver):
            def on_connected(self, agora_rtc_conn, conn_info, reason):
                print(f"[agora_rtc] Connected to channel (reason={reason})")
                client._connected = True
                agora_rtc_conn.publish_audio()

            def on_disconnected(self, agora_rtc_conn, conn_info, reason):
                print(f"[agora_rtc] Disconnected (reason={reason})")
                client._connected = False

            def on_user_joined(self, agora_rtc_conn, user_id):
                print(f"[agora_rtc] User joined: {user_id}")

            def on_user_left(self, agora_rtc_conn, user_id, reason):
                print(f"[agora_rtc] User left: {user_id}")

        # Audio frame observer for receiving TTS audio
        class AudioObserver(IAudioFrameObserver):
            def on_playback_audio_frame_before_mixing(self, agora_local_user, channelId, uid, frame, vad_result_state, vad_result_bytearray):
                client._handle_tts_audio(frame)
                return 1

        self._connection.register_observer(ConnObserver())
        self._connection.local_user.register_audio_frame_observer(AudioObserver())

        # Subscribe to all remote audio
        self._connection.local_user.subscribe_all_audio()

        # Connect
        self._running = True
        ret = self._connection.connect(token, channel_name, user_id)
        print(f"[agora_rtc] Connecting to channel '{channel_name}' as uid={user_id} (ret={ret})")

        # Start mic capture thread
        self._mic_thread = threading.Thread(target=self._mic_capture_loop, daemon=True)
        self._mic_thread.start()

        # Start TTS speaking monitor
        self._tts_monitor_thread = threading.Thread(target=self._tts_monitor_loop, daemon=True)
        self._tts_monitor_thread.start()

    def leave(self):
        """Leave the channel and clean up."""
        self._running = False
        if self._connection:
            self._connection.disconnect()
        if self._paplay_proc:
            try:
                self._paplay_proc.stdin.close()
                self._paplay_proc.wait(timeout=2)
            except Exception:
                pass
        if self._mic_thread:
            self._mic_thread.join(timeout=3)
        if self._connection:
            self._connection.release()
            self._connection = None
        if self._service:
            self._service.release()
            self._service = None
        print("[agora_rtc] Left channel.")

    def _mic_capture_loop(self):
        """Capture mic audio and push to Agora channel."""
        import sounddevice as sd
        import scipy.signal

        device_idx = _find_hollyland_device()
        block_size = int(CAPTURE_RATE * BLOCK_DURATION_MS / 1000)  # samples per block

        print("[agora_rtc] Starting mic capture...")

        def callback(indata, frames, time_info, status):
            if status:
                pass  # ignore overflow warnings
            if not self._connected or not self._running:
                return

            # Convert float32 to 16kHz int16 PCM
            chunk = indata[:, 0].copy()  # mono
            # Resample 48kHz -> 16kHz
            num_samples = int(len(chunk) * AGORA_RATE / CAPTURE_RATE)
            resampled = scipy.signal.resample(chunk, num_samples).astype(np.float32)
            # Convert to int16 PCM bytes
            pcm_int16 = (resampled * 32767).astype(np.int16)
            pcm_bytes = bytearray(pcm_int16.tobytes())

            try:
                self._connection.push_audio_pcm_data(
                    pcm_bytes,
                    sample_rate=AGORA_RATE,
                    channels=CHANNELS,
                )
            except Exception as e:
                print(f"[agora_rtc] Push audio error: {e}")

        try:
            with sd.InputStream(
                samplerate=CAPTURE_RATE,
                channels=CHANNELS,
                dtype="float32",
                blocksize=block_size,
                device=device_idx,
                callback=callback,
            ):
                print("[agora_rtc] Mic capture running.")
                while self._running:
                    time.sleep(0.1)
        except Exception as e:
            print(f"[agora_rtc] Mic capture error: {e}")

    def _handle_tts_audio(self, frame):
        """Receive TTS audio from Agora agent and play via paplay."""
        if frame.buffer is None or len(frame.buffer) == 0:
            return

        self._last_audio_time = time.time()
        if not self._tts_speaking:
            self._tts_speaking = True
            if self.on_tts_state_change:
                self.on_tts_state_change(True)

        # Start paplay process if not running
        if self._paplay_proc is None or self._paplay_proc.poll() is not None:
            self._paplay_proc = subprocess.Popen(
                ["paplay", "--raw", f"--rate={AGORA_RATE}", "--format=s16le", "--channels=1"],
                stdin=subprocess.PIPE,
            )

        try:
            if isinstance(frame.buffer, bytearray):
                self._paplay_proc.stdin.write(bytes(frame.buffer))
            else:
                self._paplay_proc.stdin.write(frame.buffer)
            self._paplay_proc.stdin.flush()
        except (BrokenPipeError, OSError):
            self._paplay_proc = None

    def _tts_monitor_loop(self):
        """Monitor TTS state — mark as not speaking after silence."""
        while self._running:
            if self._tts_speaking and (time.time() - self._last_audio_time > 0.5):
                self._tts_speaking = False
                if self.on_tts_state_change:
                    self.on_tts_state_change(False)
            time.sleep(0.1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Agora RTC Client Test")
    parser.add_argument("--app-id", required=True, help="Agora App ID")
    parser.add_argument("--channel", default="jarvis-test", help="Channel name")
    parser.add_argument("--token", default="", help="RTC token (empty for testing)")
    parser.add_argument("--uid", default="12345", help="User ID")
    args = parser.parse_args()

    def on_tts(speaking):
        print(f"[test] TTS speaking: {speaking}")

    client = AgoraRTCClient(app_id=args.app_id, on_tts_state_change=on_tts)
    client.join(args.channel, token=args.token, user_id=args.uid)

    try:
        print("[test] Running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        client.leave()
