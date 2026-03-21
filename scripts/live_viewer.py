#!/usr/bin/env python3
"""Standalone live viewer using tkinter — no browser needed.
Connects to the dashboard MJPEG stream and displays it in a window.

Usage: python3 scripts/live_viewer.py [--host localhost] [--port 8080]
"""

import argparse
import io
import json
import threading
import time
import tkinter as tk
from tkinter import font as tkfont
from urllib.request import urlopen, Request
from urllib.error import URLError
from PIL import Image, ImageTk


class LiveViewer:
    def __init__(self, host="localhost", port=8080):
        self.base_url = f"http://{host}:{port}"
        self.running = True

        # Main window
        self.root = tk.Tk()
        self.root.title("The Witness - Live Dashboard")
        self.root.configure(bg="#1a1a2e")
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

        # Video label
        self.video_label = tk.Label(self.root, bg="#1a1a2e")
        self.video_label.pack(padx=10, pady=(10, 5))

        # Status frame
        status_frame = tk.Frame(self.root, bg="#16213e", padx=10, pady=8)
        status_frame.pack(fill=tk.X, padx=10, pady=5)

        mono = tkfont.Font(family="monospace", size=11)
        self.mode_label = tk.Label(status_frame, text="MODE: --", fg="#00d2ff",
                                   bg="#16213e", font=mono, anchor="w")
        self.mode_label.pack(fill=tk.X)
        self.face_label = tk.Label(status_frame, text="Faces: --", fg="#00ff00",
                                   bg="#16213e", font=mono, anchor="w")
        self.face_label.pack(fill=tk.X)
        self.speaker_label = tk.Label(status_frame, text="Speaker: --", fg="#ffffff",
                                      bg="#16213e", font=mono, anchor="w")
        self.speaker_label.pack(fill=tk.X)

        # Events text
        events_frame = tk.Frame(self.root, bg="#16213e", padx=10, pady=8)
        events_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        tk.Label(events_frame, text="Event Log", fg="#e94560", bg="#16213e",
                 font=mono).pack(anchor="w")
        self.events_text = tk.Text(events_frame, height=8, bg="#0f0f23", fg="#aaaaaa",
                                   font=tkfont.Font(family="monospace", size=9),
                                   wrap=tk.WORD, state=tk.DISABLED)
        self.events_text.pack(fill=tk.BOTH, expand=True)

        # Status label (connection)
        self.conn_label = tk.Label(self.root, text="Connecting...", fg="#888",
                                   bg="#1a1a2e", font=tkfont.Font(family="monospace", size=9))
        self.conn_label.pack(pady=(0, 5))

        self._photo = None  # Keep reference to prevent GC

    def start(self):
        # Start MJPEG reader thread
        threading.Thread(target=self._mjpeg_loop, daemon=True).start()
        # Start status poller thread
        threading.Thread(target=self._status_loop, daemon=True).start()
        self.root.mainloop()

    def quit(self):
        self.running = False
        self.root.destroy()

    def _mjpeg_loop(self):
        while self.running:
            try:
                self.conn_label.config(text="Connecting to video stream...")
                resp = urlopen(f"{self.base_url}/video", timeout=5)
                boundary = b"--frame"
                buf = b""

                while self.running:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    buf += chunk

                    # Find JPEG frames
                    while True:
                        start = buf.find(b'\xff\xd8')  # JPEG start
                        if start == -1:
                            break
                        end = buf.find(b'\xff\xd9', start)  # JPEG end
                        if end == -1:
                            break
                        jpg_data = buf[start:end + 2]
                        buf = buf[end + 2:]

                        try:
                            img = Image.open(io.BytesIO(jpg_data))
                            # Resize if too large
                            w, h = img.size
                            max_w = 900
                            if w > max_w:
                                scale = max_w / w
                                img = img.resize((max_w, int(h * scale)), Image.LANCZOS)
                            self._photo = ImageTk.PhotoImage(img)
                            self.video_label.config(image=self._photo)
                            self.conn_label.config(text="Connected")
                        except Exception:
                            pass

            except (URLError, ConnectionError, OSError) as e:
                self.conn_label.config(text=f"Waiting for dashboard... ({e})")
                time.sleep(2)
            except Exception as e:
                self.conn_label.config(text=f"Error: {e}")
                time.sleep(2)

    def _status_loop(self):
        while self.running:
            try:
                resp = urlopen(f"{self.base_url}/api/status", timeout=3)
                data = json.loads(resp.read())

                mode = data.get("mode", "?").upper()
                color = "#e94560" if mode == "ENGAGED" else "#00d2ff"
                self.mode_label.config(text=f"MODE: {mode}  |  Uptime: {data.get('uptime', '?')}",
                                       fg=color)

                faces = data.get("faces", [])
                if faces:
                    names = [f.get("name", f.get("face_id", "?")) for f in faces]
                    self.face_label.config(text=f"Faces: {', '.join(names)}")
                else:
                    self.face_label.config(text="Faces: none")

                speaker = data.get("current_face") or "--"
                last_speech = data.get("last_speech") or ""
                if last_speech:
                    speaker += f'  |  "{last_speech[:60]}"'
                self.speaker_label.config(text=f"Speaker: {speaker}")

                # Append events
                events = data.get("events", [])
                if events:
                    self.events_text.config(state=tk.NORMAL)
                    for ev in events:
                        color_tag = ev.get("type", "system")
                        line = f"[{ev['time']}] {ev['text']}\n"
                        self.events_text.insert("1.0", line)
                    # Trim to 200 lines
                    lines = int(self.events_text.index('end-1c').split('.')[0])
                    if lines > 200:
                        self.events_text.delete("200.0", tk.END)
                    self.events_text.config(state=tk.DISABLED)

            except Exception:
                pass
            time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description="Live viewer for The Witness dashboard")
    parser.add_argument("--host", default="localhost", help="Dashboard host")
    parser.add_argument("--port", type=int, default=8080, help="Dashboard port")
    args = parser.parse_args()

    # Check PIL is available
    try:
        from PIL import Image, ImageTk
    except ImportError:
        print("Need Pillow: pip install Pillow")
        return

    viewer = LiveViewer(host=args.host, port=args.port)
    viewer.start()


if __name__ == "__main__":
    main()
