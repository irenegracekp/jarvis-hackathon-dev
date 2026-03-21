"""Live web dashboard — MJPEG stream + status overlay. No external dependencies."""

import cv2
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO

# Colors (BGR)
GREEN = (0, 255, 0)
YELLOW = (0, 255, 255)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ORANGE = (0, 165, 255)

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<title>The Witness - Live Dashboard</title>
<meta charset="utf-8">
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #1a1a2e; color: #eee; font-family: 'Courier New', monospace; }
  .container { display: flex; flex-direction: column; align-items: center; padding: 20px; }
  h1 { color: #e94560; margin-bottom: 10px; font-size: 1.5em; }
  .video-container { position: relative; margin-bottom: 15px; }
  img { border: 2px solid #333; border-radius: 8px; max-width: 95vw; }
  #status { width: 800px; max-width: 95vw; display: grid; grid-template-columns: 1fr 1fr;
            gap: 10px; }
  .panel { background: #16213e; border: 1px solid #333; border-radius: 8px; padding: 12px; }
  .panel h2 { color: #e94560; font-size: 1em; margin-bottom: 8px; border-bottom: 1px solid #333;
              padding-bottom: 4px; }
  .mode-ambient { color: #00d2ff; }
  .mode-engaged { color: #e94560; }
  .fact { color: #aaa; font-size: 0.85em; }
  .event { padding: 2px 0; font-size: 0.85em; }
  .event-arrive { color: #0f0; }
  .event-depart { color: #f55; }
  .event-memory { color: #ff0; }
  #events { max-height: 200px; overflow-y: auto; }
  .face-entry { margin-bottom: 6px; }
  .face-name { color: #0f0; font-weight: bold; }
  .face-unknown { color: #ff0; }
</style>
</head>
<body>
<div class="container">
  <h1>THE WITNESS - Live Dashboard</h1>
  <div class="video-container">
    <img src="/video" width="800">
  </div>
  <div id="status">
    <div class="panel">
      <h2>System State</h2>
      <div id="state">Loading...</div>
    </div>
    <div class="panel">
      <h2>Faces Present</h2>
      <div id="faces">None</div>
    </div>
    <div class="panel" style="grid-column: span 2;">
      <h2>Event Log</h2>
      <div id="events"></div>
    </div>
  </div>
</div>
<script>
function poll() {
  fetch('/api/status').then(r => r.json()).then(d => {
    // State
    let modeClass = d.mode === 'engaged' ? 'mode-engaged' : 'mode-ambient';
    let stateHtml = `<div>Mode: <span class="${modeClass}">${d.mode.toUpperCase()}</span></div>`;
    if (d.current_face) stateHtml += `<div>Speaker: ${d.current_face}</div>`;
    if (d.last_speech) stateHtml += `<div class="fact">Last speech: ${d.last_speech}</div>`;
    stateHtml += `<div class="fact">Uptime: ${d.uptime}</div>`;
    document.getElementById('state').innerHTML = stateHtml;

    // Faces
    let facesHtml = '';
    if (d.faces.length === 0) facesHtml = '<div class="fact">No faces detected</div>';
    d.faces.forEach(f => {
      let cls = f.is_known ? 'face-name' : 'face-unknown';
      facesHtml += `<div class="face-entry"><span class="${cls}">${f.name || f.face_id}</span>`;
      if (f.times_seen > 1) facesHtml += ` <span class="fact">(seen ${f.times_seen}x)</span>`;
      if (f.facts && f.facts.length) facesHtml += `<div class="fact">${f.facts.join(', ')}</div>`;
      facesHtml += '</div>';
    });
    document.getElementById('faces').innerHTML = facesHtml;

    // Events
    let evEl = document.getElementById('events');
    d.events.forEach(e => {
      let cls = e.type === 'arrive' ? 'event-arrive' : e.type === 'depart' ? 'event-depart' : 'event-memory';
      let div = document.createElement('div');
      div.className = 'event ' + cls;
      div.textContent = `[${e.time}] ${e.text}`;
      evEl.insertBefore(div, evEl.firstChild);
    });
  }).catch(() => {});
  setTimeout(poll, 1000);
}
poll();
</script>
</body>
</html>"""


class DashboardState:
    """Shared state between the main pipeline and the dashboard."""

    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None  # Latest annotated frame (BGR)
        self._frame_jpg = None  # JPEG-encoded
        self._mode = "ambient"
        self._current_face = None
        self._last_speech = ""
        self._faces_present = []  # list of dicts
        self._events = []  # recent events (max 50)
        self._start_time = time.time()
        self._new_events = []  # events not yet polled

    def update_frame(self, frame, faces=None, memory=None):
        """Update the displayed frame with face annotations."""
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        if faces:
            for face in faces:
                bbox = face["bbox"]
                x, y, fw, fh = bbox
                fid = face["face_id"]
                is_known = face.get("is_known", False)

                # Draw bbox
                color = GREEN if is_known else YELLOW
                cv2.rectangle(annotated, (x, y), (x + fw, y + fh), color, 2)

                # Label
                label = fid
                if memory:
                    person = memory.get_person(fid)
                    if person:
                        label = person["name"]
                        if person.get("times_seen", 0) > 1:
                            label += f" (x{person['times_seen']})"

                # Draw label background
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(annotated, (x, y - th - 8), (x + tw + 4, y), color, -1)
                cv2.putText(annotated, label, (x + 2, y - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLACK, 1, cv2.LINE_AA)

        # Mode banner
        mode_text = f"MODE: {self._mode.upper()}"
        mode_color = RED if self._mode == "engaged" else ORANGE
        cv2.rectangle(annotated, (0, 0), (200, 30), mode_color, -1)
        cv2.putText(annotated, mode_text, (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)

        # Face count
        n = len(faces) if faces else 0
        count_text = f"Faces: {n}"
        cv2.putText(annotated, count_text, (w - 120, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2, cv2.LINE_AA)

        # Encode to JPEG
        _, jpg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        with self._lock:
            self._frame = annotated
            self._frame_jpg = jpg.tobytes()

    def update_state(self, mode=None, current_face=None, last_speech=None, faces_present=None):
        with self._lock:
            if mode is not None:
                self._mode = mode
            if current_face is not None:
                self._current_face = current_face
            if last_speech is not None:
                self._last_speech = last_speech
            if faces_present is not None:
                self._faces_present = faces_present

    def add_event(self, event_type, text):
        ts = time.strftime("%H:%M:%S")
        event = {"type": event_type, "time": ts, "text": text}
        with self._lock:
            self._events.append(event)
            if len(self._events) > 200:
                self._events = self._events[-100:]
            self._new_events.append(event)

    def get_frame_jpg(self):
        with self._lock:
            return self._frame_jpg

    def get_api_status(self):
        with self._lock:
            uptime = int(time.time() - self._start_time)
            mins, secs = divmod(uptime, 60)
            events = list(self._new_events)
            self._new_events.clear()
            return {
                "mode": self._mode,
                "current_face": self._current_face,
                "last_speech": self._last_speech,
                "faces": list(self._faces_present),
                "events": events,
                "uptime": f"{mins}m {secs}s",
            }


class DashboardHandler(BaseHTTPRequestHandler):
    dashboard_state = None  # Set by DashboardServer

    def do_GET(self):
        try:
            if self.path == '/':
                self._serve_html()
            elif self.path == '/video':
                self._serve_mjpeg()
            elif self.path == '/api/status':
                self._serve_api()
            else:
                self.send_error(404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_html(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html')
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode())

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=frame')
        self.end_headers()
        try:
            while True:
                jpg = self.dashboard_state.get_frame_jpg()
                if jpg:
                    self.wfile.write(b'--frame\r\n')
                    self.wfile.write(b'Content-Type: image/jpeg\r\n')
                    self.wfile.write(f'Content-Length: {len(jpg)}\r\n\r\n'.encode())
                    self.wfile.write(jpg)
                    self.wfile.write(b'\r\n')
                time.sleep(0.1)  # ~10 FPS
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _serve_api(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        data = self.dashboard_state.get_api_status()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, format, *args):
        pass  # Suppress HTTP logs


class DashboardServer:
    def __init__(self, port=8080):
        self.port = port
        self.state = DashboardState()
        self._server = None
        self._thread = None

    def start(self):
        DashboardHandler.dashboard_state = self.state
        self._server = HTTPServer(('0.0.0.0', self.port), DashboardHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        print(f"[dashboard] Live at http://localhost:{self.port}")

    def stop(self):
        if self._server:
            self._server.shutdown()
