"""
Drone Vision Control System
Web-based UI to control both cameras simultaneously
Regular Camera: Person, Obstacle, Landing Pad, Color Target
Thermal Camera: Human Heat, Fire, Hot Objects
Author: Sue Sha
"""

import cv2
import numpy as np
import time
import threading
from flask import Flask, Response, jsonify, request
from datetime import datetime

app = Flask(__name__)

# ─────────────────────────────────────────────
# REGULAR CAMERA DETECTOR
# ─────────────────────────────────────────────

class RegularCameraDetector:
    def __init__(self):
        self.camera = None
        self.mode = 'person'
        self.frame_width = 640
        self.frame_height = 480
        self.fps = 0
        self.detection_count = 0
        self.lock = threading.Lock()
        self.last_frame = None
        self.running = False

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.body_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_fullbody.xml'
        )

    def init_camera(self, camera_id=0):
        try:
            self.camera = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, 15)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.camera.isOpened():
                print("❌ Regular camera failed to open")
                return False
            print("✅ Regular camera initialized")
            return True
        except Exception as e:
            print(f"❌ Regular camera error: {e}")
            return False

    def detect_person(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # CLAHE gives better contrast than equalizeHist, less over-brightening
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # Face detection — higher minNeighbors = fewer false positives
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=8,
            minSize=(50, 50), maxSize=(350, 350)
        )

        # Upper body cascade is FAR more reliable than fullbody
        upper_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_upperbody.xml'
        )
        upper_bodies = upper_cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=6,
            minSize=(80, 60)
        )

        detections = []
        for (x, y, w, h) in faces:
            detections.append({'bbox': (x, y, w, h), 'label': 'Human Face',
                               'color': (0, 255, 80), 'center': (x+w//2, y+h//2)})

        # Avoid duplicate detections — skip upper body if face already found nearby
        for (x, y, w, h) in upper_bodies:
            center = (x+w//2, y+h//2)
            overlap = False
            for det in detections:
                dx = abs(det['center'][0] - center[0])
                dy = abs(det['center'][1] - center[1])
                if dx < w//2 and dy < h//2:
                    overlap = True
                    break
            if not overlap:
                detections.append({'bbox': (x, y, w, h), 'label': 'Human Body',
                                   'color': (0, 200, 255), 'center': center})
        return detections

    def detect_obstacles(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        # Higher thresholds = only strong edges, less background noise
        edges = cv2.Canny(blurred, 60, 180)
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx, cy = self.frame_width // 2, self.frame_height // 2
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Raised from 2000 to 5000 - ignore small noise/texture
            if area > 5000:
                x, y, w, h = cv2.boundingRect(contour)
                # Filter out objects too wide/flat (likely floor/ceiling)
                aspect = w / h if h > 0 else 0
                if aspect > 6:
                    continue
                center = (x + w // 2, y + h // 2)
                dist = np.sqrt((center[0]-cx)**2 + (center[1]-cy)**2)
                threat = 'HIGH' if dist < 120 else 'MED' if dist < 220 else 'LOW'
                color = (0, 0, 255) if threat == 'HIGH' else (0, 165, 255) if threat == 'MED' else (0, 255, 255)
                detections.append({'bbox': (x, y, w, h), 'label': f'Obstacle {threat}',
                                   'color': color, 'center': center})
        return detections

    def detect_landing_pad(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Orange landing pad
        mask = cv2.inRange(hsv, np.array([5, 120, 120]), np.array([25, 255, 255]))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx, cy = self.frame_width // 2, self.frame_height // 2
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 3000:
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)
                offset_x = center[0] - cx
                offset_y = center[1] - cy
                aligned = abs(offset_x) < 25 and abs(offset_y) < 25
                label = 'PAD ✓ ALIGNED' if aligned else f'PAD ({offset_x:+d},{offset_y:+d})'
                detections.append({'bbox': (x, y, w, h), 'label': label,
                                   'color': (0, 255, 255), 'center': center})
        return detections

    def detect_color_target(self, frame, color='red'):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_ranges = {
            'red':    ([0, 120, 80], [10, 255, 255]),
            'blue':   ([100, 100, 80], [130, 255, 255]),
            'green':  ([40, 60, 60], [80, 255, 255]),
            'yellow': ([20, 120, 100], [35, 255, 255])
        }
        lower, upper = color_ranges.get(color, color_ranges['red'])
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1500:  # Raised to reduce noise detections
                x, y, w, h = cv2.boundingRect(contour)
                detections.append({'bbox': (x, y, w, h),
                                   'label': f'{color.upper()} TARGET',
                                   'color': (255, 0, 255),
                                   'center': (x+w//2, y+h//2)})
        return detections

    def annotate(self, frame, detections):
        for det in detections:
            x, y, w, h = det['bbox']
            color = det['color']
            label = det['label']
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.circle(frame, det['center'], 5, color, -1)
            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x, y-th-8), (x+tw+4, y), color, -1)
            cv2.putText(frame, label, (x+2, y-4),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        return frame

    def process_frame(self):
        if not self.camera:
            return None
        ret, frame = self.camera.read()
        if not ret:
            return None

        mode = self.mode
        if mode == 'person':
            detections = self.detect_person(frame)
        elif mode == 'obstacle':
            detections = self.detect_obstacles(frame)
        elif mode == 'landing_pad':
            detections = self.detect_landing_pad(frame)
        elif mode == 'target':
            detections = self.detect_color_target(frame, 'red')
        else:
            detections = []

        self.detection_count = len(detections)
        frame = self.annotate(frame, detections)

        # Overlay info bar
        cv2.rectangle(frame, (0, 0), (self.frame_width, 32), (0, 0, 0), -1)
        cv2.putText(frame, f"MODE: {mode.upper()}  |  Detected: {self.detection_count}  |  FPS: {self.fps:.1f}",
                   (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return frame


# ─────────────────────────────────────────────
# THERMAL CAMERA DETECTOR
# ─────────────────────────────────────────────

class ThermalCameraDetector:
    def __init__(self):
        self.camera = None
        self.mode = 'human'
        self.fps = 0
        self.detection_count = 0
        self.lock = threading.Lock()

    def init_camera(self, camera_id=2):
        try:
            self.camera = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'YUYV'))
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)
            self.camera.set(cv2.CAP_PROP_FPS, 25)
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.camera.isOpened():
                print("❌ Thermal camera failed to open")
                return False
            print("✅ Thermal camera initialized")
            return True
        except Exception as e:
            print(f"❌ Thermal camera error: {e}")
            return False

    def process_raw_frame(self, frame):
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            bgr = frame
        elif len(frame.shape) == 3 and frame.shape[2] == 2:
            bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
        elif len(frame.shape) == 2:
            bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        else:
            return None, None
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)
        return colored, gray

    def detect_human(self, gray, colored):
        """
        Detect human heat signature.
        Thermal cameras output 8-bit gray where brighter = hotter.
        Humans are warm but NOT the hottest thing — we exclude fire-range pixels.
        We use the TOP 15-50% brightness range to find warm bodies.
        """
        # Normalize: find the actual min/max of this frame
        frame_min = int(np.min(gray))
        frame_max = int(np.max(gray))
        frame_range = frame_max - frame_min if frame_max > frame_min else 1

        # Human bodies are in the upper-warm range but below fire
        # Target: pixels in 40%-75% of the frame brightness range
        lower = frame_min + int(frame_range * 0.40)
        upper = frame_min + int(frame_range * 0.75)
        mask = cv2.inRange(gray, lower, upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            # Min area raised to avoid noise (scaled for 512x384)
            if area > 300:
                x, y, w, h = cv2.boundingRect(contour)
                aspect = h / w if w > 0 else 0
                # Human silhouette: taller than wide, not extremely thin
                if 0.6 < aspect < 4.0:
                    center = (x+w//2, y+h//2)
                    # Show relative warmth as percentage
                    pixel_val = int(gray[center[1], center[0]])
                    warmth = int((pixel_val - frame_min) / frame_range * 100)
                    detections.append({"bbox": (x, y, w, h),
                                       "label": f"HUMAN {warmth}% warm",
                                       "color": (0, 255, 80),
                                       "center": center})
        return detections

    def detect_fire(self, gray, colored):
        """Detect fire: very high temperature >60C equivalent"""
        threshold = int(60 * 2.55)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 30:
                x, y, w, h = cv2.boundingRect(contour)
                center = (x+w//2, y+h//2)
                temp_val = int(gray[center[1], center[0]] / 2.55)
                detections.append({'bbox': (x, y, w, h),
                                   'label': f'🔥 FIRE ~{temp_val}C',
                                   'color': (0, 0, 255),
                                   'center': center})
        return detections

    def detect_hot(self, gray, colored):
        """Detect any hot object >35C"""
        threshold = int(35 * 2.55)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                x, y, w, h = cv2.boundingRect(contour)
                center = (x+w//2, y+h//2)
                temp_val = int(gray[center[1], center[0]] / 2.55)
                detections.append({'bbox': (x, y, w, h),
                                   'label': f'HOT ~{temp_val}C',
                                   'color': (0, 165, 255),
                                   'center': center})
        return detections

    def annotate(self, frame, detections):
        for det in detections:
            x, y, w, h = det['bbox']
            color = det['color']
            label = det['label']
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.circle(frame, det['center'], 4, color, -1)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(frame, (x, y-th-6), (x+tw+4, y), color, -1)
            cv2.putText(frame, label, (x+2, y-3),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
        return frame

    def process_frame(self):
        if not self.camera:
            return None
        ret, frame = self.camera.read()
        if not ret:
            return None

        colored, gray = self.process_raw_frame(frame)
        if colored is None:
            return None

        # Scale up for better visibility
        colored = cv2.resize(colored, (512, 384))
        gray = cv2.resize(gray, (512, 384))

        mode = self.mode
        if mode == 'human':
            detections = self.detect_human(gray, colored)
        elif mode == 'fire':
            detections = self.detect_fire(gray, colored)
        elif mode == 'hot':
            detections = self.detect_hot(gray, colored)
        else:
            detections = []

        self.detection_count = len(detections)
        colored = self.annotate(colored, detections)

        # Info bar
        cv2.rectangle(colored, (0, 0), (512, 30), (0, 0, 0), -1)
        cv2.putText(colored, f"THERMAL: {mode.upper()}  |  Detected: {self.detection_count}  |  FPS: {self.fps:.1f}",
                   (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return colored


# ─────────────────────────────────────────────
# INIT CAMERAS
# ─────────────────────────────────────────────

regular_cam = RegularCameraDetector()
thermal_cam = ThermalCameraDetector()

regular_cam_ok = regular_cam.init_camera(0)
thermal_cam_ok = thermal_cam.init_camera(2)


# ─────────────────────────────────────────────
# STREAM GENERATORS
# ─────────────────────────────────────────────

def generate_regular():
    prev_time = time.time()
    while True:
        frame = regular_cam.process_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        now = time.time()
        regular_cam.fps = 1.0 / (now - prev_time + 0.001)
        prev_time = now
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')


def generate_thermal():
    prev_time = time.time()
    while True:
        frame = thermal_cam.process_frame()
        if frame is None:
            time.sleep(0.05)
            continue
        now = time.time()
        thermal_cam.fps = 1.0 / (now - prev_time + 0.001)
        prev_time = now
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() + b'\r\n')


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/video/regular')
def video_regular():
    return Response(generate_regular(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video/thermal')
def video_thermal():
    return Response(generate_thermal(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/set_mode', methods=['POST'])
def set_mode():
    data = request.json
    cam = data.get('camera')
    mode = data.get('mode')
    if cam == 'regular':
        regular_cam.mode = mode
    elif cam == 'thermal':
        thermal_cam.mode = mode
    return jsonify({'status': 'ok', 'camera': cam, 'mode': mode})

@app.route('/stats')
def stats():
    return jsonify({
        'regular': {
            'mode': regular_cam.mode,
            'fps': round(regular_cam.fps, 1),
            'detections': regular_cam.detection_count,
            'active': regular_cam_ok
        },
        'thermal': {
            'mode': thermal_cam.mode,
            'fps': round(thermal_cam.fps, 1),
            'detections': thermal_cam.detection_count,
            'active': thermal_cam_ok
        }
    })

@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Drone Vision Control</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap" rel="stylesheet">
<style>
  :root {
    --bg: #080c10;
    --surface: #0d1117;
    --border: #1e2d3d;
    --accent-blue: #00d4ff;
    --accent-green: #00ff88;
    --accent-red: #ff3355;
    --accent-orange: #ff8c00;
    --text: #c9d1d9;
    --text-dim: #586069;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Rajdhani', sans-serif;
    min-height: 100vh;
    overflow-x: hidden;
  }
  /* Scanline effect */
  body::before {
    content: '';
    position: fixed;
    top: 0; left: 0; right: 0; bottom: 0;
    background: repeating-linear-gradient(
      0deg, transparent, transparent 2px,
      rgba(0,212,255,0.015) 2px, rgba(0,212,255,0.015) 4px
    );
    pointer-events: none;
    z-index: 1000;
  }

  header {
    padding: 16px 24px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--surface);
  }
  .logo {
    font-family: 'Share Tech Mono', monospace;
    font-size: 18px;
    color: var(--accent-blue);
    letter-spacing: 2px;
  }
  .logo span { color: var(--accent-green); }
  .status-bar {
    display: flex;
    gap: 16px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 12px;
  }
  .status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--accent-green);
    animation: pulse 1.5s infinite;
    margin-right: 6px;
  }
  .status-dot.off { background: var(--accent-red); animation: none; }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
  }

  .main {
    display: grid;
    grid-template-columns: 1fr 1fr;
    grid-template-rows: auto 1fr;
    gap: 0;
    padding: 20px;
    gap: 20px;
  }

  .camera-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 4px;
    overflow: hidden;
    position: relative;
  }
  .camera-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent-blue);
  }
  .camera-panel.thermal::before {
    background: var(--accent-orange);
  }

  .panel-header {
    padding: 10px 16px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .panel-title {
    font-size: 13px;
    font-weight: 700;
    letter-spacing: 2px;
    color: var(--accent-blue);
    font-family: 'Share Tech Mono', monospace;
  }
  .thermal .panel-title { color: var(--accent-orange); }

  .stats-row {
    display: flex;
    gap: 16px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
    color: var(--text-dim);
  }
  .stat-val { color: var(--accent-green); font-weight: bold; }

  .video-wrapper {
    position: relative;
    background: #000;
    aspect-ratio: 4/3;
    overflow: hidden;
  }
  .video-wrapper img {
    width: 100%;
    height: 100%;
    object-fit: contain;
    display: block;
  }
  .video-overlay {
    position: absolute;
    top: 8px; right: 8px;
    background: rgba(0,0,0,0.7);
    border: 1px solid var(--border);
    padding: 4px 8px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: var(--accent-blue);
    border-radius: 2px;
  }
  .thermal .video-overlay { color: var(--accent-orange); }

  .controls {
    padding: 12px 16px;
    border-top: 1px solid var(--border);
  }
  .controls-label {
    font-size: 10px;
    letter-spacing: 2px;
    color: var(--text-dim);
    margin-bottom: 8px;
    font-family: 'Share Tech Mono', monospace;
  }
  .btn-group {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }
  .btn {
    padding: 6px 14px;
    border: 1px solid var(--border);
    background: transparent;
    color: var(--text);
    font-family: 'Rajdhani', sans-serif;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 1px;
    cursor: pointer;
    border-radius: 2px;
    transition: all 0.15s;
    text-transform: uppercase;
  }
  .btn:hover {
    border-color: var(--accent-blue);
    color: var(--accent-blue);
    background: rgba(0, 212, 255, 0.05);
  }
  .btn.active {
    border-color: var(--accent-blue);
    background: rgba(0, 212, 255, 0.12);
    color: var(--accent-blue);
  }
  .thermal .btn:hover, .thermal .btn.active {
    border-color: var(--accent-orange);
    color: var(--accent-orange);
    background: rgba(255, 140, 0, 0.1);
  }

  /* Corner decorations */
  .corner {
    position: absolute;
    width: 12px; height: 12px;
    border-color: var(--accent-blue);
    border-style: solid;
    opacity: 0.5;
  }
  .corner.tl { top: 4px; left: 4px; border-width: 1px 0 0 1px; }
  .corner.tr { top: 4px; right: 4px; border-width: 1px 1px 0 0; }
  .corner.bl { bottom: 4px; left: 4px; border-width: 0 0 1px 1px; }
  .corner.br { bottom: 4px; right: 4px; border-width: 0 1px 1px 0; }

  .detection-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    background: rgba(0, 255, 136, 0.1);
    border: 1px solid var(--accent-green);
    color: var(--accent-green);
    padding: 2px 8px;
    border-radius: 2px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 11px;
  }
  .detection-badge.zero {
    background: transparent;
    border-color: var(--text-dim);
    color: var(--text-dim);
  }
</style>
</head>
<body>

<header>
  <div class="logo">DRONE_<span>VISION</span>_SYS</div>
  <div class="status-bar">
    <span id="reg-status"><span class="status-dot" id="reg-dot"></span>CAM_REG</span>
    <span id="thm-status"><span class="status-dot thermal" id="thm-dot"></span>CAM_THERMAL</span>
    <span id="clock" style="color:var(--text-dim)"></span>
  </div>
</header>

<div class="main">

  <!-- REGULAR CAMERA -->
  <div class="camera-panel">
    <div class="panel-header">
      <span class="panel-title">◈ REGULAR CAMERA</span>
      <div class="stats-row">
        <span>FPS: <span class="stat-val" id="reg-fps">0</span></span>
        <span>DETECT: <span class="stat-val" id="reg-det">0</span></span>
        <span>MODE: <span class="stat-val" id="reg-mode">PERSON</span></span>
      </div>
    </div>
    <div class="video-wrapper">
      <img id="reg-img" src="/video/regular" alt="Regular Camera">
      <div class="video-overlay">RGB CAM</div>
      <div class="corner tl"></div>
      <div class="corner tr"></div>
      <div class="corner bl"></div>
      <div class="corner br"></div>
    </div>
    <div class="controls">
      <div class="controls-label">// DETECTION MODE</div>
      <div class="btn-group">
        <button class="btn active" onclick="setMode('regular','person',this)">👤 Person</button>
        <button class="btn" onclick="setMode('regular','obstacle',this)">⚠ Obstacle</button>
        <button class="btn" onclick="setMode('regular','landing_pad',this)">🎯 Landing</button>
        <button class="btn" onclick="setMode('regular','target',this)">🔴 Target</button>
      </div>
    </div>
  </div>

  <!-- THERMAL CAMERA -->
  <div class="camera-panel thermal">
    <div class="panel-header">
      <span class="panel-title">◈ THERMAL CAMERA</span>
      <div class="stats-row">
        <span>FPS: <span class="stat-val" id="thm-fps">0</span></span>
        <span>DETECT: <span class="stat-val" id="thm-det">0</span></span>
        <span>MODE: <span class="stat-val" id="thm-mode">HUMAN</span></span>
      </div>
    </div>
    <div class="video-wrapper">
      <img id="thm-img" src="/video/thermal" alt="Thermal Camera">
      <div class="video-overlay">THERMAL</div>
      <div class="corner tl" style="border-color:var(--accent-orange)"></div>
      <div class="corner tr" style="border-color:var(--accent-orange)"></div>
      <div class="corner bl" style="border-color:var(--accent-orange)"></div>
      <div class="corner br" style="border-color:var(--accent-orange)"></div>
    </div>
    <div class="controls">
      <div class="controls-label">// DETECTION MODE</div>
      <div class="btn-group">
        <button class="btn active" onclick="setMode('thermal','human',this)">🧍 Human</button>
        <button class="btn" onclick="setMode('thermal','fire',this)">🔥 Fire</button>
        <button class="btn" onclick="setMode('thermal','hot',this)">♨ Hot Object</button>
      </div>
    </div>
  </div>

</div>

<script>
  // Clock
  setInterval(() => {
    document.getElementById('clock').textContent = new Date().toLocaleTimeString();
  }, 1000);

  // Mode buttons
  function setMode(camera, mode, btn) {
    fetch('/set_mode', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({camera, mode})
    });
    // Update active button in same group
    const panel = btn.closest('.camera-panel');
    panel.querySelectorAll('.btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    if (camera === 'regular') {
      document.getElementById('reg-mode').textContent = mode.toUpperCase();
    } else {
      document.getElementById('thm-mode').textContent = mode.toUpperCase();
    }
  }

  // Stats polling
  setInterval(() => {
    fetch('/stats').then(r => r.json()).then(data => {
      document.getElementById('reg-fps').textContent = data.regular.fps;
      document.getElementById('reg-det').textContent = data.regular.detections;
      document.getElementById('thm-fps').textContent = data.thermal.fps;
      document.getElementById('thm-det').textContent = data.thermal.detections;

      const regDot = document.getElementById('reg-dot');
      const thmDot = document.getElementById('thm-dot');
      regDot.className = 'status-dot' + (data.regular.active ? '' : ' off');
      thmDot.className = 'status-dot' + (data.thermal.active ? '' : ' off');
    }).catch(() => {});
  }, 1000);
</script>
</body>
</html>'''

if __name__ == '__main__':
    print("=" * 60)
    print("🚁 Drone Vision Control System")
    print("=" * 60)
    print(f"Regular camera: {'✅ OK' if regular_cam_ok else '❌ Failed'}")
    print(f"Thermal camera: {'✅ OK' if thermal_cam_ok else '❌ Failed'}")
    print("\nOpen in browser: http://172.20.10.2:5000")
    print("=" * 60)
    app.run(host='0.0.0.0', port=5000, threaded=True)