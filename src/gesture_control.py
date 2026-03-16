"""
Gesture Control - TFLite Hand Landmark Skeleton
Uses Google's official hand landmark model via tflite_runtime.
21 hand keypoints for accurate finger detection.

3 gestures:
  OPEN HAND (5 fingers up) = FLY UP
  FIST      (0 fingers up) = FLY DOWN
  PEACE     (2 fingers up) = HOVER / STOP

Run with: python3.11 gesture_control.py

Author: Sue Sha
"""

import cv2
import numpy as np
import time
import os
from flask import Flask, Response, jsonify
from datetime import datetime

# TFLite import
try:
    from tflite_runtime.interpreter import Interpreter
    TFLITE_OK = True
    print("TFLite loaded OK")
except ImportError:
    TFLITE_OK = False
    print("TFLite not found! Run: python3.11 -m pip install tflite_runtime")

app = Flask(__name__)

# Model paths
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hand_model')
PALM_MODEL = os.path.join(MODEL_DIR, 'palm_detection.tflite')
LANDMARK_MODEL = os.path.join(MODEL_DIR, 'hand_landmark.tflite')

# MediaPipe hand landmark indices
# 21 landmarks: 0=wrist, 1-4=thumb, 5-8=index, 9-12=middle, 13-16=ring, 17-20=pinky
FINGER_TIPS  = [4, 8, 12, 16, 20]
FINGER_PIPS  = [3, 6, 10, 14, 18]

# Skeleton connections for drawing
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),       # thumb
    (0,5),(5,6),(6,7),(7,8),       # index
    (0,9),(9,10),(10,11),(11,12),  # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20),# pinky
    (5,9),(9,13),(13,17),(0,17)    # palm
]

# Colors per finger for skeleton
FINGER_COLORS = [
    (255, 100, 100),  # thumb - red
    (100, 255, 100),  # index - green
    (100, 100, 255),  # middle - blue
    (255, 255, 100),  # ring - yellow
    (255, 100, 255),  # pinky - magenta
]


# ─────────────────────────────────────────────
# DRONE CONTROLLER (Simulation)
# ─────────────────────────────────────────────

class DroneController:
    def __init__(self):
        self.altitude = 1.5
        self.mode = 'GUIDED'
        self.armed = True
        self.last_command = 'NONE'

    def fly_up(self):
        self.altitude = min(self.altitude + 0.5, 10.0)
        self._log("FLY UP")

    def fly_down(self):
        self.altitude = max(self.altitude - 0.5, 0.0)
        self._log("FLY DOWN")

    def hover(self):
        self._log("HOVER")

    def land(self):
        self.altitude = 0.0
        self.mode = 'LANDED'
        self._log("LAND")

    def takeoff(self):
        self.altitude = 1.5
        self.mode = 'GUIDED'
        self.armed = True
        self._log("TAKEOFF")

    def _log(self, cmd):
        self.last_command = cmd
        print(f"[{datetime.now().strftime('%H:%M:%S')}] CMD: {cmd}")

    def get_status(self):
        return {
            'mode': self.mode,
            'armed': self.armed,
            'altitude': round(self.altitude, 2),
            'last_command': self.last_command
        }


# ─────────────────────────────────────────────
# HAND LANDMARK DETECTOR (TFLite)
# ─────────────────────────────────────────────

class HandLandmarkDetector:
    def __init__(self):
        self.palm_interpreter = None
        self.landmark_interpreter = None
        self.camera = None
        self.fps = 0
        self.current_gesture = 'NONE'
        self.finger_count = 0

        # Gesture hold + cooldown
        self.pending_gesture = 'NONE'
        self.gesture_hold_start = 0
        self.gesture_hold_required = 1.0
        self.last_trigger_time = 0
        self.cooldown = 1.5

    def load_models(self):
        if not TFLITE_OK:
            return False
        if not os.path.exists(PALM_MODEL) or not os.path.exists(LANDMARK_MODEL):
            print(f"Model files not found in {MODEL_DIR}")
            print("Run: wget commands to download models")
            return False
        try:
            self.palm_interpreter = Interpreter(model_path=PALM_MODEL)
            self.palm_interpreter.allocate_tensors()
            self.palm_input  = self.palm_interpreter.get_input_details()
            self.palm_output = self.palm_interpreter.get_output_details()

            self.landmark_interpreter = Interpreter(model_path=LANDMARK_MODEL)
            self.landmark_interpreter.allocate_tensors()
            self.lm_input  = self.landmark_interpreter.get_input_details()
            self.lm_output = self.landmark_interpreter.get_output_details()

            # Get expected input sizes
            self.palm_size = self.palm_input[0]['shape'][1]      # usually 192
            self.lm_size   = self.lm_input[0]['shape'][1]        # usually 224

            print(f"Palm model input: {self.palm_size}x{self.palm_size}")
            print(f"Landmark model input: {self.lm_size}x{self.lm_size}")
            print("Models loaded OK")
            return True
        except Exception as e:
            print(f"Model load error: {e}")
            return False

    def init_camera(self, camera_id=0):
        self.camera = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 15)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok = self.camera.isOpened()
        print(f"Camera: {'OK' if ok else 'FAILED'}")
        return ok

    def detect_palm(self, frame):
        """
        Run palm detection to find hand bounding box.
        Returns (x, y, w, h) of hand region or None.
        """
        h, w = frame.shape[:2]
        img = cv2.resize(frame, (self.palm_size, self.palm_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, axis=0)

        self.palm_interpreter.set_tensor(self.palm_input[0]['index'], img)
        self.palm_interpreter.invoke()

        # Get score — check if hand is present
        scores = self.palm_interpreter.get_tensor(self.palm_output[0]['index'])
        score = float(np.max(scores))

        if score < 0.5:
            return None

        # Get bounding box from second output
        try:
            boxes = self.palm_interpreter.get_tensor(self.palm_output[1]['index'])
            box = boxes[0][np.argmax(scores[0])]
            # box format: [ymin, xmin, ymax, xmax] normalized
            ymin, xmin, ymax, xmax = box[:4]
            # Add padding
            pad = 0.15
            xmin = max(0, xmin - pad)
            ymin = max(0, ymin - pad)
            xmax = min(1, xmax + pad)
            ymax = min(1, ymax + pad)

            bx = int(xmin * w)
            by = int(ymin * h)
            bw = int((xmax - xmin) * w)
            bh = int((ymax - ymin) * h)
            return (bx, by, bw, bh)
        except:
            # If box extraction fails, use full frame center
            margin = 0.1
            return (int(w*margin), int(h*margin),
                    int(w*(1-2*margin)), int(h*(1-2*margin)))

    def get_landmarks(self, frame, bbox):
        """
        Run landmark detection on hand crop.
        Returns list of 21 (x, y) pixel coordinates.
        """
        h, w = frame.shape[:2]
        bx, by, bw, bh = bbox

        # Crop hand region
        bx = max(0, bx); by = max(0, by)
        bw = min(bw, w - bx); bh = min(bh, h - by)
        if bw <= 0 or bh <= 0:
            return None

        crop = frame[by:by+bh, bx:bx+bw]
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_resized = cv2.resize(crop_rgb, (self.lm_size, self.lm_size))
        inp = crop_resized.astype(np.float32) / 255.0
        inp = np.expand_dims(inp, axis=0)

        self.landmark_interpreter.set_tensor(self.lm_input[0]['index'], inp)
        self.landmark_interpreter.invoke()

        # Landmarks output: shape (1, 63) = 21 points * 3 (x, y, z)
        landmarks_raw = self.landmark_interpreter.get_tensor(self.lm_output[0]['index'])
        landmarks = landmarks_raw[0].reshape(21, 3)

        # Convert normalized coords back to frame pixels
        points = []
        for lm in landmarks:
            px = int(bx + lm[0] / self.lm_size * bw)
            py = int(by + lm[1] / self.lm_size * bh)
            points.append((px, py))

        return points

    def count_fingers(self, points):
        """
        Count extended fingers.
        Finger is UP if tip Y < pip Y (higher on screen).
        Thumb uses X comparison.
        """
        if not points or len(points) < 21:
            return 0

        count = 0

        # Thumb: tip x vs ip x
        if points[4][0] < points[3][0]:
            count += 1

        # Other fingers: tip y vs pip y
        for tip, pip in zip(FINGER_TIPS[1:], FINGER_PIPS[1:]):
            if points[tip][1] < points[pip][1] - 10:
                count += 1

        return count

    def classify_gesture(self, fingers):
        if fingers == 5:
            return 'OPEN_HAND'
        elif fingers == 0:
            return 'FIST'
        elif fingers == 2:
            return 'PEACE'
        else:
            return 'NONE'

    def draw_skeleton(self, frame, points):
        """Draw 21-point hand skeleton."""
        if not points:
            return frame

        # Draw connections
        for i, (a, b) in enumerate(CONNECTIONS):
            if a < len(points) and b < len(points):
                cv2.line(frame, points[a], points[b], (200, 200, 200), 1)

        # Draw keypoints colored by finger
        finger_ranges = [(0,4,'thumb'),(5,8,'index'),(9,12,'middle'),(13,16,'ring'),(17,20,'pinky')]
        colors = [(255,100,100),(100,255,100),(100,150,255),(255,255,100),(255,100,255)]

        for i, pt in enumerate(points):
            # Find which finger this point belongs to
            color = (0, 255, 200)  # default wrist color
            for fi, (start, end, _) in enumerate(finger_ranges):
                if start <= i <= end:
                    color = colors[fi]
                    break

            size = 7 if i in FINGER_TIPS else 4
            cv2.circle(frame, pt, size, color, -1)

        return frame

    def process_frame(self, drone):
        ret, frame = self.camera.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        gesture = 'NONE'
        fingers = 0
        command_triggered = False

        if self.palm_interpreter and self.landmark_interpreter:
            bbox = self.detect_palm(frame)
            if bbox:
                # Draw bounding box
                bx, by, bw, bh = bbox
                cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (0, 200, 100), 1)

                points = self.get_landmarks(frame, bbox)
                if points:
                    frame = self.draw_skeleton(frame, points)
                    fingers = self.count_fingers(points)
                    gesture = self.classify_gesture(fingers)

                    # Gesture hold + cooldown
                    now = time.time()
                    if gesture != self.pending_gesture:
                        self.pending_gesture = gesture
                        self.gesture_hold_start = now
                    else:
                        hold = now - self.gesture_hold_start
                        cooldown_ok = (now - self.last_trigger_time) > self.cooldown
                        if hold >= self.gesture_hold_required and cooldown_ok and gesture != 'NONE':
                            self._execute(gesture, drone)
                            self.last_trigger_time = now
                            command_triggered = True

        self.current_gesture = gesture
        self.finger_count = fingers
        frame = self._draw_overlay(frame, gesture, fingers, command_triggered, drone)
        return frame

    def _execute(self, gesture, drone):
        print(f"[GESTURE] {gesture}")
        if gesture == 'OPEN_HAND': drone.fly_up()
        elif gesture == 'FIST':    drone.fly_down()
        elif gesture == 'PEACE':   drone.hover()

    def _draw_overlay(self, frame, gesture, fingers, triggered, drone):
        h, w = frame.shape[:2]
        g_colors = {
            'OPEN_HAND': (0, 255, 80),
            'FIST':      (60, 60, 255),
            'PEACE':     (0, 220, 255),
            'NONE':      (80, 80, 80)
        }
        g_color = g_colors.get(gesture, (80, 80, 80))

        bar_color = (0, 130, 0) if triggered else (0, 0, 0)
        cv2.rectangle(frame, (0, 0), (w, 44), bar_color, -1)
        cv2.putText(frame, f"GESTURE: {gesture}  FINGERS: {fingers}",
                   (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, g_color, 2)
        cv2.putText(frame, f"CMD:{drone.last_command}  ALT:{drone.altitude:.1f}m  FPS:{self.fps:.0f}",
                   (w - 320, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

        cv2.rectangle(frame, (0, h - 36), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "OPEN HAND=UP    FIST=DOWN    PEACE(2 fingers)=HOVER",
                   (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

        now = time.time()
        if gesture != 'NONE':
            hold = min(now - self.gesture_hold_start, self.gesture_hold_required)
            progress = int((hold / self.gesture_hold_required) * (w - 20))
            cv2.rectangle(frame, (10, h - 8), (w - 10, h - 3), (40, 40, 40), -1)
            cv2.rectangle(frame, (10, h - 8), (10 + progress, h - 3), g_color, -1)

        return frame


# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────

drone = DroneController()
detector = HandLandmarkDetector()
cam_ok = detector.init_camera(0)
model_ok = detector.load_models()
drone.takeoff()


# ─────────────────────────────────────────────
# FLASK
# ─────────────────────────────────────────────

def generate():
    prev = time.time()
    while True:
        frame = detector.process_frame(drone)
        if frame is None:
            time.sleep(0.05)
            continue
        now = time.time()
        detector.fps = 1.0 / (now - prev + 0.001)
        prev = now
        _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buf.tobytes() + b'\r\n')


@app.route('/video')
def video():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/status')
def status():
    return jsonify({
        'gesture': detector.current_gesture,
        'fingers': detector.finger_count,
        'fps': round(detector.fps, 1),
        'drone': drone.get_status(),
        'model_ok': model_ok
    })

@app.route('/takeoff')
def takeoff():
    drone.takeoff()
    return jsonify({'ok': True})

@app.route('/land')
def land():
    drone.land()
    return jsonify({'ok': True})

@app.route('/')
def index():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Gesture Control</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@500;700&display=swap" rel="stylesheet">
<style>
  :root{--bg:#080c10;--surface:#0d1117;--border:#1e2d3d;--green:#00ff88;--blue:#00d4ff;--red:#ff3355;--orange:#ff8c00;--text:#c9d1d9;--dim:#586069}
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:var(--bg);color:var(--text);font-family:'Rajdhani',sans-serif;min-height:100vh}
  body::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,212,255,.012) 2px,rgba(0,212,255,.012) 4px);pointer-events:none;z-index:1000}
  header{padding:14px 24px;border-bottom:1px solid var(--border);background:var(--surface);display:flex;align-items:center;justify-content:space-between}
  .logo{font-family:'Share Tech Mono',monospace;font-size:16px;color:var(--green);letter-spacing:2px}
  .badges{display:flex;gap:8px}
  .badge{padding:3px 10px;border-radius:2px;font-family:'Share Tech Mono',monospace;font-size:11px}
  .sim{background:rgba(255,140,0,.15);border:1px solid var(--orange);color:var(--orange)}
  .ok{background:rgba(0,255,136,.1);border:1px solid var(--green);color:var(--green)}
  .err{background:rgba(255,51,85,.1);border:1px solid var(--red);color:var(--red)}
  .layout{display:grid;grid-template-columns:1fr 300px;gap:20px;padding:20px}
  .video-panel{background:var(--surface);border:1px solid var(--border);border-radius:4px;overflow:hidden;position:relative}
  .video-panel::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--green)}
  .panel-title{padding:10px 16px;border-bottom:1px solid var(--border);font-family:'Share Tech Mono',monospace;font-size:12px;color:var(--green);letter-spacing:2px}
  .video-wrapper{background:#000;aspect-ratio:4/3}
  .video-wrapper img{width:100%;height:100%;object-fit:contain;display:block}
  .right-panel{display:flex;flex-direction:column;gap:16px}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:16px}
  .card-title{font-family:'Share Tech Mono',monospace;font-size:11px;color:var(--dim);letter-spacing:2px;margin-bottom:12px}
  .sgrid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .stat{background:#0a0f16;border:1px solid var(--border);padding:8px 10px;border-radius:2px}
  .slabel{font-size:10px;color:var(--dim);font-family:'Share Tech Mono',monospace}
  .sval{font-size:20px;font-weight:700;color:var(--green);font-family:'Share Tech Mono',monospace}
  .sval.sm{font-size:13px;color:var(--blue)}
  .gcards{display:flex;flex-direction:column;gap:8px}
  .gcard{padding:12px;border-radius:2px;border:1px solid var(--border);background:#0a0f16;transition:all .2s;display:flex;align-items:center;gap:10px}
  .gcard.active{border-color:var(--green);background:rgba(0,255,136,.08)}
  .dot{width:10px;height:10px;border-radius:50%;background:var(--dim);flex-shrink:0;transition:all .2s}
  .gcard.active .dot{background:var(--green);box-shadow:0 0 8px var(--green)}
  .gname{font-family:'Share Tech Mono',monospace;font-size:12px}
  .gcmd{font-size:11px;color:var(--dim);margin-left:auto}
  .gcard.active .gcmd{color:var(--green)}
  .alt-wrap{height:90px;display:flex;gap:8px;align-items:flex-end;margin-top:10px}
  .alt-bg{flex:1;height:100%;background:#0a0f16;border:1px solid var(--border);border-radius:2px;position:relative;overflow:hidden}
  .alt-fill{position:absolute;bottom:0;left:0;right:0;background:linear-gradient(to top,var(--green),rgba(0,255,136,.3));transition:height .5s;border-radius:2px}
  .alt-lbl{font-family:'Share Tech Mono',monospace;font-size:10px;color:var(--dim);writing-mode:vertical-rl}
  .btn-row{display:flex;gap:8px}
  .btn{flex:1;padding:10px;border:1px solid var(--border);background:transparent;color:var(--text);font-family:'Rajdhani',sans-serif;font-size:13px;font-weight:700;letter-spacing:1px;cursor:pointer;border-radius:2px;transition:all .15s;text-transform:uppercase}
  .btn.g:hover{border-color:var(--green);color:var(--green);background:rgba(0,255,136,.08)}
  .btn.r:hover{border-color:var(--red);color:var(--red);background:rgba(255,51,85,.08)}
</style>
</head>
<body>
<header>
  <div class="logo">GESTURE_CONTROL_SYS</div>
  <div class="badges">
    <div class="badge sim">SIMULATION</div>
    <div class="badge" id="model-badge">CHECKING...</div>
  </div>
</header>
<div class="layout">
  <div class="video-panel">
    <div class="panel-title">HAND SKELETON — TFLITE LANDMARKS</div>
    <div class="video-wrapper"><img src="/video"></div>
  </div>
  <div class="right-panel">
    <div class="card">
      <div class="card-title">// DRONE STATUS</div>
      <div class="sgrid">
        <div class="stat"><div class="slabel">MODE</div><div class="sval" id="d-mode">--</div></div>
        <div class="stat"><div class="slabel">ALTITUDE</div><div class="sval" id="d-alt">0.0m</div></div>
      </div>
      <div class="stat" style="margin-top:8px"><div class="slabel">LAST COMMAND</div><div class="sval sm" id="d-cmd">NONE</div></div>
      <div class="alt-wrap">
        <div class="alt-lbl">ALT</div>
        <div class="alt-bg"><div class="alt-fill" id="alt-bar" style="height:15%"></div></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">// GESTURES (hold 1 sec)</div>
      <div class="gcards">
        <div class="gcard" id="g-OPEN_HAND"><div class="dot"></div><span class="gname">OPEN HAND (5 fingers)</span><span class="gcmd">FLY UP</span></div>
        <div class="gcard" id="g-FIST"><div class="dot"></div><span class="gname">FIST (0 fingers)</span><span class="gcmd">FLY DOWN</span></div>
        <div class="gcard" id="g-PEACE"><div class="dot"></div><span class="gname">PEACE (2 fingers)</span><span class="gcmd">HOVER</span></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">// CONTROLS</div>
      <div class="btn-row">
        <button class="btn g" onclick="fetch('/takeoff')">TAKEOFF</button>
        <button class="btn r" onclick="fetch('/land')">LAND</button>
      </div>
    </div>
  </div>
</div>
<script>
setInterval(()=>{
  fetch('/status').then(r=>r.json()).then(d=>{
    document.getElementById('d-mode').textContent = d.drone.mode||'--';
    document.getElementById('d-alt').textContent = (d.drone.altitude||0).toFixed(1)+'m';
    document.getElementById('d-cmd').textContent = d.drone.last_command||'NONE';
    document.getElementById('alt-bar').style.height = Math.min((d.drone.altitude/10)*100,100)+'%';
    const mb = document.getElementById('model-badge');
    mb.textContent = d.model_ok ? 'TFLITE OK' : 'MODEL ERROR';
    mb.className = 'badge ' + (d.model_ok ? 'ok' : 'err');
    ['OPEN_HAND','FIST','PEACE'].forEach(g=>{
      const el = document.getElementById('g-'+g);
      if(el) el.classList.toggle('active', d.gesture===g);
    });
  }).catch(()=>{});
}, 500);
</script>
</body>
</html>'''


if __name__ == '__main__':
    print("=" * 50)
    print("Gesture Control - TFLite Hand Skeleton")
    print(f"TFLite:  {'OK' if TFLITE_OK else 'NOT INSTALLED'}")
    print(f"Models:  {'OK' if model_ok else 'NOT FOUND'}")
    print(f"Camera:  {'OK' if cam_ok else 'FAILED'}")
    print("Open: http://<PI_IP>:5002")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5002, threaded=True)