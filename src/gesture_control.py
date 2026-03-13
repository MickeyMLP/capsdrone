"""
Gesture Control - OpenCV DNN Hand Skeleton
Uses Caffe hand keypoint model (22 landmarks) for accurate skeleton detection.

3 gestures only:
  OPEN HAND (fingers up) = FLY UP
  FIST (fingers down)    = FLY DOWN  
  PEACE/2 fingers        = HOVER/STOP

Run setup first:
  python3 gesture_control.py --setup

Then run normally:
  python3 gesture_control.py

Author: Sue Sha
"""

import cv2
import numpy as np
import time
import sys
import os
from flask import Flask, Response, jsonify
from datetime import datetime

app = Flask(__name__)

# ─────────────────────────────────────────────
# MODEL SETUP
# ─────────────────────────────────────────────

MODEL_DIR = os.path.join(os.path.dirname(__file__), 'hand_model')
PROTO_FILE = os.path.join(MODEL_DIR, 'pose_deploy.prototxt')
WEIGHTS_FILE = os.path.join(MODEL_DIR, 'pose_iter_102000.caffemodel')

# 22 hand keypoints from the Caffe model
# 0=Wrist, 1-4=Thumb, 5-8=Index, 9-12=Middle, 13-16=Ring, 17-20=Pinky, 21=Palm
KEYPOINT_NAMES = [
    'Wrist',
    'ThumbMetacarpal', 'ThumbProximal', 'ThumbMiddle', 'ThumbTip',
    'IndexMetacarpal', 'IndexProximal', 'IndexMiddle', 'IndexTip',
    'MiddleMetacarpal', 'MiddleProximal', 'MiddleMiddle', 'MiddleTip',
    'RingMetacarpal', 'RingProximal', 'RingMiddle', 'RingTip',
    'PinkyMetacarpal', 'PinkyProximal', 'PinkyMiddle', 'PinkyTip',
    'Palm'
]

# Finger tip and base keypoint indices
FINGER_TIPS  = [4, 8, 12, 16, 20]   # thumb, index, middle, ring, pinky tips
FINGER_BASES = [2, 5,  9, 13, 17]   # corresponding base joints

# Skeleton connections to draw
POSE_PAIRS = [
    [0, 1],[1, 2],[2, 3],[3, 4],       # thumb
    [0, 5],[5, 6],[6, 7],[7, 8],       # index
    [0, 9],[9,10],[10,11],[11,12],     # middle
    [0,13],[13,14],[14,15],[15,16],    # ring
    [0,17],[17,18],[18,19],[19,20],    # pinky
    [0,21],[5,9],[9,13],[13,17]        # palm connections
]


def setup_model():
    """Download model files if not present."""
    os.makedirs(MODEL_DIR, exist_ok=True)

    proto_url = "https://raw.githubusercontent.com/CMU-Perceptual-Computing-Lab/openpose/master/models/hand/pose_deploy.prototxt"
    weights_url = "https://www.dropbox.com/s/n7hbavl44f7l5ny/pose_iter_102000.caffemodel"

    if not os.path.exists(PROTO_FILE):
        print("Downloading prototxt...")
        import urllib.request
        urllib.request.urlretrieve(proto_url, PROTO_FILE)
        print("Done!")

    if not os.path.exists(WEIGHTS_FILE):
        print("Downloading caffemodel (~10MB)...")
        print("If this fails, manually download from:")
        print(weights_url)
        print(f"Save to: {WEIGHTS_FILE}")
        try:
            import urllib.request
            urllib.request.urlretrieve(weights_url, WEIGHTS_FILE)
            print("Done!")
        except Exception as e:
            print(f"Download failed: {e}")
            print("Please download manually.")
            return False
    return True


def model_exists():
    return os.path.exists(PROTO_FILE) and os.path.exists(WEIGHTS_FILE)


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
# HAND SKELETON DETECTOR
# ─────────────────────────────────────────────

class HandSkeletonDetector:
    def __init__(self):
        self.camera = None
        self.net = None
        self.fps = 0
        self.current_gesture = 'NONE'
        self.keypoints = []

        # Gesture hold logic
        self.pending_gesture = 'NONE'
        self.gesture_hold_start = 0
        self.gesture_hold_required = 1.0
        self.last_trigger_time = 0
        self.cooldown = 1.5

        # Input size for model
        self.in_width = 368
        self.in_height = 368
        self.threshold = 0.15  # keypoint confidence threshold

    def load_model(self):
        if not model_exists():
            print("Model files not found! Run: python3 gesture_control.py --setup")
            return False
        try:
            self.net = cv2.dnn.readNetFromCaffe(PROTO_FILE, WEIGHTS_FILE)
            # Use CPU (OpenCV DNN default)
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            print("Hand skeleton model loaded!")
            return True
        except Exception as e:
            print(f"Model load failed: {e}")
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

    def detect_keypoints(self, frame):
        """Run DNN inference and return list of (x,y) or None per keypoint."""
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0/255,
            (self.in_width, self.in_height),
            (0, 0, 0), swapRB=False, crop=False
        )
        self.net.setInput(blob)
        output = self.net.forward()  # shape: (1, 22, H/8, W/8)

        points = []
        for i in range(21):  # 21 keypoints (skip 21=palm center)
            heatmap = output[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatmap)
            x = int((point[0] / output.shape[3]) * w)
            y = int((point[1] / output.shape[2]) * h)
            if conf > self.threshold:
                points.append((x, y))
            else:
                points.append(None)
        return points

    def count_fingers_from_keypoints(self, points):
        """
        Count extended fingers using tip vs base keypoint positions.
        A finger is UP if its tip Y is significantly above its base Y.
        Lower Y = higher on screen.
        """
        if not points or all(p is None for p in points):
            return 0

        fingers_up = 0
        for tip_idx, base_idx in zip(FINGER_TIPS, FINGER_BASES):
            tip = points[tip_idx]
            base = points[base_idx]
            if tip is None or base is None:
                continue
            # Tip is above base = finger extended
            if tip[1] < base[1] - 15:  # 15px threshold
                fingers_up += 1

        return fingers_up

    def classify_gesture(self, fingers_up, points):
        """
        3 gestures:
          5 fingers up = OPEN HAND = FLY UP
          0 fingers up = FIST      = FLY DOWN
          2 fingers up = PEACE     = HOVER
        """
        if fingers_up == 5:
            return 'OPEN_HAND'
        elif fingers_up == 0:
            return 'FIST'
        elif fingers_up == 2:
            return 'PEACE'
        else:
            return 'NONE'

    def draw_skeleton(self, frame, points):
        """Draw hand skeleton on frame."""
        # Draw connections
        for pair in POSE_PAIRS:
            a, b = pair
            if a < len(points) and b < len(points):
                pt_a = points[a]
                pt_b = points[b]
                if pt_a and pt_b:
                    cv2.line(frame, pt_a, pt_b, (0, 255, 200), 2)

        # Draw keypoints
        for i, pt in enumerate(points):
            if pt:
                # Tips in different color
                if i in FINGER_TIPS:
                    cv2.circle(frame, pt, 7, (0, 80, 255), -1)
                elif i == 0:  # wrist
                    cv2.circle(frame, pt, 8, (255, 200, 0), -1)
                else:
                    cv2.circle(frame, pt, 5, (0, 200, 255), -1)

        return frame

    def process_frame(self, drone):
        ret, frame = self.camera.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        gesture = 'NONE'
        fingers_up = 0
        command_triggered = False

        if self.net is not None:
            points = self.detect_keypoints(frame)
            fingers_up = self.count_fingers_from_keypoints(points)
            gesture = self.classify_gesture(fingers_up, points)

            # Draw skeleton if any keypoints found
            if any(p is not None for p in points):
                frame = self.draw_skeleton(frame, points)

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
        frame = self._draw_overlay(frame, gesture, fingers_up, command_triggered, drone)
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

        # Top bar
        bar_color = (0, 120, 0) if triggered else (0, 0, 0)
        cv2.rectangle(frame, (0, 0), (w, 44), bar_color, -1)
        cv2.putText(frame, f"GESTURE: {gesture}  FINGERS: {fingers}",
                   (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, g_color, 2)
        cv2.putText(frame, f"CMD:{drone.last_command} ALT:{drone.altitude:.1f}m FPS:{self.fps:.0f}",
                   (w - 310, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        # Gesture guide at bottom
        cv2.rectangle(frame, (0, h-36), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "OPEN HAND=UP   FIST=DOWN   PEACE(2 fingers)=HOVER",
                   (8, h-12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

        # Hold progress bar
        now = time.time()
        if gesture != 'NONE':
            hold = min(now - self.gesture_hold_start, self.gesture_hold_required)
            progress = int((hold / self.gesture_hold_required) * (w - 20))
            cv2.rectangle(frame, (10, h-8), (w-10, h-3), (40, 40, 40), -1)
            cv2.rectangle(frame, (10, h-8), (10+progress, h-3), g_color, -1)

        return frame


# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────

drone = DroneController()
detector = HandSkeletonDetector()
cam_ok = detector.init_camera(0)
model_ok = detector.load_model()
drone.takeoff()

if not model_ok:
    print("\nWARNING: Model not loaded! Run setup first:")
    print("  python3 gesture_control.py --setup\n")


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
    return jsonify({'gesture': detector.current_gesture,
                    'fps': round(detector.fps, 1),
                    'drone': drone.get_status(),
                    'model_loaded': model_ok})

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
  header{padding:14px 24px;border-bottom:1px solid var(--border);background:var(--surface);display:flex;align-items:center;justify-content:space-between;gap:12px}
  .logo{font-family:'Share Tech Mono',monospace;font-size:16px;color:var(--green);letter-spacing:2px}
  .badge{padding:3px 10px;border-radius:2px;font-family:'Share Tech Mono',monospace;font-size:11px}
  .sim{background:rgba(255,140,0,.15);border:1px solid var(--orange);color:var(--orange)}
  .model-ok{background:rgba(0,255,136,.1);border:1px solid var(--green);color:var(--green)}
  .model-err{background:rgba(255,51,85,.1);border:1px solid var(--red);color:var(--red)}
  .layout{display:grid;grid-template-columns:1fr 300px;gap:20px;padding:20px}
  .video-panel{background:var(--surface);border:1px solid var(--border);border-radius:4px;overflow:hidden;position:relative}
  .video-panel::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--green)}
  .panel-title{padding:10px 16px;border-bottom:1px solid var(--border);font-family:'Share Tech Mono',monospace;font-size:12px;color:var(--green);letter-spacing:2px}
  .video-wrapper{background:#000;aspect-ratio:4/3}
  .video-wrapper img{width:100%;height:100%;object-fit:contain;display:block}
  .right-panel{display:flex;flex-direction:column;gap:16px}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:16px}
  .card-title{font-family:'Share Tech Mono',monospace;font-size:11px;color:var(--dim);letter-spacing:2px;margin-bottom:12px}
  .status-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .stat{background:#0a0f16;border:1px solid var(--border);padding:8px 10px;border-radius:2px}
  .stat-label{font-size:10px;color:var(--dim);font-family:'Share Tech Mono',monospace}
  .stat-val{font-size:20px;font-weight:700;color:var(--green);font-family:'Share Tech Mono',monospace}
  .stat-val.sm{font-size:13px;color:var(--blue)}
  .gesture-cards{display:flex;flex-direction:column;gap:8px}
  .g-card{padding:12px;border-radius:2px;border:1px solid var(--border);background:#0a0f16;transition:all .2s;display:flex;align-items:center;gap:12px}
  .g-card.active{border-color:var(--green);background:rgba(0,255,136,.08)}
  .g-icon{font-size:24px;width:36px;text-align:center}
  .g-info{flex:1}
  .g-name{font-family:'Share Tech Mono',monospace;font-size:12px;color:var(--text)}
  .g-cmd{font-size:11px;color:var(--dim);margin-top:2px}
  .g-card.active .g-cmd{color:var(--green)}
  .dot{width:10px;height:10px;border-radius:50%;background:var(--dim);flex-shrink:0}
  .g-card.active .dot{background:var(--green);box-shadow:0 0 8px var(--green)}
  .alt-wrap{height:90px;display:flex;gap:8px;align-items:flex-end;margin-top:10px}
  .alt-bg{flex:1;height:100%;background:#0a0f16;border:1px solid var(--border);border-radius:2px;position:relative;overflow:hidden}
  .alt-fill{position:absolute;bottom:0;left:0;right:0;background:linear-gradient(to top,var(--green),rgba(0,255,136,.3));transition:height .5s;border-radius:2px}
  .alt-label{font-family:'Share Tech Mono',monospace;font-size:10px;color:var(--dim);writing-mode:vertical-rl}
  .btn-row{display:flex;gap:8px}
  .btn{flex:1;padding:10px;border:1px solid var(--border);background:transparent;color:var(--text);font-family:'Rajdhani',sans-serif;font-size:13px;font-weight:700;letter-spacing:1px;cursor:pointer;border-radius:2px;transition:all .15s;text-transform:uppercase}
  .btn.g:hover{border-color:var(--green);color:var(--green);background:rgba(0,255,136,.08)}
  .btn.r:hover{border-color:var(--red);color:var(--red);background:rgba(255,51,85,.08)}
</style>
</head>
<body>
<header>
  <div class="logo">GESTURE_CONTROL_SYS</div>
  <div style="display:flex;gap:8px">
    <div class="badge sim">SIMULATION</div>
    <div class="badge" id="model-badge">CHECKING MODEL...</div>
  </div>
</header>
<div class="layout">
  <div class="video-panel">
    <div class="panel-title">HAND SKELETON CAMERA</div>
    <div class="video-wrapper"><img src="/video"></div>
  </div>
  <div class="right-panel">
    <div class="card">
      <div class="card-title">// DRONE STATUS</div>
      <div class="status-grid">
        <div class="stat"><div class="stat-label">MODE</div><div class="stat-val" id="d-mode">--</div></div>
        <div class="stat"><div class="stat-label">ALTITUDE</div><div class="stat-val" id="d-alt">0.0m</div></div>
      </div>
      <div class="stat" style="margin-top:8px"><div class="stat-label">LAST COMMAND</div><div class="stat-val sm" id="d-cmd">NONE</div></div>
      <div class="alt-wrap">
        <div class="alt-label">ALT</div>
        <div class="alt-bg"><div class="alt-fill" id="alt-bar" style="height:15%"></div></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">// GESTURES (hold 1 sec)</div>
      <div class="gesture-cards">
        <div class="g-card" id="g-OPEN_HAND">
          <div class="dot"></div>
          <div class="g-info">
            <div class="g-name">OPEN HAND (5 fingers)</div>
            <div class="g-cmd">FLY UP</div>
          </div>
        </div>
        <div class="g-card" id="g-FIST">
          <div class="dot"></div>
          <div class="g-info">
            <div class="g-name">FIST (0 fingers)</div>
            <div class="g-cmd">FLY DOWN</div>
          </div>
        </div>
        <div class="g-card" id="g-PEACE">
          <div class="dot"></div>
          <div class="g-info">
            <div class="g-name">PEACE (2 fingers)</div>
            <div class="g-cmd">HOVER / STOP</div>
          </div>
        </div>
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
    document.getElementById('d-mode').textContent=d.drone.mode||'--';
    document.getElementById('d-alt').textContent=(d.drone.altitude||0).toFixed(1)+'m';
    document.getElementById('d-cmd').textContent=d.drone.last_command||'NONE';
    document.getElementById('alt-bar').style.height=Math.min((d.drone.altitude/10)*100,100)+'%';
    const mb=document.getElementById('model-badge');
    mb.textContent=d.model_loaded?'MODEL OK':'MODEL NOT LOADED';
    mb.className='badge '+(d.model_loaded?'model-ok':'model-err');
    ['OPEN_HAND','FIST','PEACE'].forEach(g=>{
      const el=document.getElementById('g-'+g);
      if(el)el.classList.toggle('active',d.gesture===g);
    });
  }).catch(()=>{});
},500);
</script>
</body>
</html>'''


if __name__ == '__main__':
    if '--setup' in sys.argv:
        print("Setting up hand skeleton model...")
        setup_model()
        sys.exit(0)

    print("="*50)
    print("Gesture Control - Hand Skeleton DNN Mode")
    print(f"Camera: {'OK' if cam_ok else 'FAILED'}")
    print(f"Model:  {'OK' if model_ok else 'NOT LOADED - run with --setup'}")
    print("Open: http://172.20.10.2:5002")
    print("="*50)
    app.run(host='0.0.0.0', port=5002, threaded=True)