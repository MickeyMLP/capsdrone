"""
Gesture Control Module - OpenCV Only Version
No MediaPipe needed! Uses skin color detection + convexity defects.

Gestures:
  Open hand in TOP zone    → Fly Up
  Open hand in BOTTOM zone → Fly Down
  Open hand in LEFT zone   → Move Left
  Open hand in RIGHT zone  → Move Right
  FIST (0-1 fingers)       → Hover / Stop

Author: Sue Sha
"""

import cv2
import numpy as np
import time
import math
from flask import Flask, Response, jsonify
from datetime import datetime

app = Flask(__name__)

# ─────────────────────────────────────────────
# DRONE CONTROLLER (Simulation)
# ─────────────────────────────────────────────

class DroneController:
    def __init__(self):
        self.altitude = 1.5
        self.position_x = 0.0
        self.position_y = 0.0
        self.mode = 'GUIDED'
        self.armed = True
        self.last_command = 'NONE'
        print("Simulation mode active")

    def fly_up(self):
        self.altitude = min(self.altitude + 0.5, 10.0)
        self._log("FLY UP")

    def fly_down(self):
        self.altitude = max(self.altitude - 0.5, 0.0)
        self._log("FLY DOWN")

    def move_left(self):
        self.position_y -= 0.5
        self._log("MOVE LEFT")

    def move_right(self):
        self.position_y += 0.5
        self._log("MOVE RIGHT")

    def hover(self):
        self._log("HOVER")

    def land(self):
        self.altitude = 0.0
        self.mode = 'LANDED'
        self.armed = False
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
            'position_x': round(self.position_x, 2),
            'position_y': round(self.position_y, 2),
            'last_command': self.last_command,
            'simulation': True
        }


# ─────────────────────────────────────────────
# GESTURE DETECTOR (OpenCV only)
# ─────────────────────────────────────────────

class GestureDetector:
    def __init__(self):
        self.camera = None
        self.fps = 0
        self.current_gesture = 'NONE'
        self.finger_count = 0

        # Gesture hold logic
        self.pending_gesture = 'NONE'
        self.gesture_hold_start = 0
        self.gesture_hold_required = 1.0  # hold 1 sec before triggering
        self.last_trigger_time = 0
        self.cooldown = 1.5

        # Skin color HSV range - works for most skin tones indoors
        self.skin_lower = np.array([0, 20, 70], dtype=np.uint8)
        self.skin_upper = np.array([20, 255, 255], dtype=np.uint8)

    def init_camera(self, camera_id=0):
        self.camera = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 15)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok = self.camera.isOpened()
        print(f"Camera: {'OK' if ok else 'FAILED'}")
        return ok

    def count_fingers(self, contour, frame_shape):
        """Count extended fingers using convexity defects."""
        if contour is None or len(contour) < 5:
            return 0, 0.5, 0.5

        h, w = frame_shape[:2]
        x, y, bw, bh = cv2.boundingRect(contour)
        wrist_x = (x + bw // 2) / w
        wrist_y = (y + bh) / h  # bottom of hand bounding box = wrist

        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is None or len(hull) < 3:
            return 0, wrist_x, wrist_y

        try:
            defects = cv2.convexityDefects(contour, hull)
        except:
            return 0, wrist_x, wrist_y

        if defects is None:
            return 0, wrist_x, wrist_y

        finger_gaps = 0
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])

            b = math.dist(far, start)
            c = math.dist(far, end)
            a = math.dist(start, end)
            depth = d / 256.0

            if b * c == 0:
                continue

            angle = math.acos(max(-1, min(1, (b**2 + c**2 - a**2) / (2 * b * c))))

            if angle < math.pi / 2 and depth > 20:
                finger_gaps += 1

        fingers = min(finger_gaps + 1, 5)
        return fingers, wrist_x, wrist_y

    def detect_hand(self, frame):
        """Detect hand via skin color. Returns (contour, mask)."""
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.skin_lower, self.skin_upper)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, mask

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 8000:
            return None, mask

        return largest, mask

    def classify_gesture(self, fingers, wrist_x, wrist_y):
        """Return gesture string from finger count + wrist position."""
        if fingers <= 1:
            return 'FIST'
        if fingers >= 3:
            if wrist_y < 0.35:
                return 'HAND_UP'
            elif wrist_y > 0.65:
                return 'HAND_DOWN'
            elif wrist_x < 0.30:
                return 'HAND_LEFT'
            elif wrist_x > 0.70:
                return 'HAND_RIGHT'
            else:
                return 'OPEN_CENTER'
        return 'NONE'

    def process_frame(self, drone):
        ret, frame = self.camera.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        contour, mask = self.detect_hand(frame)
        gesture = 'NONE'
        fingers = 0
        wrist_x, wrist_y = 0.5, 0.5
        command_triggered = False

        if contour is not None:
            fingers, wrist_x, wrist_y = self.count_fingers(contour, frame.shape)
            gesture = self.classify_gesture(fingers, wrist_x, wrist_y)

            cv2.drawContours(frame, [contour], -1, (0, 255, 120), 2)
            hull_pts = cv2.convexHull(contour)
            cv2.drawContours(frame, [hull_pts], -1, (0, 200, 255), 1)
            cv2.circle(frame, (int(wrist_x * w), int(wrist_y * h)), 10, (255, 100, 0), -1)

            now = time.time()
            if gesture != self.pending_gesture:
                self.pending_gesture = gesture
                self.gesture_hold_start = now
            else:
                hold = now - self.gesture_hold_start
                cooldown_ok = (now - self.last_trigger_time) > self.cooldown
                if hold >= self.gesture_hold_required and cooldown_ok and gesture not in ['NONE', 'OPEN_CENTER']:
                    self._execute(gesture, drone)
                    self.last_trigger_time = now
                    command_triggered = True

        self.current_gesture = gesture
        self.finger_count = fingers
        frame = self._draw_overlay(frame, gesture, fingers, wrist_x, wrist_y, command_triggered, drone)
        return frame

    def _execute(self, gesture, drone):
        print(f"[GESTURE] {gesture}")
        if gesture == 'HAND_UP':     drone.fly_up()
        elif gesture == 'HAND_DOWN': drone.fly_down()
        elif gesture == 'HAND_LEFT': drone.move_left()
        elif gesture == 'HAND_RIGHT':drone.move_right()
        elif gesture == 'FIST':      drone.hover()

    def _draw_overlay(self, frame, gesture, fingers, wrist_x, wrist_y, triggered, drone):
        h, w = frame.shape[:2]
        overlay = frame.copy()

        cv2.rectangle(overlay, (0, 0), (w, int(h*0.35)), (0, 200, 0), -1)
        cv2.rectangle(overlay, (0, int(h*0.65)), (w, h), (0, 0, 200), -1)
        cv2.rectangle(overlay, (0, int(h*0.35)), (int(w*0.30), int(h*0.65)), (0, 120, 255), -1)
        cv2.rectangle(overlay, (int(w*0.70), int(h*0.35)), (w, int(h*0.65)), (180, 0, 255), -1)
        frame = cv2.addWeighted(overlay, 0.13, frame, 0.87, 0)

        cv2.putText(frame, "FLY UP",    (w//2-40, 30),   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,80), 2)
        cv2.putText(frame, "FLY DOWN",  (w//2-55, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60,60,255), 2)
        cv2.putText(frame, "LEFT",      (8, h//2),        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,160,255), 2)
        cv2.putText(frame, "RIGHT",     (w-70, h//2),     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,0,255), 2)
        cv2.putText(frame, "FIST=HOVER",(w//2-65, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

        g_colors = {
            'HAND_UP':(0,255,80),'HAND_DOWN':(60,60,255),
            'HAND_LEFT':(0,160,255),'HAND_RIGHT':(200,0,255),
            'FIST':(0,220,255),'OPEN_CENTER':(180,180,180),'NONE':(80,80,80)
        }
        g_color = g_colors.get(gesture, (80,80,80))
        bar_color = (0, 150, 0) if triggered else (0, 0, 0)
        cv2.rectangle(frame, (0, 0), (w, 40), bar_color, -1)
        cv2.putText(frame, f"GESTURE: {gesture}  FINGERS: {fingers}",
                   (8, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.65, g_color, 2)
        cv2.putText(frame, f"CMD:{drone.last_command} ALT:{drone.altitude:.1f}m FPS:{self.fps:.0f}",
                   (w-310, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)

        # Hold progress bar
        now = time.time()
        if gesture not in ['NONE', 'OPEN_CENTER']:
            hold = min(now - self.gesture_hold_start, self.gesture_hold_required)
            progress = int((hold / self.gesture_hold_required) * (w - 20))
            cv2.rectangle(frame, (10, h-8), (w-10, h-3), (40,40,40), -1)
            cv2.rectangle(frame, (10, h-8), (10+progress, h-3), g_color, -1)

        return frame


# ─────────────────────────────────────────────
# INIT
# ─────────────────────────────────────────────

drone = DroneController()
detector = GestureDetector()
cam_ok = detector.init_camera(0)
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
    return jsonify({'gesture': detector.current_gesture, 'fingers': detector.finger_count,
                    'fps': round(detector.fps, 1), 'drone': drone.get_status()})

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
  .sim-badge{background:rgba(255,140,0,.15);border:1px solid var(--orange);color:var(--orange);padding:3px 10px;border-radius:2px;font-family:'Share Tech Mono',monospace;font-size:11px}
  .layout{display:grid;grid-template-columns:1fr 320px;gap:20px;padding:20px}
  .video-panel{background:var(--surface);border:1px solid var(--border);border-radius:4px;overflow:hidden;position:relative}
  .video-panel::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:var(--green)}
  .panel-title{padding:10px 16px;border-bottom:1px solid var(--border);font-family:'Share Tech Mono',monospace;font-size:12px;color:var(--green);letter-spacing:2px}
  .video-wrapper{background:#000;aspect-ratio:4/3}
  .video-wrapper img{width:100%;height:100%;object-fit:contain;display:block}
  .right-panel{display:flex;flex-direction:column;gap:16px}
  .card{background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:16px}
  .card-title{font-family:'Share Tech Mono',monospace;font-size:11px;color:var(--dim);letter-spacing:2px;margin-bottom:12px}
  .status-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
  .status-item{background:#0a0f16;border:1px solid var(--border);padding:8px 10px;border-radius:2px}
  .status-label{font-size:10px;color:var(--dim);font-family:'Share Tech Mono',monospace}
  .status-val{font-size:18px;font-weight:700;color:var(--green);font-family:'Share Tech Mono',monospace}
  .status-val.cmd{font-size:13px;color:var(--blue)}
  .gesture-list{display:flex;flex-direction:column;gap:6px}
  .gesture-item{display:flex;align-items:center;gap:10px;padding:7px 10px;border-radius:2px;border:1px solid var(--border);background:#0a0f16;transition:all .2s}
  .gesture-item.active{border-color:var(--green);background:rgba(0,255,136,.08)}
  .gesture-dot{width:10px;height:10px;border-radius:50%;background:var(--dim);flex-shrink:0}
  .gesture-item.active .gesture-dot{background:var(--green);box-shadow:0 0 8px var(--green)}
  .gesture-name{font-family:'Share Tech Mono',monospace;font-size:11px}
  .gesture-cmd{font-size:12px;color:var(--dim);margin-left:auto}
  .gesture-item.active .gesture-cmd{color:var(--green)}
  .btn-row{display:flex;gap:8px}
  .btn{flex:1;padding:10px;border:1px solid var(--border);background:transparent;color:var(--text);font-family:'Rajdhani',sans-serif;font-size:13px;font-weight:700;letter-spacing:1px;cursor:pointer;border-radius:2px;transition:all .15s;text-transform:uppercase}
  .btn.green:hover{border-color:var(--green);color:var(--green);background:rgba(0,255,136,.08)}
  .btn.red:hover{border-color:var(--red);color:var(--red);background:rgba(255,51,85,.08)}
  .alt-wrap{height:100px;display:flex;align-items:flex-end;gap:8px;margin-top:8px}
  .alt-bg{flex:1;height:100%;background:#0a0f16;border:1px solid var(--border);border-radius:2px;position:relative;overflow:hidden}
  .alt-fill{position:absolute;bottom:0;left:0;right:0;background:linear-gradient(to top,var(--green),rgba(0,255,136,.3));transition:height .5s;border-radius:2px}
  .alt-label{font-family:'Share Tech Mono',monospace;font-size:11px;color:var(--dim);writing-mode:vertical-rl}
  .tip{background:rgba(255,140,0,.08);border:1px solid rgba(255,140,0,.3);border-radius:4px;padding:10px;font-size:12px;color:var(--orange);line-height:1.6}
</style>
</head>
<body>
<header>
  <div class="logo">GESTURE_CONTROL_SYS</div>
  <div class="sim-badge">SIMULATION MODE</div>
</header>
<div class="layout">
  <div class="video-panel">
    <div class="panel-title">GESTURE CAMERA — Show open hand in colored zones</div>
    <div class="video-wrapper"><img src="/video"></div>
  </div>
  <div class="right-panel">
    <div class="card">
      <div class="card-title">// DRONE STATUS</div>
      <div class="status-grid">
        <div class="status-item"><div class="status-label">MODE</div><div class="status-val" id="d-mode">--</div></div>
        <div class="status-item"><div class="status-label">ALTITUDE</div><div class="status-val" id="d-alt">0.0m</div></div>
        <div class="status-item"><div class="status-label">POS X</div><div class="status-val" id="d-x">0.0</div></div>
        <div class="status-item"><div class="status-label">POS Y</div><div class="status-val" id="d-y">0.0</div></div>
      </div>
      <div class="status-item" style="margin-top:8px"><div class="status-label">LAST COMMAND</div><div class="status-val cmd" id="d-cmd">NONE</div></div>
      <div class="alt-wrap">
        <div class="alt-label">ALT</div>
        <div class="alt-bg"><div class="alt-fill" id="alt-bar" style="height:15%"></div></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">// GESTURE REFERENCE</div>
      <div class="gesture-list">
        <div class="gesture-item" id="g-HAND_UP"><div class="gesture-dot"></div><span class="gesture-name">OPEN HAND — TOP ZONE</span><span class="gesture-cmd">FLY UP</span></div>
        <div class="gesture-item" id="g-HAND_DOWN"><div class="gesture-dot"></div><span class="gesture-name">OPEN HAND — BOTTOM</span><span class="gesture-cmd">FLY DOWN</span></div>
        <div class="gesture-item" id="g-HAND_LEFT"><div class="gesture-dot"></div><span class="gesture-name">OPEN HAND — LEFT</span><span class="gesture-cmd">MOVE LEFT</span></div>
        <div class="gesture-item" id="g-HAND_RIGHT"><div class="gesture-dot"></div><span class="gesture-name">OPEN HAND — RIGHT</span><span class="gesture-cmd">MOVE RIGHT</span></div>
        <div class="gesture-item" id="g-FIST"><div class="gesture-dot"></div><span class="gesture-name">FIST — ANY ZONE</span><span class="gesture-cmd">HOVER</span></div>
      </div>
    </div>
    <div class="card">
      <div class="card-title">// MANUAL CONTROLS</div>
      <div class="btn-row">
        <button class="btn green" onclick="fetch('/takeoff')">TAKEOFF</button>
        <button class="btn red" onclick="fetch('/land')">LAND</button>
      </div>
    </div>
    <div class="card">
      <div class="tip">TIP: Hold gesture steady for 1 second to trigger command. Use good lighting and plain background for best detection!</div>
    </div>
  </div>
</div>
<script>
setInterval(()=>{
  fetch('/status').then(r=>r.json()).then(d=>{
    document.getElementById('d-mode').textContent=d.drone.mode||'--';
    document.getElementById('d-alt').textContent=(d.drone.altitude||0).toFixed(1)+'m';
    document.getElementById('d-x').textContent=(d.drone.position_x||0).toFixed(1);
    document.getElementById('d-y').textContent=(d.drone.position_y||0).toFixed(1);
    document.getElementById('d-cmd').textContent=d.drone.last_command||'NONE';
    document.getElementById('alt-bar').style.height=Math.min((d.drone.altitude/10)*100,100)+'%';
    ['HAND_UP','HAND_DOWN','HAND_LEFT','HAND_RIGHT','FIST'].forEach(g=>{
      const el=document.getElementById('g-'+g);
      if(el)el.classList.toggle('active',d.gesture===g);
    });
  }).catch(()=>{});
},500);
</script>
</body>
</html>'''


if __name__ == '__main__':
    print("="*50)
    print("Gesture Control - OpenCV Mode")
    print(f"Camera: {'OK' if cam_ok else 'FAILED'}")
    print("Open: http://172.20.10.2:5002")
    print("="*50)
    app.run(host='0.0.0.0', port=5002, threaded=True)