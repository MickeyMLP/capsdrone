"""
Gesture Control - MediaPipe Hand Skeleton
Accurate 21-landmark hand detection for drone control.

3 gestures:
  OPEN HAND (5 fingers) = FLY UP
  FIST      (0 fingers) = FLY DOWN
  PEACE     (2 fingers) = HOVER / STOP

Author: Sue Sha
"""

import cv2
import numpy as np
import time
from flask import Flask, Response, jsonify
from datetime import datetime

# MediaPipe import
try:
    import mediapipe as mp
    MP_OK = True
    print("MediaPipe loaded OK")
except ImportError:
    MP_OK = False
    print("MediaPipe not found!")

app = Flask(__name__)

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
# GESTURE DETECTOR (MediaPipe)
# ─────────────────────────────────────────────

class GestureDetector:
    def __init__(self):
        self.camera = None
        self.fps = 0
        self.current_gesture = 'NONE'
        self.finger_count = 0

        # Gesture hold + cooldown
        self.pending_gesture = 'NONE'
        self.gesture_hold_start = 0
        self.gesture_hold_required = 1.0  # hold 1 sec
        self.last_trigger_time = 0
        self.cooldown = 1.5

        if MP_OK:
            self.mp_hands = mp.solutions.hands
            self.mp_draw = mp.solutions.drawing_utils
            self.mp_styles = mp.solutions.drawing_styles
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.6
            )
        else:
            self.hands = None

    def init_camera(self, camera_id=0):
        self.camera = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 15)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        ok = self.camera.isOpened()
        print(f"Camera: {'OK' if ok else 'FAILED'}")
        return ok

    def count_fingers(self, hand_landmarks):
        """
        Count extended fingers using MediaPipe 21 landmarks.
        Tip landmark Y < PIP landmark Y = finger is up.
        Thumb uses X comparison instead.
        """
        lm = hand_landmarks.landmark
        tips  = [4, 8, 12, 16, 20]
        pips  = [3, 6, 10, 14, 18]  # one joint below tip

        count = 0

        # Thumb: compare x (horizontal)
        if lm[4].x < lm[3].x:
            count += 1

        # Other 4 fingers: compare y (vertical — lower y = higher on screen)
        for tip, pip in zip(tips[1:], pips[1:]):
            if lm[tip].y < lm[pip].y:
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

    def process_frame(self, drone):
        ret, frame = self.camera.read()
        if not ret:
            return None

        frame = cv2.flip(frame, 1)
        gesture = 'NONE'
        fingers = 0
        command_triggered = False

        if self.hands:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = self.hands.process(rgb)
            rgb.flags.writeable = True

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw full skeleton with MediaPipe style
                    self.mp_draw.draw_landmarks(
                        frame,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_styles.get_default_hand_landmarks_style(),
                        self.mp_styles.get_default_hand_connections_style()
                    )

                    fingers = self.count_fingers(hand_landmarks)
                    gesture = self.classify_gesture(fingers)

                    # Gesture hold + cooldown logic
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

        # Top bar — flashes green when command triggers
        bar_color = (0, 130, 0) if triggered else (0, 0, 0)
        cv2.rectangle(frame, (0, 0), (w, 44), bar_color, -1)
        cv2.putText(frame, f"GESTURE: {gesture}  FINGERS: {fingers}",
                   (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, g_color, 2)
        cv2.putText(frame, f"CMD:{drone.last_command}  ALT:{drone.altitude:.1f}m  FPS:{self.fps:.0f}",
                   (w - 320, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)

        # Bottom guide
        cv2.rectangle(frame, (0, h - 36), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, "OPEN HAND = UP    FIST = DOWN    PEACE = HOVER",
                   (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

        # Hold progress bar
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
    return jsonify({
        'gesture': detector.current_gesture,
        'fingers': detector.finger_count,
        'fps': round(detector.fps, 1),
        'drone': drone.get_status(),
        'mediapipe': MP_OK
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
    <div class="badge" id="mp-badge">CHECKING...</div>
  </div>
</header>
<div class="layout">
  <div class="video-panel">
    <div class="panel-title">MEDIAPIPE HAND SKELETON</div>
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
    const mb = document.getElementById('mp-badge');
    mb.textContent = d.mediapipe ? 'MEDIAPIPE OK' : 'MEDIAPIPE ERROR';
    mb.className = 'badge ' + (d.mediapipe ? 'ok' : 'err');
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
    print("Gesture Control - MediaPipe Mode")
    print(f"MediaPipe: {'OK' if MP_OK else 'NOT INSTALLED'}")
    print(f"Camera:    {'OK' if cam_ok else 'FAILED'}")
    print("Open: http://<PI_IP>:5002")
    print("=" * 50)
    app.run(host='0.0.0.0', port=5002, threaded=True)