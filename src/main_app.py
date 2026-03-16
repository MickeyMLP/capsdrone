"""
CAPSDRONE - Unified Control System
Single Flask app combining:
  - Thermal Camera Detection (port 5000 → merged here)
  - Regular Camera Detection
  - Gesture Control (TFLite hand skeleton)

Auto-detects Python version and available modules.
Run with: python3 main_app.py (or python3.11 for gesture)

Author: Sue Sha
"""

import cv2
import numpy as np
import time
import sys
import os
import threading
from flask import Flask, Response, jsonify, request
from datetime import datetime

print(f"Python {sys.version}")

# ─── TFLite (needed for gesture, Python 3.11 only) ───
TFLITE_OK = False
if sys.version_info[:2] == (3, 11):
    try:
        from tflite_runtime.interpreter import Interpreter
        TFLITE_OK = True
        print("TFLite: OK")
    except ImportError:
        print("TFLite: not available")
else:
    print(f"TFLite: skipped (needs Python 3.11, running {sys.version_info[0]}.{sys.version_info[1]})")

app = Flask(__name__)

# ─── Global active mode ───
active_mode = "thermal"  # thermal | regular | gesture

# ─── Model paths ───
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_model")
PALM_MODEL = os.path.join(MODEL_DIR, "palm_detection.tflite")
LANDMARK_MODEL = os.path.join(MODEL_DIR, "hand_landmark.tflite")

# Hand skeleton constants
FINGER_TIPS = [4, 8, 12, 16, 20]
FINGER_PIPS = [3, 6, 10, 14, 18]
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]
FINGER_COLORS = [(255,100,100),(100,255,100),(100,150,255),(255,255,100),(255,100,255)]


# ════════════════════════════════════════════
# DRONE CONTROLLER (Simulation)
# ════════════════════════════════════════════

class DroneController:
    def __init__(self):
        self.altitude = 1.5
        self.mode = "GUIDED"
        self.armed = True
        self.last_command = "NONE"
        self.position_x = 0.0

    def fly_up(self):
        self.altitude = min(self.altitude + 0.5, 10.0)
        self._log("FLY UP")

    def fly_down(self):
        self.altitude = max(self.altitude - 0.5, 0.0)
        self._log("FLY DOWN")

    def move_forward(self):
        self.position_x += 0.5
        self._log("MOVE FORWARD")

    def move_backward(self):
        self.position_x -= 0.5
        self._log("MOVE BACKWARD")

    def land(self):
        self.altitude = 0.0
        self.mode = "LANDED"
        self._log("LAND")

    def takeoff(self):
        self.altitude = 1.5
        self.mode = "GUIDED"
        self.armed = True
        self._log("TAKEOFF")

    def _log(self, cmd):
        self.last_command = cmd
        print(f"[{datetime.now().strftime('%H:%M:%S')}] CMD: {cmd}")

    def get_status(self):
        return {
            "mode": self.mode,
            "armed": self.armed,
            "altitude": round(self.altitude, 2),
            "last_command": self.last_command,
            "position_x": round(self.position_x, 2),
        }

drone = DroneController()


# ════════════════════════════════════════════
# CAMERA MANAGER
# ════════════════════════════════════════════

class CameraManager:
    def __init__(self):
        self.cameras = {}  # id → VideoCapture
        self.lock = threading.Lock()

    def get(self, cam_id, width=640, height=480):
        with self.lock:
            if cam_id not in self.cameras or not self.cameras[cam_id].isOpened():
                cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                cap.set(cv2.CAP_PROP_FPS, 15)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                self.cameras[cam_id] = cap
            return self.cameras[cam_id]

    def release_all(self):
        with self.lock:
            for cap in self.cameras.values():
                cap.release()
            self.cameras.clear()

cam_mgr = CameraManager()


# ════════════════════════════════════════════
# THERMAL DETECTION
# ════════════════════════════════════════════

class ThermalDetector:
    def __init__(self):
        self.mode = "human"  # human | fire | hot
        self.fps = 0
        self.detections = 0

    def detect_human(self, gray):
        frame_min = int(np.min(gray))
        frame_max = int(np.max(gray))
        frame_range = frame_max - frame_min if frame_max > frame_min else 1
        lower = frame_min + int(frame_range * 0.40)
        upper = frame_min + int(frame_range * 0.75)
        mask = cv2.inRange(gray, lower, upper)
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 300:
                x,y,w,h = cv2.boundingRect(c)
                aspect = h/w if w > 0 else 0
                if 0.6 < aspect < 4.0:
                    cx,cy = x+w//2, y+h//2
                    pv = int(gray[cy,cx])
                    warmth = int((pv-frame_min)/frame_range*100)
                    detections.append({"bbox":(x,y,w,h),"label":f"HUMAN {warmth}%","color":(0,255,80)})
        return detections

    def detect_fire(self, gray):
        frame_max = int(np.max(gray))
        if frame_max < 180:
            return []
        frame_min = int(np.min(gray))
        frame_range = frame_max - frame_min if frame_max > frame_min else 1
        threshold = frame_min + int(frame_range * 0.95)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 100:
                x,y,w,h = cv2.boundingRect(c)
                cx,cy = x+w//2, y+h//2
                pv = int(gray[cy,cx])
                detections.append({"bbox":(x,y,w,h),"label":f"FIRE px:{pv}","color":(0,0,255)})
        return detections

    def detect_hot(self, gray):
        frame_min = int(np.min(gray))
        frame_max = int(np.max(gray))
        frame_range = frame_max - frame_min if frame_max > frame_min else 1
        threshold = frame_min + int(frame_range * 0.75)
        _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((4,4), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 200:
                x,y,w,h = cv2.boundingRect(c)
                cx,cy = x+w//2, y+h//2
                pv = int(gray[cy,cx])
                warmth = int((pv-frame_min)/frame_range*100)
                detections.append({"bbox":(x,y,w,h),"label":f"HOT {warmth}%","color":(0,165,255)})
        return detections

    def process_frame(self):
        cap = cam_mgr.get(2, 256, 192)
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.resize(frame, (512, 384))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thermal_colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

        if self.mode == "human":
            dets = self.detect_human(gray)
        elif self.mode == "fire":
            dets = self.detect_fire(gray)
        else:
            dets = self.detect_hot(gray)

        self.detections = len(dets)
        for d in dets:
            x,y,w,h = d["bbox"]
            cv2.rectangle(thermal_colored, (x,y), (x+w,y+h), d["color"], 2)
            cv2.putText(thermal_colored, d["label"], (x,y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, d["color"], 1)

        h_f, w_f = thermal_colored.shape[:2]
        cv2.rectangle(thermal_colored, (0,0), (w_f, 36), (0,0,0), -1)
        cv2.putText(thermal_colored, f"THERMAL: {self.mode.upper()}  DETECT:{self.detections}  FPS:{self.fps:.0f}",
                   (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,180), 1)
        return thermal_colored

thermal_detector = ThermalDetector()


# ════════════════════════════════════════════
# REGULAR CAMERA DETECTION
# ════════════════════════════════════════════

class RegularDetector:
    def __init__(self):
        self.mode = "person"
        self.fps = 0
        self.detections = 0
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.upper_body = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')

    def process_frame(self):
        cap = cam_mgr.get(0, 640, 480)
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_eq = clahe.apply(gray)
        dets = []

        if self.mode == "person":
            faces = self.face_cascade.detectMultiScale(gray_eq, 1.1, 8, minSize=(30,30))
            bodies = self.upper_body.detectMultiScale(gray_eq, 1.1, 8, minSize=(60,60))
            for (x,y,w,h) in faces:
                dets.append({"bbox":(x,y,w,h),"label":"FACE","color":(0,255,80)})
            for (x,y,w,h) in bodies:
                dets.append({"bbox":(x,y,w,h),"label":"PERSON","color":(0,200,255)})

        elif self.mode == "obstacle":
            blurred = cv2.GaussianBlur(gray, (5,5), 0)
            edges = cv2.Canny(blurred, 60, 180)
            kernel = np.ones((3,3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area > 5000:
                    x,y,w,h = cv2.boundingRect(c)
                    aspect = w/h if h > 0 else 0
                    if aspect < 6:
                        dets.append({"bbox":(x,y,w,h),"label":"OBSTACLE","color":(0,80,255)})

        elif self.mode == "landing":
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, np.array([5,100,100]), np.array([25,255,255]))
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                area = cv2.contourArea(c)
                if area > 2000:
                    x,y,w,h = cv2.boundingRect(c)
                    dets.append({"bbox":(x,y,w,h),"label":"LANDING PAD","color":(0,255,255)})

        self.detections = len(dets)
        for d in dets:
            x,y,w,h = d["bbox"]
            cv2.rectangle(frame, (x,y), (x+w,y+h), d["color"], 2)
            cv2.putText(frame, d["label"], (x,y-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, d["color"], 1)

        h_f, w_f = frame.shape[:2]
        cv2.rectangle(frame, (0,0), (w_f, 36), (0,0,0), -1)
        cv2.putText(frame, f"CAMERA: {self.mode.upper()}  DETECT:{self.detections}  FPS:{self.fps:.0f}",
                   (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,200,255), 1)
        return frame

regular_detector = RegularDetector()


# ════════════════════════════════════════════
# GESTURE DETECTOR (TFLite)
# ════════════════════════════════════════════

class GestureDetector:
    def __init__(self):
        self.palm_interpreter = None
        self.landmark_interpreter = None
        self.palm_size = 192
        self.lm_size = 224
        self.fps = 0
        self.current_gesture = "NONE"
        self.finger_count = 0
        self.pending_gesture = "NONE"
        self.gesture_hold_start = 0.0
        self.gesture_hold_required = 1.0
        self.last_trigger_time = 0.0
        self.cooldown = 1.5

    def load_models(self):
        if not TFLITE_OK:
            return False
        if not os.path.exists(PALM_MODEL) or not os.path.exists(LANDMARK_MODEL):
            print(f"Hand models not found in {MODEL_DIR}")
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
            self.palm_size = int(self.palm_input[0]["shape"][1])
            self.lm_size   = int(self.lm_input[0]["shape"][1])
            print("Gesture models loaded OK")
            return True
        except Exception as e:
            print(f"Gesture model error: {e}")
            return False

    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def _preprocess(self, img_rgb, input_details):
        dtype = input_details[0]["dtype"]
        arr = img_rgb.astype(np.float32) / 127.5 - 1.0 if dtype == np.float32 else img_rgb.astype(np.uint8)
        return np.expand_dims(arr, axis=0)

    def detect_palm(self, frame):
        if self.palm_interpreter is None:
            return None
        h, w = frame.shape[:2]
        img = cv2.resize(frame, (self.palm_size, self.palm_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        inp = self._preprocess(img, self.palm_input)
        self.palm_interpreter.set_tensor(self.palm_input[0]["index"], inp)
        self.palm_interpreter.invoke()
        outputs = [self.palm_interpreter.get_tensor(d["index"]) for d in self.palm_output]
        if not outputs:
            return None
        outputs_sorted = sorted(outputs, key=lambda t: t.shape[-1] if len(t.shape) > 0 else 9999)
        score_tensor = outputs_sorted[0]
        box_tensor = outputs_sorted[-1] if len(outputs_sorted) > 1 else None
        scores = np.array(score_tensor).reshape(-1)
        if scores.size == 0:
            return None
        if np.max(scores) > 1.0 or np.min(scores) < 0.0:
            scores = self._sigmoid(scores)
        best_idx = int(np.argmax(scores))
        if float(scores[best_idx]) < 0.55:
            return None
        if box_tensor is None:
            margin = 0.15
            return (int(w*margin), int(h*margin), int(w*(1-2*margin)), int(h*(1-2*margin)))
        box_arr = np.array(box_tensor).reshape(-1, np.array(box_tensor).shape[-1])
        if best_idx >= len(box_arr):
            return (int(w*0.15), int(h*0.10), int(w*0.70), int(h*0.80))
        raw = np.array(box_arr[best_idx]).flatten()
        if raw.size >= 4:
            vals = raw[:4]
            if np.all(vals >= -0.05) and np.all(vals <= 1.05) and vals[2] > vals[0] and vals[3] > vals[1]:
                ymin,xmin,ymax,xmax = vals.tolist()
                pad = 0.15
                bx = int(max(0,xmin-pad)*w); by = int(max(0,ymin-pad)*h)
                bw = int((min(1,xmax+pad)-max(0,xmin-pad))*w)
                bh = int((min(1,ymax+pad)-max(0,ymin-pad))*h)
                if bw > 10 and bh > 10:
                    return (bx, by, bw, bh)
        return (int(w*0.15), int(h*0.10), int(w*0.70), int(h*0.80))

    def get_landmarks(self, frame, bbox):
        if self.landmark_interpreter is None:
            return None
        h, w = frame.shape[:2]
        bx,by,bw,bh = bbox
        bx=max(0,bx); by=max(0,by); bw=min(bw,w-bx); bh=min(bh,h-by)
        if bw<=0 or bh<=0:
            return None
        crop = frame[by:by+bh, bx:bx+bw]
        if crop.size == 0:
            return None
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_r = cv2.resize(crop_rgb, (self.lm_size, self.lm_size))
        inp = self._preprocess(crop_r, self.lm_input)
        self.landmark_interpreter.set_tensor(self.lm_input[0]["index"], inp)
        self.landmark_interpreter.invoke()
        outputs = [self.landmark_interpreter.get_tensor(d["index"]) for d in self.lm_output]
        landmarks_raw = None
        for out in outputs:
            flat = np.array(out).reshape(-1)
            if flat.size >= 63:
                landmarks_raw = flat[:63]; break
        if landmarks_raw is None:
            return None
        landmarks = landmarks_raw.reshape(21, 3)
        xy = landmarks[:, :2]
        normalized = np.max(np.abs(xy)) <= 1.5
        points = []
        for lm in landmarks:
            if normalized:
                px = int(bx + float(lm[0]) * bw)
                py = int(by + float(lm[1]) * bh)
            else:
                px = int(bx + (float(lm[0]) / self.lm_size) * bw)
                py = int(by + (float(lm[1]) / self.lm_size) * bh)
            px = int(np.clip(px, 0, w-1))
            py = int(np.clip(py, 0, h-1))
            points.append((px, py))
        return points

    def get_finger_states(self, points):
        if not points or len(points) < 21:
            return {"thumb":False,"index":False,"middle":False,"ring":False,"pinky":False}
        index_left = points[5][0] < points[17][0]
        thumb_up = points[4][0] < points[3][0] - 5 if index_left else points[4][0] > points[3][0] + 5
        def up(tip, pip, m=10): return points[tip][1] < points[pip][1] - m
        return {"thumb":thumb_up,"index":up(8,6),"middle":up(12,10),"ring":up(16,14),"pinky":up(20,18)}

    def classify_gesture(self, states):
        count = int(sum(states.values()))
        if count == 5: return "OPEN_HAND"
        if count == 0: return "FIST"
        if states["index"] and states["middle"] and not states["ring"] and not states["pinky"]: return "FORWARD"
        if states["index"] and not states["middle"] and not states["ring"] and not states["pinky"]: return "BACKWARD"
        return "NONE"

    def draw_skeleton(self, frame, points):
        if not points: return frame
        for a,b in CONNECTIONS:
            if a < len(points) and b < len(points):
                cv2.line(frame, points[a], points[b], (200,200,200), 1)
        ranges = [(1,4,0),(5,8,1),(9,12,2),(13,16,3),(17,20,4)]
        for i, pt in enumerate(points):
            color = (0,255,200)
            for s,e,ci in ranges:
                if s <= i <= e: color = FINGER_COLORS[ci]; break
            cv2.circle(frame, pt, 7 if i in FINGER_TIPS else 4, color, -1)
        return frame

    def process_frame(self):
        cap = cam_mgr.get(0, 640, 480)
        ret, frame = cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        gesture = "NONE"; fingers = 0; command_triggered = False

        if self.palm_interpreter and self.landmark_interpreter:
            bbox = self.detect_palm(frame)
            if bbox:
                bx,by,bw,bh = bbox
                cv2.rectangle(frame, (bx,by), (bx+bw,by+bh), (0,200,100), 1)
                points = self.get_landmarks(frame, bbox)
                if points:
                    frame = self.draw_skeleton(frame, points)
                    states = self.get_finger_states(points)
                    fingers = int(sum(states.values()))
                    gesture = self.classify_gesture(states)
                    now = time.time()
                    if gesture != self.pending_gesture:
                        self.pending_gesture = gesture; self.gesture_hold_start = now
                    else:
                        hold = now - self.gesture_hold_start
                        if hold >= self.gesture_hold_required and (now-self.last_trigger_time) > self.cooldown and gesture != "NONE":
                            self._execute(gesture); self.last_trigger_time = now; command_triggered = True

        self.current_gesture = gesture; self.finger_count = fingers
        g_colors = {"OPEN_HAND":(0,255,80),"FIST":(60,60,255),"FORWARD":(0,220,255),"BACKWARD":(255,180,0),"NONE":(80,80,80)}
        g_color = g_colors.get(gesture, (80,80,80))
        h_f, w_f = frame.shape[:2]
        cv2.rectangle(frame, (0,0), (w_f,44), (0,130,0) if command_triggered else (0,0,0), -1)
        cv2.putText(frame, f"GESTURE: {gesture}  FINGERS:{fingers}  FPS:{self.fps:.0f}",
                   (8,28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, g_color, 2)
        cv2.putText(frame, f"CMD:{drone.last_command}  ALT:{drone.altitude:.1f}m",
                   (w_f-220,28), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,180,180), 1)
        cv2.rectangle(frame, (0,h_f-30), (w_f,h_f), (0,0,0), -1)
        cv2.putText(frame, "OPEN=UP  FIST=DOWN  PEACE=FWD  POINT=BWD",
                   (8,h_f-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160,160,160), 1)
        if gesture != "NONE":
            hold = min(time.time()-self.gesture_hold_start, self.gesture_hold_required)
            prog = int((hold/self.gesture_hold_required)*(w_f-20))
            cv2.rectangle(frame, (10,h_f-6), (w_f-10,h_f-2), (40,40,40), -1)
            cv2.rectangle(frame, (10,h_f-6), (10+prog,h_f-2), g_color, -1)
        return frame

    def _execute(self, gesture):
        print(f"[GESTURE] {gesture}")
        if gesture == "OPEN_HAND":  drone.fly_up()
        elif gesture == "FIST":     drone.fly_down()
        elif gesture == "FORWARD":  drone.move_forward()
        elif gesture == "BACKWARD": drone.move_backward()

gesture_detector = GestureDetector()
gesture_model_ok = gesture_detector.load_models()
drone.takeoff()


# ════════════════════════════════════════════
# FPS TRACKERS
# ════════════════════════════════════════════

fps_state = {"thermal": 0, "regular": 0, "gesture": 0}


# ════════════════════════════════════════════
# VIDEO STREAM
# ════════════════════════════════════════════

def generate():
    global active_mode
    prev = time.time()
    while True:
        mode = active_mode
        frame = None
        try:
            if mode == "thermal":
                frame = thermal_detector.process_frame()
            elif mode == "regular":
                frame = regular_detector.process_frame()
            elif mode == "gesture":
                frame = gesture_detector.process_frame()
        except Exception as e:
            print(f"Frame error ({mode}): {e}")

        if frame is None:
            frame = np.zeros((480,640,3), dtype=np.uint8)
            cv2.putText(frame, f"Mode: {mode.upper()} - No signal",
                       (30,240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80,80,80), 2)

        now = time.time()
        fps = 1.0 / (now - prev + 0.001)
        fps_state[mode] = round(fps, 1)
        if mode == "gesture":
            gesture_detector.fps = fps
        elif mode == "thermal":
            thermal_detector.fps = fps
        elif mode == "regular":
            regular_detector.fps = fps
        prev = now

        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")


# ════════════════════════════════════════════
# FLASK ROUTES
# ════════════════════════════════════════════

@app.route("/video")
def video():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/set_mode", methods=["POST"])
def set_mode():
    global active_mode
    data = request.get_json()
    mode = data.get("mode", "thermal")
    if mode in ["thermal", "regular", "gesture"]:
        active_mode = mode
    return jsonify({"mode": active_mode})

@app.route("/set_detection", methods=["POST"])
def set_detection():
    data = request.get_json()
    det = data.get("detection", "human")
    if active_mode == "thermal":
        thermal_detector.mode = det
    elif active_mode == "regular":
        regular_detector.mode = det
    return jsonify({"detection": det})

@app.route("/status")
def status():
    return jsonify({
        "active_mode": active_mode,
        "thermal_mode": thermal_detector.mode,
        "regular_mode": regular_detector.mode,
        "gesture": gesture_detector.current_gesture,
        "fingers": gesture_detector.finger_count,
        "drone": drone.get_status(),
        "gesture_available": TFLITE_OK and gesture_model_ok,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "fps": fps_state.get(active_mode, 0),
        "detections": thermal_detector.detections if active_mode == "thermal" else regular_detector.detections if active_mode == "regular" else 0,
    })

@app.route("/takeoff")
def takeoff():
    drone.takeoff(); return jsonify({"ok": True})

@app.route("/land")
def land():
    drone.land(); return jsonify({"ok": True})

@app.route("/")
def index():
    return '''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CAPSDRONE CONTROL</title>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{
  --bg:#060a0e;--surface:#0b1016;--surface2:#0f1620;
  --border:#1a2a3a;--border2:#243040;
  --green:#00ff88;--blue:#00d4ff;--red:#ff3355;
  --orange:#ff8c00;--yellow:#ffd700;--purple:#cc44ff;
  --text:#c9d1d9;--dim:#4a5568;
}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'Rajdhani',sans-serif;min-height:100vh;overflow-x:hidden}
body::before{content:'';position:fixed;inset:0;background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,212,255,.008) 2px,rgba(0,212,255,.008) 4px);pointer-events:none;z-index:999}

/* HEADER */
header{
  padding:12px 24px;border-bottom:1px solid var(--border);
  background:var(--surface);
  display:flex;align-items:center;justify-content:space-between;
  position:sticky;top:0;z-index:100;
}
.logo{font-family:'Share Tech Mono',monospace;font-size:18px;color:var(--green);letter-spacing:3px}
.logo span{color:var(--blue)}
.header-right{display:flex;align-items:center;gap:16px}
.sys-info{font-family:'Share Tech Mono',monospace;font-size:10px;color:var(--dim);line-height:1.6}
.sys-info b{color:var(--text)}

/* TABS */
.tabs{display:flex;gap:0;border-bottom:1px solid var(--border);background:var(--surface);padding:0 24px}
.tab{
  padding:12px 24px;cursor:pointer;border:none;background:transparent;
  color:var(--dim);font-family:'Rajdhani',sans-serif;font-size:14px;font-weight:600;
  letter-spacing:1px;text-transform:uppercase;border-bottom:2px solid transparent;
  transition:all .2s;position:relative;
}
.tab:hover{color:var(--text)}
.tab.active{color:var(--green);border-bottom-color:var(--green)}
.tab .tab-dot{width:6px;height:6px;border-radius:50%;background:var(--dim);display:inline-block;margin-right:8px;transition:all .2s}
.tab.active .tab-dot{background:var(--green);box-shadow:0 0 6px var(--green)}
.tab-unavailable{opacity:0.4;cursor:not-allowed}

/* MAIN LAYOUT */
.layout{display:grid;grid-template-columns:1fr 280px;gap:0;height:calc(100vh - 100px)}

/* VIDEO PANEL */
.video-section{padding:20px;display:flex;flex-direction:column;gap:16px;overflow-y:auto}
.video-card{background:var(--surface);border:1px solid var(--border);border-radius:6px;overflow:hidden;position:relative}
.video-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--green),var(--blue));z-index:1}
.video-header{padding:10px 16px;border-bottom:1px solid var(--border);display:flex;align-items:center;justify-content:space-between}
.video-title{font-family:'Share Tech Mono',monospace;font-size:11px;color:var(--green);letter-spacing:2px}
.video-stats{display:flex;gap:12px}
.vstat{font-family:'Share Tech Mono',monospace;font-size:10px;color:var(--dim)}
.vstat b{color:var(--blue)}
.video-wrapper{background:#000;position:relative}
.video-wrapper img{width:100%;display:block;max-height:480px;object-fit:contain}

/* DETECTION BUTTONS */
.det-buttons{display:flex;gap:8px;padding:12px 16px;border-top:1px solid var(--border);flex-wrap:wrap}
.det-btn{
  padding:6px 14px;border:1px solid var(--border);background:transparent;
  color:var(--dim);font-family:'Rajdhani',sans-serif;font-size:12px;font-weight:600;
  letter-spacing:1px;cursor:pointer;border-radius:2px;transition:all .15s;text-transform:uppercase;
}
.det-btn:hover{border-color:var(--blue);color:var(--blue)}
.det-btn.active{border-color:var(--green);color:var(--green);background:rgba(0,255,136,.08)}

/* RIGHT SIDEBAR */
.sidebar{border-left:1px solid var(--border);background:var(--surface);padding:16px;display:flex;flex-direction:column;gap:14px;overflow-y:auto}
.widget{background:var(--surface2);border:1px solid var(--border);border-radius:4px;padding:14px}
.widget-title{font-family:'Share Tech Mono',monospace;font-size:10px;color:var(--dim);letter-spacing:2px;margin-bottom:10px}

/* DRONE STATUS */
.drone-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.drone-stat{background:var(--bg);border:1px solid var(--border);padding:8px;border-radius:2px}
.ds-label{font-size:9px;color:var(--dim);font-family:'Share Tech Mono',monospace;letter-spacing:1px}
.ds-val{font-size:22px;font-weight:700;color:var(--green);font-family:'Share Tech Mono',monospace;line-height:1.2}
.ds-val.blue{color:var(--blue);font-size:13px}
.ds-val.orange{color:var(--orange);font-size:16px}

/* ALTITUDE BAR */
.alt-wrap{height:80px;display:flex;gap:8px;align-items:flex-end;margin-top:8px}
.alt-bg{flex:1;height:100%;background:var(--bg);border:1px solid var(--border);border-radius:2px;position:relative;overflow:hidden}
.alt-fill{position:absolute;bottom:0;left:0;right:0;background:linear-gradient(to top,var(--green),rgba(0,255,136,.2));transition:height .6s ease;border-radius:2px}
.alt-lbl{font-family:'Share Tech Mono',monospace;font-size:9px;color:var(--dim);writing-mode:vertical-rl;text-align:center}

/* GESTURE REFERENCE */
.gesture-list{display:flex;flex-direction:column;gap:6px}
.g-row{display:flex;align-items:center;gap:8px;padding:7px 10px;border:1px solid var(--border);border-radius:2px;background:var(--bg);transition:all .2s}
.g-row.active{border-color:var(--green);background:rgba(0,255,136,.06)}
.g-dot{width:8px;height:8px;border-radius:50%;background:var(--dim);flex-shrink:0;transition:all .2s}
.g-row.active .g-dot{background:var(--green);box-shadow:0 0 6px var(--green)}
.g-label{font-family:'Share Tech Mono',monospace;font-size:10px;flex:1}
.g-cmd{font-size:11px;color:var(--dim);font-weight:600}
.g-row.active .g-cmd{color:var(--green)}

/* CONTROLS */
.btn-row{display:flex;gap:8px}
.btn{flex:1;padding:10px;border:1px solid var(--border);background:transparent;color:var(--text);font-family:'Rajdhani',sans-serif;font-size:13px;font-weight:700;letter-spacing:1px;cursor:pointer;border-radius:2px;transition:all .15s;text-transform:uppercase}
.btn.g:hover{border-color:var(--green);color:var(--green);background:rgba(0,255,136,.06)}
.btn.r:hover{border-color:var(--red);color:var(--red);background:rgba(255,51,85,.06)}

/* WARNING BADGE */
.warn-badge{background:rgba(255,140,0,.1);border:1px solid rgba(255,140,0,.3);border-radius:3px;padding:8px 10px;font-size:11px;color:var(--orange);line-height:1.5}
</style>
</head>
<body>

<header>
  <div class="logo">CAPS<span>DRONE</span></div>
  <div class="header-right">
    <div class="sys-info">
      <div>PYTHON <b id="py-ver">--</b> &nbsp;|&nbsp; FPS <b id="hdr-fps">--</b> &nbsp;|&nbsp; DETECT <b id="hdr-det">--</b></div>
      <div>MODE <b id="hdr-mode">--</b> &nbsp;|&nbsp; CMD <b id="hdr-cmd">--</b></div>
    </div>
  </div>
</header>

<div class="tabs">
  <button class="tab active" id="tab-thermal" onclick="switchMode('thermal')">
    <span class="tab-dot"></span>THERMAL
  </button>
  <button class="tab" id="tab-regular" onclick="switchMode('regular')">
    <span class="tab-dot"></span>CAMERA
  </button>
  <button class="tab" id="tab-gesture" id="tab-gesture" onclick="switchMode('gesture')">
    <span class="tab-dot"></span>GESTURE
  </button>
</div>

<div class="layout">
  <div class="video-section">

    <!-- THERMAL PANEL -->
    <div id="panel-thermal" class="video-card">
      <div class="video-header">
        <div class="video-title">THERMAL CAMERA — /dev/video2</div>
        <div class="video-stats">
          <div class="vstat">FPS <b id="t-fps">--</b></div>
          <div class="vstat">DETECT <b id="t-det">--</b></div>
        </div>
      </div>
      <div class="video-wrapper"><img src="/video" id="main-feed"></div>
      <div class="det-buttons" id="det-thermal">
        <button class="det-btn active" onclick="setDetection('human')">HUMAN</button>
        <button class="det-btn" onclick="setDetection('fire')">FIRE</button>
        <button class="det-btn" onclick="setDetection('hot')">HOT OBJECT</button>
      </div>
    </div>

    <!-- REGULAR PANEL -->
    <div id="panel-regular" class="video-card" style="display:none">
      <div class="video-header">
        <div class="video-title">REGULAR CAMERA — /dev/video0</div>
        <div class="video-stats">
          <div class="vstat">FPS <b id="r-fps">--</b></div>
          <div class="vstat">DETECT <b id="r-det">--</b></div>
        </div>
      </div>
      <div class="video-wrapper"><img src="/video"></div>
      <div class="det-buttons" id="det-regular">
        <button class="det-btn active" onclick="setDetection('person')">PERSON</button>
        <button class="det-btn" onclick="setDetection('obstacle')">OBSTACLE</button>
        <button class="det-btn" onclick="setDetection('landing')">LANDING PAD</button>
      </div>
    </div>

    <!-- GESTURE PANEL -->
    <div id="panel-gesture" class="video-card" style="display:none">
      <div class="video-header">
        <div class="video-title">GESTURE CONTROL — TFLITE SKELETON</div>
        <div class="video-stats">
          <div class="vstat">FPS <b id="g-fps">--</b></div>
          <div class="vstat">FINGERS <b id="g-fingers">--</b></div>
        </div>
      </div>
      <div class="video-wrapper"><img src="/video"></div>
      <div class="det-buttons">
        <div id="gesture-unavail" class="warn-badge" style="display:none">
          Gesture control requires Python 3.11 + TFLite models. 
          Run with: <b>python3.11 main_app.py</b>
        </div>
      </div>
    </div>

  </div>

  <!-- SIDEBAR -->
  <div class="sidebar">

    <div class="widget">
      <div class="widget-title">// DRONE STATUS</div>
      <div class="drone-grid">
        <div class="drone-stat"><div class="ds-label">MODE</div><div class="ds-val orange" id="d-mode">--</div></div>
        <div class="drone-stat"><div class="ds-label">ALTITUDE</div><div class="ds-val" id="d-alt">--</div></div>
      </div>
      <div class="drone-stat" style="margin-top:8px"><div class="ds-label">LAST COMMAND</div><div class="ds-val blue" id="d-cmd">--</div></div>
      <div class="drone-stat" style="margin-top:8px"><div class="ds-label">POSITION X</div><div class="ds-val blue" id="d-x">--</div></div>
      <div class="alt-wrap">
        <div class="alt-lbl">ALT</div>
        <div class="alt-bg"><div class="alt-fill" id="alt-bar" style="height:15%"></div></div>
      </div>
    </div>

    <div class="widget" id="gesture-ref" style="display:none">
      <div class="widget-title">// GESTURE REFERENCE</div>
      <div class="gesture-list">
        <div class="g-row" id="g-OPEN_HAND"><div class="g-dot"></div><span class="g-label">OPEN HAND (5)</span><span class="g-cmd">FLY UP</span></div>
        <div class="g-row" id="g-FIST"><div class="g-dot"></div><span class="g-label">FIST (0)</span><span class="g-cmd">FLY DOWN</span></div>
        <div class="g-row" id="g-FORWARD"><div class="g-dot"></div><span class="g-label">PEACE (2)</span><span class="g-cmd">FORWARD</span></div>
        <div class="g-row" id="g-BACKWARD"><div class="g-dot"></div><span class="g-label">POINT (1)</span><span class="g-cmd">BACKWARD</span></div>
      </div>
    </div>

    <div class="widget">
      <div class="widget-title">// FLIGHT CONTROLS</div>
      <div class="btn-row">
        <button class="btn g" onclick="fetch('/takeoff')">TAKEOFF</button>
        <button class="btn r" onclick="fetch('/land')">LAND</button>
      </div>
    </div>

  </div>
</div>

<script>
let currentMode = 'thermal';
let gestureAvailable = false;

function switchMode(mode) {
  if (mode === 'gesture' && !gestureAvailable) {
    document.getElementById('gesture-unavail').style.display = 'block';
  }
  currentMode = mode;
  fetch('/set_mode', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({mode})
  });

  // Update tabs
  ['thermal','regular','gesture'].forEach(m => {
    document.getElementById('tab-'+m).classList.toggle('active', m===mode);
    document.getElementById('panel-'+m).style.display = m===mode ? 'block' : 'none';
  });

  // Show/hide gesture ref
  document.getElementById('gesture-ref').style.display = mode==='gesture' ? 'block' : 'none';
}

function setDetection(det) {
  fetch('/set_detection', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({detection: det})
  });

  // Update buttons in current panel
  const panel = document.getElementById('det-' + currentMode);
  if (panel) {
    panel.querySelectorAll('.det-btn').forEach(btn => {
      btn.classList.toggle('active', btn.textContent.toLowerCase().includes(det));
    });
  }
}

// Poll status
setInterval(() => {
  fetch('/status').then(r=>r.json()).then(d => {
    // Header
    document.getElementById('py-ver').textContent = d.python_version;
    document.getElementById('hdr-fps').textContent = d.fps;
    document.getElementById('hdr-det').textContent = d.detections;
    document.getElementById('hdr-mode').textContent = d.active_mode.toUpperCase();
    document.getElementById('hdr-cmd').textContent = d.drone.last_command;

    // Drone
    document.getElementById('d-mode').textContent = d.drone.mode;
    document.getElementById('d-alt').textContent = (d.drone.altitude||0).toFixed(1)+'m';
    document.getElementById('d-cmd').textContent = d.drone.last_command;
    document.getElementById('d-x').textContent = (d.drone.position_x||0).toFixed(1)+'m';
    document.getElementById('alt-bar').style.height = Math.min((d.drone.altitude/10)*100,100)+'%';

    // Gesture availability
    gestureAvailable = d.gesture_available;
    const gTab = document.getElementById('tab-gesture');
    if (!gestureAvailable) gTab.classList.add('tab-unavailable');
    else gTab.classList.remove('tab-unavailable');

    // Mode-specific stats
    if (d.active_mode === 'thermal') {
      document.getElementById('t-fps').textContent = d.fps;
      document.getElementById('t-det').textContent = d.detections;
    } else if (d.active_mode === 'regular') {
      document.getElementById('r-fps').textContent = d.fps;
      document.getElementById('r-det').textContent = d.detections;
    } else if (d.active_mode === 'gesture') {
      document.getElementById('g-fps').textContent = d.fps;
      document.getElementById('g-fingers').textContent = d.fingers;
      ['OPEN_HAND','FIST','FORWARD','BACKWARD'].forEach(g => {
        const el = document.getElementById('g-'+g);
        if(el) el.classList.toggle('active', d.gesture===g);
      });
    }
  }).catch(()=>{});
}, 600);
</script>
</body>
</html>'''


if __name__ == "__main__":
    print("=" * 56)
    print("CAPSDRONE Unified Control System")
    print(f"Python:   {sys.version_info.major}.{sys.version_info.minor}")
    print(f"TFLite:   {'OK' if TFLITE_OK else 'N/A (use python3.11)'}")
    print(f"Gesture:  {'OK' if gesture_model_ok else 'Models not found'}")
    print("Open:     http://<PI_IP>:5000")
    print("=" * 56)
    app.run(host="0.0.0.0", port=5000, threaded=True)