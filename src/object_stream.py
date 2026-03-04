"""
Drone Object Detection + Flask Streaming
Works with GEMBIRD AX2311 UVC camera
"""

import cv2
import numpy as np
from flask import Flask, Response, request

# ==========================================================
# YOUR ORIGINAL CLASS (UNCHANGED)
# ==========================================================

class DroneObjectDetector:
    def __init__(self):
        self.camera = None
        self.frame_width = 640
        self.frame_height = 480

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.body_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_fullbody.xml'
        )

        self.COLOR_PERSON = (0, 255, 0)
        self.COLOR_OBSTACLE = (0, 0, 255)
        self.COLOR_TARGET = (255, 0, 255)
        self.COLOR_LANDING = (0, 255, 255)

        print("✅ Object Detector initialized")

    def init_camera(self, device="/dev/video2"):
        self.camera = cv2.VideoCapture(device, cv2.CAP_V4L2)

        # Force exact supported format
        self.camera.set(cv2.CAP_PROP_FOURCC,
                        cv2.VideoWriter_fourcc(*'YUYV'))

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

        # VERY important on Pi
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.camera.isOpened():
            print("❌ Could not open camera")
            return False

        # Warmup frames (prevents select() timeout)
        for _ in range(5):
            self.camera.read()

        print("✅ Camera connected and streaming")
        return True

    def read_frame(self):
        if not self.camera:
            return None
        ret, frame = self.camera.read()
        if not ret:
            return None
        return frame

    # ================= DETECTIONS =================

    def detect_person(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        bodies = self.body_cascade.detectMultiScale(gray, 1.1, 3)

        persons = []
        for (x, y, w, h) in faces:
            persons.append({'bbox': (x, y, w, h), 'center': (x+w//2, y+h//2), 'area': w*h})
        for (x, y, w, h) in bodies:
            persons.append({'bbox': (x, y, w, h), 'center': (x+w//2, y+h//2), 'area': w*h})

        return persons

    def detect_obstacles(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        obstacles = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 1500:
                x, y, w, h = cv2.boundingRect(c)
                obstacles.append({'bbox': (x, y, w, h), 'center': (x+w//2, y+h//2), 'area': area})
        return obstacles

    def detect_landing_pad(self, frame):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([5,100,100]), np.array([25,255,255]))
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        pads = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 2000:
                x, y, w, h = cv2.boundingRect(c)
                pads.append({'bbox': (x, y, w, h), 'center': (x+w//2, y+h//2), 'area': area})
        return pads

    def detect_color_target(self, frame, color="red"):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower = np.array([0,100,100])
        upper = np.array([10,255,255])
        mask = cv2.inRange(hsv, lower, upper)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        targets = []
        for c in contours:
            area = cv2.contourArea(c)
            if area > 500:
                x, y, w, h = cv2.boundingRect(c)
                targets.append({'bbox': (x, y, w, h), 'center': (x+w//2, y+h//2), 'area': area})
        return targets

    def annotate(self, frame, detections, mode):
        color_map = {
            "person": self.COLOR_PERSON,
            "obstacle": self.COLOR_OBSTACLE,
            "landing_pad": self.COLOR_LANDING,
            "target": self.COLOR_TARGET
        }
        color = color_map.get(mode, (255,255,255))

        for d in detections:
            x,y,w,h = d["bbox"]
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)

        cv2.putText(frame, f"Mode: {mode} | Detected: {len(detections)}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

        return frame


# ==========================================================
# FLASK STREAMING
# ==========================================================

app = Flask(__name__)
detector = DroneObjectDetector()

if not detector.init_camera("/dev/video2"):   # change if needed
    raise RuntimeError("Camera failed")

def generate(mode):
    while True:
        frame = detector.read_frame()
        if frame is None:
            continue

        if mode == "person":
            detections = detector.detect_person(frame)
        elif mode == "obstacle":
            detections = detector.detect_obstacles(frame)
        elif mode == "landing_pad":
            detections = detector.detect_landing_pad(frame)
        elif mode == "target":
            detections = detector.detect_color_target(frame)
        else:
            detections = []

        frame = detector.annotate(frame, detections, mode)

        ret, buffer = cv2.imencode(".jpg", frame)
        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               buffer.tobytes() +
               b'\r\n')

@app.route("/")
def index():
    mode = request.args.get("mode", "person")
    return Response(generate(mode),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route("/<mode>")
def select_mode(mode):
    return Response(generate(mode),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    print("\n🌐 Open in browser:")
    print("http://PI_IP:5000/person")
    print("http://PI_IP:5000/obstacle")
    print("http://PI_IP:5000/landing_pad")
    print("http://PI_IP:5000/target\n")

    app.run(host="0.0.0.0", port=5000, threaded=True)