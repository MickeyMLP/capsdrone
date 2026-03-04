"""
Thermal Detection + HTTP Streaming
Keeps all detection functions
Streams annotated result over network
"""

import cv2
import numpy as np
import time
from flask import Flask, Response
from datetime import datetime

app = Flask(__name__)


class ThermalDetector:

    def __init__(self, config):
        self.config = config
        self.threshold_temp = config.get('threshold_temp', 30)
        self.detection_interval = config.get('detection_interval', 1.0)

        self.thermal_camera = None
        self.last_detection_time = 0

        self.detected_objects = []
        self.max_temp = 0
        self.min_temp = 0
        self.avg_temp = 0

        print("✅ Thermal Detector initialized")

    # --------------------------------------------------
    # CAMERA INIT
    # --------------------------------------------------

    def init_camera(self, camera_id="/dev/video0"):
        self.thermal_camera = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

        self.thermal_camera.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*'YUYV')
        )
        self.thermal_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
        self.thermal_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)
        self.thermal_camera.set(cv2.CAP_PROP_FPS, 25)
        self.thermal_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.thermal_camera.isOpened():
            print("❌ Failed to open camera")
            return False

        print("✅ Camera opened successfully")
        return True

    # --------------------------------------------------
    # FRAME PROCESSING
    # --------------------------------------------------

    def read_frame(self):
        ret, frame = self.thermal_camera.read()
        if not ret:
            return None
        return frame

    def process_thermal_frame(self, frame):

        if frame is None:
            return None, None

        try:
            # If already BGR (3 channels)
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                bgr = frame

            # If YUYV (2 channels packed)
            elif len(frame.shape) == 3 and frame.shape[2] == 2:
                bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)

            # If grayscale
            elif len(frame.shape) == 2:
                bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

            else:
                print("⚠️ Unknown frame format:", frame.shape)
                return None, None

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

            self.max_temp = float(np.max(gray))
            self.min_temp = float(np.min(gray))
            self.avg_temp = float(np.mean(gray))

            colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

            return colored, gray

        except Exception as e:
            print("❌ Frame processing error:", e)
            return None, None

    # --------------------------------------------------
    # DETECTION FUNCTIONS (UNCHANGED LOGIC)
    # --------------------------------------------------

    def detect_heat_sources(self, gray_frame, threshold=None):

        if threshold is None:
            threshold = self.threshold_temp

        _, hot_spots = cv2.threshold(
            gray_frame,
            int(threshold * 2.55),
            255,
            cv2.THRESH_BINARY
        )

        contours, _ = cv2.findContours(
            hot_spots,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        detected = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 50:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2

                temp_estimate = gray_frame[center_y, center_x]

                detected.append({
                    'position': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'area': area,
                    'temp_estimate': float(temp_estimate)
                })

        self.detected_objects = detected
        return detected

    def detect_person(self, gray_frame):
        persons = []
        heat_sources = self.detect_heat_sources(gray_frame, threshold=32)

        for source in heat_sources:
            x, y, w, h = source['bbox']
            temp = source['temp_estimate']

            aspect_ratio = h / w if w > 0 else 0

            if 1.2 < aspect_ratio < 3.5 and 32 < temp < 40:
                persons.append(source)

        return persons

    def detect_fire(self, gray_frame):
        return self.detect_heat_sources(gray_frame, threshold=60)

    # --------------------------------------------------
    # ANNOTATION
    # --------------------------------------------------

    def annotate_frame(self, frame, detections, mode):

        color_map = {
            'heat': (0, 255, 255),
            'person': (0, 255, 0),
            'fire': (0, 0, 255)
        }

        color = color_map.get(mode, (255, 255, 255))

        for det in detections:
            x, y, w, h = det['bbox']
            center = det['position']

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.circle(frame, center, 5, color, -1)

        cv2.putText(
            frame,
            f"Mode: {mode.upper()}  Detected: {len(detections)}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255,255,255),
            1
        )

        return frame


# --------------------------------------------------
# STREAMING SETUP
# --------------------------------------------------

config = {
    'threshold_temp': 30,
    'detection_interval': 1.0
}

detector = ThermalDetector(config)

MODE = "heat"  # change to heat/person/fire

if not detector.init_camera():
    exit()


def generate():

    while True:
        frame = detector.read_frame()
        if frame is None:
            continue

        colored, gray = detector.process_thermal_frame(frame)

        if gray is None:
            continue

        if MODE == 'heat':
            detections = detector.detect_heat_sources(gray)
        elif MODE == 'person':
            detections = detector.detect_person(gray)
        elif MODE == 'fire':
            detections = detector.detect_fire(gray)
        else:
            detections = []

        output = detector.annotate_frame(colored, detections, MODE)

        _, buffer = cv2.imencode('.jpg', output)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes +
               b'\r\n')


@app.route('/')
def video_feed():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == "__main__":
    print("🔥 Thermal streaming server running...")
    print("Open in browser: http://PI_IP:5000")
    app.run(host='0.0.0.0', port=5000, threaded=True)