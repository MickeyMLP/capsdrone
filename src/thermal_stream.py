from flask import Flask, Response, request
import cv2
import numpy as np
import time

app = Flask(__name__)

# --------------------------------------------------
# THERMAL DETECTOR CLASS (keep your functions)
# --------------------------------------------------

class ThermalDetector:

    def __init__(self, config):
        self.threshold_temp = config.get('threshold_temp', 30)
        self.detection_interval = config.get('detection_interval', 1.0)
        self.thermal_camera = None
        self.detected_objects = []

    def init_camera(self, camera_id="/dev/video0"):
        self.thermal_camera = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)

        self.thermal_camera.set(cv2.CAP_PROP_FOURCC,
                                cv2.VideoWriter_fourcc(*'YUYV'))
        self.thermal_camera.set(cv2.CAP_PROP_FRAME_WIDTH, 256)
        self.thermal_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 192)
        self.thermal_camera.set(cv2.CAP_PROP_FPS, 25)
        self.thermal_camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.thermal_camera.isOpened():
            print("❌ Failed to open camera")
            return False

        print("✅ Camera opened")
        return True

    def read_frame(self):
        ret, frame = self.thermal_camera.read()
        if not ret:
            return None
        return frame

    # SAFE PROCESSING
    def process_thermal_frame(self, frame):

        if frame is None:
            return None, None

        try:
            if len(frame.shape) == 3 and frame.shape[2] == 2:
                bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_YUY2)
            elif len(frame.shape) == 3:
                bgr = frame
            elif len(frame.shape) == 2:
                bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                return None, None

            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            colored = cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

            return colored, gray

        except Exception as e:
            print("Processing error:", e)
            return None, None

    # ---------------- DETECTIONS ----------------

    def detect_heat_sources(self, gray, threshold=30):
        _, hot = cv2.threshold(gray, int(threshold * 2.55),
                               255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(hot,
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        results = []
        for c in contours:
            if cv2.contourArea(c) > 50:
                x, y, w, h = cv2.boundingRect(c)
                results.append({
                    "bbox": (x, y, w, h),
                    "position": (x + w//2, y + h//2)
                })
        return results

    def detect_person(self, gray):
        return self.detect_heat_sources(gray, threshold=32)

    def detect_fire(self, gray):
        return self.detect_heat_sources(gray, threshold=60)

    def annotate(self, frame, detections, mode):

        colors = {
            "heat": (0,255,255),
            "person": (0,255,0),
            "fire": (0,0,255)
        }

        color = colors.get(mode, (255,255,255))

        for d in detections:
            x, y, w, h = d["bbox"]
            cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)

        cv2.putText(frame, f"Mode: {mode.upper()}",
                    (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255,255,255), 2)

        return frame


# --------------------------------------------------
# INITIALIZE
# --------------------------------------------------

detector = ThermalDetector({
    "threshold_temp": 30,
    "detection_interval": 1.0
})

if not detector.init_camera():
    exit()


# --------------------------------------------------
# STREAM GENERATOR
# --------------------------------------------------

def generate(mode):

    while True:
        frame = detector.read_frame()
        if frame is None:
            continue

        colored, gray = detector.process_thermal_frame(frame)
        if gray is None:
            continue

        if mode == "heat":
            detections = detector.detect_heat_sources(gray)
        elif mode == "person":
            detections = detector.detect_person(gray)
        elif mode == "fire":
            detections = detector.detect_fire(gray)
        else:
            detections = []

        output = detector.annotate(colored, detections, mode)

        ret, buffer = cv2.imencode('.jpg', output)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               frame_bytes + b'\r\n')


# --------------------------------------------------
# ROUTES
# --------------------------------------------------

@app.route('/')
def index():
    mode = request.args.get("mode", "heat")
    return Response(generate(mode),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/<mode>')
def mode_route(mode):
    return Response(generate(mode),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


# --------------------------------------------------

if __name__ == "__main__":
    print("🔥 Thermal Streaming Server Running")
    print("Available modes:")
    print("  http://PI_IP:5000/heat")
    print("  http://PI_IP:5000/person")
    print("  http://PI_IP:5000/fire")
    app.run(host="0.0.0.0", port=5000, threaded=True)