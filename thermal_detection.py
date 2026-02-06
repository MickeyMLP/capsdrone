"""
Thermal Detection Module
Process thermal camera data for object detection and tracking
Author: Sue Sha
"""

import cv2
import numpy as np
from datetime import datetime
import time

class ThermalDetector:
    """
    Thermal camera processing for:
    - Heat source detection
    - Person detection (search & rescue)
    - Temperature monitoring
    - Fire detection
    """
    
    def __init__(self, config):
        self.config = config
        self.enabled = config.get('enabled', True)
        self.threshold_temp = config.get('threshold_temp', 30)  # Celsius
        self.detection_interval = config.get('detection_interval', 1.0)  # seconds
        
        # Camera initialization
        self.thermal_camera = None
        self.last_detection_time = 0
        
        # Display settings - BIGGER WINDOW
        self.frame_width = 1280
        self.frame_height = 720
        
        # Detection results
        self.detected_objects = []
        self.max_temp = 0
        self.min_temp = 0
        self.avg_temp = 0
        
        print("✅ Thermal Detector initialized")
        print(f"   Threshold: {self.threshold_temp}°C")
        print(f"   Detection interval: {self.detection_interval}s")
        print(f"   Display size: {self.frame_width}x{self.frame_height}")
    
    def init_camera(self, camera_id=0):
        """
        Initialize thermal camera
        Args:
            camera_id: Camera device ID or path
        """
        try:
            # For regular USB thermal camera
            self.thermal_camera = cv2.VideoCapture(camera_id)
            
            # Set resolution for bigger display
            self.thermal_camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.thermal_camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            if not self.thermal_camera.isOpened():
                print("⚠️ Could not open thermal camera")
                return False
            
            print("✅ Thermal camera initialized")
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return False
    
    def read_frame(self):
        """Read frame from thermal camera"""
        if not self.thermal_camera:
            return None
        
        ret, frame = self.thermal_camera.read()
        if not ret:
            return None
        
        return frame
    
    def process_thermal_frame(self, frame):
        """
        Process thermal frame to extract temperature data
        Returns: processed frame with annotations
        """
        if frame is None:
            return None, None
        
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Apply colormap for visualization
        thermal_colored = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        
        # Calculate temperature statistics
        self.max_temp = np.max(gray)
        self.min_temp = np.min(gray)
        self.avg_temp = np.mean(gray)
        
        return thermal_colored, gray
    
    def detect_heat_sources(self, gray_frame, threshold=None):
        """
        Detect heat sources above threshold
        Returns: list of detected heat source locations
        """
        if threshold is None:
            threshold = self.threshold_temp
        
        # Normalize grayscale to approximate temperature
        # This is simplified - actual conversion depends on camera calibration
        temp_normalized = (gray_frame / 255.0) * 100  # Assume 0-100°C range
        
        # Threshold to find hot spots
        _, hot_spots = cv2.threshold(
            gray_frame, 
            int(threshold * 2.55),  # Convert threshold to grayscale value
            255, 
            cv2.THRESH_BINARY
        )
        
        # Find contours of hot spots
        contours, _ = cv2.findContours(
            hot_spots, 
            cv2.RETR_EXTERNAL, 
            cv2.CHAIN_APPROX_SIMPLE
        )
        
        detected = []
        for contour in contours:
            # Filter by minimum area
            area = cv2.contourArea(contour)
            if area > 50:  # Minimum 50 pixels
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate center
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Estimate temperature at center
                temp_estimate = temp_normalized[center_y, center_x]
                
                detected.append({
                    'position': (center_x, center_y),
                    'bbox': (x, y, w, h),
                    'area': area,
                    'temp_estimate': temp_estimate
                })
        
        self.detected_objects = detected
        return detected
    
    def detect_person(self, gray_frame):
        """
        Detect human heat signature
        Returns: list of detected persons with confidence
        """
        # Typical human heat signature characteristics:
        # - Temperature range: 33-37°C (surface temp)
        # - Shape: roughly vertical rectangle
        # - Size: depends on distance
        
        persons = []
        heat_sources = self.detect_heat_sources(gray_frame, threshold=32)
        
        for source in heat_sources:
            x, y, w, h = source['bbox']
            temp = source['temp_estimate']
            
            # Check if it matches human characteristics
            aspect_ratio = h / w if w > 0 else 0
            
            # Human typically has aspect ratio between 1.5 and 3.0
            if 1.2 < aspect_ratio < 3.5 and 32 < temp < 40:
                confidence = self._calculate_person_confidence(
                    aspect_ratio, temp, source['area']
                )
                
                persons.append({
                    'position': source['position'],
                    'bbox': source['bbox'],
                    'temp': temp,
                    'confidence': confidence
                })
        
        return persons
    
    def _calculate_person_confidence(self, aspect_ratio, temp, area):
        """Calculate confidence that detection is a person"""
        confidence = 0.0
        
        # Temperature confidence (peak at 35°C)
        temp_conf = 1.0 - abs(35 - temp) / 10.0
        temp_conf = max(0, min(1, temp_conf))
        
        # Aspect ratio confidence (peak at 2.0)
        aspect_conf = 1.0 - abs(2.0 - aspect_ratio) / 2.0
        aspect_conf = max(0, min(1, aspect_conf))
        
        # Size confidence (prefer medium sizes)
        if 500 < area < 5000:
            size_conf = 1.0
        else:
            size_conf = 0.5
        
        # Weighted average
        confidence = (temp_conf * 0.5 + aspect_conf * 0.3 + size_conf * 0.2)
        
        return confidence
    
    def detect_fire(self, gray_frame):
        """
        Detect potential fire based on temperature
        Returns: fire detection results
        """
        # Fire typically has temperature > 60°C
        fire_threshold = 60
        
        fire_sources = self.detect_heat_sources(gray_frame, threshold=fire_threshold)
        
        if fire_sources:
            print(f"🔥 FIRE DETECTED! {len(fire_sources)} sources")
            return {
                'fire_detected': True,
                'num_sources': len(fire_sources),
                'sources': fire_sources
            }
        
        return {'fire_detected': False}
    
    def annotate_frame(self, frame, detections, detection_type='heat'):
        """
        Draw annotations on frame
        Args:
            frame: Frame to annotate
            detections: List of detection results
            detection_type: 'heat', 'person', or 'fire'
        """
        annotated = frame.copy()
        
        color_map = {
            'heat': (0, 255, 255),    # Yellow
            'person': (0, 255, 0),     # Green
            'fire': (0, 0, 255)        # Red
        }
        
        color = color_map.get(detection_type, (255, 255, 255))
        
        for det in detections:
            x, y, w, h = det['bbox']
            center = det['position']
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 3)
            
            # Draw center point
            cv2.circle(annotated, center, 8, color, -1)
            
            # Add label with bigger font
            if 'temp_estimate' in det:
                label = f"{det['temp_estimate']:.1f}C"
            elif 'confidence' in det:
                label = f"Person {det['confidence']*100:.0f}%"
            else:
                label = detection_type
            
            cv2.putText(
                annotated, label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, color, 2
            )
        
        return annotated
    
    def run_detection(self, mode='heat'):
        """
        Run thermal detection in real-time
        Args:
            mode: 'heat', 'person', or 'fire'
        """
        if not self.thermal_camera:
            print("⚠️ Camera not initialized")
            return
        
        print(f"\n🔍 Starting thermal detection: {mode} mode")
        print("Press 'q' to quit, 's' to save frame, 'f' for fullscreen")
        print(f"Window size: {self.frame_width}x{self.frame_height}\n")
        
        # Create resizable window
        cv2.namedWindow('Thermal Detection', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Thermal Detection', self.frame_width, self.frame_height)
        
        fullscreen = False
        
        while True:
            # Read frame
            frame = self.read_frame()
            if frame is None:
                print("⚠️ Failed to read frame")
                break
            
            # Process thermal data
            thermal_colored, gray = self.process_thermal_frame(frame)
            
            if thermal_colored is None or gray is None:
                continue
            
            # Perform detection based on mode
            current_time = time.time()
            if current_time - self.last_detection_time > self.detection_interval:
                if mode == 'heat':
                    detections = self.detect_heat_sources(gray)
                elif mode == 'person':
                    detections = self.detect_person(gray)
                elif mode == 'fire':
                    fire_result = self.detect_fire(gray)
                    detections = fire_result.get('sources', [])
                else:
                    detections = []
                
                self.last_detection_time = current_time
            else:
                detections = self.detected_objects
            
            # Annotate frame
            if detections:
                display_frame = self.annotate_frame(thermal_colored, detections, mode)
            else:
                display_frame = thermal_colored
            
            # Add temperature info with bigger font
            info_text = [
                f"Mode: {mode.upper()}",
                f"Max: {self.max_temp:.1f}",
                f"Min: {self.min_temp:.1f}",
                f"Avg: {self.avg_temp:.1f}",
                f"Detected: {len(detections)}"
            ]
            
            y_offset = 40
            for text in info_text:
                cv2.putText(
                    display_frame, text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (255, 255, 255), 2
                )
                y_offset += 35
            
            # Display
            cv2.imshow('Thermal Detection', display_frame)
            
            # Handle key press
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"thermal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, display_frame)
                print(f"💾 Saved: {filename}")
            elif key == ord('f'):
                # Toggle fullscreen
                fullscreen = not fullscreen
                if fullscreen:
                    cv2.setWindowProperty('Thermal Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    print("🖥️  Fullscreen ON")
                else:
                    cv2.setWindowProperty('Thermal Detection', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    print("🪟 Fullscreen OFF")
        
        cv2.destroyAllWindows()
    
    def get_detection_report(self):
        """Get summary of current detections"""
        return {
            'num_detections': len(self.detected_objects),
            'max_temp': self.max_temp,
            'min_temp': self.min_temp,
            'avg_temp': self.avg_temp,
            'detections': self.detected_objects
        }
    
    def cleanup(self):
        """Release camera resources"""
        if self.thermal_camera:
            self.thermal_camera.release()
        cv2.destroyAllWindows()
        print("✅ Thermal camera cleanup complete")


# Test function
if __name__ == "__main__":
    print("="*60)
    print("🌡️  Thermal Detection System")
    print("="*60)
    print("Testing Thermal Detector")
    print("Note: This will use your webcam as a simulated thermal camera")
    
    config = {
        'enabled': True,
        'threshold_temp': 30,
        'detection_interval': 1.0
    }
    
    detector = ThermalDetector(config)
    
    # Initialize camera (use 0 for default webcam)
    if detector.init_camera(0):
        print("\nStarting detection...")
        print("Available modes:")
        print("  1. heat - Detect all heat sources")
        print("  2. person - Detect people")
        print("  3. fire - Detect fire")
        
        mode = input("\nSelect mode (heat/person/fire): ").strip().lower()
        if mode not in ['heat', 'person', 'fire']:
            mode = 'heat'
        
        try:
            detector.run_detection(mode=mode)
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
        finally:
            detector.cleanup()
    else:
        print("❌ Failed to initialize camera")