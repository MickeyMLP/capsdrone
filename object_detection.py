"""
Object Detection Module for Drone Operations
Detects: People, Obstacles, Landing Pads, Follow Targets
Author: Sue Sha
"""

import cv2
import numpy as np
from datetime import datetime
import time

class DroneObjectDetector:
    """
    Object detection for drone operations:
    - Person detection (search & rescue, follow-me)
    - Obstacle detection (collision avoidance)
    - Landing pad detection (auto-landing)
    - Color-based target tracking
    """
    
    def __init__(self):
        # Camera initialization
        self.camera = None
        self.frame_width = 640
        self.frame_height = 480
        
        # Detection settings
        self.detection_mode = 'person'  # person, obstacle, landing_pad, target
        
        # Face/Person cascade classifier (built-in OpenCV)
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.body_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_fullbody.xml'
        )
        
        # Detection results
        self.detected_objects = []
        self.target_locked = False
        self.target_position = None
        
        # Colors (BGR)
        self.COLOR_PERSON = (0, 255, 0)      # Green
        self.COLOR_OBSTACLE = (0, 0, 255)    # Red
        self.COLOR_TARGET = (255, 0, 255)    # Magenta
        self.COLOR_LANDING = (0, 255, 255)   # Yellow
        
        print("✅ Object Detector initialized")
    
    def init_camera(self, camera_id=0):
        """Initialize camera"""
        try:
            self.camera = cv2.VideoCapture(camera_id)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            
            if not self.camera.isOpened():
                print("❌ Could not open camera")
                return False
            
            print("✅ Camera initialized")
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization failed: {e}")
            return False
    
    def read_frame(self):
        """Read frame from camera"""
        if not self.camera:
            return None
        
        ret, frame = self.camera.read()
        if not ret:
            return None
        
        return frame
    
    # ==================== PERSON DETECTION ====================
    
    def detect_person(self, frame):
        """
        Detect people in frame for search & rescue or follow-me mode
        Returns: list of detected persons with positions
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces (more reliable for close range)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        # Detect full bodies (better for far range)
        bodies = self.body_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=3, minSize=(50, 100)
        )
        
        persons = []
        
        # Process face detections
        for (x, y, w, h) in faces:
            center_x = x + w // 2
            center_y = y + h // 2
            
            persons.append({
                'type': 'face',
                'bbox': (x, y, w, h),
                'center': (center_x, center_y),
                'area': w * h,
                'distance_estimate': self._estimate_distance(w, h, 'face')
            })
        
        # Process body detections
        for (x, y, w, h) in bodies:
            center_x = x + w // 2
            center_y = y + h // 2
            
            persons.append({
                'type': 'body',
                'bbox': (x, y, w, h),
                'center': (center_x, center_y),
                'area': w * h,
                'distance_estimate': self._estimate_distance(w, h, 'body')
            })
        
        return persons
    
    def _estimate_distance(self, width, height, detection_type):
        """
        Estimate distance based on detection size
        (Rough approximation - needs calibration with real drone)
        """
        if detection_type == 'face':
            # Average face width ~15cm, focal length approximation
            known_width = 15  # cm
            focal_length = 600  # pixels (approximate)
            if width > 0:
                distance = (known_width * focal_length) / width
                return distance / 100  # convert to meters
        
        elif detection_type == 'body':
            # Average body height ~170cm
            known_height = 170  # cm
            focal_length = 600
            if height > 0:
                distance = (known_height * focal_length) / height
                return distance / 100
        
        return 0
    
    # ==================== OBSTACLE DETECTION ====================
    
    def detect_obstacles(self, frame):
        """
        Detect obstacles for collision avoidance
        Uses edge detection and contour analysis
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to connect nearby objects
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        obstacles = []
        
        # Filter significant obstacles
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by minimum area (ignore small noise)
            if area > 1000:
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Determine threat level based on position
                # Objects in center are more dangerous
                frame_center_x = self.frame_width // 2
                frame_center_y = self.frame_height // 2
                distance_from_center = np.sqrt(
                    (center_x - frame_center_x)**2 + 
                    (center_y - frame_center_y)**2
                )
                
                threat_level = 'low'
                if distance_from_center < 100:
                    threat_level = 'high'
                elif distance_from_center < 200:
                    threat_level = 'medium'
                
                obstacles.append({
                    'bbox': (x, y, w, h),
                    'center': (center_x, center_y),
                    'area': area,
                    'threat_level': threat_level,
                    'distance_from_center': distance_from_center
                })
        
        return obstacles
    
    # ==================== LANDING PAD DETECTION ====================
    
    def detect_landing_pad(self, frame):
        """
        Detect landing pad for auto-landing
        Looks for specific color/pattern (e.g., orange circle or 'H' marker)
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Define orange color range for landing pad (adjust as needed)
        lower_orange = np.array([5, 100, 100])
        upper_orange = np.array([25, 255, 255])
        
        # Create mask for orange color
        mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        landing_pads = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size (landing pad should be reasonably large)
            if area > 2000:
                # Check if circular (landing pad characteristic)
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                # Calculate offset from frame center (for alignment)
                frame_center_x = self.frame_width // 2
                frame_center_y = self.frame_height // 2
                offset_x = center_x - frame_center_x
                offset_y = center_y - frame_center_y
                
                landing_pads.append({
                    'bbox': (x, y, w, h),
                    'center': (center_x, center_y),
                    'area': area,
                    'circularity': circularity,
                    'offset': (offset_x, offset_y),
                    'aligned': abs(offset_x) < 20 and abs(offset_y) < 20
                })
        
        return landing_pads
    
    # ==================== COLOR TARGET TRACKING ====================
    
    def detect_color_target(self, frame, color='red'):
        """
        Track colored object for follow-me or target tracking
        Colors: 'red', 'blue', 'green', 'yellow'
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Color ranges (HSV)
        color_ranges = {
            'red': ([0, 100, 100], [10, 255, 255]),
            'blue': ([100, 100, 100], [130, 255, 255]),
            'green': ([40, 50, 50], [80, 255, 255]),
            'yellow': ([20, 100, 100], [40, 255, 255])
        }
        
        if color not in color_ranges:
            color = 'red'
        
        lower, upper = color_ranges[color]
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        targets = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area > 500:  # Minimum target size
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                
                targets.append({
                    'bbox': (x, y, w, h),
                    'center': (center_x, center_y),
                    'area': area,
                    'color': color
                })
        
        return targets
    
    # ==================== VISUALIZATION ====================
    
    def annotate_detections(self, frame, detections, detection_type):
        """Draw annotations on frame"""
        annotated = frame.copy()
        
        color_map = {
            'person': self.COLOR_PERSON,
            'obstacle': self.COLOR_OBSTACLE,
            'landing_pad': self.COLOR_LANDING,
            'target': self.COLOR_TARGET
        }
        
        color = color_map.get(detection_type, (255, 255, 255))
        
        for det in detections:
            x, y, w, h = det['bbox']
            center = det['center']
            
            # Draw bounding box
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            
            # Draw center point
            cv2.circle(annotated, center, 5, color, -1)
            
            # Draw crosshair at center
            cv2.line(annotated, (center[0] - 10, center[1]), (center[0] + 10, center[1]), color, 2)
            cv2.line(annotated, (center[0], center[1] - 10), (center[0], center[1] + 10), color, 2)
            
            # Label
            if detection_type == 'person':
                label = f"{det['type']} {det['distance_estimate']:.1f}m"
            elif detection_type == 'obstacle':
                label = f"Obstacle - {det['threat_level']}"
            elif detection_type == 'landing_pad':
                label = f"PAD {'✓' if det['aligned'] else '✗'}"
            elif detection_type == 'target':
                label = f"{det['color'].upper()}"
            else:
                label = detection_type
            
            cv2.putText(annotated, label, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return annotated
    
    def draw_guidance_overlay(self, frame, detections, detection_type):
        """Draw guidance information for drone control"""
        overlay = frame.copy()
        
        # Draw center crosshair
        center_x = self.frame_width // 2
        center_y = self.frame_height // 2
        cv2.line(overlay, (center_x - 20, center_y), (center_x + 20, center_y), (0, 255, 0), 2)
        cv2.line(overlay, (center_x, center_y - 20), (center_x, center_y + 20), (0, 255, 0), 2)
        
        # Draw safe zones (for obstacle avoidance)
        if detection_type == 'obstacle':
            cv2.circle(overlay, (center_x, center_y), 100, (0, 255, 255), 2)
            cv2.circle(overlay, (center_x, center_y), 200, (0, 165, 255), 1)
        
        # Draw alignment guides (for landing)
        if detection_type == 'landing_pad':
            cv2.rectangle(overlay, (center_x - 30, center_y - 30), 
                         (center_x + 30, center_y + 30), (0, 255, 255), 2)
        
        # Find largest/closest detection
        if detections:
            largest = max(detections, key=lambda x: x['area'])
            target_center = largest['center']
            
            # Draw line from center to target
            cv2.line(overlay, (center_x, center_y), target_center, (255, 0, 255), 2)
            
            # Calculate offset
            offset_x = target_center[0] - center_x
            offset_y = target_center[1] - center_y
            
            # Draw guidance text
            if offset_x > 20:
                direction = "→ MOVE RIGHT"
            elif offset_x < -20:
                direction = "← MOVE LEFT"
            else:
                direction = "✓ CENTERED"
            
            cv2.putText(overlay, direction, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return overlay
    
    # ==================== MAIN DETECTION LOOP ====================
    
    def run_detection(self, mode='person'):
        """
        Run real-time detection
        Modes: 'person', 'obstacle', 'landing_pad', 'target'
        """
        if not self.camera:
            print("❌ Camera not initialized")
            return
        
        self.detection_mode = mode
        print(f"\n🎯 Starting Object Detection: {mode.upper()} mode")
        print("Controls:")
        print("  q - Quit")
        print("  s - Save screenshot")
        print("  1 - Person detection")
        print("  2 - Obstacle detection")
        print("  3 - Landing pad detection")
        print("  4 - Color target tracking")
        print("\n")
        
        target_color = 'red'  # For target tracking mode
        
        while True:
            frame = self.read_frame()
            if frame is None:
                print("❌ Failed to read frame")
                break
            
            # Perform detection based on mode
            if mode == 'person':
                detections = self.detect_person(frame)
            elif mode == 'obstacle':
                detections = self.detect_obstacles(frame)
            elif mode == 'landing_pad':
                detections = self.detect_landing_pad(frame)
            elif mode == 'target':
                detections = self.detect_color_target(frame, target_color)
            else:
                detections = []
            
            # Annotate frame
            if detections:
                display_frame = self.annotate_detections(frame, detections, mode)
                display_frame = self.draw_guidance_overlay(display_frame, detections, mode)
            else:
                display_frame = frame.copy()
                cv2.putText(display_frame, f"No {mode} detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            
            # Display info
            info_text = f"Mode: {mode.upper()} | Detected: {len(detections)}"
            cv2.putText(display_frame, info_text, (10, display_frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Drone Object Detection', display_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"detection_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(filename, display_frame)
                print(f"💾 Saved: {filename}")
            elif key == ord('1'):
                mode = 'person'
                print(f"🔄 Switched to: {mode.upper()}")
            elif key == ord('2'):
                mode = 'obstacle'
                print(f"🔄 Switched to: {mode.upper()}")
            elif key == ord('3'):
                mode = 'landing_pad'
                print(f"🔄 Switched to: {mode.upper()}")
            elif key == ord('4'):
                mode = 'target'
                print(f"🔄 Switched to: {mode.upper()}")
        
        cv2.destroyAllWindows()
    
    def get_detection_command(self, detections, mode):
        """
        Convert detections to drone control commands
        Returns: dict with movement commands
        """
        if not detections:
            return {'action': 'hover', 'reason': 'no_target'}
        
        # Get primary target (largest/closest)
        target = max(detections, key=lambda x: x['area'])
        center = target['center']
        
        # Calculate offset from frame center
        center_x = self.frame_width // 2
        center_y = self.frame_height // 2
        offset_x = center[0] - center_x
        offset_y = center[1] - center_y
        
        command = {
            'action': 'track',
            'offset_x': offset_x,
            'offset_y': offset_y,
            'target_center': center,
            'move_left': offset_x < -30,
            'move_right': offset_x > 30,
            'move_up': offset_y < -30,
            'move_down': offset_y > 30,
            'centered': abs(offset_x) < 30 and abs(offset_y) < 30
        }
        
        return command
    
    def cleanup(self):
        """Release camera resources"""
        if self.camera:
            self.camera.release()
        cv2.destroyAllWindows()
        print("✅ Object detector cleanup complete")


# Test function
if __name__ == "__main__":
    print("="*60)
    print("🎯 Drone Object Detection System")
    print("="*60)
    
    detector = DroneObjectDetector()
    
    if detector.init_camera(0):
        print("\nSelect detection mode:")
        print("  1 - Person detection (search & rescue, follow-me)")
        print("  2 - Obstacle detection (collision avoidance)")
        print("  3 - Landing pad detection (auto-landing)")
        print("  4 - Color target tracking")
        
        choice = input("\nEnter choice (1-4): ").strip()
        
        mode_map = {
            '1': 'person',
            '2': 'obstacle',
            '3': 'landing_pad',
            '4': 'target'
        }
        
        mode = mode_map.get(choice, 'person')
        
        try:
            detector.run_detection(mode=mode)
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
        finally:
            detector.cleanup()
    else:
        print("❌ Failed to initialize camera")