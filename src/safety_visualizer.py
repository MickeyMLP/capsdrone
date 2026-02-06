"""
Visual Safety Monitor Dashboard
Real-time visualization of drone safety status
Author: Sue Sha
"""

import cv2
import numpy as np
import math
import time
from datetime import datetime
import sys
sys.path.append('..')

from safety_system import SafetyMonitor

class SafetyVisualizer:
    """
    Create visual dashboard for safety monitoring
    """
    
    def __init__(self, safety_monitor):
        self.safety = safety_monitor
        
        # Window size
        self.width = 1200
        self.height = 800
        
        # Colors (BGR format)
        self.COLOR_SAFE = (0, 255, 0)      # Green
        self.COLOR_WARNING = (0, 165, 255)  # Orange
        self.COLOR_DANGER = (0, 0, 255)     # Red
        self.COLOR_BG = (40, 40, 40)        # Dark gray
        self.COLOR_TEXT = (255, 255, 255)   # White
        
        print("✅ Safety Visualizer initialized")
    
    def draw_gauge(self, img, x, y, radius, value, max_value, label, unit=""):
        """
        Draw circular gauge
        Args:
            img: Image to draw on
            x, y: Center position
            radius: Gauge radius
            value: Current value
            max_value: Maximum value
            label: Gauge label
            unit: Unit string (e.g., "m", "%")
        """
        # Calculate percentage
        percent = min(value / max_value, 1.0) if max_value > 0 else 0
        
        # Determine color based on percentage
        if percent < 0.5:
            color = self.COLOR_SAFE
        elif percent < 0.8:
            color = self.COLOR_WARNING
        else:
            color = self.COLOR_DANGER
        
        # Draw outer circle
        cv2.circle(img, (x, y), radius, (100, 100, 100), 3)
        
        # Draw filled arc for value
        angle = int(percent * 270) - 135  # -135 to 135 degrees
        
        if percent > 0:
            # Create arc points
            pts = [np.array([x, y])]
            for i in range(-135, angle):
                rad = math.radians(i)
                px = int(x + (radius - 10) * math.cos(rad))
                py = int(y + (radius - 10) * math.sin(rad))
                pts.append([px, py])
            pts = np.array(pts)
            
            cv2.fillPoly(img, [pts], color)
        
        # Draw center circle
        cv2.circle(img, (x, y), radius - 15, self.COLOR_BG, -1)
        
        # Draw value text
        value_text = f"{value:.1f}{unit}"
        text_size = cv2.getTextSize(value_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = x - text_size[0] // 2
        text_y = y + text_size[1] // 2
        cv2.putText(img, value_text, (text_x, text_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.COLOR_TEXT, 2)
        
        # Draw label
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
        label_x = x - label_size[0] // 2
        label_y = y + radius + 30
        cv2.putText(img, label, (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TEXT, 1)
    
    def draw_tilt_indicator(self, img, x, y, size, pitch, roll):
        """
        Draw attitude indicator (artificial horizon)
        """
        # Draw outer box
        cv2.rectangle(img, (x - size//2, y - size//2), 
                     (x + size//2, y + size//2), (100, 100, 100), 3)
        
        # Calculate tilt
        max_tilt = 45.0
        pitch_offset = int((pitch / max_tilt) * size // 2)
        roll_rad = math.radians(roll)
        
        # Draw horizon line
        horizon_y = y + pitch_offset
        
        # Calculate rotated line endpoints
        half_width = size // 2
        x1 = int(x - half_width * math.cos(roll_rad))
        y1 = int(horizon_y + half_width * math.sin(roll_rad))
        x2 = int(x + half_width * math.cos(roll_rad))
        y2 = int(horizon_y - half_width * math.sin(roll_rad))
        
        # Draw sky (blue) and ground (brown)
        points_sky = np.array([[x - size//2, y - size//2],
                              [x + size//2, y - size//2],
                              [x2, y2], [x1, y1]])
        points_ground = np.array([[x - size//2, y + size//2],
                                 [x + size//2, y + size//2],
                                 [x2, y2], [x1, y1]])
        
        cv2.fillPoly(img, [points_sky], (139, 69, 19))      # Brown (ground)
        cv2.fillPoly(img, [points_ground], (200, 150, 100))  # Sky blue
        
        # Draw horizon line
        cv2.line(img, (x1, y1), (x2, y2), self.COLOR_TEXT, 3)
        
        # Draw center marker
        cv2.circle(img, (x, y), 5, self.COLOR_WARNING, -1)
        cv2.circle(img, (x, y), 8, self.COLOR_TEXT, 2)
        
        # Draw pitch/roll values
        cv2.putText(img, f"P: {pitch:.1f}", (x - size//2 + 10, y - size//2 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1)
        cv2.putText(img, f"R: {roll:.1f}", (x - size//2 + 10, y - size//2 + 45),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1)
        
        # Label
        cv2.putText(img, "ATTITUDE", (x - 40, y + size//2 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TEXT, 1)
    
    def draw_geofence_map(self, img, x, y, size, distance, max_distance):
        """
        Draw geofence visualization (top-down map view)
        """
        # Draw geofence circle
        cv2.circle(img, (x, y), size//2, (100, 100, 100), 2)
        
        # Draw distance rings
        for i in range(1, 4):
            radius = int(size//2 * i / 3)
            cv2.circle(img, (x, y), radius, (60, 60, 60), 1)
        
        # Draw home position
        cv2.circle(img, (x, y), 8, self.COLOR_SAFE, -1)
        cv2.putText(img, "HOME", (x - 25, y - 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_TEXT, 1)
        
        # Draw drone position
        if max_distance > 0:
            drone_distance = min(distance, max_distance)
            drone_radius = int((drone_distance / max_distance) * size//2)
            
            # Drone position (for visualization, assume it's to the right)
            drone_x = x + drone_radius
            drone_y = y
            
            # Draw line from home to drone
            cv2.line(img, (x, y), (drone_x, drone_y), self.COLOR_WARNING, 2)
            
            # Draw drone
            color = self.COLOR_SAFE if distance < max_distance * 0.8 else self.COLOR_DANGER
            cv2.circle(img, (drone_x, drone_y), 10, color, -1)
            cv2.circle(img, (drone_x, drone_y), 12, self.COLOR_TEXT, 2)
        
        # Draw distance text
        cv2.putText(img, f"{distance:.1f}m / {max_distance}m", 
                   (x - size//2, y + size//2 + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TEXT, 1)
        
        # Label
        cv2.putText(img, "GEOFENCE", (x - 45, y - size//2 - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TEXT, 1)
    
    def draw_status_panel(self, img, x, y, width, height, status):
        """
        Draw overall status panel
        """
        # Background
        cv2.rectangle(img, (x, y), (x + width, y + height), (60, 60, 60), -1)
        cv2.rectangle(img, (x, y), (x + width, y + height), (100, 100, 100), 3)

        # Overall status
        is_safe = bool(status.get('safe', False))
        status_color = self.COLOR_SAFE if is_safe else self.COLOR_DANGER
        status_text = "ALL SYSTEMS NOMINAL" if is_safe else "WARNING"  # ASCII only

        cv2.putText(img, status_text, (x + 20, y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)

        # Mode (normalize object -> readable string)
        mode = status.get('mode', 'UNKNOWN')
        if hasattr(mode, "name"):
            mode = mode.name
        else:
            mode = str(mode)

        cv2.putText(img, f"Mode: {mode}", (x + 20, y + 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TEXT, 1)

        # Individual checks
        checks = [
            ("Altitude", status.get('altitude', 0) <= self.safety.max_altitude),
            ("Battery", status.get('battery', 0) >= self.safety.min_battery),
            ("Tilt", abs(status.get('tilt_pitch', 0)) <= self.safety.max_tilt_angle and
                    abs(status.get('tilt_roll', 0)) <= self.safety.max_tilt_angle),
            ("Geofence", status.get('distance_from_home', 0) <= self.safety.geofence_radius)
        ]

        y_offset = 120
        for check_name, is_ok in checks:
            color = self.COLOR_SAFE if is_ok else self.COLOR_DANGER
            symbol = "[OK]" if is_ok else "[X]"   # ASCII only
            cv2.putText(img, f"{symbol} {check_name}", (x + 20, y + y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 35

        # Warning count
        cv2.putText(img, f"Warnings: {status.get('warnings', 0)}", (x + 20, y + height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WARNING, 1)

    
    def draw_safety_log(self, img, x, y, width, height):
        """
        Draw recent safety warnings
        """
        # Background
        cv2.rectangle(img, (x, y), (x + width, y + height), (50, 50, 50), -1)
        cv2.rectangle(img, (x, y), (x + width, y + height), (100, 100, 100), 2)
        
        # Title
        cv2.putText(img, "SAFETY LOG", (x + 10, y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.COLOR_TEXT, 1)
        
        # Show last 5 warnings
        log_entries = self.safety.safety_log[-5:] if self.safety.safety_log else []
        
        y_offset = 55
        for entry in log_entries:
            # Truncate if too long
            display_text = entry if len(entry) < 60 else entry[:57] + "..."
            # Extract just the message part (after timestamp)
            if "]" in display_text:
                display_text = display_text.split("]", 1)[1].strip()
            
            cv2.putText(img, display_text, (x + 10, y + y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.COLOR_WARNING, 1)
            y_offset += 25
        
        if not log_entries:
            cv2.putText(img, "No warnings", (x + 10, y + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    def create_dashboard(self, status):
        """
        Create complete safety dashboard
        """
        # Create blank canvas
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = self.COLOR_BG
        
        # Title
        cv2.putText(img, "DRONE SAFETY MONITOR", (self.width//2 - 200, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, self.COLOR_TEXT, 2)
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(img, timestamp, (self.width - 250, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        # Draw gauges (top row)
        gauge_y = 180
        self.draw_gauge(img, 150, gauge_y, 80, 
                       status['altitude'], self.safety.max_altitude,
                       "ALTITUDE", "m")
        
        self.draw_gauge(img, 350, gauge_y, 80,
                       100 - status['battery'], 100,
                       "BATTERY USED", "%")
        
        # Tilt indicator (center)
        self.draw_tilt_indicator(img, 600, 200, 200,
                                status['tilt_pitch'], status['tilt_roll'])
        
        # Geofence map (right)
        self.draw_geofence_map(img, 950, 200, 200,
                              status['distance_from_home'],
                              self.safety.geofence_radius)
        
        # Status panel (bottom left)
        self.draw_status_panel(img, 50, 400, 350, 320, status)
        
        # Safety log (bottom right)
        self.draw_safety_log(img, 450, 400, 700, 320)
        
        # Instructions
        cv2.putText(img, "Press 'q' to quit | 's' to save screenshot | 't' to trigger test warning",
                   (20, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return img
    
    def run(self):
        """
        Run live dashboard
        """
        print("\n🎨 Starting Safety Visualizer...")
        print("Controls:")
        print("  q - Quit")
        print("  s - Save screenshot")
        print("  t - Trigger test warning")
        print("  Arrow keys - Simulate drone movement")
        print("\n")
        
        # Start safety monitoring
        self.safety.start_monitoring()
        
        # Simulation variables
        sim_altitude = 5.0
        sim_battery = 80.0
        sim_pitch = 0.0
        sim_roll = 0.0
        sim_distance = 20.0
        
        try:
            while True:
                # Get status (or use mock data)
                if self.safety.vehicle:
                    status = self.safety.get_safety_status()
                else:
                    # Simulate changing values
                    status = {
                        'safe': (sim_altitude <= self.safety.max_altitude and 
                                sim_battery >= self.safety.min_battery and 
                                abs(sim_pitch) <= self.safety.max_tilt_angle and 
                                abs(sim_roll) <= self.safety.max_tilt_angle and 
                                sim_distance <= self.safety.geofence_radius),
                        'mode': 'GUIDED',
                        'altitude': sim_altitude,
                        'battery': sim_battery,
                        'tilt_pitch': sim_pitch,
                        'tilt_roll': sim_roll,
                        'distance_from_home': sim_distance,
                        'warnings': len(self.safety.safety_log)
                    }
                
                # Create dashboard
                dashboard = self.create_dashboard(status)
                
                # Display
                cv2.imshow('Drone Safety Monitor', dashboard)
                
                # Handle keyboard input
                key = cv2.waitKey(100) & 0xFF
                
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f"safety_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
                    cv2.imwrite(filename, dashboard)
                    print(f"💾 Screenshot saved: {filename}")
                elif key == ord('t'):
                    # Trigger test warning
                    self.safety.log_warning("Test warning triggered manually")
                    print("⚠️ Test warning added to log")
                
                # Simulate drone movement with arrow keys
                elif key == 82:  # Up arrow - increase altitude
                    sim_altitude = min(sim_altitude + 0.5, 15)
                elif key == 84:  # Down arrow - decrease altitude
                    sim_altitude = max(sim_altitude - 0.5, 0)
                elif key == 81:  # Left arrow - increase distance
                    sim_distance = min(sim_distance + 5, 150)
                elif key == 83:  # Right arrow - decrease distance
                    sim_distance = max(sim_distance - 5, 0)
                
                # Simulate battery drain
                sim_battery = max(sim_battery - 0.01, 0)
                
                # Simulate slight attitude changes
                import random
                sim_pitch += random.uniform(-0.5, 0.5)
                sim_roll += random.uniform(-0.5, 0.5)
                sim_pitch = max(-10, min(10, sim_pitch))
                sim_roll = max(-10, min(10, sim_roll))
                
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n⏹️ Stopped by user")
        finally:
            self.safety.stop_monitoring()
            cv2.destroyAllWindows()
            print("✅ Visualizer closed")


# Main execution
if __name__ == "__main__":
    print("="*60)
    print("🎨 Drone Safety Monitor - Visual Dashboard")
    print("="*60)
    
    # Mock vehicle for testing
    class MockVehicle:
        class Location:
            class GlobalRelativeFrame:
                lat = 13.7563
                lon = 100.5018
                alt = 5.0
            
            global_relative_frame = GlobalRelativeFrame()  # Important!
        
        class Battery:
            voltage = 12.6
            current = 15.0
            level = 80
        
        class Attitude:
            pitch = 0.1
            roll = 0.05
            yaw = 1.57
        
        class Mode:
            name = "GUIDED"
        
        location = Location()
        battery = Battery()
        attitude = Attitude()
        mode = Mode()
        velocity = [0, 0, 0]
        groundspeed = 0
        airspeed = 0
        armed = False
        is_armable = True
        ekf_ok = True
    
    # Configuration
    config = {
        'max_altitude': 10,
        'max_distance': 100,
        'min_battery': 20,
        'max_tilt_angle': 45,
        'geofence_enabled': True,
        'geofence_radius': 100
    }
    
    # Create safety monitor with mock vehicle
    vehicle = MockVehicle()
    safety = SafetyMonitor(vehicle, config)
    safety.set_home_position()
    
    # Create and run visualizer
    visualizer = SafetyVisualizer(safety)
    visualizer.run()