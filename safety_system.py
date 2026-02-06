"""
Safety System Module
Monitors drone safety parameters and triggers emergency actions
Author: Sue Sha
"""

import time
import math
import threading
from datetime import datetime

class SafetyMonitor:
    """
    Monitors critical safety parameters:
    - Altitude limits
    - Battery level
    - Tilt angle (crash detection)
    - Geofencing
    - Signal strength
    """
    
    def __init__(self, vehicle, config):
        self.vehicle = vehicle
        self.config = config
        
        # Safety thresholds
        self.max_altitude = config.get('max_altitude', 10)  # meters
        self.max_distance = config.get('max_distance', 100)  # meters
        self.min_battery = config.get('min_battery', 20)  # percentage
        self.max_tilt_angle = config.get('max_tilt_angle', 45)  # degrees
        self.geofence_enabled = config.get('geofence_enabled', True)
        self.geofence_radius = config.get('geofence_radius', 100)  # meters
        
        # Home position (set at takeoff)
        self.home_lat = None
        self.home_lon = None
        
        # Monitoring flags
        self.monitoring = False
        self.emergency_triggered = False
        self.monitor_thread = None
        
        # Safety log
        self.safety_log = []
        
        print("✅ Safety Monitor initialized")
        print(f"   Max altitude: {self.max_altitude}m")
        print(f"   Min battery: {self.min_battery}%")
        print(f"   Max tilt: {self.max_tilt_angle}°")
        print(f"   Geofence: {self.geofence_radius}m")
    
    def set_home_position(self):
        """Set current position as home (call at takeoff)"""
        if self.vehicle:
            self.home_lat = self.vehicle.location.global_relative_frame.lat
            self.home_lon = self.vehicle.location.global_relative_frame.lon
            print(f"🏠 Home position set: {self.home_lat:.6f}, {self.home_lon:.6f}")
        else:
            print("⚠️ No vehicle connected, using mock home position")
            self.home_lat = 0.0
            self.home_lon = 0.0
    
    def start_monitoring(self):
        """Start safety monitoring in background thread"""
        if self.monitoring:
            print("⚠️ Monitoring already running")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("🔍 Safety monitoring started")
    
    def stop_monitoring(self):
        """Stop safety monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        print("⏹️ Safety monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop (runs in background)"""
        while self.monitoring:
            if self.vehicle:
                # Check all safety conditions
                self.check_altitude()
                self.check_battery()
                self.check_tilt()
                self.check_geofence()
                
            time.sleep(0.5)  # Check twice per second
    
    def check_altitude(self):
        """Check if altitude exceeds limit"""
        if not self.vehicle:
            return True
        
        current_alt = self.vehicle.location.global_relative_frame.alt
        
        if current_alt > self.max_altitude:
            self.log_warning(f"Altitude exceeded: {current_alt:.1f}m > {self.max_altitude}m")
            return False
        return True
    
    def check_battery(self):
        """Check battery level"""
        if not self.vehicle:
            return True
        
        battery_level = self.vehicle.battery.level
        
        if battery_level < self.min_battery:
            self.log_warning(f"Low battery: {battery_level}% < {self.min_battery}%")
            return False
        return True
    
    def check_tilt(self):
        """Check drone tilt angle (crash detection)"""
        if not self.vehicle:
            return True
        
        # Get pitch and roll in radians
        pitch = self.vehicle.attitude.pitch
        roll = self.vehicle.attitude.roll
        
        # Convert to degrees
        pitch_deg = math.degrees(pitch)
        roll_deg = math.degrees(roll)
        
        # Check if tilt exceeds threshold
        if abs(pitch_deg) > self.max_tilt_angle or abs(roll_deg) > self.max_tilt_angle:
            self.log_warning(f"Excessive tilt detected! Pitch: {pitch_deg:.1f}°, Roll: {roll_deg:.1f}°")
            return False
        return True
    
    def check_geofence(self):
        """Check if drone is within geofence"""
        if not self.vehicle or not self.geofence_enabled:
            return True
        
        if self.home_lat is None or self.home_lon is None:
            return True  # Home not set yet
        
        current_lat = self.vehicle.location.global_relative_frame.lat
        current_lon = self.vehicle.location.global_relative_frame.lon
        
        distance = self.calculate_distance(
            self.home_lat, self.home_lon,
            current_lat, current_lon
        )
        
        if distance > self.geofence_radius:
            self.log_warning(f"Geofence breach: {distance:.1f}m > {self.geofence_radius}m")
            return False
        return True
    
    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two GPS coordinates (Haversine formula)"""
        R = 6371000  # Earth radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        distance = R * c
        return distance
    
    def is_safe(self):
        """Check if all safety conditions are met"""
        if not self.vehicle:
            return True  # Simulation mode
        
        return (self.check_altitude() and 
                self.check_battery() and 
                self.check_tilt() and 
                self.check_geofence())
    
    def handle_emergency(self):
        """Execute emergency protocol"""
        if self.emergency_triggered:
            return  # Already handling emergency
        
        self.emergency_triggered = True
        print("🚨 EMERGENCY PROTOCOL ACTIVATED")
        
        if not self.vehicle:
            print("   [SIMULATION] Would execute emergency landing")
            return
        
        try:
            # Check what triggered emergency
            if not self.check_battery():
                print("   → Low battery: Returning to home")
                self.return_to_home()
            
            elif not self.check_tilt():
                print("   → Excessive tilt: Emergency landing")
                self.emergency_land()
            
            elif not self.check_geofence():
                print("   → Geofence breach: Returning to home")
                self.return_to_home()
            
            elif not self.check_altitude():
                print("   → Altitude limit: Descending")
                self.descend_to_safe_altitude()
            
        except Exception as e:
            print(f"❌ Emergency handling failed: {e}")
            self.emergency_land()
        
        finally:
            self.emergency_triggered = False
    
    def return_to_home(self):
        """Command drone to return to home position"""
        if self.vehicle:
            print("🏠 Returning to home...")
            self.vehicle.mode = 'RTL'  # Return To Launch mode
    
    def emergency_land(self):
        """Command immediate landing"""
        if self.vehicle:
            print("🛬 Emergency landing...")
            self.vehicle.mode = 'LAND'
    
    def descend_to_safe_altitude(self):
        """Descend to safe altitude"""
        if self.vehicle:
            safe_alt = self.max_altitude * 0.8  # 80% of max
            print(f"⬇️ Descending to safe altitude: {safe_alt}m")
            # Implementation would depend on Section 1 flight control
    
    def log_warning(self, message):
        """Log safety warning"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}"
        self.safety_log.append(log_entry)
        print(f"⚠️ {log_entry}")
    
    def get_safety_status(self):
        """Get current safety status as dictionary"""
        if not self.vehicle:
            return {
                'safe': True,
                'mode': 'SIMULATION',
                'altitude': 0,
                'battery': 100,
                'tilt': 0,
                'distance_from_home': 0
            }
        
        current_lat = self.vehicle.location.global_relative_frame.lat
        current_lon = self.vehicle.location.global_relative_frame.lon
        
        if self.home_lat and self.home_lon:
            distance = self.calculate_distance(
                self.home_lat, self.home_lon,
                current_lat, current_lon
            )
        else:
            distance = 0
        
        return {
            'safe': self.is_safe(),
            'mode': str(self.vehicle.mode),
            'altitude': self.vehicle.location.global_relative_frame.alt,
            'battery': self.vehicle.battery.level,
            'tilt_pitch': math.degrees(self.vehicle.attitude.pitch),
            'tilt_roll': math.degrees(self.vehicle.attitude.roll),
            'distance_from_home': distance,
            'warnings': len(self.safety_log)
        }
    
    def save_log(self, filename='safety_log.txt'):
        """Save safety log to file"""
        with open(filename, 'w') as f:
            f.write("=== Drone Safety Log ===\n\n")
            for entry in self.safety_log:
                f.write(entry + "\n")
        print(f"📝 Safety log saved to {filename}")


# Test function
if __name__ == "__main__":
    print("Testing Safety Monitor (Simulation Mode)")
    
    # Mock vehicle for testing
    class MockVehicle:
        class Location:
            class GlobalRelativeFrame:
                lat = 13.7563
                lon = 100.5018
                alt = 5.0
            
            global_relative_frame = GlobalRelativeFrame()
        
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
    
    # Test configuration
    config = {
        'max_altitude': 10,
        'max_distance': 100,
        'min_battery': 20,
        'max_tilt_angle': 45,
        'geofence_enabled': True,
        'geofence_radius': 100
    }
    
    # Create safety monitor
    vehicle = MockVehicle()
    safety = SafetyMonitor(vehicle, config)
    safety.set_home_position()
    
    # Test safety checks
    print("\n--- Testing Safety Checks ---")
    print(f"Altitude safe: {safety.check_altitude()}")
    print(f"Battery safe: {safety.check_battery()}")
    print(f"Tilt safe: {safety.check_tilt()}")
    print(f"Geofence safe: {safety.check_geofence()}")
    print(f"Overall safe: {safety.is_safe()}")
    
    # Get status
    print("\n--- Safety Status ---")
    status = safety.get_safety_status()
    for key, value in status.items():
        print(f"{key}: {value}")
    
    print("\n✅ Test complete")