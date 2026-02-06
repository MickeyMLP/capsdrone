"""
Autonomous Mission Planner
Create and execute GPS waypoint missions
Author: Sue Sha
"""

import time
import math
from dronekit import LocationGlobalRelative, Command
from pymavlink import mavutil

class MissionPlanner:
    """
    Plan and execute autonomous waypoint missions
    Features:
    - Create waypoint missions
    - Upload to drone
    - Monitor mission progress
    - Modify missions in-flight
    """
    
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.current_mission = []
        self.mission_active = False
        self.current_waypoint = 0
        
        print("✅ Mission Planner initialized")
    
    def create_square_mission(self, center_lat, center_lon, altitude, side_length):
        """
        Create a square flight pattern
        Args:
            center_lat, center_lon: Center point coordinates
            altitude: Flight altitude in meters
            side_length: Length of square side in meters
        """
        print(f"📐 Creating square mission: {side_length}m sides at {altitude}m altitude")
        
        # Calculate offset in degrees (approximate)
        # 1 degree latitude ≈ 111km
        offset = side_length / 111000.0
        
        waypoints = [
            (center_lat + offset, center_lon + offset, altitude),  # NE corner
            (center_lat + offset, center_lon - offset, altitude),  # NW corner
            (center_lat - offset, center_lon - offset, altitude),  # SW corner
            (center_lat - offset, center_lon + offset, altitude),  # SE corner
            (center_lat + offset, center_lon + offset, altitude),  # Back to NE
        ]
        
        self.current_mission = waypoints
        print(f"   Created {len(waypoints)} waypoints")
        return waypoints
    
    def create_line_mission(self, start_lat, start_lon, end_lat, end_lon, altitude, num_points=5):
        """
        Create a straight line mission with multiple waypoints
        """
        print(f"📏 Creating line mission: {num_points} waypoints at {altitude}m")
        
        waypoints = []
        for i in range(num_points):
            t = i / (num_points - 1)  # 0 to 1
            lat = start_lat + t * (end_lat - start_lat)
            lon = start_lon + t * (end_lon - start_lon)
            waypoints.append((lat, lon, altitude))
        
        self.current_mission = waypoints
        print(f"   Created {len(waypoints)} waypoints")
        return waypoints
    
    def create_circle_mission(self, center_lat, center_lon, altitude, radius, num_points=8):
        """
        Create a circular flight pattern
        """
        print(f"⭕ Creating circle mission: {radius}m radius, {num_points} points")
        
        waypoints = []
        for i in range(num_points + 1):  # +1 to close the circle
            angle = 2 * math.pi * i / num_points
            
            # Calculate offset in degrees
            lat_offset = (radius * math.cos(angle)) / 111000.0
            lon_offset = (radius * math.sin(angle)) / (111000.0 * math.cos(math.radians(center_lat)))
            
            waypoints.append((
                center_lat + lat_offset,
                center_lon + lon_offset,
                altitude
            ))
        
        self.current_mission = waypoints
        print(f"   Created {len(waypoints)} waypoints")
        return waypoints
    
    def create_custom_mission(self, waypoints):
        """
        Create custom mission from list of (lat, lon, alt) tuples
        """
        print(f"🎯 Creating custom mission: {len(waypoints)} waypoints")
        self.current_mission = waypoints
        return waypoints
    
    def upload_mission(self):
        """Upload current mission to drone"""
        if not self.vehicle:
            print("⚠️ No vehicle connected")
            return False
        
        if not self.current_mission:
            print("⚠️ No mission created")
            return False
        
        print("📤 Uploading mission to drone...")
        
        try:
            # Get mission commands
            cmds = self.vehicle.commands
            cmds.clear()
            
            # Add takeoff command (first waypoint)
            first_wp = self.current_mission[0]
            cmds.add(Command(
                0, 0, 0,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
                0, 0, 0, 0, 0, 0,
                first_wp[0], first_wp[1], first_wp[2]
            ))
            
            # Add waypoints
            for wp in self.current_mission:
                cmds.add(Command(
                    0, 0, 0,
                    mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                    mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                    0, 0, 0, 0, 0, 0,
                    wp[0], wp[1], wp[2]
                ))
            
            # Add RTL command at end
            cmds.add(Command(
                0, 0, 0,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT,
                mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
                0, 0, 0, 0, 0, 0, 0, 0, 0
            ))
            
            # Upload
            cmds.upload()
            print(f"✅ Mission uploaded: {len(self.current_mission)} waypoints")
            return True
            
        except Exception as e:
            print(f"❌ Mission upload failed: {e}")
            return False
    
    def start_mission(self):
        """Start executing the uploaded mission"""
        if not self.vehicle:
            print("⚠️ No vehicle connected")
            return False
        
        try:
            print("🚀 Starting mission...")
            
            # Set mode to AUTO
            self.vehicle.mode = 'AUTO'
            self.mission_active = True
            self.current_waypoint = 0
            
            print("✅ Mission started")
            return True
            
        except Exception as e:
            print(f"❌ Failed to start mission: {e}")
            return False
    
    def pause_mission(self):
        """Pause current mission"""
        if self.vehicle:
            print("⏸️ Pausing mission...")
            self.vehicle.mode = 'GUIDED'
            self.mission_active = False
    
    def resume_mission(self):
        """Resume paused mission"""
        if self.vehicle:
            print("▶️ Resuming mission...")
            self.vehicle.mode = 'AUTO'
            self.mission_active = True
    
    def abort_mission(self):
        """Abort mission and return to launch"""
        if self.vehicle:
            print("🛑 Aborting mission...")
            self.vehicle.mode = 'RTL'
            self.mission_active = False
    
    def get_mission_progress(self):
        """Get current mission progress"""
        if not self.vehicle or not self.mission_active:
            return None
        
        try:
            current_wp = self.vehicle.commands.next
            total_wp = self.vehicle.commands.count
            
            if total_wp > 0:
                progress = (current_wp / total_wp) * 100
            else:
                progress = 0
            
            return {
                'current_waypoint': current_wp,
                'total_waypoints': total_wp,
                'progress_percent': progress,
                'mission_active': self.mission_active
            }
            
        except Exception as e:
            print(f"⚠️ Error getting mission progress: {e}")
            return None
    
    def monitor_mission(self, callback=None):
        """
        Monitor mission execution
        Args:
            callback: Function to call with progress updates
        """
        if not self.vehicle:
            print("⚠️ No vehicle connected")
            return
        
        print("👀 Monitoring mission...")
        
        try:
            while self.mission_active:
                progress = self.get_mission_progress()
                
                if progress:
                    print(f"   Waypoint {progress['current_waypoint']}/{progress['total_waypoints']} "
                          f"({progress['progress_percent']:.1f}%)")
                    
                    if callback:
                        callback(progress)
                    
                    # Check if mission complete
                    if progress['current_waypoint'] >= progress['total_waypoints']:
                        print("✅ Mission complete!")
                        self.mission_active = False
                        break
                
                time.sleep(2)  # Update every 2 seconds
                
        except KeyboardInterrupt:
            print("\n⏹️ Monitoring stopped")
    
    def goto_waypoint(self, lat, lon, alt):
        """
        Go to specific waypoint immediately (GUIDED mode)
        """
        if not self.vehicle:
            print("⚠️ No vehicle connected")
            return False
        
        try:
            print(f"🎯 Going to waypoint: ({lat:.6f}, {lon:.6f}, {alt}m)")
            
            # Switch to GUIDED mode
            self.vehicle.mode = 'GUIDED'
            time.sleep(1)
            
            # Create location and command
            target = LocationGlobalRelative(lat, lon, alt)
            self.vehicle.simple_goto(target)
            
            print("✅ Command sent")
            return True
            
        except Exception as e:
            print(f"❌ Failed to go to waypoint: {e}")
            return False
    
    def calculate_mission_distance(self):
        """Calculate total distance of current mission"""
        if not self.current_mission or len(self.current_mission) < 2:
            return 0
        
        total_distance = 0
        for i in range(len(self.current_mission) - 1):
            wp1 = self.current_mission[i]
            wp2 = self.current_mission[i + 1]
            
            distance = self._calculate_distance(
                wp1[0], wp1[1], wp2[0], wp2[1]
            )
            total_distance += distance
        
        return total_distance
    
    def _calculate_distance(self, lat1, lon1, lat2, lon2):
        """Calculate distance between two GPS points (Haversine)"""
        R = 6371000  # Earth radius in meters
        
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lat = math.radians(lat2 - lat1)
        delta_lon = math.radians(lon2 - lon1)
        
        a = (math.sin(delta_lat/2)**2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * 
             math.sin(delta_lon/2)**2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        
        return R * c
    
    def save_mission(self, filename):
        """Save current mission to file"""
        if not self.current_mission:
            print("⚠️ No mission to save")
            return
        
        with open(filename, 'w') as f:
            f.write("# Drone Mission File\n")
            f.write("# Format: lat,lon,alt\n")
            for wp in self.current_mission:
                f.write(f"{wp[0]},{wp[1]},{wp[2]}\n")
        
        print(f"💾 Mission saved to {filename}")
    
    def load_mission(self, filename):
        """Load mission from file"""
        try:
            waypoints = []
            with open(filename, 'r') as f:
                for line in f:
                    if line.startswith('#'):
                        continue
                    parts = line.strip().split(',')
                    if len(parts) == 3:
                        waypoints.append((
                            float(parts[0]),
                            float(parts[1]),
                            float(parts[2])
                        ))
            
            self.current_mission = waypoints
            print(f"📂 Mission loaded from {filename}: {len(waypoints)} waypoints")
            return waypoints
            
        except Exception as e:
            print(f"❌ Failed to load mission: {e}")
            return None


# Test function
if __name__ == "__main__":
    print("Testing Mission Planner (Simulation Mode)")
    
    # Mock vehicle
    class MockVehicle:
        class Commands:
            count = 0
            next = 0
            
            def clear(self):
                print("   Commands cleared")
            
            def add(self, cmd):
                self.count += 1
                print(f"   Added command {self.count}")
            
            def upload(self):
                print("   Commands uploaded")
        
        commands = Commands()
        mode = "GUIDED"
        
        def simple_goto(self, location):
            print(f"   Going to: {location.lat:.6f}, {location.lon:.6f}, {location.alt}m")
    
    vehicle = MockVehicle()
    planner = MissionPlanner(vehicle)
    
    # Test creating missions
    print("\n--- Test Square Mission ---")
    center_lat, center_lon = 13.7563, 100.5018  # Bangkok
    waypoints = planner.create_square_mission(center_lat, center_lon, 10, 50)
    print(f"Total distance: {planner.calculate_mission_distance():.1f}m")
    
    print("\n--- Test Circle Mission ---")
    waypoints = planner.create_circle_mission(center_lat, center_lon, 10, 30, 8)
    print(f"Total distance: {planner.calculate_mission_distance():.1f}m")
    
    print("\n--- Test Upload ---")
    planner.upload_mission()
    
    print("\n--- Test Save/Load ---")
    planner.save_mission("test_mission.txt")
    planner.load_mission("test_mission.txt")
    
    print("\n✅ Test complete")