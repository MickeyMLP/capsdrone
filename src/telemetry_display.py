"""
Telemetry Display Module
Real-time display of drone telemetry data
Author: Sue Sha
"""

import time
import threading
import math
from datetime import datetime
import json

class TelemetryDisplay:
    """
    Display real-time drone telemetry:
    - GPS position
    - Altitude
    - Battery level
    - Speed
    - Attitude (pitch, roll, yaw)
    - Flight mode
    """
    
    def __init__(self, vehicle, update_rate=10):
        self.vehicle = vehicle
        self.update_rate = update_rate  # Hz
        self.running = False
        self.display_thread = None
        
        # Telemetry data storage
        self.telemetry_log = []
        self.max_log_entries = 1000
        
        print("✅ Telemetry Display initialized")
        print(f"   Update rate: {update_rate} Hz")
    
    def start(self):
        """Start telemetry display in background thread"""
        if self.running:
            print("⚠️ Telemetry already running")
            return
        
        self.running = True
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()
        print("📊 Telemetry display started")
    
    def stop(self):
        """Stop telemetry display"""
        self.running = False
        if self.display_thread:
            self.display_thread.join(timeout=2)
        print("⏹️ Telemetry display stopped")
    
    def _display_loop(self):
        """Main display loop (runs in background)"""
        while self.running:
            self.update()
            time.sleep(1.0 / self.update_rate)
    
    def update(self):
        """Update and display telemetry"""
        if not self.vehicle:
            return
        
        data = self.get_telemetry_data()
        self.log_telemetry(data)
        
        # Don't print every frame to avoid spam
        # You can implement GUI display here instead
    
    def get_telemetry_data(self):
        """Get current telemetry data as dictionary"""
        if not self.vehicle:
            return self._get_mock_data()
        
        try:
            # GPS and position
            location = self.vehicle.location.global_relative_frame
            gps_lat = location.lat
            gps_lon = location.lon
            altitude = location.alt
            
            # Battery
            battery_voltage = self.vehicle.battery.voltage
            battery_current = self.vehicle.battery.current
            battery_level = self.vehicle.battery.level
            
            # Attitude
            pitch = math.degrees(self.vehicle.attitude.pitch)
            roll = math.degrees(self.vehicle.attitude.roll)
            yaw = math.degrees(self.vehicle.attitude.yaw)
            
            # Velocity
            velocity = self.vehicle.velocity
            groundspeed = self.vehicle.groundspeed
            airspeed = self.vehicle.airspeed
            
            # Flight status
            mode = str(self.vehicle.mode.name)
            armed = self.vehicle.armed
            is_armable = self.vehicle.is_armable
            
            # System status
            ekf_ok = self.vehicle.ekf_ok
            
            data = {
                'timestamp': datetime.now().isoformat(),
                'gps': {
                    'lat': gps_lat,
                    'lon': gps_lon,
                    'alt': altitude
                },
                'battery': {
                    'voltage': battery_voltage,
                    'current': battery_current,
                    'level': battery_level
                },
                'attitude': {
                    'pitch': pitch,
                    'roll': roll,
                    'yaw': yaw
                },
                'velocity': {
                    'vx': velocity[0] if velocity else 0,
                    'vy': velocity[1] if velocity else 0,
                    'vz': velocity[2] if velocity else 0,
                    'groundspeed': groundspeed,
                    'airspeed': airspeed
                },
                'status': {
                    'mode': mode,
                    'armed': armed,
                    'armable': is_armable,
                    'ekf_ok': ekf_ok
                }
            }
            
            return data
            
        except Exception as e:
            print(f"⚠️ Error reading telemetry: {e}")
            return self._get_mock_data()
    
    def _get_mock_data(self):
        """Return mock data for testing"""
        return {
            'timestamp': datetime.now().isoformat(),
            'gps': {'lat': 13.7563, 'lon': 100.5018, 'alt': 5.0},
            'battery': {'voltage': 12.6, 'current': 15.0, 'level': 80},
            'attitude': {'pitch': 0.0, 'roll': 0.0, 'yaw': 90.0},
            'velocity': {'vx': 0, 'vy': 0, 'vz': 0, 'groundspeed': 0, 'airspeed': 0},
            'status': {'mode': 'GUIDED', 'armed': False, 'armable': True, 'ekf_ok': True}
        }
    
    def log_telemetry(self, data):
        """Log telemetry data"""
        self.telemetry_log.append(data)
        
        # Keep log size manageable
        if len(self.telemetry_log) > self.max_log_entries:
            self.telemetry_log.pop(0)
    
    def print_telemetry(self, data=None):
        """Print telemetry in formatted way"""
        if data is None:
            data = self.get_telemetry_data()
        
        print("\n" + "="*60)
        print(f"🚁 DRONE TELEMETRY - {data['timestamp']}")
        print("="*60)
        
        # GPS
        print(f"\n📍 GPS Position:")
        print(f"   Latitude:  {data['gps']['lat']:.6f}°")
        print(f"   Longitude: {data['gps']['lon']:.6f}°")
        print(f"   Altitude:  {data['gps']['alt']:.1f} m")
        
        # Battery
        print(f"\n🔋 Battery:")
        print(f"   Voltage: {data['battery']['voltage']:.1f} V")
        print(f"   Current: {data['battery']['current']:.1f} A")
        print(f"   Level:   {data['battery']['level']:.0f}%")
        
        # Attitude
        print(f"\n🎯 Attitude:")
        print(f"   Pitch: {data['attitude']['pitch']:6.1f}°")
        print(f"   Roll:  {data['attitude']['roll']:6.1f}°")
        print(f"   Yaw:   {data['attitude']['yaw']:6.1f}°")
        
        # Velocity
        print(f"\n💨 Velocity:")
        print(f"   Ground Speed: {data['velocity']['groundspeed']:.1f} m/s")
        print(f"   Air Speed:    {data['velocity']['airspeed']:.1f} m/s")
        print(f"   Vertical:     {data['velocity']['vz']:.1f} m/s")
        
        # Status
        print(f"\n⚙️  Status:")
        print(f"   Mode:    {data['status']['mode']}")
        print(f"   Armed:   {'✓' if data['status']['armed'] else '✗'}")
        print(f"   Armable: {'✓' if data['status']['armable'] else '✗'}")
        print(f"   EKF OK:  {'✓' if data['status']['ekf_ok'] else '✗'}")
        
        print("="*60)
    
    def save_log(self, filename='telemetry_log.json'):
        """Save telemetry log to file"""
        try:
            with open(filename, 'w') as f:
                json.dump(self.telemetry_log, f, indent=2)
            print(f"💾 Telemetry log saved: {filename}")
            print(f"   Total entries: {len(self.telemetry_log)}")
        except Exception as e:
            print(f"❌ Failed to save log: {e}")
    
    def save_csv(self, filename='telemetry_log.csv'):
        """Save telemetry log as CSV for analysis"""
        try:
            import csv
            
            if not self.telemetry_log:
                print("⚠️ No telemetry data to save")
                return
            
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                
                # Header
                writer.writerow([
                    'timestamp', 'lat', 'lon', 'alt',
                    'battery_voltage', 'battery_current', 'battery_level',
                    'pitch', 'roll', 'yaw',
                    'groundspeed', 'airspeed',
                    'mode', 'armed'
                ])
                
                # Data rows
                for entry in self.telemetry_log:
                    writer.writerow([
                        entry['timestamp'],
                        entry['gps']['lat'],
                        entry['gps']['lon'],
                        entry['gps']['alt'],
                        entry['battery']['voltage'],
                        entry['battery']['current'],
                        entry['battery']['level'],
                        entry['attitude']['pitch'],
                        entry['attitude']['roll'],
                        entry['attitude']['yaw'],
                        entry['velocity']['groundspeed'],
                        entry['velocity']['airspeed'],
                        entry['status']['mode'],
                        entry['status']['armed']
                    ])
            
            print(f"💾 Telemetry CSV saved: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to save CSV: {e}")
    
    def get_summary(self):
        """Get flight summary statistics"""
        if not self.telemetry_log:
            return None
        
        altitudes = [e['gps']['alt'] for e in self.telemetry_log]
        speeds = [e['velocity']['groundspeed'] for e in self.telemetry_log]
        battery_levels = [e['battery']['level'] for e in self.telemetry_log]
        
        return {
            'flight_time': len(self.telemetry_log) / self.update_rate,
            'max_altitude': max(altitudes),
            'max_speed': max(speeds),
            'battery_used': battery_levels[0] - battery_levels[-1] if len(battery_levels) > 1 else 0,
            'total_samples': len(self.telemetry_log)
        }
    
    def print_summary(self):
        """Print flight summary"""
        summary = self.get_summary()
        
        if not summary:
            print("⚠️ No flight data available")
            return
        
        print("\n" + "="*60)
        print("📊 FLIGHT SUMMARY")
        print("="*60)
        print(f"Flight Time:    {summary['flight_time']:.1f} seconds")
        print(f"Max Altitude:   {summary['max_altitude']:.1f} m")
        print(f"Max Speed:      {summary['max_speed']:.1f} m/s")
        print(f"Battery Used:   {summary['battery_used']:.1f}%")
        print(f"Total Samples:  {summary['total_samples']}")
        print("="*60)


# Test function
if __name__ == "__main__":
    print("Testing Telemetry Display (Mock Mode)")
    
    # Mock vehicle
    class MockVehicle:
        class Location:
            class GlobalRelativeFrame:
                lat = 13.7563
                lon = 100.5018
                alt = 5.0
        
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
    
    vehicle = MockVehicle()
    telemetry = TelemetryDisplay(vehicle, update_rate=1)
    
    print("\n--- Single Telemetry Display ---")
    telemetry.print_telemetry()
    
    print("\n--- Logging for 5 seconds ---")
    telemetry.start()
    time.sleep(5)
    telemetry.stop()
    
    print("\n--- Flight Summary ---")
    telemetry.print_summary()
    
    print("\n--- Saving Logs ---")
    telemetry.save_log('test_telemetry.json')
    telemetry.save_csv('test_telemetry.csv')
    
    print("\n✅ Test complete")