"""
Drone Control Panel
GUI interface for controlling drone operations
Compatible with Raspberry Pi
Author: Sue Sha
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import time
from datetime import datetime
import json

# Import drone modules (will work after integration)
try:
    from .safety_system import SafetyMonitor
    from .autonomous_mission import MissionPlanner
    from .telemetry_display import TelemetryDisplay
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    print("⚠️ Drone modules not found - running in demo mode")

class DroneControlPanel:
    """
    Main control panel for drone operations
    Features:
    - Arm/Disarm drone
    - Takeoff/Land controls
    - Mission planning
    - Safety monitoring
    - Real-time telemetry
    - Emergency controls
    """
    
    def __init__(self, root):
        self.root = root
        self.root.title("🚁 Drone Control Panel")
        self.root.geometry("1024x600")  # Raspberry Pi screen size
        
        # Drone state
        self.connected = False
        self.armed = False
        self.flying = False
        self.mode = "STABILIZE"
        
        # Telemetry data
        self.telemetry = {
            'altitude': 0.0,
            'battery': 100,
            'gps_lat': 0.0,
            'gps_lon': 0.0,
            'speed': 0.0,
            'mode': 'STABILIZE'
        }
        
        # Update thread
        self.update_thread = None
        self.running = False
        
        # Setup UI
        self.setup_ui()
        
        print("✅ Control Panel initialized")
    
    def setup_ui(self):
        """Create user interface"""
        
        # ==================== TOP BAR ====================
        top_frame = tk.Frame(self.root, bg='#2c3e50', height=60)
        top_frame.pack(fill='x', side='top')
        
        title = tk.Label(top_frame, text="🚁 DRONE CONTROL PANEL", 
                        font=('Arial', 20, 'bold'), bg='#2c3e50', fg='white')
        title.pack(pady=15)
        
        # ==================== LEFT PANEL - CONTROLS ====================
        left_frame = tk.Frame(self.root, bg='#34495e', width=350)
        left_frame.pack(fill='y', side='left', padx=5, pady=5)
        
        # Connection Status
        connection_frame = tk.LabelFrame(left_frame, text="Connection", 
                                        font=('Arial', 12, 'bold'), bg='#34495e', fg='white')
        connection_frame.pack(fill='x', padx=10, pady=10)
        
        self.connection_status = tk.Label(connection_frame, text="⚫ DISCONNECTED", 
                                         font=('Arial', 11), bg='#34495e', fg='#e74c3c')
        self.connection_status.pack(pady=5)
        
        btn_connect = tk.Button(connection_frame, text="Connect to Drone", 
                               command=self.connect_drone, bg='#3498db', fg='white',
                               font=('Arial', 10, 'bold'), width=20)
        btn_connect.pack(pady=5)
        
        # Arm/Disarm
        arm_frame = tk.LabelFrame(left_frame, text="Arm/Disarm", 
                                 font=('Arial', 12, 'bold'), bg='#34495e', fg='white')
        arm_frame.pack(fill='x', padx=10, pady=10)
        
        self.arm_status = tk.Label(arm_frame, text="⚫ DISARMED", 
                                  font=('Arial', 11), bg='#34495e', fg='#95a5a6')
        self.arm_status.pack(pady=5)
        
        btn_arm = tk.Button(arm_frame, text="ARM", command=self.arm_drone,
                           bg='#27ae60', fg='white', font=('Arial', 10, 'bold'), width=9)
        btn_arm.pack(side='left', padx=5, pady=5)
        
        btn_disarm = tk.Button(arm_frame, text="DISARM", command=self.disarm_drone,
                              bg='#e67e22', fg='white', font=('Arial', 10, 'bold'), width=9)
        btn_disarm.pack(side='left', padx=5, pady=5)
        
        # Flight Controls
        flight_frame = tk.LabelFrame(left_frame, text="Flight Controls", 
                                    font=('Arial', 12, 'bold'), bg='#34495e', fg='white')
        flight_frame.pack(fill='x', padx=10, pady=10)
        
        # Altitude input
        alt_label = tk.Label(flight_frame, text="Altitude (m):", 
                            bg='#34495e', fg='white', font=('Arial', 10))
        alt_label.pack(pady=5)
        
        self.altitude_entry = tk.Entry(flight_frame, font=('Arial', 12), width=10)
        self.altitude_entry.insert(0, "5")
        self.altitude_entry.pack(pady=5)
        
        btn_takeoff = tk.Button(flight_frame, text="🛫 TAKEOFF", 
                               command=self.takeoff, bg='#27ae60', fg='white',
                               font=('Arial', 11, 'bold'), width=20)
        btn_takeoff.pack(pady=5)
        
        btn_land = tk.Button(flight_frame, text="🛬 LAND", 
                            command=self.land, bg='#e67e22', fg='white',
                            font=('Arial', 11, 'bold'), width=20)
        btn_land.pack(pady=5)
        
        btn_rtl = tk.Button(flight_frame, text="🏠 RETURN HOME", 
                           command=self.return_home, bg='#3498db', fg='white',
                           font=('Arial', 11, 'bold'), width=20)
        btn_rtl.pack(pady=5)
        
        # Emergency
        emergency_frame = tk.LabelFrame(left_frame, text="⚠️ EMERGENCY", 
                                       font=('Arial', 12, 'bold'), bg='#34495e', fg='white')
        emergency_frame.pack(fill='x', padx=10, pady=10)
        
        btn_emergency = tk.Button(emergency_frame, text="🚨 EMERGENCY STOP", 
                                 command=self.emergency_stop, bg='#c0392b', fg='white',
                                 font=('Arial', 12, 'bold'), width=20)
        btn_emergency.pack(pady=10)
        
        # ==================== CENTER PANEL - TELEMETRY ====================
        center_frame = tk.Frame(self.root, bg='#ecf0f1')
        center_frame.pack(fill='both', expand=True, side='left', padx=5, pady=5)
        
        # Telemetry Display
        telemetry_frame = tk.LabelFrame(center_frame, text="📊 Real-Time Telemetry", 
                                       font=('Arial', 14, 'bold'), bg='#ecf0f1')
        telemetry_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create telemetry labels
        self.telem_labels = {}
        
        telemetry_items = [
            ('Altitude', 'altitude', 'm'),
            ('Battery', 'battery', '%'),
            ('GPS Lat', 'gps_lat', '°'),
            ('GPS Lon', 'gps_lon', '°'),
            ('Speed', 'speed', 'm/s'),
            ('Mode', 'mode', '')
        ]
        
        for i, (name, key, unit) in enumerate(telemetry_items):
            # Label
            label = tk.Label(telemetry_frame, text=f"{name}:", 
                           font=('Arial', 12, 'bold'), bg='#ecf0f1')
            label.grid(row=i, column=0, sticky='w', padx=20, pady=10)
            
            # Value
            value_label = tk.Label(telemetry_frame, text=f"0 {unit}", 
                                  font=('Arial', 12), bg='#ecf0f1', fg='#2c3e50')
            value_label.grid(row=i, column=1, sticky='w', padx=20, pady=10)
            
            self.telem_labels[key] = (value_label, unit)
        
        # Mission Controls
        mission_frame = tk.LabelFrame(center_frame, text="🗺️ Mission Planning", 
                                     font=('Arial', 14, 'bold'), bg='#ecf0f1')
        mission_frame.pack(fill='x', padx=10, pady=10)
        
        mission_buttons = [
            ("Square Mission", self.square_mission),
            ("Circle Mission", self.circle_mission),
            ("Abort Mission", self.abort_mission)
        ]
        
        for text, command in mission_buttons:
            btn = tk.Button(mission_frame, text=text, command=command,
                          bg='#9b59b6', fg='white', font=('Arial', 10, 'bold'), width=15)
            btn.pack(side='left', padx=10, pady=10)
        
        # ==================== RIGHT PANEL - LOGS ====================
        right_frame = tk.Frame(self.root, bg='#34495e', width=300)
        right_frame.pack(fill='y', side='right', padx=5, pady=5)
        
        log_frame = tk.LabelFrame(right_frame, text="📝 Activity Log", 
                                 font=('Arial', 12, 'bold'), bg='#34495e', fg='white')
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Scrolled text for logs
        self.log_text = scrolledtext.ScrolledText(log_frame, height=30, width=35,
                                                  font=('Courier', 9), bg='#2c3e50', fg='#ecf0f1')
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Add initial log
        self.log("Control panel started")
    
    # ==================== LOGGING ====================
    
    def log(self, message):
        """Add message to activity log"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        self.log_text.insert(tk.END, log_entry)
        self.log_text.see(tk.END)
        print(log_entry.strip())
    
    # ==================== CONNECTION ====================
    
    def connect_drone(self):
        """Connect to drone"""
        self.log("Attempting to connect to drone...")
        
        # Simulate connection (replace with actual dronekit connection)
        if MODULES_AVAILABLE:
            # TODO: Add actual connection code
            # self.vehicle = connect('udp:127.0.0.1:14550', wait_ready=True)
            pass
        
        # Simulate connection success
        time.sleep(1)
        self.connected = True
        self.connection_status.config(text="🟢 CONNECTED", fg='#27ae60')
        self.log("✅ Connected to drone")
        
        # Start telemetry updates
        self.start_telemetry_updates()
    
    # ==================== ARM/DISARM ====================
    
    def arm_drone(self):
        """Arm the drone"""
        if not self.connected:
            messagebox.showwarning("Warning", "Please connect to drone first!")
            return
        
        self.log("Arming drone...")
        
        # TODO: Add actual arm command
        # self.vehicle.armed = True
        
        self.armed = True
        self.arm_status.config(text="🟢 ARMED", fg='#27ae60')
        self.log("✅ Drone ARMED")
    
    def disarm_drone(self):
        """Disarm the drone"""
        if not self.connected:
            return
        
        self.log("Disarming drone...")
        
        # TODO: Add actual disarm command
        # self.vehicle.armed = False
        
        self.armed = False
        self.arm_status.config(text="⚫ DISARMED", fg='#95a5a6')
        self.log("✅ Drone DISARMED")
    
    # ==================== FLIGHT CONTROLS ====================
    
    def takeoff(self):
        """Execute takeoff"""
        if not self.connected:
            messagebox.showwarning("Warning", "Please connect to drone first!")
            return
        
        if not self.armed:
            messagebox.showwarning("Warning", "Please arm drone first!")
            return
        
        try:
            altitude = float(self.altitude_entry.get())
        except ValueError:
            messagebox.showerror("Error", "Invalid altitude value!")
            return
        
        self.log(f"🛫 Taking off to {altitude}m...")
        
        # TODO: Add actual takeoff command
        # self.vehicle.simple_takeoff(altitude)
        
        self.flying = True
        self.log(f"✅ Takeoff command sent: {altitude}m")
    
    def land(self):
        """Execute landing"""
        if not self.connected or not self.flying:
            return
        
        self.log("🛬 Landing...")
        
        # TODO: Add actual land command
        # self.vehicle.mode = VehicleMode("LAND")
        
        self.flying = False
        self.log("✅ Landing command sent")
    
    def return_home(self):
        """Return to launch point"""
        if not self.connected or not self.flying:
            return
        
        self.log("🏠 Returning to home...")
        
        # TODO: Add actual RTL command
        # self.vehicle.mode = VehicleMode("RTL")
        
        self.log("✅ RTL command sent")
    
    def emergency_stop(self):
        """Emergency stop - immediate action"""
        confirm = messagebox.askyesno("EMERGENCY STOP", 
                                     "This will immediately stop all motors!\n\nContinue?")
        if confirm:
            self.log("🚨 EMERGENCY STOP ACTIVATED")
            
            # TODO: Add actual emergency stop
            # self.vehicle.mode = VehicleMode("LAND")
            # self.vehicle.armed = False
            
            self.flying = False
            self.armed = False
            self.arm_status.config(text="⚫ DISARMED", fg='#95a5a6')
            self.log("⚠️ Emergency stop complete")
    
    # ==================== MISSIONS ====================
    
    def square_mission(self):
        """Execute square mission"""
        if not self.connected:
            messagebox.showwarning("Warning", "Please connect to drone first!")
            return
        
        self.log("🗺️ Starting square mission...")
        
        # TODO: Add actual mission code
        # Use MissionPlanner from section2_advanced_functions
        
        self.log("✅ Square mission uploaded")
    
    def circle_mission(self):
        """Execute circle mission"""
        if not self.connected:
            messagebox.showwarning("Warning", "Please connect to drone first!")
            return
        
        self.log("🗺️ Starting circle mission...")
        
        # TODO: Add actual mission code
        
        self.log("✅ Circle mission uploaded")
    
    def abort_mission(self):
        """Abort current mission"""
        self.log("⚠️ Aborting mission...")
        
        # TODO: Add actual abort code
        # self.vehicle.mode = VehicleMode("GUIDED")
        
        self.log("✅ Mission aborted")
    
    # ==================== TELEMETRY UPDATES ====================
    
    def start_telemetry_updates(self):
        """Start background thread for telemetry updates"""
        self.running = True
        self.update_thread = threading.Thread(target=self.update_telemetry_loop, daemon=True)
        self.update_thread.start()
    
    def update_telemetry_loop(self):
        """Update telemetry data in background"""
        while self.running:
            self.update_telemetry()
            time.sleep(0.5)  # Update twice per second
    
    def update_telemetry(self):
        """Update telemetry display"""
        # TODO: Get actual telemetry from vehicle
        # For now, simulate data
        
        if self.connected and self.flying:
            # Simulate increasing altitude
            self.telemetry['altitude'] = min(self.telemetry['altitude'] + 0.1, 10)
            self.telemetry['battery'] = max(self.telemetry['battery'] - 0.01, 0)
        
        # Update labels
        for key, (label, unit) in self.telem_labels.items():
            value = self.telemetry.get(key, 0)
            if isinstance(value, float):
                text = f"{value:.2f} {unit}"
            else:
                text = f"{value} {unit}"
            label.config(text=text)
    
    # ==================== CLEANUP ====================
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        if self.update_thread:
            self.update_thread.join(timeout=2)
        self.log("Control panel closed")


# ==================== MAIN ====================

def main():
    """Main entry point"""
    root = tk.Tk()
    
    # Set icon and appearance
    try:
        root.iconbitmap('drone_icon.ico')  # Optional
    except:
        pass
    
    # Create control panel
    panel = DroneControlPanel(root)
    
    # Handle window close
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            panel.cleanup()
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Run
    root.mainloop()


if __name__ == "__main__":
    print("="*60)
    print("🚁 Drone Control Panel")
    print("="*60)
    main()