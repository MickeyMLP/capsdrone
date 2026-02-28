"""
Main Launcher UI - Drone Control System
Two-camera system: Thermal Camera + Regular Camera
Author: Sue Sha
"""

import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import sys
import os

# Get the directory where main_launcher.py is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

class MainLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Drone Vision System")
        self.root.geometry("980x620")
        self.root.minsize(880, 560)

        self._configure_theme()
        self._build_ui()
        self.center_window()

        print("Main Launcher initialized")

    def _configure_theme(self):
        self.root.configure(bg="#0f172a")  # deep navy

        style = ttk.Style(self.root)
        # Try a modern-ish theme if available
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure("App.TFrame", background="#0f172a")
        style.configure("Surface.TFrame", background="#111827")  # slightly lighter
        style.configure("Card.TFrame", background="#111827")

        style.configure("Title.TLabel", background="#0f172a", foreground="#f8fafc",
                        font=("Segoe UI", 26, "bold"))
        style.configure("SubTitle.TLabel", background="#0f172a", foreground="#94a3b8",
                        font=("Segoe UI", 11))

        style.configure("CardTitle.TLabel", background="#111827", foreground="#f8fafc",
                        font=("Segoe UI", 15, "bold"))
        style.configure("CardBody.TLabel", background="#111827", foreground="#cbd5e1",
                        font=("Segoe UI", 11))

        style.configure("Footer.TLabel", background="#0f172a", foreground="#94a3b8",
                        font=("Segoe UI", 9))

        style.configure("Primary.TButton", font=("Segoe UI", 11, "bold"), padding=(14, 10))
        style.configure("Secondary.TButton", font=("Segoe UI", 10, "bold"), padding=(12, 8))

    def center_window(self):
        self.root.update_idletasks()
        w = self.root.winfo_width()
        h = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (w // 2)
        y = (self.root.winfo_screenheight() // 2) - (h // 2)
        self.root.geometry(f"{w}x{h}+{x}+{y}")

    def _build_ui(self):
        outer = ttk.Frame(self.root, style="App.TFrame")
        outer.pack(fill="both", expand=True)

        # ---------- Header ----------
        header = ttk.Frame(outer, style="App.TFrame")
        header.pack(fill="x", padx=28, pady=(24, 10))

        ttk.Label(header, text="Drone Vision System", style="Title.TLabel").pack(anchor="w")
        ttk.Label(header, text="Choose a camera system to launch", style="SubTitle.TLabel").pack(anchor="w", pady=(6, 0))

        # ---------- Content ----------
        content = ttk.Frame(outer, style="App.TFrame")
        content.pack(fill="both", expand=True, padx=28, pady=18)

        content.columnconfigure(0, weight=1, uniform="cards")
        content.columnconfigure(1, weight=1, uniform="cards")
        content.rowconfigure(0, weight=1)

        thermal_card = self._make_card(
            parent=content,
            title="Thermal Camera",
            body="Heat source detection\nPerson detection\nFire detection\nSearch and rescue",
            accent="#ef4444",
            button_text="Launch Thermal",
            button_cmd=self.launch_thermal
        )
        thermal_card.grid(row=0, column=0, sticky="nsew", padx=(0, 14), pady=(0, 12))

        regular_card = self._make_card(
            parent=content,
            title="Regular Camera",
            body="Object detection\nObstacle detection\nLanding pad detection\nColor tracking",
            accent="#3b82f6",
            button_text="Launch Vision",
            button_cmd=self.launch_regular
        )
        regular_card.grid(row=0, column=1, sticky="nsew", padx=(14, 0), pady=(0, 12))

        # ---------- Footer ----------
        footer = ttk.Frame(outer, style="App.TFrame")
        footer.pack(fill="x", padx=28, pady=(0, 18))

        footer.columnconfigure(0, weight=1)
        footer.columnconfigure(1, weight=1)
        footer.columnconfigure(2, weight=1)

        control_btn = ttk.Button(footer, text="Drone Control Panel", style="Secondary.TButton",
                                 command=self.launch_control_panel)
        safety_btn = ttk.Button(footer, text="Safety Monitor", style="Secondary.TButton",
                                command=self.launch_safety)

        control_btn.grid(row=0, column=0, sticky="w")
        ttk.Label(footer, text="Capstone Project", style="Footer.TLabel").grid(row=0, column=1)
        safety_btn.grid(row=0, column=2, sticky="e")

    def _make_card(self, parent, title, body, accent, button_text, button_cmd):
        card = ttk.Frame(parent, style="Card.TFrame")
        card.configure(padding=18)

        # Accent bar
        bar = tk.Frame(card, bg=accent, height=4)
        bar.pack(fill="x", side="top")
        tk.Frame(card, height=10, bg="#111827").pack(fill="x")  # spacer

        ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")

        ttk.Label(card, text=body, style="CardBody.TLabel", justify="left").pack(
            anchor="w", pady=(10, 14)
        )

        # Button row
        btn = ttk.Button(card, text=button_text, style="Primary.TButton", command=button_cmd)
        btn.pack(anchor="w")

        # Make it feel like a card with a subtle border
        # Tkinter ttk borders are limited; using highlight on underlying tk widget works well.
        wrapper = tk.Frame(parent, bg="#0f172a")
        # Put ttk card inside a tk frame to simulate border
        bordered = tk.Frame(wrapper, bg="#1f2937", padx=1, pady=1)  # border color
        bordered.pack(fill="both", expand=True)
        inner = tk.Frame(bordered, bg="#111827")
        inner.pack(fill="both", expand=True)
        card_master = card

        # Re-parenting ttk widgets is painful; easiest is: return a tk.Frame that contains the ttk card
        # So we rebuild inside "inner":
        card.destroy()
        card = ttk.Frame(inner, style="Card.TFrame")
        card.configure(padding=18)

        bar = tk.Frame(card, bg=accent, height=4)
        bar.pack(fill="x", side="top")
        tk.Frame(card, height=10, bg="#111827").pack(fill="x")

        ttk.Label(card, text=title, style="CardTitle.TLabel").pack(anchor="w")
        ttk.Label(card, text=body, style="CardBody.TLabel", justify="left").pack(anchor="w", pady=(10, 14))
        ttk.Button(card, text=button_text, style="Primary.TButton", command=button_cmd).pack(anchor="w")

        card.pack(fill="both", expand=True)
        return wrapper

    # ---------- Launch functions (same logic you had) ----------
    def launch_thermal(self):
        print("Launching Thermal Camera System...")
        try:
            self.show_thermal_menu()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch thermal system:\n{e}")

    def show_thermal_menu(self):
        w = tk.Toplevel(self.root)
        w.title("Thermal Camera System")
        w.geometry("520x420")
        w.configure(bg="#0f172a")

        frm = ttk.Frame(w, style="App.TFrame", padding=22)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Thermal Camera", style="Title.TLabel").pack(anchor="w")
        ttk.Label(frm, text="Choose a mode", style="SubTitle.TLabel").pack(anchor="w", pady=(6, 14))

        buttons = [
            ("Heat Source Detection", lambda: self.run_thermal("heat")),
            ("Person Detection", lambda: self.run_thermal("person")),
            ("Fire Detection", lambda: self.run_thermal("fire")),
        ]
        for txt, cmd in buttons:
            ttk.Button(frm, text=txt, style="Primary.TButton", command=cmd).pack(fill="x", pady=8)

    def run_thermal(self, mode):
        print(f"Starting thermal detection: {mode} mode")
        thermal_path = os.path.join(SCRIPT_DIR, "thermal_detection.py")
        subprocess.Popen([sys.executable, thermal_path, mode])


    def launch_regular(self):
        print("Launching Regular Camera System...")
        try:
            self.show_regular_menu()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch vision system:\n{e}")

    def show_regular_menu(self):
        w = tk.Toplevel(self.root)
        w.title("Regular Camera System")
        w.geometry("520x520")
        w.configure(bg="#0f172a")

        frm = ttk.Frame(w, style="App.TFrame", padding=22)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Regular Camera", style="Title.TLabel").pack(anchor="w")
        ttk.Label(frm, text="Choose a mode", style="SubTitle.TLabel").pack(anchor="w", pady=(6, 14))

        options = [
            ("Person Detection", lambda: self.run_detection("person")),
            ("Obstacle Detection", lambda: self.run_detection("obstacle")),
            ("Landing Pad Detection", lambda: self.run_detection("landing_pad")),
            ("Color Target Tracking", lambda: self.run_detection("target")),
        ]
        for txt, cmd in options:
            ttk.Button(frm, text=txt, style="Primary.TButton", command=cmd).pack(fill="x", pady=8)

    def run_detection(self, mode):
        print(f"Starting object detection: {mode} mode")
        detection_path = os.path.join(SCRIPT_DIR, "object_detection.py")
        subprocess.Popen([sys.executable, detection_path, mode])

    def launch_control_panel(self):
        print("Launching Control Panel...")
        control_path = os.path.join(SCRIPT_DIR, "control_panel.py")
        try:
            subprocess.Popen([sys.executable, control_path])
            print("✅ Control Panel launched")
        except FileNotFoundError:
            messagebox.showerror("Error", f"control_panel.py not found at:\n{control_path}")


    def launch_safety(self):
        print("Launching Safety Monitor...")
        safety_path = os.path.join(SCRIPT_DIR, "safety_visualizer.py")
        try:
            subprocess.Popen([sys.executable, safety_path])
            print("✅ Safety Monitor launched")
        except FileNotFoundError:
            messagebox.showerror("Error", f"safety_visualizer.py not found at:\n{safety_path}")

def main():
    root = tk.Tk()
    try:
        root.iconbitmap("drone_icon.ico")
    except Exception:
        pass

    MainLauncher(root)
    root.mainloop()


if __name__ == "__main__":
    print("=" * 60)
    print("Drone Vision System - Main Launcher")
    print("=" * 60)
    main()