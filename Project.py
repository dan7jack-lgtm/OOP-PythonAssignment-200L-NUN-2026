import sys
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

def calculate_trajectory(v0, angle_deg, h0, drag_coeff, mass, g=9.81, dt=0.01):
    angle_rad = np.radians(angle_deg)
    x, y = [0.0], [h0]
    vx = v0 * np.cos(angle_rad)
    vy = v0 * np.sin(angle_rad)
    t = [0.0]
    k = drag_coeff
    
    while y[-1] >= 0:
        v = np.sqrt(vx**2 + vy**2)
        ax = -(k / mass) * v * vx
        ay = -g - (k / mass) * v * vy
        vx += ax * dt
        vy += ay * dt
        new_x = x[-1] + vx * dt
        new_y = y[-1] + vy * dt
        
        if new_y < 0:
            fraction = (0 - y[-1]) / (new_y - y[-1])
            x.append(x[-1] + fraction * (new_x - x[-1]))
            y.append(0.0)
            t.append(t[-1] + fraction * dt)
            break
            
        x.append(new_x)
        y.append(new_y)
        t.append(t[-1] + dt)
        
    return np.array(x), np.array(y), np.array(t)

class ProjectileApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Projectile Motion Simulator")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)
        
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.history = []
        
        self.create_layout()
        self.create_sidebar_inputs()
        self.create_plot_area()
        self.create_status_bar()
        self.run_simulation()

    def create_layout(self):
        self.main_container = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        self.sidebar = ttk.Frame(self.main_container, width=320, padding=10)
        self.plot_panel = ttk.Frame(self.main_container, padding=10)
        self.main_container.add(self.sidebar, weight=1)
        self.main_container.add(self.plot_panel, weight=4)

    def create_sidebar_inputs(self):
        ttk.Label(self.sidebar, text="Simulation Controls", font=("Arial", 14, "bold")).pack(pady=(0, 15), anchor=tk.W)
        tabs = ttk.Notebook(self.sidebar)
        tabs.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        tab_launcher = ttk.Frame(tabs, padding=10)
        tab_env = ttk.Frame(tabs, padding=10)
        tabs.add(tab_launcher, text="Launcher Specs")
        tabs.add(tab_env, text="Environment")
        
        ttk.Label(tab_launcher, text="Initial Velocity (m/s):", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(5, 2))
        self.val_v0 = tk.DoubleVar(value=30.0)
        tk.Scale(tab_launcher, from_=1, to=150, orient=tk.HORIZONTAL, variable=self.val_v0, resolution=0.5).pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(tab_launcher, text="Launch Angle (degrees):", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(5, 2))
        self.val_angle = tk.DoubleVar(value=45.0)
        tk.Scale(tab_launcher, from_=0, to=90, orient=tk.HORIZONTAL, variable=self.val_angle, resolution=0.5).pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(tab_launcher, text="Initial Height (m):", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(5, 2))
        self.val_h0 = tk.DoubleVar(value=0.0)
        tk.Scale(tab_launcher, from_=0, to=100, orient=tk.HORIZONTAL, variable=self.val_h0, resolution=1).pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(tab_env, text="Projectile Mass (kg):", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(5, 2))
        self.val_mass = tk.DoubleVar(value=1.0)
        tk.Scale(tab_env, from_=0.1, to=50, orient=tk.HORIZONTAL, variable=self.val_mass, resolution=0.1).pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(tab_env, text="Air Drag Coeff (k):", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(5, 2))
        self.val_drag = tk.DoubleVar(value=0.005)
        tk.Scale(tab_env, from_=0.0, to=0.1, orient=tk.HORIZONTAL, variable=self.val_drag, resolution=0.001).pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(tab_env, text="Target Celestial Body:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(5, 2))
        self.gravity_choices = {"Earth (9.81 m/s²)": 9.81, "Moon (1.62 m/s²)": 1.62, "Mars (3.71 m/s²)": 3.71, "Jupiter (24.79 m/s²)": 24.79}
        self.val_gravity = tk.StringVar(value="Earth (9.81 m/s²)")
        ttk.Combobox(tab_env, textvariable=self.val_gravity, values=list(self.gravity_choices.keys()), state="readonly").pack(fill=tk.X, pady=(0, 10))

        btn_frame = ttk.Frame(self.sidebar)
        btn_frame.pack(fill=tk.X, pady=10)
        ttk.Button(btn_frame, text="🚀 Simulate", command=self.run_simulation).pack(fill=tk.X, pady=2)
        ttk.Button(btn_frame, text="🔄 Clear History", command=self.clear_plots).pack(fill=tk.X, pady=2)
        
        readout_frame = ttk.LabelFrame(self.sidebar, text="Live Telemetry Data", padding=10)
        readout_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.lbl_range = ttk.Label(readout_frame, text="Max Range: --", font=("Courier", 10))
        self.lbl_range.pack(anchor=tk.W, pady=2)
        self.lbl_height = ttk.Label(readout_frame, text="Max Height: --", font=("Courier", 10))
        self.lbl_height.pack(anchor=tk.W, pady=2)
        self.lbl_time = ttk.Label(readout_frame, text="Flight Duration: --", font=("Courier", 10))
        self.lbl_time.pack(anchor=tk.W, pady=2)

    def create_plot_area(self):
        self.fig, self.ax = plt.subplots(figsize=(6, 4), dpi=100)
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_panel)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.plot_panel)
        self.toolbar.update()

    def create_status_bar(self):
        self.status = ttk.Label(self.root, text="Ready.", relief=tk.SUNKEN, anchor=tk.W, padding=3)
        self.status.pack(side=tk.BOTTOM, fill=tk.X)

    def run_simulation(self):
        try:
            v0 = self.val_v0.get()
            angle = self.val_angle.get()
            h0 = self.val_h0.get()
            mass = self.val_mass.get()
            drag = self.val_drag.get()
            g = self.gravity_choices[self.val_gravity.get()]
            
            x, y, t = calculate_trajectory(v0, angle, h0, drag, mass, g)
            
            self.lbl_range.config(text=f"Max Range:       {x[-1]:.2f} m")
            self.lbl_height.config(text=f"Max Height:      {np.max(y):.2f} m")
            self.lbl_time.config(text=f"Flight Duration: {t[-1]:.2f} s")
            
            self.history.append((x, y, f"v0={v0}m/s, θ={angle}°, k={drag}"))
            self.render_graphs()
            self.status.config(text="Simulation updated.")
        except Exception as err:
            messagebox.showerror("Error", str(err))

    def render_graphs(self):
        self.ax.clear()
        self.ax.set_title("Projectile Trajectory Profile", fontsize=12, fontweight='bold')
        self.ax.set_xlabel("Horizontal Distance (meters)")
        self.ax.set_ylabel("Vertical Altitude (meters)")
        self.ax.grid(True, linestyle='--', alpha=0.5)
        
        for index, (x, y, label) in enumerate(self.history):
            alpha_val = 1.0 if index == len(self.history) - 1 else 0.35
            lw_val = 2.5 if index == len(self.history) - 1 else 1.2
            self.ax.plot(x, y, label=label, alpha=alpha_val, linewidth=lw_val)
            if index == len(self.history) - 1:
                self.ax.plot(x[np.argmax(y)], np.max(y), 'ro', markersize=6)
        
        self.ax.set_ylim(bottom=0)
        self.ax.set_xlim(left=0)
        if self.history:
            self.ax.legend(loc="upper right", prop={'size': 8})
        self.canvas.draw()

    def clear_plots(self):
        self.history.clear()
        self.render_graphs()
        self.lbl_range.config(text="Max Range: --")
        self.lbl_height.config(text="Max Height: --")
        self.lbl_time.config(text="Flight Duration: --")
        self.status.config(text="History cleared.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ProjectileApp(root)
    root.mainloop()