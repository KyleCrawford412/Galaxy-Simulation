"""GUI application using tkinter."""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
from typing import Optional
from galaxy_sim.backends.factory import get_backend, list_available_backends
from galaxy_sim.physics.simulator import Simulator
from galaxy_sim.physics.integrators.euler import EulerIntegrator
from galaxy_sim.physics.integrators.verlet import VerletIntegrator
from galaxy_sim.physics.integrators.rk4 import RK4Integrator
from galaxy_sim.presets import SpiralGalaxy, CollisionScenario, GlobularCluster, GalaxyCluster
from galaxy_sim.render.manager import RenderManager
from galaxy_sim.utils.reproducibility import set_all_seeds


class GalaxySimGUI:
    """Main GUI application."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Galaxy Simulator")
        self.root.geometry("1200x800")
        
        self.simulator: Optional[Simulator] = None
        self.renderer: Optional[RenderManager] = None
        self.running = False
        self.sim_thread: Optional[threading.Thread] = None
        
        self._create_widgets()
        self._setup_layout()
    
    def _create_widgets(self):
        """Create GUI widgets."""
        # Control panel (left)
        self.control_frame = ttk.LabelFrame(self.root, text="Controls", padding=10)
        
        # Preset selection
        ttk.Label(self.control_frame, text="Preset:").grid(row=0, column=0, sticky='w', pady=5)
        self.preset_var = tk.StringVar(value="spiral")
        preset_combo = ttk.Combobox(self.control_frame, textvariable=self.preset_var,
                                   values=["spiral", "collision", "globular", "cluster"],
                                   state="readonly", width=15)
        preset_combo.grid(row=0, column=1, pady=5)
        
        # Particle count
        ttk.Label(self.control_frame, text="Particles:").grid(row=1, column=0, sticky='w', pady=5)
        self.particles_var = tk.IntVar(value=1000)
        particles_spin = ttk.Spinbox(self.control_frame, from_=100, to=100000, 
                                    textvariable=self.particles_var, width=15)
        particles_spin.grid(row=1, column=1, pady=5)
        
        # Backend selection
        ttk.Label(self.control_frame, text="Backend:").grid(row=2, column=0, sticky='w', pady=5)
        self.backend_var = tk.StringVar(value="numpy")
        backends = list_available_backends()
        backend_combo = ttk.Combobox(self.control_frame, textvariable=self.backend_var,
                                    values=backends, state="readonly", width=15)
        backend_combo.grid(row=2, column=1, pady=5)
        
        # Integrator selection
        ttk.Label(self.control_frame, text="Integrator:").grid(row=3, column=0, sticky='w', pady=5)
        self.integrator_var = tk.StringVar(value="verlet")
        integrator_combo = ttk.Combobox(self.control_frame, textvariable=self.integrator_var,
                                       values=["euler", "verlet", "rk4"],
                                       state="readonly", width=15)
        integrator_combo.grid(row=3, column=1, pady=5)
        
        # Timestep
        ttk.Label(self.control_frame, text="Timestep:").grid(row=4, column=0, sticky='w', pady=5)
        self.dt_var = tk.DoubleVar(value=0.01)
        dt_scale = ttk.Scale(self.control_frame, from_=0.001, to=0.1, 
                             variable=self.dt_var, orient='horizontal', length=150)
        dt_scale.grid(row=4, column=1, pady=5)
        self.dt_label = ttk.Label(self.control_frame, text="0.01")
        self.dt_label.grid(row=4, column=2, pady=5)
        dt_scale.configure(command=lambda v: self.dt_label.config(text=f"{float(v):.4f}"))
        
        # Steps per frame (decouple sim from render)
        ttk.Label(self.control_frame, text="Steps/Frame:").grid(row=5, column=0, sticky='w', pady=5)
        self.steps_per_frame_var = tk.IntVar(value=10)
        steps_spin = ttk.Spinbox(self.control_frame, from_=1, to=50,
                                 textvariable=self.steps_per_frame_var, width=8)
        steps_spin.grid(row=5, column=1, pady=5)
        ttk.Label(self.control_frame, text="(5–20 typical)").grid(row=5, column=2, sticky='w')
        
        # Seed
        ttk.Label(self.control_frame, text="Seed:").grid(row=6, column=0, sticky='w', pady=5)
        self.seed_var = tk.StringVar(value="")
        seed_entry = ttk.Entry(self.control_frame, textvariable=self.seed_var, width=15)
        seed_entry.grid(row=6, column=1, pady=5)
        
        # Render mode
        ttk.Label(self.control_frame, text="Render Mode:").grid(row=7, column=0, sticky='w', pady=5)
        self.render_mode_var = tk.StringVar(value="2d")
        render_mode_combo = ttk.Combobox(self.control_frame, textvariable=self.render_mode_var,
                                        values=["2d", "3d"], state="readonly", width=15)
        render_mode_combo.grid(row=7, column=1, pady=5)
        
        # Profiling
        self.profile_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_frame, text="Profile (timing)", variable=self.profile_var).grid(row=8, column=0, columnspan=2, sticky='w', pady=2)
        
        # Buttons
        self.init_button = ttk.Button(self.control_frame, text="Initialize", command=self.initialize)
        self.init_button.grid(row=9, column=0, columnspan=2, pady=10, sticky='ew')
        
        self.play_button = ttk.Button(self.control_frame, text="Play", command=self.toggle_play,
                                     state='disabled')
        self.play_button.grid(row=10, column=0, columnspan=2, pady=5, sticky='ew')
        
        self.step_button = ttk.Button(self.control_frame, text="Step", command=self.step_once,
                                     state='disabled')
        self.step_button.grid(row=11, column=0, columnspan=2, pady=5, sticky='ew')
        
        # Status
        self.status_label = ttk.Label(self.control_frame, text="Ready", foreground="green")
        self.status_label.grid(row=12, column=0, columnspan=2, pady=10)
        
        # Info display
        self.info_frame = ttk.LabelFrame(self.root, text="Simulation Info", padding=10)
        self.time_label = ttk.Label(self.info_frame, text="Time: 0.00")
        self.time_label.pack(anchor='w')
        self.steps_label = ttk.Label(self.info_frame, text="Steps: 0")
        self.steps_label.pack(anchor='w')
        self.energy_label = ttk.Label(self.info_frame, text="Energy: 0.00")
        self.energy_label.pack(anchor='w')
        self.timing_label = ttk.Label(self.info_frame, text="")
        self.timing_label.pack(anchor='w')
    
    def _setup_layout(self):
        """Setup window layout."""
        # Control panel on left
        self.control_frame.pack(side='left', fill='y', padx=10, pady=10)
        
        # Info panel below controls
        self.info_frame.pack(side='left', fill='y', padx=10, pady=10)
        
        # Renderer will be shown via matplotlib (separate window)
    
    def _get_integrator(self):
        """Get integrator instance."""
        integrator_name = self.integrator_var.get()
        integrators = {
            'euler': EulerIntegrator,
            'verlet': VerletIntegrator,
            'rk4': RK4Integrator
        }
        return integrators[integrator_name]()
    
    def _get_preset(self):
        """Get preset instance."""
        backend = get_backend(self.backend_var.get())
        preset_name = self.preset_var.get()
        n_particles = self.particles_var.get()
        
        seed = None
        if self.seed_var.get():
            try:
                seed = int(self.seed_var.get())
            except ValueError:
                messagebox.showerror("Error", "Seed must be an integer")
                return None
        
        presets = {
            'spiral': SpiralGalaxy,
            'collision': CollisionScenario,
            'globular': GlobularCluster,
            'cluster': GalaxyCluster
        }
        
        preset_class = presets[preset_name]
        return preset_class(backend, n_particles=n_particles, seed=seed), backend
    
    def initialize(self):
        """Initialize simulation."""
        try:
            preset, backend = self._get_preset()
            if preset is None:
                return
            
            if self.seed_var.get():
                try:
                    seed = int(self.seed_var.get())
                    set_all_seeds(seed, backend)
                except ValueError:
                    pass
            
            positions, velocities, masses = preset.generate()
            
            integrator = self._get_integrator()
            dt = self.dt_var.get()
            
            self.simulator = Simulator(backend, integrator, dt=dt)
            self.simulator.initialize(positions, velocities, masses)
            
            # Create renderer
            if self.renderer:
                self.renderer.close()
            
            self.renderer = RenderManager(
                mode=self.render_mode_var.get(),
                show_trails=False,
                color_by_velocity=True
            )
            
            # Initial render
            pos, vel, mass = self.simulator.system.get_state()[:3]
            self.renderer.render(pos, vel, mass)
            
            self.play_button.config(state='normal')
            self.step_button.config(state='normal')
            self.status_label.config(text="Initialized", foreground="green")
            self.update_info()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to initialize: {str(e)}")
    
    def toggle_play(self):
        """Toggle play/pause."""
        if not self.simulator:
            return
        
        if self.running:
            self.simulator.pause()
            self.running = False
            self.play_button.config(text="Play")
            self.status_label.config(text="Paused", foreground="orange")
        else:
            self.simulator.resume()
            self.running = True
            self.play_button.config(text="Pause")
            self.status_label.config(text="Running", foreground="green")
            
            if self.sim_thread is None or not self.sim_thread.is_alive():
                self.sim_thread = threading.Thread(target=self._run_loop, daemon=True)
                self.sim_thread.start()
    
    def _run_loop(self):
        """Main simulation loop: run K steps per frame, then render (conversion only at render)."""
        while self.running and self.simulator:
            self.simulator.set_timestep(self.dt_var.get())
            self.simulator.set_profiling(self.profile_var.get())
            try:
                k = max(1, min(50, self.steps_per_frame_var.get()))
            except (tk.TclError, AttributeError):
                k = 10
            self.simulator.run_steps(k)
            # Single host transfer for this frame (get_state -> numpy for render)
            pos, vel, mass = self.simulator.get_state()[:3]
            self.renderer.render(pos, vel, mass)
            self.root.after(0, self.update_info)
    
    def step_once(self):
        """Perform one simulation step."""
        if not self.simulator:
            return
        
        self.simulator.step()
        pos, vel, mass = self.simulator.system.get_state()[:3]
        self.renderer.render(pos, vel, mass)
        self.update_info()
    
    def update_info(self):
        """Update info display (energy/diagnostics trigger to_numpy only here)."""
        if not self.simulator:
            return
        
        self.time_label.config(text=f"Time: {self.simulator.time:.2f}")
        self.steps_label.config(text=f"Steps: {self.simulator.step_count}")
        
        if self.simulator._profile:
            t = self.simulator.get_timing()
            fs, it = t.get("forces_ms"), t.get("integrator_ms")
            if fs is not None and it is not None:
                self.timing_label.config(text=f"Forces: {fs:.2f} ms | Integrator: {it:.2f} ms")
            else:
                self.timing_label.config(text="")
        else:
            self.timing_label.config(text="")
        
        try:
            energy = self.simulator.get_energy()
            self.energy_label.config(text=f"Energy: {energy:.6f}")
        except Exception:
            self.energy_label.config(text="Energy: ...")


def run_gui():
    """Run GUI application."""
    root = tk.Tk()
    app = GalaxySimGUI(root)
    root.mainloop()


if __name__ == '__main__':
    run_gui()
