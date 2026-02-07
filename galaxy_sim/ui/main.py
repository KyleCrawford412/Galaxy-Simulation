"""GUI application using tkinter."""

import tkinter as tk
from tkinter import ttk, messagebox
import threading
import numpy as np
from typing import Optional
from galaxy_sim.backends.factory import get_backend, list_available_backends
from galaxy_sim.physics.simulator import Simulator
from galaxy_sim.physics.nbody import NBodySystem
from galaxy_sim.physics.integrators.euler import EulerIntegrator
from galaxy_sim.physics.integrators.verlet import VerletIntegrator
from galaxy_sim.physics.integrators.rk4 import RK4Integrator
from galaxy_sim.presets import SpiralGalaxy, CollisionScenario, GlobularCluster, GalaxyCluster
from galaxy_sim.render.manager import RenderManager
from galaxy_sim.physics.spiral_potential import SpiralPotential
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
                                   values=["spiral", "spiral_arms", "collision", "globular", "cluster"],
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
        self.dt_var = tk.DoubleVar(value=0.005)  # Smaller default for better inner orbit accuracy
        dt_scale = ttk.Scale(self.control_frame, from_=0.001, to=0.1, 
                             variable=self.dt_var, orient='horizontal', length=150)
        dt_scale.grid(row=4, column=1, pady=5)
        self.dt_label = ttk.Label(self.control_frame, text="0.005")
        self.dt_label.grid(row=4, column=2, pady=5)
        dt_scale.configure(command=lambda v: self.dt_label.config(text=f"{float(v):.4f}"))
        ttk.Label(self.control_frame, text="(0.002–0.02 typical)").grid(row=4, column=3, sticky='w')
        
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
        
        # Space visuals toggles
        self.trails_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.control_frame, text="Trails", variable=self.trails_var).grid(row=8, column=0, sticky='w', pady=2)
        self.trail_length_var = tk.IntVar(value=25)
        ttk.Spinbox(self.control_frame, from_=5, to=100, textvariable=self.trail_length_var, width=6).grid(row=8, column=1, sticky='w')
        
        self.density_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.control_frame, text="Density", variable=self.density_var).grid(row=9, column=0, sticky='w', pady=2)
        self.density_res_var = tk.IntVar(value=256)
        ttk.Spinbox(self.control_frame, from_=64, to=512, textvariable=self.density_res_var, width=6).grid(row=9, column=1, sticky='w')
        self.density_blur_var = tk.DoubleVar(value=1.2)
        ttk.Entry(self.control_frame, textvariable=self.density_blur_var, width=6).grid(row=9, column=2, sticky='w')
        
        self.starfield_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.control_frame, text="Starfield", variable=self.starfield_var).grid(row=10, column=0, sticky='w', pady=2)
        self.starfield_count_var = tk.IntVar(value=5000)
        ttk.Spinbox(self.control_frame, from_=500, to=20000, textvariable=self.starfield_count_var, width=6).grid(row=10, column=1, sticky='w')
        self.starfield_layers_var = tk.IntVar(value=3)
        ttk.Spinbox(self.control_frame, from_=1, to=5, textvariable=self.starfield_layers_var, width=6).grid(row=10, column=2, sticky='w')
        
        ttk.Label(self.control_frame, text="Color Mode:").grid(row=11, column=0, sticky='w', pady=2)
        self.color_mode_var = tk.StringVar(value="component")
        ttk.Combobox(self.control_frame, textvariable=self.color_mode_var,
                     values=["component", "radius", "speed", "bound"],
                     state="readonly", width=12).grid(row=11, column=1, sticky='w')
        
        self.camera_follow_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.control_frame, text="Follow COM", variable=self.camera_follow_var).grid(row=12, column=0, sticky='w', pady=2)
        self.auto_zoom_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(self.control_frame, text="Auto Zoom", variable=self.auto_zoom_var).grid(row=12, column=1, sticky='w', pady=2)
        
        ttk.Label(self.control_frame, text="Render Every:").grid(row=13, column=0, sticky='w', pady=2)
        self.render_every_var = tk.IntVar(value=10)
        ttk.Spinbox(self.control_frame, from_=1, to=50, textvariable=self.render_every_var, width=6).grid(row=13, column=1, sticky='w')
        
        # Gravity mode
        ttk.Label(self.control_frame, text="Gravity Mode:").grid(row=8, column=0, sticky='w', pady=5)
        self.gravity_mode_var = tk.StringVar(value="test_particles")
        gravity_mode_combo = ttk.Combobox(
            self.control_frame,
            textvariable=self.gravity_mode_var,
            values=["test_particles", "hybrid", "full_nbody"],
            state="readonly",
            width=15
        )
        gravity_mode_combo.grid(row=8, column=1, pady=5)
        ttk.Label(self.control_frame, text="(auto low N)").grid(row=8, column=2, sticky='w')

        # Profiling
        self.profile_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_frame, text="Profile (timing)", variable=self.profile_var).grid(row=14, column=0, columnspan=2, sticky='w', pady=2)

        # Debug inner disk
        self.debug_inner_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(self.control_frame, text="Debug inner disk", variable=self.debug_inner_var).grid(row=15, column=0, columnspan=2, sticky='w', pady=2)
        
        # Buttons
        self.init_button = ttk.Button(self.control_frame, text="Initialize", command=self.initialize)
        self.init_button.grid(row=16, column=0, columnspan=2, pady=10, sticky='ew')
        
        self.play_button = ttk.Button(self.control_frame, text="Play", command=self.toggle_play,
                                     state='disabled')
        self.play_button.grid(row=17, column=0, columnspan=2, pady=5, sticky='ew')
        
        self.step_button = ttk.Button(self.control_frame, text="Step", command=self.step_once,
                                     state='disabled')
        self.step_button.grid(row=18, column=0, columnspan=2, pady=5, sticky='ew')
        
        # Status
        self.status_label = ttk.Label(self.control_frame, text="Ready", foreground="green")
        self.status_label.grid(row=19, column=0, columnspan=2, pady=10)
        
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
            'spiral_arms': SpiralGalaxy,
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
            
            n_particles = self.particles_var.get()
            gravity_mode = self.gravity_mode_var.get()
            preset_name = self.preset_var.get()
            if n_particles < NBodySystem.LOW_N_THRESHOLD and gravity_mode == "full_nbody":
                gravity_mode = "test_particles"
                self.gravity_mode_var.set(gravity_mode)

            use_analytic_bulge = gravity_mode in ("test_particles", "hybrid")
            use_analytic_disk = gravity_mode == "hybrid"
            preset.use_analytic_bulge = use_analytic_bulge
            preset.use_analytic_disk = use_analytic_disk
            preset.gravity_mode = gravity_mode

            positions, velocities, masses = preset.generate()
            
            integrator = self._get_integrator()
            dt = self.dt_var.get()
            # dt safety cap based on inner orbital period
            r_min_disk = getattr(preset, "min_radius_non_core", None)
            eps_central = getattr(preset, "eps_central_est", None)
            if r_min_disk is not None and eps_central is not None and r_min_disk > 0:
                T_inner = 2 * np.pi * np.sqrt(((r_min_disk ** 2 + eps_central ** 2) ** 1.5) / (preset.core_mass))
                dt_cap = T_inner / 500.0
                if dt > dt_cap:
                    dt = dt_cap
                    self.dt_var.set(dt)
                    self.dt_label.config(text=f"{float(dt):.4f}")

            self_gravity = gravity_mode == "full_nbody"
            self.simulator = Simulator(backend, integrator, dt=dt, self_gravity=self_gravity)
            # Attach preset so component-wise virialization can use it
            self.simulator.preset = preset
            # Spiral arms preset: enable rotating spiral potential
            if preset_name == "spiral_arms":
                self.simulator.system.spiral_potential = SpiralPotential()
            self.simulator.debug_inner = self.debug_inner_var.get()
            self.simulator.debug_interval = 50
            self.simulator.debug_fraction = 0.1
            particle_types = getattr(preset, "particle_types", None)
            virialize = n_particles < NBodySystem.LOW_N_THRESHOLD
            target_Q = 1.1 if n_particles < NBodySystem.LOW_N_THRESHOLD else 1.0
            self.simulator.initialize(
                positions,
                velocities,
                masses,
                virialize=virialize,
                target_Q=target_Q,
                particle_types=particle_types,
            )
            
            # Create renderer
            if self.renderer:
                self.renderer.close()
            
            render_mode = self.render_mode_var.get()
            if render_mode == "3d":
                self.renderer = RenderManager(
                    mode=render_mode,
                    show_trails=self.trails_var.get(),
                    trail_length=self.trail_length_var.get(),
                    color_by_velocity=True,
                    trails=self.trails_var.get(),
                    density=self.density_var.get(),
                    density_res=self.density_res_var.get(),
                    density_blur_sigma=self.density_blur_var.get(),
                    density_alpha=0.25,
                    starfield=self.starfield_var.get(),
                    starfield_count=self.starfield_count_var.get(),
                    starfield_layers=self.starfield_layers_var.get(),
                    color_mode=self.color_mode_var.get(),
                    camera_follow_com=self.camera_follow_var.get(),
                    auto_zoom=self.auto_zoom_var.get(),
                    render_every_k_steps=self.render_every_var.get(),
                    fps_overlay=True
                )
            else:
                self.renderer = RenderManager(
                    mode=render_mode,
                    show_trails=self.trails_var.get(),
                    trail_length=self.trail_length_var.get(),
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
        """Main simulation loop: run K steps per frame, then schedule render on main thread."""
        while self.running and self.simulator:
            self.simulator.set_timestep(self.dt_var.get())
            self.simulator.set_profiling(self.profile_var.get())
            try:
                k = max(1, min(50, self.steps_per_frame_var.get()))
            except (tk.TclError, AttributeError):
                k = 10
            self.simulator.run_steps(k)
            # Snapshot state and schedule render on main thread (matplotlib is not thread-safe)
            pos, vel, mass = self.simulator.get_state()[:3]
            p = np.asarray(pos).copy()
            v = np.asarray(vel).copy()
            m = np.asarray(mass).copy()
            self.root.after(0, lambda p=p, v=v, m=m: self._render_frame(p, v, m))
    
    def _render_frame(self, pos, vel, mass):
        """Run on main thread only: render snapshot and update info labels."""
        if not self.simulator or not self.renderer:
            return
        self.renderer.render(pos, vel, mass)
        self.update_info()
    
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
