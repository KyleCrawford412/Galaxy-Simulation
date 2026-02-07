"""Demo: spiral arms via rotating potential; exports GIF if imageio installed."""

import numpy as np
from galaxy_sim.backends.numpy_backend import NumPyBackend
from galaxy_sim.physics.simulator import Simulator
from galaxy_sim.physics.integrators.verlet import VerletIntegrator
from galaxy_sim.physics.spiral_potential import SpiralPotential
from galaxy_sim.presets.spiral import SpiralGalaxy
from galaxy_sim.render.manager import RenderManager
from galaxy_sim.io.gif_exporter import GIFExporter


def main():
    backend = NumPyBackend()
    preset = SpiralGalaxy(backend, n_particles=2000, seed=42)
    preset.use_analytic_bulge = True
    preset.use_analytic_disk = True

    positions, velocities, masses = preset.generate()
    sim = Simulator(backend, VerletIntegrator(), dt=0.01, self_gravity=False)
    sim.preset = preset
    sim.system.spiral_potential = SpiralPotential(
        pitch_angle_deg=18.0,
        amplitude=0.05,
        r_ref=preset.disk_scale_radius * 2.5,
        r_sigma=preset.disk_scale_radius,
    )
    sim.initialize(positions, velocities, masses, particle_types=getattr(preset, "particle_types", None))

    renderer = RenderManager(mode="2d", show_trails=False, color_by_velocity=True)
    gif = GIFExporter("spiral_arms_demo.gif", fps=20)

    total_steps = 600
    render_every = 5
    for step in range(total_steps):
        sim.step()
        if step % render_every == 0:
            pos, vel, mass = sim.system.get_state()[:3]
            renderer.render(pos, vel, mass)
            frame = renderer.capture_frame()
            gif.add_frame(frame)

    gif.export()
    renderer.close()
    print("Saved spiral_arms_demo.gif")


if __name__ == "__main__":
    main()
