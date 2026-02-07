"""Demo: space-themed 3D renderer with spiral arms; exports GIF."""

from pathlib import Path
from galaxy_sim.backends.numpy_backend import NumPyBackend
from galaxy_sim.physics.simulator import Simulator
from galaxy_sim.physics.integrators.verlet import VerletIntegrator
from galaxy_sim.physics.spiral_potential import SpiralPotential
from galaxy_sim.presets.spiral import SpiralGalaxy
from galaxy_sim.render.manager import RenderManager
from galaxy_sim.io.gif_exporter import GIFExporter


def main():
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "space_visuals_demo.gif"

    backend = NumPyBackend()
    preset = SpiralGalaxy(backend, n_particles=3000, seed=42)
    preset.use_analytic_bulge = True
    preset.use_analytic_disk = True

    positions, velocities, masses = preset.generate()
    sim = Simulator(backend, VerletIntegrator(), dt=0.01, self_gravity=False)
    sim.preset = preset
    sim.system.spiral_potential = SpiralPotential(
        m=2,
        pitch_angle_deg=18.0,
        pattern_speed=0.35,
        amplitude=0.05,
        r_ref=preset.disk_scale_radius * 2.5,
        r_sigma=preset.disk_scale_radius,
    )
    sim.initialize(positions, velocities, masses, particle_types=getattr(preset, "particle_types", None))

    renderer = RenderManager(
        mode="3d",
        show_trails=True,
        trail_length=25,
        density=True,
        density_res=256,
        density_blur_sigma=1.2,
        density_alpha=0.25,
        starfield=True,
        starfield_count=5000,
        starfield_layers=3,
        color_mode="component",
        camera_follow_com=True,
        auto_zoom=True,
        render_every_k_steps=10,
        fps_overlay=False,
    )
    gif = GIFExporter(str(output_path), fps=20)

    total_steps = 600
    render_every = 10
    for step in range(total_steps):
        sim.step()
        if step % render_every == 0:
            pos, vel, mass = sim.system.get_state()[:3]
            renderer.render(pos, vel, mass)
            frame = renderer.capture_frame()
            gif.add_frame(frame)

    gif.export()
    renderer.close()
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
