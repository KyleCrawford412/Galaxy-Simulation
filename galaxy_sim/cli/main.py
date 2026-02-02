"""CLI main entry point."""

import argparse
import sys
from pathlib import Path
from galaxy_sim.backends.factory import get_backend, list_available_backends
from galaxy_sim.physics.simulator import Simulator
from galaxy_sim.physics.integrators.euler import EulerIntegrator
from galaxy_sim.physics.integrators.verlet import VerletIntegrator
from galaxy_sim.physics.integrators.rk4 import RK4Integrator
from galaxy_sim.presets import SpiralGalaxy, CollisionScenario, GlobularCluster, GalaxyCluster
from galaxy_sim.render.manager import RenderManager
from galaxy_sim.io.video_exporter import VideoExporter
from galaxy_sim.io.gif_exporter import GIFExporter
from galaxy_sim.io.state_io import save_state
from galaxy_sim.utils.reproducibility import set_all_seeds


def get_integrator(name: str):
    """Get integrator by name."""
    integrators = {
        'euler': EulerIntegrator,
        'verlet': VerletIntegrator,
        'rk4': RK4Integrator
    }
    return integrators.get(name.lower())


def get_preset(name: str, backend, n_particles: int, seed: int = None):
    """Get preset by name."""
    presets = {
        'spiral': SpiralGalaxy,
        'collision': CollisionScenario,
        'globular': GlobularCluster,
        'cluster': GalaxyCluster
    }
    preset_class = presets.get(name.lower())
    if preset_class is None:
        raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")
    return preset_class(backend, n_particles=n_particles, seed=seed)


def run_simulation(args):
    """Run a simulation."""
    # Get backend
    backend = get_backend(args.backend, prefer_gpu=args.gpu)
    
    # Set seed
    if args.seed is not None:
        set_all_seeds(args.seed, backend)
    
    # Get integrator
    integrator_class = get_integrator(args.integrator)
    if integrator_class is None:
        print(f"Unknown integrator: {args.integrator}")
        sys.exit(1)
    integrator = integrator_class()
    
    # Create simulator
    sim = Simulator(backend, integrator, dt=args.dt)
    
    # Generate preset
    preset = get_preset(args.preset, backend, args.particles, args.seed)
    positions, velocities, masses = preset.generate()
    sim.initialize(positions, velocities, masses)
    
    # Create renderer
    renderer = None
    if args.render:
        renderer = RenderManager(
            mode=args.render_mode,
            show_trails=args.trails,
            color_by_velocity=True,
            size_by_mass=False  # Disable for now to avoid size issues
        )
    
    # Exporters
    video_exporter = None
    gif_exporter = None
    
    if args.export_video:
        video_exporter = VideoExporter(args.output + ".mp4", fps=args.fps)
    
    if args.export_gif:
        gif_exporter = GIFExporter(args.output + ".gif", fps=args.fps)
    
    # Run simulation
    print(f"Running simulation: {args.preset} with {args.particles} particles")
    print(f"Backend: {backend.name}, Integrator: {integrator.name}, dt: {args.dt}")
    
    for step in range(args.steps):
        sim.step()
        
        if step % args.render_every == 0:
            pos, vel, mass = sim.system.get_state()[:3]
            
            if renderer:
                renderer.render(pos, vel, mass)
            
            if video_exporter:
                frame = renderer.capture_frame() if renderer else None
                if frame is not None:
                    video_exporter.add_frame(frame)
            
            if gif_exporter:
                frame = renderer.capture_frame() if renderer else None
                if frame is not None:
                    gif_exporter.add_frame(frame)
        
        if step % 100 == 0:
            energy = sim.get_energy()
            print(f"Step {step}/{args.steps}, Time: {sim.time:.2f}, Energy: {energy:.6f}")
    
    # Export
    if video_exporter:
        print(f"Exporting video to {video_exporter.output_path}...")
        video_exporter.export()
    
    if gif_exporter:
        print(f"Exporting GIF to {gif_exporter.output_path}...")
        gif_exporter.export()
    
    # Save state
    if args.save_state:
        pos, vel, mass = sim.system.get_state()[:3]
        save_state(pos, vel, mass, args.save_state, metadata={
            'time': sim.time,
            'steps': sim.step_count,
            'preset': args.preset,
            'integrator': args.integrator,
            'backend': backend.name
        })
        print(f"State saved to {args.save_state}")
    
    if renderer:
        renderer.close()
    
    print("Simulation complete!")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Galaxy Simulator - N-body galaxy simulation")
    
    # Simulation parameters
    parser.add_argument('--preset', type=str, default='spiral',
                       choices=['spiral', 'collision', 'globular', 'cluster'],
                       help='Preset scenario')
    parser.add_argument('--particles', type=int, default=1000,
                       help='Number of particles')
    parser.add_argument('--steps', type=int, default=1000,
                       help='Number of simulation steps')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='Time step')
    
    # Backend and integrator
    parser.add_argument('--backend', type=str, default=None,
                       choices=['numpy', 'jax', 'pytorch', 'cupy'],
                       help='Compute backend (auto-select if not specified)')
    parser.add_argument('--gpu', action='store_true',
                       help='Prefer GPU backends')
    parser.add_argument('--integrator', type=str, default='verlet',
                       choices=['euler', 'verlet', 'rk4'],
                       help='Numerical integrator')
    
    # Rendering
    parser.add_argument('--render', action='store_true',
                       help='Enable real-time rendering')
    parser.add_argument('--render-mode', type=str, default='2d',
                       choices=['2d', '3d'],
                       help='Rendering mode')
    parser.add_argument('--render-every', type=int, default=1,
                       help='Render every N steps')
    parser.add_argument('--trails', action='store_true',
                       help='Show particle trails')
    
    # Export
    parser.add_argument('--export-video', action='store_true',
                       help='Export to MP4 video')
    parser.add_argument('--export-gif', action='store_true',
                       help='Export to animated GIF')
    parser.add_argument('--output', type=str, default='output',
                       help='Output file base name')
    parser.add_argument('--fps', type=int, default=30,
                       help='Frames per second for export')
    
    # Reproducibility
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--save-state', type=str, default=None,
                       help='Save final state to file')
    
    # Info
    parser.add_argument('--list-backends', action='store_true',
                       help='List available backends and exit')
    
    args = parser.parse_args()
    
    if args.list_backends:
        backends = list_available_backends()
        print("Available backends:")
        for backend in backends:
            print(f"  - {backend}")
        return
    
    run_simulation(args)


if __name__ == '__main__':
    main()
