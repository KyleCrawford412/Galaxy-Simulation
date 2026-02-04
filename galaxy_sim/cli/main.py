"""CLI main entry point."""

import argparse
import sys
from pathlib import Path
from galaxy_sim.backends.factory import get_backend, list_available_backends
from galaxy_sim.physics.simulator import Simulator
from galaxy_sim.physics.integrators.euler import EulerIntegrator
from galaxy_sim.physics.integrators.verlet import VerletIntegrator
from galaxy_sim.physics.integrators.rk4 import RK4Integrator
from galaxy_sim.presets import (
    SpiralGalaxy, CollisionScenario, GlobularCluster, GalaxyCluster, StableDisk,
    SpiralDiskGalaxy, DiskPlusBulgeGalaxy, CollidingGalaxies
)
from galaxy_sim.render.manager import RenderManager
from galaxy_sim.io.video_exporter import VideoExporter
from galaxy_sim.io.gif_exporter import GIFExporter
from galaxy_sim.io.state_io import save_state
from galaxy_sim.utils.reproducibility import set_all_seeds
from galaxy_sim.physics.halo_potential import HaloPotential


def get_integrator(name: str):
    """Get integrator by name."""
    integrators = {
        'euler': EulerIntegrator,
        'verlet': VerletIntegrator,
        'rk4': RK4Integrator
    }
    return integrators.get(name.lower())


def get_preset(name: str, backend, n_particles: int, seed: int = None, **kwargs):
    """Get preset by name."""
    presets = {
        'spiral': SpiralGalaxy,
        'collision': CollisionScenario,
        'globular': GlobularCluster,
        'cluster': GalaxyCluster,
        'stable_disk': StableDisk,
        'spiral_disk': SpiralDiskGalaxy,
        'disk_plus_bulge': DiskPlusBulgeGalaxy,
        'colliding_galaxies': CollidingGalaxies
    }
    preset_class = presets.get(name.lower())
    if preset_class is None:
        raise ValueError(f"Unknown preset: {name}. Available: {list(presets.keys())}")
    return preset_class(backend, n_particles=n_particles, seed=seed, **kwargs)


def run_simulation(args):
    """Run a simulation."""
    # Parse self-gravity flag
    self_gravity = (args.self_gravity == 'on')
    
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
    
    # Generate preset with optional parameters
    preset_kwargs = {}
    if args.M_center is not None:
        preset_kwargs['M_center'] = args.M_center
    if args.epsilon is not None:
        preset_kwargs['epsilon'] = args.epsilon
    
    # Parameters for disk_plus_bulge preset
    if args.preset == 'disk_plus_bulge':
        if args.N_disk is not None:
            preset_kwargs['N_disk'] = args.N_disk
        if args.N_bulge is not None:
            preset_kwargs['N_bulge'] = args.N_bulge
        if args.R_d is not None:
            preset_kwargs['R_d'] = args.R_d
        if args.R_b is not None:
            preset_kwargs['R_b'] = args.R_b
        if args.sigma_b is not None:
            preset_kwargs['sigma_b'] = args.sigma_b
        if args.disk_sigma_r is not None:
            preset_kwargs['disk_sigma_r'] = args.disk_sigma_r
        if args.disk_sigma_t is not None:
            preset_kwargs['disk_sigma_t'] = args.disk_sigma_t
        if args.vary_masses:
            preset_kwargs['vary_masses'] = True
    
    # For multi-component galaxies, check if they want analytic halo
    halo_potential = None
    halo_enabled = (args.halo == 'on' or args.halo_enabled)
    
    if args.preset in ['spiral_disk', 'disk_plus_bulge', 'colliding_galaxies']:
        if halo_enabled:
            preset_kwargs['use_analytic_halo'] = True
            preset_kwargs['halo_v0'] = args.halo_v0
            preset_kwargs['halo_rc'] = args.halo_rc
    elif halo_enabled:
        # For other presets, use standalone halo potential
        halo_core = args.halo_core if args.halo_core is not None else args.halo_rc
        halo_potential = HaloPotential(
            model=args.halo_model,
            v_0=args.halo_v0,
            r_c=args.halo_rc,
            M=args.halo_mass,
            a=halo_core,
            enabled=True
        )
    
    # For presets that support halo potential in velocity calculation
    if args.preset in ['spiral_disk', 'disk_plus_bulge', 'colliding_galaxies']:
        if halo_enabled:
            # Pass halo potential to preset for velocity calculation
            preset_kwargs['halo_potential'] = HaloPotential(
                model=args.halo_model,
                v_0=args.halo_v0,
                r_c=args.halo_rc,
                M=args.halo_mass,
                a=args.halo_core if args.halo_core is not None else args.halo_rc,
                enabled=True
            )
    
    preset = get_preset(args.preset, backend, args.particles, args.seed, **preset_kwargs)
    
    # Create simulator with self-gravity setting
    sim = Simulator(backend, integrator, dt=args.dt, halo_potential=halo_potential, self_gravity=self_gravity)
    positions, velocities, masses = preset.generate()
    
    # Store preset reference for virialization (to access particle_types)
    sim.preset = preset
    
    # Get particle types from preset if available
    particle_types = None
    if hasattr(preset, 'particle_types'):
        particle_types = preset.particle_types
    
    sim.initialize(positions, velocities, masses, virialize=args.virialize, target_Q=args.target_Q, particle_types=particle_types)
    
    # Store epsilon for debug output
    debug_epsilon = args.epsilon if args.epsilon is not None else sim.system.epsilon
    
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
    print(f"Backend: {backend.name}, Integrator: {integrator.name}, dt: {args.dt}, eps: {debug_epsilon:.4f}")
    
    # Use consistent diagnostics
    from galaxy_sim.physics.diagnostics import Diagnostics
    
    diagnostics = Diagnostics(
        backend,
        G=sim.system.G,
        epsilon=sim.system.epsilon,
        halo_potential=halo_potential,
        self_gravity=self_gravity,
        particle_types=particle_types
    )
    
    # Compute initial energies using consistent diagnostics
    K0, U0, E0 = diagnostics.compute_energies(
        sim.system.positions,
        sim.system.velocities,
        sim.system.masses
    )
    Q0 = diagnostics.compute_virial_ratio(
        sim.system.positions,
        sim.system.velocities,
        sim.system.masses
    )
    
    # Compute initial angular momentum (returns magnitude)
    L0 = sim.system.compute_angular_momentum()
    Lz0 = float(L0)  # Returns magnitude (Lz for 2D, |L| for 3D)
    
    print(f"{'Step':<8} {'Time':<10} {'K':<12} {'U':<12} {'E':<12} {'Q':<8} {'Lz':<12} {'dE/E0':<10}")
    print("-" * 90)
    print(f"{0:<8} {0.0:<10.2f} {K0:<12.2f} {U0:<12.2f} {E0:<12.2f} {Q0:<8.4f} {Lz0:<12.2f} {0.0:<10.2f}%")
    
    initial_energy = E0
    initial_ang_mom = L0
    previous_energy = initial_energy
    
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
        
        # Debug output every N frames
        if step % args.debug_every == 0:
            # Use consistent diagnostics
            K, U, E = diagnostics.compute_energies(
                sim.system.positions,
                sim.system.velocities,
                sim.system.masses
            )
            Q = diagnostics.compute_virial_ratio(
                sim.system.positions,
                sim.system.velocities,
                sim.system.masses
            )
            L = sim.system.compute_angular_momentum()
            if isinstance(L, (int, float)) or (hasattr(L, '__len__') and len(L) == 1):
                Lz = float(L) if not hasattr(L, '__len__') else float(L[0])
            else:
                Lz = float(L[-1]) if len(L) > 2 else float(L[0])
            dE_total = (E - initial_energy) / abs(initial_energy) * 100 if abs(initial_energy) > 0 else 0
            
            print(f"{step:<8} {sim.time:<10.2f} {K:<12.2f} {U:<12.2f} {E:<12.2f} {Q:<8.4f} {Lz:<12.2f} {dE_total:<10.2f}%")
            previous_energy = E
    
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
                       choices=['spiral', 'collision', 'globular', 'cluster', 'stable_disk',
                                'spiral_disk', 'disk_plus_bulge', 'colliding_galaxies'],
                       help='Preset scenario')
    parser.add_argument('--particles', type=int, default=1000,
                       help='Number of particles')
    parser.add_argument('--steps', type=int, default=1000,
                       help='Number of simulation steps')
    parser.add_argument('--dt', type=float, default=0.01,
                       help='Time step')
    parser.add_argument('--M-center', type=float, default=None,
                       help='Central mass (for stable_disk preset, default: 1000)')
    parser.add_argument('--epsilon', type=float, default=None,
                       help='Softening parameter epsilon (default: adaptive)')
    parser.add_argument('--debug-every', type=int, default=10,
                       help='Print debug info every N steps')
    
    # Halo potential
    parser.add_argument('--halo', type=str, choices=['on', 'off'], default='off',
                       help='Enable/disable analytic halo potential (default: off)')
    parser.add_argument('--halo-enabled', action='store_true',
                       help='Enable analytic halo potential (deprecated, use --halo on)')
    parser.add_argument('--halo-model', type=str, choices=['flat', 'plummer'], default='flat',
                       help='Halo potential model: flat (rotation curve) or plummer (default: flat)')
    parser.add_argument('--halo-v0', type=float, default=1.0,
                       help='Halo asymptotic circular velocity v₀ (for flat model, default: 1.0)')
    parser.add_argument('--halo-rc', type=float, default=1.0,
                       help='Halo core radius r_c (for flat model, default: 1.0)')
    parser.add_argument('--halo-mass', type=float, default=1000.0,
                       help='Halo mass M (for Plummer model, default: 1000.0)')
    parser.add_argument('--halo-core', type=float, default=None,
                       help='Halo core/scale radius a (for Plummer model, uses --halo-rc if not specified)')
    
    # Parameters for disk_plus_bulge preset
    parser.add_argument('--N-disk', type=int, default=None,
                       help='Number of disk particles (for disk_plus_bulge preset)')
    parser.add_argument('--N-bulge', type=int, default=None,
                       help='Number of bulge particles (for disk_plus_bulge preset)')
    parser.add_argument('--R-d', type=float, default=None,
                       help='Disk scale radius R_d (for disk_plus_bulge preset, default: 15.0)')
    parser.add_argument('--R-b', type=float, default=None,
                       help='Bulge scale radius R_b (for disk_plus_bulge preset, default: 4.0)')
    parser.add_argument('--sigma-b', type=float, default=None,
                       help='Bulge velocity dispersion σ_b (for disk_plus_bulge preset, default: 0.8)')
    parser.add_argument('--disk-sigma-r', type=float, default=None,
                       help='Disk radial velocity dispersion (default: 0.05 * v_circ)')
    parser.add_argument('--disk-sigma-t', type=float, default=None,
                       help='Disk tangential velocity dispersion (default: 0.02 * v_circ)')
    parser.add_argument('--vary-masses', action='store_true',
                       help='Use lognormal mass distribution instead of uniform (for disk_plus_bulge preset)')
    
    # Virialization
    parser.add_argument('--virialize', action='store_true',
                       help='Rescale initial velocities to achieve virial equilibrium (Q=1.0)')
    # Determine default target_Q based on preset (if preset is known)
    # Disk simulations need Q=1.2-1.6 for stability (low-N disks unstable at Q=1.0)
    # Isotropic clouds use Q=1.0
    default_target_Q = 1.0  # Default for unknown presets
    
    parser.add_argument('--target-Q', type=float, default=None,
                       help='Target virial ratio Q = 2K/|U|. Default: 1.3 for disk simulations (stable_disk, spiral_disk, disk_plus_bulge, spiral, colliding_galaxies), 1.0 for isotropic clouds (globular, cluster). Low-N disks are unstable at Q=1.0, use 1.2-1.6 for stability.')
    
    # Self-gravity
    parser.add_argument('--self-gravity', type=str, default='on', choices=['on', 'off'],
                       help='Enable/disable self-gravity for disk particles. If off, disk particles only feel analytic potentials (halo + bulge + central), preventing two-body relaxation. Default: on')
    
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
    
    # Set default target_Q based on preset type if not specified
    if args.target_Q is None:
        # Disk simulations need higher Q for stability (low-N disks unstable at Q=1.0)
        disk_presets = ['stable_disk', 'spiral_disk', 'disk_plus_bulge', 'spiral', 'colliding_galaxies']
        if args.preset.lower() in disk_presets:
            args.target_Q = 1.3  # Default for disk simulations
        else:
            args.target_Q = 1.0  # Default for isotropic clouds
    
    run_simulation(args)


if __name__ == '__main__':
    main()
