"""Basic example of using the galaxy simulator."""

from galaxy_sim import Simulator, get_backend
from galaxy_sim.physics.integrators import VerletIntegrator
from galaxy_sim.presets import SpiralGalaxy

def main():
    """Run a basic spiral galaxy simulation."""
    # Get backend (NumPy is always available)
    backend = get_backend("numpy")
    
    # Create a spiral galaxy preset
    preset = SpiralGalaxy(
        backend,
        n_particles=2000,
        seed=42,
        disk_radius=10.0,
        bulge_radius=2.0
    )
    
    # Generate initial conditions
    positions, velocities, masses = preset.generate()
    
    # Create simulator with Verlet integrator
    sim = Simulator(
        backend,
        VerletIntegrator(),
        dt=0.01
    )
    
    # Initialize simulation
    sim.initialize(positions, velocities, masses)
    
    # Run simulation
    print("Running simulation...")
    print(f"Initial energy: {sim.get_energy():.6f}")
    
    for step in range(500):
        sim.step()
        if step % 100 == 0:
            energy = sim.get_energy()
            print(f"Step {step}: Time={sim.time:.2f}, Energy={energy:.6f}")
    
    print(f"Final energy: {sim.get_energy():.6f}")
    print("Simulation complete!")

if __name__ == "__main__":
    main()
