"""Example with real-time rendering."""

from galaxy_sim import Simulator, get_backend
from galaxy_sim.physics.integrators import VerletIntegrator
from galaxy_sim.presets import CollisionScenario
from galaxy_sim.render.manager import RenderManager

def main():
    """Run a collision simulation with rendering."""
    backend = get_backend("numpy")
    
    # Create collision preset
    preset = CollisionScenario(
        backend,
        n_particles=3000,
        seed=123,
        galaxy_separation=20.0,
        relative_velocity=0.5
    )
    
    positions, velocities, masses = preset.generate()
    
    # Create simulator
    sim = Simulator(backend, VerletIntegrator(), dt=0.01)
    sim.initialize(positions, velocities, masses)
    
    # Create 2D renderer
    renderer = RenderManager(mode="2d", show_trails=False, color_by_velocity=True)
    
    print("Running simulation with rendering...")
    print("Close the matplotlib window to stop.")
    
    try:
        for step in range(1000):
            sim.step()
            
            # Render every 5 steps for better performance
            if step % 5 == 0:
                pos, vel, mass = sim.system.get_state()[:3]
                renderer.render(pos, vel, mass)
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        renderer.close()
        print("Simulation complete!")

if __name__ == "__main__":
    main()
