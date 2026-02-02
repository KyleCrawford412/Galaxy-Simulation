# Quick Start Guide

## Installation

1. **Navigate to the project directory:**
   ```bash
   cd "Galaxy Simulation"
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install the package in development mode:**
   ```bash
   pip install -e .
   ```

## Running the Simulator

### Option 1: Command Line Interface (CLI)

After installation, you can use the `galaxy-sim` command:

**Basic simulation with rendering:**
```bash
galaxy-sim --preset spiral --particles 2000 --steps 1000 --render --render-mode 2d
```

**Export to video:**
```bash
galaxy-sim --preset collision --particles 5000 --steps 2000 --export-video --output collision.mp4
```

**List available backends:**
```bash
galaxy-sim --list-backends
```

**Full options:**
```bash
galaxy-sim --preset spiral --particles 5000 --steps 1000 --dt 0.01 --integrator verlet --backend numpy --render --render-mode 2d --seed 42
```

### Option 2: GUI Interface

Launch the graphical interface:
```bash
galaxy-sim-gui
```

Or run directly:
```bash
python -m galaxy_sim.ui.main
```

### Option 3: Run Example Scripts

**Basic example (no rendering):**
```bash
python examples/basic_example.py
```

**Example with rendering:**
```bash
python examples/render_example.py
```

### Option 4: Python API

Create your own script:

```python
from galaxy_sim import Simulator, get_backend
from galaxy_sim.physics.integrators import VerletIntegrator
from galaxy_sim.presets import SpiralGalaxy

# Get backend
backend = get_backend("numpy")

# Create preset
preset = SpiralGalaxy(backend, n_particles=2000, seed=42)
positions, velocities, masses = preset.generate()

# Create simulator
sim = Simulator(backend, VerletIntegrator(), dt=0.01)
sim.initialize(positions, velocities, masses)

# Run simulation
for step in range(1000):
    sim.step()
    if step % 100 == 0:
        print(f"Step {step}: Energy = {sim.get_energy():.6f}")
```

## CLI Options

```
--preset {spiral,collision,globular,cluster}  Preset scenario
--particles N                                  Number of particles
--steps N                                      Number of simulation steps
--dt FLOAT                                     Time step (default: 0.01)
--integrator {euler,verlet,rk4}                Integrator (default: verlet)
--backend {numpy,jax,pytorch,cupy}             Backend (auto-select if not specified)
--gpu                                          Prefer GPU backends
--render                                       Enable real-time rendering
--render-mode {2d,3d}                          Rendering mode
--render-every N                               Render every N steps
--trails                                       Show particle trails
--export-video                                 Export to MP4
--export-gif                                   Export to animated GIF
--output PATH                                   Output file base name
--fps N                                        Frames per second for export
--seed N                                       Random seed for reproducibility
--save-state PATH                              Save final state to file
--list-backends                                List available backends
```

## Troubleshooting

**If `galaxy-sim` command not found:**
- Make sure you installed with `pip install -e .`
- Or run directly: `python -m galaxy_sim.cli.main`

**If matplotlib window doesn't show:**
- Make sure you're using a GUI environment (not SSH without X11 forwarding)
- Try: `python -c "import matplotlib; matplotlib.use('TkAgg')"`

**If import errors:**
- Make sure you're in the project directory or have installed the package
- Check that all dependencies are installed: `pip install -r requirements.txt`
