# Galaxy Simulator

An N-body galaxy simulation framework built in Python. This project implements modular architecture, multiple compute backends, and comprehensive features.

## Features

### Core Capabilities
- **N-body Physics**: Full gravitational force calculations with configurable softening
- **Multiple Integrators**: Euler (baseline), Verlet/Leapfrog (better energy conservation), Runge-Kutta 4th order (high accuracy)
- **Multiple Compute Backends**: Unified API supporting NumPy, JAX, PyTorch, and CuPy
- **2D and 3D Rendering**: Real-time visualization with matplotlib
- **Preset Scenarios**: Spiral galaxies, collisions, globular clusters, galaxy clusters
- **Export Capabilities**: MP4 video and animated GIF export
- **Reproducibility**: Seed management and state save/load
- **CLI and GUI**: Both command-line and graphical interfaces

### Architecture Highlights
- **Modular Design**: Clean separation of physics, rendering, I/O, and UI
- **Backend Abstraction**: Same simulation code runs on different execution engines
- **Extensible**: Easy to add new integrators, presets, or backends

## Installation

### Basic Installation
```bash
pip install -r requirements.txt
```

### With GPU Support
```bash
pip install -r requirements.txt
pip install jax jaxlib  # For JAX backend
# or
pip install torch  # For PyTorch backend
# or
pip install cupy  # For CuPy backend (CUDA only)
```

### With Export Support
```bash
pip install imageio opencv-python
```

### Full Installation
```bash
pip install -e .
pip install -e ".[gpu,export,config]"
```

## Quick Start

### Command Line Interface

Run a spiral galaxy simulation:
```bash
galaxy-sim --preset spiral --particles 5000 --steps 1000 --render --render-mode 2d
```

Export to video:
```bash
galaxy-sim --preset collision --particles 10000 --steps 2000 --export-video --output collision.mp4
```

List available backends:
```bash
galaxy-sim --list-backends
```

### GUI Interface

Launch the graphical interface:
```bash
galaxy-sim-gui
```

Or from Python:
```python
from galaxy_sim.ui import run_gui
run_gui()
```

### Python API

```python
from galaxy_sim import Simulator, get_backend
from galaxy_sim.physics.integrators import VerletIntegrator
from galaxy_sim.presets import SpiralGalaxy

# Get backend
backend = get_backend("numpy")  # or "jax", "pytorch", "cupy"

# Create preset
preset = SpiralGalaxy(backend, n_particles=5000, seed=42)
positions, velocities, masses = preset.generate()

# Create simulator
sim = Simulator(backend, VerletIntegrator(), dt=0.01)
sim.initialize(positions, velocities, masses)

# Run simulation
for _ in range(1000):
    sim.step()
    # Render or export frames as needed
```

## Preset Scenarios

### Spiral Galaxy
```python
from galaxy_sim.presets import SpiralGalaxy

preset = SpiralGalaxy(
    backend,
    n_particles=10000,
    disk_radius=10.0,
    bulge_radius=2.0,
    spiral_arms=2
)
```

### Galaxy Collision
```python
from galaxy_sim.presets import CollisionScenario

preset = CollisionScenario(
    backend,
    n_particles=10000,
    galaxy_separation=20.0,
    relative_velocity=0.5
)
```

### Globular Cluster
```python
from galaxy_sim.presets import GlobularCluster

preset = GlobularCluster(
    backend,
    n_particles=5000,
    radius=5.0
)
```

### Galaxy Cluster
```python
from galaxy_sim.presets import GalaxyCluster

preset = GalaxyCluster(
    backend,
    n_particles=20000,
    n_galaxies=5,
    cluster_radius=30.0
)
```

### Disk Plus Bulge Galaxy
```python
from galaxy_sim.presets import DiskPlusBulgeGalaxy

preset = DiskPlusBulgeGalaxy(
    backend,
    n_particles=1000,
    N_disk=750,      # Number of disk particles
    N_bulge=150,     # Number of bulge particles
    R_d=15.0,        # Disk scale radius
    R_b=4.0,         # Bulge scale radius
    sigma_b=0.8,     # Bulge velocity dispersion
    vary_masses=False,  # Use lognormal mass distribution (optional)
    seed=42
)
```

**CLI Usage:**
```bash
# Basic usage
galaxy-sim --preset disk_plus_bulge --particles 1000 --steps 1000 --render

# With custom parameters
galaxy-sim --preset disk_plus_bulge \
  --particles 2000 \
  --N-disk 1500 \
  --N-bulge 300 \
  --R-d 20.0 \
  --R-b 5.0 \
  --sigma-b 1.0 \
  --vary-masses \
  --render
```

**Features:**
- **Disk**: Exponential radial distribution `r = -R_d * log(1-u)` with uniform angles
- **Bulge**: Compact Plummer distribution centered at origin
- **Velocities**: Disk particles have tangential (rotational) velocities; bulge particles have isotropic random velocities with dispersion σ_b
- **Masses**: Uniform by default, or lognormal distribution if `--vary-masses` is used
- **All parameters configurable**: N_disk, N_bulge, R_d, R_b, σ_b, seed

## Integrators

### Euler Method
- **Order**: O(h)
- **Use Case**: Baseline, fast but less accurate
- **Energy Conservation**: Poor

### Verlet/Leapfrog
- **Order**: O(h²)
- **Use Case**: Good default choice
- **Energy Conservation**: Good (symplectic)

### Runge-Kutta 4th Order
- **Order**: O(h⁴)
- **Use Case**: High accuracy requirements
- **Energy Conservation**: Excellent

## Backends

### NumPy (Default)
- Always available
- CPU only
- Good for small to medium simulations (< 10K particles)

### JAX
- GPU support
- JIT compilation
- Excellent for large simulations

### PyTorch
- GPU support
- Familiar API
- Good for integration with ML workflows

### CuPy
- CUDA GPU only
- NumPy-like API
- High performance for CUDA devices

## Export

### Video Export
```python
from galaxy_sim.io import VideoExporter

exporter = VideoExporter("output.mp4", fps=30)
for frame in frames:
    exporter.add_frame(frame)
exporter.export()
```

### GIF Export
```python
from galaxy_sim.io import GIFExporter

exporter = GIFExporter("output.gif", fps=10)
for frame in frames:
    exporter.add_frame(frame)
exporter.export()
```

### State I/O
```python
from galaxy_sim.io import save_state, load_state

# Save
save_state(positions, velocities, masses, "state.npz", metadata={"time": 10.0})

# Load
positions, velocities, masses, metadata = load_state("state.npz")
```

## Reproducibility

Set seeds for deterministic simulations:
```python
from galaxy_sim.utils.reproducibility import set_all_seeds

set_all_seeds(42, backend)
```

## Performance Tuning

### Particle Count
- **< 1K**: Real-time on CPU
- **1K-10K**: Good performance on CPU, excellent on GPU
- **10K-100K**: Requires GPU for real-time
- **> 100K**: GPU recommended, may need optimizations

### Timestep
- Smaller timesteps = more accurate but slower
- Typical range: 0.001 - 0.1
- Adaptive timestep (future feature)

### Backend Selection
- **NumPy**: Best for small simulations, debugging
- **JAX**: Best for large simulations with GPU
- **PyTorch**: Good if already using PyTorch ecosystem
- **CuPy**: Best for CUDA-only workflows

## Project Structure

```
galaxy_sim/
├── backends/          # Compute backend abstractions
├── physics/           # N-body physics and integrators
├── render/            # 2D and 3D rendering
├── presets/           # Preset scenario generators
├── io/                # Export and state I/O
├── cli/               # Command-line interface
├── ui/                # GUI interface
└── utils/             # Utilities and configuration
```

## Contributing

Contributions welcome! Areas for improvement:
- Barnes-Hut tree for O(N log N) force calculations
- Adaptive timestep
- More preset scenarios
- Additional integrators
- Performance optimizations
- Documentation improvements

## License

MIT License

## Acknowledgments

Built as a demonstration of senior-level Python engineering practices including:
- Modular architecture
- Abstract interfaces
- Multiple backend support
- Comprehensive feature set
- Clean code organization
