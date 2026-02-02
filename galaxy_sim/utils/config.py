"""Configuration management."""

import json
import yaml
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict


@dataclass
class Config:
    """Simulation configuration."""
    # Simulation parameters
    n_particles: int = 1000
    dt: float = 0.01
    integrator: str = "verlet"
    backend: str = "numpy"
    
    # Preset parameters
    preset: str = "spiral"
    preset_params: Dict[str, Any] = None
    
    # Rendering parameters
    render_mode: str = "2d"
    show_trails: bool = False
    color_by_velocity: bool = True
    
    # Export parameters
    export_video: bool = False
    export_gif: bool = False
    output_path: str = "output"
    
    # Reproducibility
    seed: Optional[int] = None
    
    def __post_init__(self):
        if self.preset_params is None:
            self.preset_params = {}


def load_config(config_path: str) -> Config:
    """Load configuration from file.
    
    Args:
        config_path: Path to config file (.json or .yaml)
        
    Returns:
        Config object
    """
    config_path = Path(config_path)
    
    with open(config_path, 'r') as f:
        if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
            try:
                import yaml
                data = yaml.safe_load(f)
            except ImportError:
                raise ImportError("YAML support requires PyYAML. Install with: pip install pyyaml")
        else:
            data = json.load(f)
    
    return Config(**data)


def save_config(config: Config, output_path: str):
    """Save configuration to file.
    
    Args:
        config: Config object
        output_path: Output file path (.json or .yaml)
    """
    output_path = Path(output_path)
    data = asdict(config)
    
    with open(output_path, 'w') as f:
        if output_path.suffix == '.yaml' or output_path.suffix == '.yml':
            try:
                import yaml
                yaml.dump(data, f, default_flow_style=False)
            except ImportError:
                raise ImportError("YAML support requires PyYAML. Install with: pip install pyyaml")
        else:
            json.dump(data, f, indent=2)
