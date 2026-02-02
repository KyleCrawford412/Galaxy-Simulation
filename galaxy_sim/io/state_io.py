"""State I/O for saving and loading simulation states."""

import numpy as np
import json
from typing import Tuple, Dict, Any, Optional
from pathlib import Path


def save_state(
    positions: np.ndarray,
    velocities: np.ndarray,
    masses: np.ndarray,
    output_path: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """Save simulation state to file.
    
    Args:
        positions: Particle positions
        velocities: Particle velocities
        masses: Particle masses
        output_path: Output file path (.npz or .json)
        metadata: Optional metadata dictionary
    """
    output_path = Path(output_path)
    
    if output_path.suffix == '.npz':
        # NumPy compressed format
        save_dict = {
            'positions': positions,
            'velocities': velocities,
            'masses': masses
        }
        if metadata:
            # Convert metadata to arrays/strings for npz
            for key, value in metadata.items():
                if isinstance(value, (int, float, str)):
                    save_dict[f'metadata_{key}'] = value
        np.savez_compressed(output_path, **save_dict)
    
    elif output_path.suffix == '.json':
        # JSON format (less efficient but human-readable)
        state_dict = {
            'positions': positions.tolist(),
            'velocities': velocities.tolist(),
            'masses': masses.tolist(),
            'metadata': metadata or {}
        }
        with open(output_path, 'w') as f:
            json.dump(state_dict, f, indent=2)
    
    else:
        raise ValueError(f"Unsupported file format: {output_path.suffix}. Use .npz or .json")


def load_state(input_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Load simulation state from file.
    
    Args:
        input_path: Input file path
        
    Returns:
        Tuple of (positions, velocities, masses, metadata)
    """
    input_path = Path(input_path)
    
    if input_path.suffix == '.npz':
        data = np.load(input_path)
        positions = data['positions']
        velocities = data['velocities']
        masses = data['masses']
        
        # Extract metadata
        metadata = {}
        for key in data.keys():
            if key.startswith('metadata_'):
                metadata[key[9:]] = data[key].item() if hasattr(data[key], 'item') else data[key]
        
        return positions, velocities, masses, metadata
    
    elif input_path.suffix == '.json':
        with open(input_path, 'r') as f:
            state_dict = json.load(f)
        
        positions = np.array(state_dict['positions'])
        velocities = np.array(state_dict['velocities'])
        masses = np.array(state_dict['masses'])
        metadata = state_dict.get('metadata', {})
        
        return positions, velocities, masses, metadata
    
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}. Use .npz or .json")
