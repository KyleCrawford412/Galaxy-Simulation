"""Preset scenario generators for galaxy simulations."""

from galaxy_sim.presets.spiral import SpiralGalaxy
from galaxy_sim.presets.collision import CollisionScenario
from galaxy_sim.presets.globular import GlobularCluster
from galaxy_sim.presets.cluster import GalaxyCluster
from galaxy_sim.presets.stable_disk import StableDisk
from galaxy_sim.presets.multi_component import (
    MultiComponentGalaxy,
    SpiralDiskGalaxy,
    DiskPlusBulgeGalaxy
)
from galaxy_sim.presets.colliding_galaxies import CollidingGalaxies

__all__ = [
    "SpiralGalaxy",
    "CollisionScenario",
    "GlobularCluster",
    "GalaxyCluster",
    "StableDisk",
    "MultiComponentGalaxy",
    "SpiralDiskGalaxy",
    "DiskPlusBulgeGalaxy",
    "CollidingGalaxies"
]
