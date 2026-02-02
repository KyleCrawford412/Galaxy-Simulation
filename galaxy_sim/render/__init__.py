"""Rendering system for 2D and 3D visualization."""

from galaxy_sim.render.renderer_2d import Renderer2D
from galaxy_sim.render.renderer_3d import Renderer3D
from galaxy_sim.render.manager import RenderManager

__all__ = ["Renderer2D", "Renderer3D", "RenderManager"]
