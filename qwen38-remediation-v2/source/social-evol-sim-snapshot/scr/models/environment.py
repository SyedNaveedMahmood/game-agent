"""
Environment models for the simulation.

This module re-exports all environment-related models from their
respective modules for backward compatibility.
"""

# Re-export all models from their respective modules
from scr.models.resource import Resource
from scr.models.plant import PlantNode, PlantItem

# For backward compatibility
__all__ = [
    'Resource',
    'PlantNode',
    'PlantItem'
]