"""
Environment Models Module.

This module contains all environment-related models, including
physical environment, social environment, resources,
and plants.
"""

from .physical import PhysicalEnvironment
from .social import SocialEnvironment
from ..core.base_models import InventoryItem

__all__ = [
    'PhysicalEnvironment',
    'SocialEnvironment',
    'InventoryItem'
] 