"""
Base Models Module.

This module contains base models that are shared across different modules
to avoid circular imports.
"""

from pydantic import BaseModel
from typing import List, Dict, Optional, Literal

class InventoryItem(BaseModel):
    """
    Represents an aggregated item in an agent's inventory.
    
    Instead of tracking individual items with unique IDs, this model
    tracks the total quantity of each type of item.
    
    Attributes:
        type (str): The type of item ('plant' or 'meat')
        quantity (int): The number of items of this type
        hp_restore (int): Amount of HP each item restores when consumed
    """
    type: Literal['plant', 'meat']
    quantity: int = 1
    hp_restore: int = 5  # Default for plants, overridden for meat
    
    @classmethod
    def create_plant_item(cls, quantity: int = 1, hp_restore: int = 5) -> 'InventoryItem':
        """Create a plant item with the specified quantity and hp_restore value."""
        return cls(
            type='plant', 
            quantity=quantity,
            hp_restore=hp_restore
        )
    
    @classmethod
    def create_meat_item(cls, quantity: int = 1, hp_restore: int = 10) -> 'InventoryItem':
        """Create a meat item with the specified quantity and hp_restore value."""
        return cls(
            type='meat', 
            quantity=quantity, 
            hp_restore=hp_restore
        ) 