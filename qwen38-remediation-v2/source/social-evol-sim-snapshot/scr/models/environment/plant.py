"""
Plant Module.

This module defines plant-related models for the simulation.
"""

from pydantic import BaseModel
from scr.utils.random_utils import shared_random

class PlantNode(BaseModel):
    """Represents a plant resource in the simulation."""
    id: str
    quantity: int
    nutrition: int = 1  # HP restored when eaten
    depletion_turn: int = -1  # Turn when plant was depleted (-1 means never depleted)
    respawn_delay: int  # Turns to wait before respawn after depletion (must be set from config)
    capacity: int = 5  # Maximum quantity before stopping growth
    has_logged_death: bool = False  # Track if death has been logged

    @property
    def is_dead(self) -> bool:
        """Check if the plant is dead (quantity is 0)."""
        return self.quantity <= 0

    @property
    def is_at_capacity(self) -> bool:
        """Check if the plant has reached its maximum capacity."""
        return self.quantity >= self.capacity

    def advance_growth(self, current_turn: int) -> None:
        """
        Advance the plant's growth.
        
        This method:
        1. If plant is depleted, tracks the depletion turn
        2. After respawn_delay turns, marks plant for respawn
        3. Otherwise, increases quantity by 1 up to capacity
        
        Args:
            current_turn (int): The current simulation turn
        """
        if self.is_dead:
            if self.depletion_turn == -1:
                # First time being depleted, record the turn
                self.depletion_turn = current_turn
            elif current_turn - self.depletion_turn >= self.respawn_delay:
                # Respawn delay has passed, mark for respawn
                self.quantity = 0  # Keep as dead until respawned
                self.depletion_turn = -1  # Reset depletion turn
                self.has_logged_death = False  # Reset death logging
        elif not self.is_at_capacity:
            # Normal growth, but only if not at capacity
            self.quantity += 1 