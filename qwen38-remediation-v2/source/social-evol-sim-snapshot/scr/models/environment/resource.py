"""
Resource models for the simulation environment.

This module contains the base Resource model that represents
any collectible resource in the simulation.
"""

from pydantic import BaseModel, Field
from typing import Optional

class Resource(BaseModel):
    """
    Base model for all collectible resources in the environment.
    
    Resources are objects that agents can interact with, collect,
    and potentially consume. Each resource has a unique ID, a type,
    and a quantity.
    
    Attributes:
        id (str): Unique identifier for the resource
        type (str): The type of resource (e.g., "plants", "water", "minerals")
        quantity (int): The amount of the resource available
    """
    id: str = Field(..., description="Unique identifier for the resource")
    type: str = Field(..., description="Type of resource")
    quantity: int = Field(default=10, ge=0, description="Amount of resource available")
    
    def __str__(self) -> str:
        """String representation of the resource."""
        return f"{self.type}({self.id}, qty={self.quantity})"
    
    def is_depleted(self) -> bool:
        """
        Check if the resource is depleted (quantity = 0).
        
        Returns:
            bool: True if the resource is depleted, False otherwise
        """
        return self.quantity <= 0
    
    def collect(self, amount: int) -> int:
        """
        Collect a specified amount of the resource.
        
        Args:
            amount (int): The amount to collect
            
        Returns:
            int: The actual amount collected (may be less than requested if not enough available)
        """
        collected = min(amount, self.quantity)
        self.quantity -= collected
        return collected 