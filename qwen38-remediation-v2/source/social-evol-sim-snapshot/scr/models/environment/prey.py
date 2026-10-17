"""
Prey Module.

This module defines prey animal models for the hunting feature in the simulation.
"""

from pydantic import BaseModel

class PreyAnimal(BaseModel):
    """
    Represents a prey animal resource in the simulation.
    
    Prey animals can be hunted by agents for food, but unlike plants,
    they can defend themselves and may cause damage to the agent.
    """
    id: str
    hp: int = 4  # Current health points of the prey
    max_hp: int = 4  # Original maximum health points
    physical_ability: int = 1  # Damage inflicted on failed hunt
    # nutrition: int = 2  # HP restored per meat unit
    # meat_units: int = 2  # Number of meat units this prey yields when hunted
    num_agents_to_kill: int = 1  # Number of agents that

    @property
    def is_dead(self) -> bool:
        """Check if the prey is dead (hp is 0)."""
        return self.hp <= 0
    
    def take_damage(self, damage: int) -> int:
        """
        Inflict damage on the prey animal.
        
        Args:
            damage (int): Amount of damage to inflict
            
        Returns:
            int: The actual damage dealt
        """
        actual_damage = min(damage, self.hp)
        self.hp -= actual_damage
        return actual_damage
    
    def get_reward_hp(self) -> int:
        """
        Get the HP reward provided when this prey is killed.
        
        Returns:
            int: HP reward provided by the prey
        """
        return self.max_hp

    def get_meat_units(self) -> int:
        """
        Deprecated compatibility alias for the old meat-unit API.

        The current simulation rewards hunting by adding HP directly to the
        killing agent. New code should use get_reward_hp() or max_hp instead.

        Returns:
            int: HP reward provided by the prey
        """
        return self.get_reward_hp()
    
    def counter_fight(self) -> int:
        """
        Get the counter-fight damage the prey deals when a hunt fails.
        
        Returns:
            int: Amount of damage dealt to the agent
        """
        return self.physical_ability 
