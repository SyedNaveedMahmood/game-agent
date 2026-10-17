"""
Agent Module.

This module defines the agent models for the simulation.
"""

from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import random
import numpy as np

from ..core.base_models import InventoryItem
from ..core.logs import Log
from ..core.config import Config
from .actions import Action
from .memory import Memory
from .responses import Response

class AgentState(BaseModel):
    """
    Represents the current state of an agent in the simulation.
    
    The state parameters are derived from simulation_parameters in settings.json:
    - hp: Health points (from agent_hp, default 45). Agents die when hp reaches 0.
    - max_hp: Maximum health points (from agent_max_hp, default 60).
    - age: Age in timesteps (from agent_age, default 10).
    - max_age: Maximum age before death (from agent_age_max, default 100).
    - min_reproduction_hp: Minimum HP required to reproduce (from min_reproduction_hp, default 11)
    - reproduction_hp_cost: HP cost for reproduction (from reproduction_hp_cost, default 10)
    - min_reproduction_age: Minimum age required to reproduce (from min_reproduction_age, default 4)
    - offspring_initial_hp: Initial HP for offspring (from offspring_initial_hp, default 3)
    - physical_ability: Damage dealt by the agent when fighting (normally distributed with mean=5, sd=2.5, range 1-9)
    - phisical_ability_scaling: Scaling factors for physical ability comparison (slope=5, intercept=0.1)
    
    These values can be configured in the simulation_parameters section of settings.json.
    """
    hp: int = 45  # Current health points of the agent
    max_hp: int = 60  # Maximum health points the agent can have
    age: int = 10  # Age of the agent in timesteps
    max_age: int = 100  # Maximum age before death
    min_reproduction_hp: int = 11  # Minimum HP required to reproduce
    reproduction_hp_cost: int = 10  # HP cost for reproduction
    min_reproduction_age: int = 4  # Minimum age required to reproduce
    offspring_initial_hp: int = 3  # Initial HP for offspring
    physical_ability: float = 5.0  # Damage dealt by the agent when fighting
    phisical_ability_scaling: Dict[str, float] = {"slope": 5, "intercept": 0.1}  # Scaling factors for physical ability

class Family(BaseModel):
    """Represents a family of agents."""
    parent_id: Optional[str] = None
    children_ids: List[str] = []

class Agent(BaseModel):
    """Represents an agent in the simulation."""
    id: str
    type: str
    state: Optional[AgentState] = None
    memory: Memory = Memory()
    family: Family = Family()
    response_history: List[Response] = []
    logs: Log = Log()
    
    def is_alive(self) -> bool:
        """
        Check if the agent is alive.
        
        An agent is considered alive if:
        1. The agent has a state
        2. The agent's HP is greater than 0
        
        Returns:
            bool: True if the agent is alive, False otherwise
        """
        return self.state is not None and self.state.hp > 0

    def add_response(self, response: Response) -> None:
        """
        Add a response to the agent's response history.
        Store all responses without limiting the history size.
        
        Args:
            response (Response): The response to add
        """
        self.response_history.append(response)

    def get_last_response(self) -> Response:
        """
        Get the most recent response from the agent's response history.
        
        Returns:
            Response: The most recent response
        """
        return self.response_history[-1] if self.response_history else None
        
    def get_recent_responses(self, limit: int = 10) -> List[Response]:
        """
        Get the most recent responses, limited to the specified number.
        
        Args:
            limit (int): Maximum number of responses to return
            
        Returns:
            List[Response]: The most recent responses, newest first
        """
        return self.response_history[-limit:] if self.response_history else []

    @classmethod
    def initialize(cls, 
                   agent_type: str, 
                   agent_id: str, 
                   config: Config, 
                   hp: Optional[int] = None,
                   age: Optional[int] = None) -> 'Agent':
        """Initialize a new agent from the given configuration.
        
        Args:
            agent_type: The type of agent to create (e.g., 'moral' or 'immoral')
            agent_id: A unique identifier for this agent
            config: The simulation configuration object
            hp: Optional starting health points
            age: Optional starting age
        
        Returns:
            A new Agent instance
        """
        # Get configuration values from the new structure
        max_hp: int = config.agent.hp.max
        initial_hp: int = config.agent.hp.initial
        initial_age: int = config.agent.age.initial
        max_age: int = config.agent.age.max
        min_reproduction_hp: int = config.agent.reproduction.min_hp
        reproduction_hp_cost: int = config.agent.reproduction.hp_cost
        min_reproduction_age: int = config.agent.reproduction.min_age
        offspring_initial_hp: int = config.agent.reproduction.offspring_initial_hp
        
        # Generate fight power from Gaussian distribution
        physical_ability_mean: float = config.agent.physical_ability.values.mean
        physical_ability_sd: float = config.agent.physical_ability.values.sd
        physical_ability_scaling: Dict[str, float] = config.agent.physical_ability.scaling
        
        # Generate fight power until we get a value between 1 and 9
        while True:
            physical_ability: float = np.random.normal(physical_ability_mean, physical_ability_sd)
            # Round to nearest integer and ensure it's between 1 and 9
            physical_ability = int(round(physical_ability))
            if 1 <= physical_ability <= 9:
                break
        
        # Set default values if none provided
        hp_value: int = hp if hp is not None else initial_hp
        age_value: int = age if age is not None else initial_age
        
        # Create agent state
        agent_state = AgentState(
            hp=hp_value,
            max_hp=max_hp,
            age=age_value,
            max_age=max_age,
            min_reproduction_hp=min_reproduction_hp,
            reproduction_hp_cost=reproduction_hp_cost,
            min_reproduction_age=min_reproduction_age,
            offspring_initial_hp=offspring_initial_hp,
            physical_ability=physical_ability,
            phisical_ability_scaling=physical_ability_scaling
        )
        
        # Create the agent
        agent = cls(
            id=agent_id,
            type=agent_type,
            state=agent_state
        )
        
        return agent