# scr/models/simulation/checkpoint.py
"""
Checkpoint Module for the Morality-AI Simulation.

This module manages the simulation state, including the physical and social
environments, metadata, and logging.
"""

import json
from datetime import datetime, timezone
from typing import List, Optional, Any, Tuple, Union
from pydantic import BaseModel, Field, ValidationError
import time
import os

from ..core.config import Config
from ..core.metadata import Metadata
from ..core.logs import Logs, Events
from ..environment import PhysicalEnvironment, SocialEnvironment
from scr.utils.logger import get_logger
from scr.utils.random_utils import shared_random

logger = get_logger(__name__)

class Checkpoint(BaseModel):
    """
    Represents a complete simulation state checkpoint:
    - PhysicalEnvironment
    - SocialEnvironment
    - Metadata
    - Events & Logs
    - Configuration
    - A growing list of textual observations
    """
    metadata: Optional[Metadata] = None
    physical_environment: Optional[PhysicalEnvironment] = None
    social_environment: Optional[SocialEnvironment] = None
    events: Optional[Events] = None
    logs: Optional[Logs] = None
    configuration: Optional[Config] = None
    observations: List[str] = Field(default_factory=list)

    @classmethod
    def initialize_from_config(cls, config_dir: str, config_overrides: dict | None = None) -> "Checkpoint":
        """
        Initialize a new checkpoint from a configuration directory.

        Args:
            config_dir (str): Path to the configuration directory
            config_overrides: Optional dict of dotted config keys to override
                before environments/agents are created.

        Returns:
            Checkpoint: A new initialized checkpoint
        """
        try:
            # Create a blank checkpoint instance
            checkpoint = cls()

            # Load configuration
            checkpoint.configuration = Config.load_from_dir(config_dir)
            logger.info(f"Loaded configuration from {config_dir}")

            # Apply overrides BEFORE initializing environments so that
            # agent.initial_count and other structural params take effect.
            if config_overrides:
                checkpoint.configuration.apply_overrides(config_overrides)

            
            # Initialize environments
            checkpoint._initialize_environments()
            
            # Initialize metadata 
            checkpoint._initialize_metadata()
            
            # Initialize logging
            checkpoint._initialize_logging()
            
            return checkpoint
        except Exception as e:
            logger.error(f"Failed to initialize checkpoint from '{config_dir}': {e}")
            raise RuntimeError(f"Checkpoint initialization failed: {e}")
    
    def _initialize_environments(self) -> None:
        """Initialize physical and social environments."""
        # physical environment comes with its own initial observations
        phy_env, init_obs = PhysicalEnvironment.initialize(self.configuration)
        self.physical_environment = phy_env
        self.observations.extend(init_obs)

        # initialize social environment
        self.social_environment = SocialEnvironment.initialize(self.configuration)
        logger.info("Initialized environments")

    def _initialize_metadata(self) -> None:
        """Initialize simulation metadata."""
        self.metadata = Metadata(
            current_time_step=1,
            execution_queue=[agent.id for agent in self.social_environment.agents],
            run_id=datetime.now(timezone.utc).strftime("%m%d-%H%M%S"),
            current_agent_index=0
        )

    def _initialize_logging(self) -> None:
        """Initialize events and logs."""
        self.events = Events()
        self.logs = Logs()
        logger.info("Initialized events and logs")

    def add_observation(self, step: int, agent_id: str, details: str) -> None:
        """
        Record an observation of an agent action.
        
        Args:
            step (int): Current simulation step
            agent_id (str): ID of the agent performing the action
            details (str): String with action-specific details
        """
        observation = f"Step {step}: Agent {agent_id} {details}"
        self.observations.append(observation)
        logger.debug(f"Added observation: {observation}")

    def remove_dead_agents(self, death_reason: str = "natural_causes") -> None:
        """
        Remove dead agents from the checkpoint.
        
        This method:
        1. Identifies agents with HP <= 0
        2. Records their deaths in the social environment
        3. Removes them from the agents list
        4. Updates the execution queue
        
        Args:
            death_reason (str): The reason for the agent's death
        """
        if not self.social_environment:
            logger.warning("Cannot remove dead agents: social environment not initialized")
            return
            
        # Find dead agents
        dead_agents = [agent for agent in self.social_environment.agents if not agent.is_alive()]
        
        if not dead_agents:
            logger.info("No dead agents to remove from the checkpoint")
            return
            
        # Record deaths and remove agents
        for agent in dead_agents:
            self.social_environment.record_death(agent, death_reason, self.metadata.current_time_step)
            
            # Add observation for agent death
            details = f"died at age {agent.state.age} with HP {agent.state.hp}, death reason: {death_reason}"
            self.add_observation(
                step=self.metadata.current_time_step,
                agent_id=agent.id,
                details=details
            )
            
            self.social_environment.agents.remove(agent)
            
        # Update execution queue
        dead_agent_ids = [agent.id for agent in dead_agents]
        self.metadata.execution_queue = [
            agent_id for agent_id in self.metadata.execution_queue 
            if agent_id not in dead_agent_ids
        ]
        
        # Reset current agent index if needed
        if self.metadata.current_agent_index >= len(self.metadata.execution_queue):
            self.metadata.current_agent_index = 0
            
        logger.info(f"Removed {len(dead_agents)} dead agents from the checkpoint: {', '.join(dead_agent_ids)}")
