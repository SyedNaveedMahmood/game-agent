"""
Social Environment Module for the Morality-AI Simulation.

This module manages the social aspects of the simulation environment,
including agent interactions, relationships, and social dynamics.
"""

from typing import List, Dict, Optional, Tuple, cast
import uuid
from pydantic import BaseModel
import math

from scr.models.agent.agent import AgentState, Agent
from ..core.config import Config
from scr.utils.random_utils import shared_random

class DeathRecord(BaseModel):
    """Record of an agent's death with metadata."""
    agent: Agent
    death_reason: str
    dead_time_step: int

class SocialEnvironment(BaseModel):
    """
    Manages the social environment of the simulation.
    
    This includes:
    - Agent management and interactions
    - Social relationships and dynamics
    - Communication and decision-making
    """
    agents: List[Agent] = []
    relationships: Dict[str, Dict[str, float]] = {}
    cemetery: List[DeathRecord] = []

    def record_death(self, agent: Agent, death_reason: str, dead_time_step: int) -> None:
        """
        Record a dead agent in the cemetery.
        
        Args:
            agent (Agent): The agent that died
            death_reason (str): The reason for the agent's death
            dead_time_step (int): The time step when the agent died
        """
        death_record = DeathRecord(
            agent=agent,
            death_reason=death_reason,
            dead_time_step=dead_time_step
        )
        self.cemetery.append(death_record)


    def get_agent_by_id(self, agent_id: str) -> Agent:
        """
        Retrieve an agent from the environment by ID.
        
        Args:
            agent_id (str): The ID of the agent to find
            
        Returns:
            Agent: The agent if found
            
        Raises:
            ValueError: If agent with specified ID is not found
        """
        agent = next((agent for agent in self.agents if agent.id == agent_id), None)
        if agent is None:
            raise ValueError(f"Agent with ID {agent_id} not found")
        return agent

    @classmethod
    def initialize(cls, config: Config) -> 'SocialEnvironment':
        """
        Initialize a new social environment from configuration.
        
        Args:
            config (Config): Configuration settings
            
        Returns:
            SocialEnvironment: The initialized environment
        """
        agents: List[Agent] = []
        
        # Get agent parameters from config structure
        initial_agents: int = config.agent.initial_count
        agent_hp: int = config.agent.hp.initial
        agent_age: int = config.agent.age.initial

        def allocate_agent_counts(ratios: Dict[str, float], total: int) -> Dict[str, int]:
            raw_counts = {t: total * ratios[t] for t in ratios}
            floored_counts = {t: int(math.floor(raw_counts[t])) for t in ratios}
            allocated = sum(floored_counts.values())
            remaining = total - allocated
            remainders = {t: raw_counts[t] - floored_counts[t] for t in ratios}
            sorted_types = sorted(remainders.items(), key=lambda x: x[1], reverse=True)

            for i in range(remaining):
                floored_counts[sorted_types[i][0]] += 1

            return floored_counts

        # Define agent type ratios from config
        ratios = config.agent.ratio.model_dump() 
        agent_types = list(ratios.keys())

        # Calculate count per type
        agent_type_counts = allocate_agent_counts(ratios, initial_agents)

        # Adjust rounding if sum is not equal to total_agents
        allocated = sum(agent_type_counts.values())
        if allocated < initial_agents:
            # Add the missing agents to the first types until filled
            for agent_type in agent_types:
                agent_type_counts[agent_type] += 1
                allocated += 1
                if allocated == initial_agents:
                    break
        index = 0
        for agent_type, count in agent_type_counts.items():
            for _ in range(count):
                # Use the Agent.initialize method instead of direct construction
                agent = Agent.initialize(
                    agent_type=agent_type,
                    agent_id=f"agent_{index+1}",
                    config=config,
                    hp=agent_hp,
                    age=agent_age
                )
                agents.append(agent)
                index += 1
            
        return cls(
            agents=agents,
            relationships={}
        ) 