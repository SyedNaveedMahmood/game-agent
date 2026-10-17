"""
Physical Environment Module for the Morality-AI Simulation.

This module manages the physical aspects of the simulation environment,
including resources.
"""

from typing import List, Dict, Optional, Tuple
from pydantic import BaseModel
import uuid

from ..core.config import Config, PlantConfig, PreyConfig
from scr.models.environment.plant import PlantNode
from scr.models.environment.prey import PreyAnimal
from scr.utils.random_utils import shared_random
from scr.utils.logger import get_logger

logger = get_logger(__name__)

class PhysicalEnvironment(BaseModel):
    """
    Manages the physical environment of the simulation.
    
    This includes:
    - Resource distribution and management
    """
    resources: List[PlantNode] = []
    prey_animals: List[PreyAnimal] = []
    plant_config: Optional[PlantConfig] = None
    prey_config: Optional[PreyConfig] = None

    @classmethod
    def _generate_plant_resources(cls, plant_config: PlantConfig) -> Tuple[List[PlantNode], List[str]]:
        """
        Generate plant resources for the environment based on config.
        
        Args:
            plant_config (PlantConfig): Configuration for plants
            
        Returns:
            Tuple[List[PlantNode], List[str]]: List of generated plant resources and list of observations
        """
        resources = []
        observations = []
        num_resources = plant_config.initial_quantity
        
        # If initial quantity is 0, return empty list
        if num_resources <= 0:
            return resources, observations
        
        # Create resources
        for i in range(num_resources):
            # Start with a reasonable initial quantity (1-3)
            initial_quantity = shared_random.randint(1, min(3, plant_config.capacity))
            
            resource = PlantNode(
                id=f"plant_{i+1}",
                quantity=initial_quantity,
                nutrition=plant_config.nutrition,
                respawn_delay=plant_config.respawn_delay,
                capacity=plant_config.capacity
            )
            resources.append(resource)
            
            # Create observation for initial plant creation
            details = (
                f"Plant {resource.id} found "
                f"with quantity={resource.quantity}/{resource.capacity}, "
                f"nutrition={resource.nutrition} per unit"
            )
            logger.observation(details)
            
            # Add observation to list
            observation = f"Step 1: {details}"
            observations.append(observation)
        
        return resources, observations
    
    @classmethod
    def _generate_prey_resources(cls, prey_config: PreyConfig) -> Tuple[List[PreyAnimal], List[str]]:
        """
        Generate prey animals for the environment based on config.
        
        Args:
            prey_config (PreyConfig): Configuration for prey animals
            
        Returns:
            Tuple[List[PreyAnimal], List[str]]: List of generated prey animals and list of observations
        """
        prey_animals = []
        observations = []
        num_prey = prey_config.initial_quantity
        
        # If initial quantity is 0, return empty list
        if num_prey <= 0:
            return prey_animals, observations
        
        # Create prey animals
        for i in range(num_prey):
            # Generate random HP value based on config with normal distribution
            random_hp = max(1, int(shared_random.gauss(prey_config.hp * prey_config.difficulty+3, prey_config.hp_std)))
            
            # Calculate meat units based on max_hp and meat_unit_hp
            # meat_units = max(1, random_hp // prey_config.meat_unit_hp)
            
            prey = PreyAnimal(
                id=f"prey_{uuid.uuid4().hex[:8]}",
                hp=random_hp,
                max_hp=random_hp,  # Set max_hp to the same as initial hp
                physical_ability=prey_config.physical_ability,
                num_agents_to_kill=prey_config.difficulty*2
                # nutrition=prey_config.nutrition,
                # meat_units=meat_units
            )
            prey_animals.append(prey)
            
            # Create observation for initial prey creation
            details = (
                f"Prey {prey.id} found "
                f"with HP={prey.hp}/{prey.max_hp}, physical_ability={prey.physical_ability}"
                # f"meat_units={prey.get_meat_units()}, "
                # f"nutrition={prey.nutrition} per meat unit"
            )
            logger.observation(details)
            
            # Add observation to list
            observation = f"Step 1: {details}"
            observations.append(observation)
        
        return prey_animals, observations

    @classmethod
    def initialize(cls, config: Config) -> Tuple['PhysicalEnvironment', List[str]]:
        """
        Initialize a new physical environment from configuration.
        
        Args:
            config (Config): Configuration settings
            
        Returns:
            Tuple[PhysicalEnvironment, List[str]]: The initialized environment and list of observations
        """
        # Initialize lists for resources and observations
        resources: List[PlantNode] = []
        prey_animals: List[PreyAnimal] = []
        observations: List[str] = []
        
        abundance = config.resource.abundance

        # Get plant config and generate plant resources
        plant_config: PlantConfig = config.resource.plant
        plant_config_adapted = PlantConfig(
            type='plant',
            initial_quantity=plant_config.initial_quantity * abundance,
            capacity=plant_config.capacity + abundance,
            respawn_delay=int(plant_config.respawn_delay + -5 * abundance),
            nutrition=plant_config.nutrition
        )
        resources, plant_observations = cls._generate_plant_resources(plant_config_adapted)
        observations.extend(plant_observations)
            
        # Get prey config and generate prey resources
        prey_config: PreyConfig = config.resource.prey
        prey_config_adapted = PreyConfig(
            type='prey',
            initial_quantity=prey_config.initial_quantity * abundance,
            hp=prey_config.hp,
            hp_std=prey_config.hp_std,
            physical_ability=prey_config.physical_ability,
            respawn_rate=prey_config.respawn_rate * abundance,
            # meat_unit_hp=prey_config.meat_unit_hp,
            # nutrition=prey_config.nutrition,
            max_quantity=prey_config.max_quantity * abundance,
            difficulty=prey_config.difficulty
        )
        prey_animals, prey_observations = cls._generate_prey_resources(prey_config_adapted)
        observations.extend(prey_observations)
        
        return cls(
            resources=resources,
            prey_animals=prey_animals,
            plant_config=plant_config_adapted,
            prey_config=prey_config_adapted
        ), observations
    
    def spawn_new_prey(self) -> None:
        """
        Potentially spawn new prey animals in the environment.
        This is called at each time step with a small chance of creating new prey.
        """
        if not self.prey_config:
            return  # Skip if no prey config available
            
        # Get spawn chance from config
        spawn_chance = self.prey_config.respawn_rate
        
        # Check if we're already at max quantity
        if len(self.prey_animals) >= self.prey_config.max_quantity:
            logger.info(f"Already at max quantity of {self.prey_config.max_quantity} prey animals")
            return  # Skip spawning if we're at max quantity
            
        if shared_random.random() < spawn_chance:
            # Generate random HP value based on config with normal distribution
            random_hp = max(1, int(shared_random.gauss(self.prey_config.hp * self.prey_config.difficulty + 3, self.prey_config.hp_std)))
            
            # # Calculate meat units based on max_hp and meat_unit_hp
            # meat_units = max(1, random_hp // self.prey_config.meat_unit_hp)
            
            new_prey = PreyAnimal(
                id=f"prey_{uuid.uuid4().hex[:8]}",
                hp=random_hp,
                max_hp=random_hp,  # Set max_hp to the same as initial hp
                physical_ability=self.prey_config.physical_ability,
                # nutrition=self.prey_config.nutrition,
                # meat_units=meat_units
                num_agents_to_kill=self.prey_config.difficulty*2
            )
            self.prey_animals.append(new_prey)
    
    def remove_dead_prey(self) -> None:
        """Remove any prey animals that have been killed (hp <= 0)."""
        self.prey_animals = [prey for prey in self.prey_animals if not prey.is_dead] 