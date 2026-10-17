"""
Configuration Module for the Morality-AI Simulation.

This module defines the configuration parameters and settings for the simulation.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Literal, Type
from pydantic import BaseModel, Field, root_validator, ConfigDict
from scr.utils.logger import get_logger
from scr.utils.prompt_loader import load_prompts

logger = get_logger(__name__)

# 1. A global registry
_RESOURCE_REGISTRY: Dict[str, BaseModel] = {}

def register_resource(cls: Type[BaseModel]):
    """Class decorator to register a resource model by its own .type field."""
    _RESOURCE_REGISTRY[cls.__fields__['type'].default] = cls
    return cls


# 2. The discriminated base
class ResourceBase(BaseModel):
    """Base class for all resource configurations"""
    type: str = Field(..., description="Resource kind; drives which subclass is used")
    initial_quantity: int = Field(0, ge=0, description="Number of resources to place at simulation start")


# 3. Two concrete resources
@register_resource
class PlantConfig(ResourceBase):
    """
    Configuration for plant resources in the simulation.
    
    Parameters from settings.json:
    - type: Always 'plant' for this resource type
    - initial_quantity: Number of plants to place at start (can be 0)
    - capacity: Maximum units a plant can provide
    - respawn_delay: Time steps until respawn after depletion
    - nutrition: HP restored when an agent consumes a plant
    
    These values control the availability and value of plant resources.
    """
    type: Literal['plant'] = 'plant'
    capacity: int = 5          # Maximum units before plant depletion
    respawn_delay: int = 10    # Time steps until respawn after depletion
    nutrition: int = 1         # HP restored when consumed by agent


@register_resource
class PreyConfig(ResourceBase):
    """
    Configuration for prey animal resources in the simulation.
    
    Parameters from settings.json:
    - type: Always 'prey' for this resource type
    - initial_quantity: Number of prey animals to place at start (can be 0)
    - hp: Health points of the prey (determines hunting difficulty)
    - fight: Damage dealt to agent on failed hunt
    - respawn_rate: Probability per step of new prey spawning
    - meat_unit_hp: How much prey HP converts to one meat unit (used in calculation, not stored directly)
    - nutrition: HP restored when an agent consumes meat
    - max_quantity: Maximum number of prey animals allowed in the simulation
    
    These values balance the risk/reward of hunting versus gathering.
    """
    type: Literal['prey'] = 'prey'
    hp: int = 4                # Prey health points
    physical_ability: int = 1            # Damage dealt to agent on failed hunt
    respawn_rate: float = 0.1  # Per-step spawn probability at empty location
    # meat_unit_hp: int = 2      # Used to calculate meat units (max_hp // meat_unit_hp)
    # nutrition: int = 2         # HP restored per meat unit when consumed
    max_quantity: int = 10     # Maximum number of prey animals allowed
    hp_std : int = 1
    difficulty: int = 1
    
class AgentRatio(BaseModel):
    universal_group_focused_moral: float
    reciprocal_group_focused_moral: float
    kin_focused_moral: float
    reproductive_selfish: float
    reproduction_averse_selfish: float

class HPConfig(BaseModel):
    """HP configuration for agents"""
    initial: int = 45
    max: int = 60

class AgeConfig(BaseModel):
    """Age configuration for agents"""
    initial: int = 10
    max: int = 100  # Maximum age before death

class InventoryConfig(BaseModel):
    """Inventory configuration for agents"""
    max_size: int = 10
    initial: Dict[str, int] = Field(
        default_factory=lambda: {"meat": 0, "plant": 0},
        description="Initial quantities of meat and plant items in agent inventory"
    )

class ReproductionConfig(BaseModel):
    """Reproduction configuration for agents"""
    min_hp: int = 11
    hp_cost: int = 10
    min_age: int = 4
    offspring_initial_hp: int = 3

class PhysicalAbilityValueConfig(BaseModel):
    """Physical ability values distribution configuration"""
    mean: float = 5.0
    sd: float = 2.5

class PhysicalAbilityConfig(BaseModel):
    """Physical ability configuration for agents"""
    values: PhysicalAbilityValueConfig = PhysicalAbilityValueConfig()
    scaling: Dict[str, float] = Field(
        default_factory=lambda: {"slope": 5, "intercept": 0.1},
        description="Scaling factors for physical ability comparison"
    )

class ViewConfig(BaseModel):
    """View configuration for agents"""
    show_other_agent_type: bool = False
    visible_steps: int = 3  # Number of previous steps to show, including current step

class AgentConfig(BaseModel):
    """
    Configuration for agents in the simulation.
    
    Parameters from settings.json agent section:
    - initial_count: Number of agents to start with
    - ratio: Ratio of moral to immoral agents
    - hp: HP configuration (initial and max)
    - age: Age configuration (initial)
    - inventory: Inventory configuration (max_size)
    - reproduction: Reproduction configuration (min_hp, hp_cost, min_age, offspring_initial_hp)
    - physical_ability: Physical_ability configuration (power)
    - max_collect_quantity: Maximum items collectible in a single collect action
    - view: View configuration for agent visibility
    """
    initial_count: int = 5
    ratio: AgentRatio
    hp: HPConfig = HPConfig()
    age: AgeConfig = AgeConfig()
    inventory: InventoryConfig = InventoryConfig()
    reproduction: ReproductionConfig = ReproductionConfig()
    physical_ability: PhysicalAbilityConfig = PhysicalAbilityConfig()
    max_collect_quantity: int = 2
    view: ViewConfig = ViewConfig()

class WorldConfig(BaseModel):
    """
    Configuration for the world in the simulation.
    
    Parameters from settings.json world section:
    - max_life_steps: Maximum number of time steps to run
    - communication_and_sharing_steps: Number of steps for communication and sharing
    """
    max_life_steps: int = 100
    communication_and_sharing_steps: int = 1

class SimulationConfig(BaseModel):
    """
    Core simulation configuration.
    
    Parameters from settings.json simulation section:
    - name: Identifying name for the simulation
    - version: Simulation version
    - description: Brief description of the simulation purpose
    """
    name: str = "Default Simulation"
    version: str = "1.0.0"
    description: str = "A simulation of agent interactions in a moral framework"

class ResourceConfig(BaseModel):
    """
    Configuration for resources in the simulation.
    
    Parameters from settings.json resource section:
    - plant: Configuration for plant resources
    - prey: Configuration for prey animals
    """
    plant: PlantConfig = PlantConfig(initial_quantity=5)
    prey: PreyConfig = PreyConfig(initial_quantity=3)
    abundance: int = 1

class AsyncConfig(BaseModel):
    """
    Configuration for async/parallel LLM execution.

    Parameters from settings.json llm.async section:
    - max_concurrent_calls: Maximum simultaneous LLM API calls
    - call_timeout_seconds: Per-call timeout
    - retry_backoff_base: Base for exponential backoff
    - retry_backoff_max: Maximum backoff delay
    - enable_dashboard: Whether to enable Rich Live dashboard
    """
    max_concurrent_calls: int = 10
    call_timeout_seconds: float = 120.0
    retry_backoff_base: float = 2.0
    retry_backoff_max: float = 30.0
    enable_dashboard: bool = False


class LLMConfig(BaseModel):
    """
    Configuration for LLM models.

    Parameters from settings.json llm section:
    - provider: LLM service provider
    - chat_model: Model for agent interactions
    - max_retries: Retry attempts for API calls
    - two_stage_model: Whether to use a two-stage model approach
    - async_config: Configuration for async/parallel execution (optional, has defaults)
    """
    provider: str
    reasoning_model: Optional[str] = None
    chat_model: str
    max_retries: int = 3
    two_stage_model: bool
    async_config: AsyncConfig = AsyncConfig()

class Morality_Prompt(BaseModel):   
    universal_group_focused_moral: str = "Default pure moral prompt"
    reciprocal_group_focused_moral: str = "Default group moral prompt"
    kin_focused_moral: str = "Default kin moral prompt"
    reproductive_selfish: str = "Default reproductive_selfish prompt"
    reproduction_averse_selfish: str = "Default pure immoral prompt"

class Prompts(BaseModel):
    morality: Morality_Prompt = Morality_Prompt()
    rules: str = "Default rules"
    strategies: str = "Default strategies"

class Config(BaseModel):
    """
    Main configuration class for the simulation.
    
    This class loads and represents all settings from the settings.json file with the new structure:
    
    simulation: Core simulation settings
      - name: Identifying name for the simulation
      - version: Simulation version
      - description: Brief description of the simulation purpose
    
    world: World parameters
      - max_life_steps: Maximum simulation duration
    
    agent: Agent configuration
      - initial_count: Number of agents to create
      - ratio: Distribution between moral and immoral agents
      - hp: HP configuration (initial and max)
      - age: Age configuration (initial)
      - inventory: Inventory configuration (max_size)
      - reproduction: Reproduction parameters
      - max_collect_quantity: Maximum items collected at once
    
    resource: Resource configuration
      - plant: Plant resource configuration
      - prey: Prey animal configuration
    
    llm: LLM configuration (formerly ai)
      - provider: LLM service provider
      - chat_model: Model for agent interactions
      - max_retries: Retry attempts for API calls
    
    This configuration drives all aspects of simulation behavior.
    """
    simulation: SimulationConfig = SimulationConfig()
    world: WorldConfig = WorldConfig()
    agent: AgentConfig
    resource: ResourceConfig = ResourceConfig()
    llm: LLMConfig
    prompts: Prompts = Prompts()
    random_seed: Optional[int] = datetime.now().microsecond
    start_date: Optional[str] = None
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def calc_rest_communication_rounds(self, step: int) -> int:
        divisor = self.world.communication_and_sharing_steps + 1
        return divisor -  step % divisor - 1
    
    def isSocialInteractionStep(self, step: int) -> bool:
        divisor = self.world.communication_and_sharing_steps + 1
        if step % divisor == 0:
            return False
        else:
            return True
    
    # Legacy compatibility properties
    @property
    def simulation_name(self) -> str:
        return self.simulation.name
        
    @property
    def version(self) -> str:
        return self.simulation.version
        
    @property
    def description(self) -> str:
        return self.simulation.description
    
    @property
    def resources(self) -> List[ResourceBase]:
        """Convert new resource structure to old flat resources list format for backward compatibility"""
        return [
            self.resource.plant,
            self.resource.prey
        ]
    
    @property
    def termination_conditions(self) -> dict:
        """Convert new structure to old termination_conditions format for backward compatibility"""
        return {
            "max_life_steps": self.world.max_life_steps,
            "min_agents": 1
        }

    def apply_overrides(self, overrides: dict) -> None:
        """Apply CLI overrides to this config, merging non-None values.

        Args:
            overrides: Dict with dotted keys, e.g. {"world.max_life_steps": 5}
        """
        for key, value in overrides.items():
            if value is not None:
                parts = key.split(".")
                obj = self
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)

    @classmethod
    def load_from_dir(cls, config_dir: str) -> 'Config':
        """
        Load configuration from a config directory.
        
        Args:
            config_dir (str): Name of the config directory (e.g., 'config_00')
            
        Returns:
            Config: Loaded configuration object
            
        Raises:
            FileNotFoundError: If config files are not found
            ValueError: If config files are invalid
            PermissionError: If files cannot be accessed
        """
        try:
            # Validate config_dir name
            if not config_dir or not isinstance(config_dir, str):
                raise ValueError("config_dir must be a non-empty string")
            
            # Construct paths
            base_config_dir = os.path.join('config', config_dir)
            settings_path = os.path.join(base_config_dir, 'settings.json')
            prompts_dir = os.path.join(base_config_dir, 'prompts')
            
            # Check if config directory exists
            if not os.path.exists(base_config_dir):
                raise FileNotFoundError(f"Config directory not found: {base_config_dir}")
            
            # Check if settings file exists
            if not os.path.exists(settings_path):
                raise FileNotFoundError(f"Settings file not found at {settings_path}")
            
            # Check if prompts directory exists
            if not os.path.exists(prompts_dir):
                raise FileNotFoundError(f"Prompts directory not found at {prompts_dir}")
            
            # Load settings
            try:
                with open(settings_path, 'r') as f:
                    settings = json.load(f)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON in settings file: {str(e)}")
            except PermissionError as e:
                raise PermissionError(f"Cannot read settings file: {str(e)}")
            
            # Load prompts
            # moral_prompt_path = os.path.join(prompts_dir, 'morality/moral.txt')
            # immoral_prompt_path = os.path.join(prompts_dir, 'morality/immoral.txt')
            morality_dir = os.path.join(prompts_dir, 'morality')
            morality_prompts = {}

            for filename in os.listdir(morality_dir):
                if filename.endswith('.txt'):
                    agent_type = os.path.splitext(filename)[0]  # e.g., 'moral' from 'moral.txt'
                    file_path = os.path.join(morality_dir, filename)
                    try:
                        morality_prompts[agent_type] = load_prompts(file_path)
                    except Exception as e:
                        raise ValueError(f"Failed to load prompt for agent type '{agent_type}': {str(e)}")
                    
            rules_prompt_path = os.path.join(prompts_dir, 'rules.txt')
            strategies_prompt_path = os.path.join(prompts_dir, 'strategies.txt')
            
            # Check if all prompt files exist
            # for path in [moral_prompt_path, immoral_prompt_path, rules_prompt_path, strategies_prompt_path]:
            for path in [rules_prompt_path, strategies_prompt_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"Prompt file not found: {path}")
            
            try:
                # moral_prompt = load_prompts(moral_prompt_path)
                # immoral_prompt = load_prompts(immoral_prompt_path)
                rules_prompt = load_prompts(rules_prompt_path)
                from jinja2 import Template, Undefined
                rules_prompt = Template(rules_prompt, undefined=Undefined).render(**settings)
                strategies_prompt = load_prompts(strategies_prompt_path)
            except Exception as e:
                raise ValueError(f"Failed to load prompts: {str(e)}")
            
            # Create prompts object
            prompts = Prompts(
                morality=Morality_Prompt(
                    universal_group_focused_moral=morality_prompts['universal_group_focused_moral'],
                    reciprocal_group_focused_moral=morality_prompts['reciprocal_group_focused_moral'],
                    kin_focused_moral=morality_prompts['kin_focused_moral'],
                    reproductive_selfish=morality_prompts['reproductive_selfish'],
                    reproduction_averse_selfish=morality_prompts['reproduction_averse_selfish']
                ),
                rules=rules_prompt,
                strategies=strategies_prompt
            )
            
            # Add prompts to settings
            settings['prompts'] = prompts.model_dump()
            
            # Create and return config object
            return cls(**settings)
                
        except (FileNotFoundError, ValueError, PermissionError) as e:
            logger.error(f"Error loading config from directory {config_dir}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading config from directory {config_dir}: {str(e)}")
            raise ValueError(f"Failed to load config: {str(e)}")

