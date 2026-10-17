# Models Directory

This directory contains all the data models for the Morality-AI simulation.

## Directory Structure

```
models/
├── core/              # Core data models
│   ├── __init__.py
│   ├── config.py      # Configuration models
│   ├── metadata.py    # Simulation metadata
│   └── logs.py        # Logging models
│
├── environment/       # Environment-related models
│   ├── __init__.py
│   ├── physical.py    # Physical environment
│   ├── social.py      # Social environment
│   ├── location.py    # Location models
│   ├── resource.py    # Resource models
│   └── plant.py       # Plant models
│
├── agent/            # Agent-related models
│   ├── __init__.py
│   ├── agent.py      # Agent model
│   ├── actions.py    # Agent actions
│   └── responses.py  # Agent responses
│
├── simulation/       # Simulation state models
│   ├── __init__.py
│   └── checkpoint.py # Simulation checkpoint
│
└── utils/           # Utility models
    └── __init__.py
```

## Module Descriptions

### Core Models
- `config.py`: Configuration settings and parameters
- `metadata.py`: Simulation metadata and timing information
- `logs.py`: Logging and event tracking

### Environment Models
- `physical.py`: Physical environment state and resources
- `social.py`: Social environment and agent interactions
- `location.py`: Location and spatial models
- `resource.py`: Resource management and types
- `plant.py`: Plant growth and management

### Agent Models
- `agent.py`: Agent state and behavior
- `actions.py`: Available agent actions
- `responses.py`: Agent response handling

### Simulation Models
- `checkpoint.py`: Complete simulation state

## Usage

```python
from scr.models.simulation import Checkpoint
from scr.models.agent import Agent
from scr.models.environment import PhysicalEnvironment

# Create a new simulation checkpoint
checkpoint = Checkpoint()

# Access environment
physical_env = checkpoint.physical_environment
social_env = checkpoint.social_environment

# Work with agents
agents = social_env.agents
```

## Dependencies
- pydantic: Data validation and settings management
- typing: Type hints and annotations
- datetime: Time and date handling

## Recent Changes

- Added `target_location` field to the `Reproduce` action model to specify where the new agent should be placed
- Removed `AgentAttributes` and replaced with direct attributes in `AgentState`
- Added `age`, `parent_id`, and `action_points` fields to `AgentState`
- Modified action handling to no longer deduct action points when actions are performed, as they are reset based on agent age at each step
- Added `children` field to the `Agent` model to track an agent's offspring 