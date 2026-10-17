# Prompt Manager

The Prompt Manager module is responsible for preparing and managing prompts for agent decision-making in the simulation.

## Overview

The module consists of several components:

1. **Constructor (`constructor.py`)**
   - Constructs the full prompt by combining:
     - Agent-specific morality prompt (moral/immoral)
     - Simulation rules
     - Strategies
     - Response schema

2. **Agent View (`agent_view.py`)**
   - Creates an agent-specific view of the simulation state
   - Formats the current state into a natural language description
   - Includes relevant information like:
     - Agent's current state
     - Nearby agents and resources
     - Recent events and observations

3. **Messages (`messages.py`)**
   - Manages the conversation history for agent interactions
   - Formats messages for the AI provider's API
   - Handles message validation and formatting

## Usage

The prompts are loaded from the simulation configuration and accessed through `checkpoint.config.prompts`. The structure is:

```python
checkpoint.config.prompts
├── morality
│   ├── moral: str      # Prompt for moral agents
│   └── immoral: str    # Prompt for immoral agents
├── rules: str          # Simulation rules
└── strategies: str     # Available strategies
```

To prepare prompts for an agent:

```python
from scr.models.prompt_manager import prepare_agent_prompts

# Get prompts for an agent
messages = prepare_agent_prompts(checkpoint, agent)

# Access the first message (combined prompt)
combined_prompt = messages.get_message_content(0)
```

## Directory Structure

```
prompt_manager/
├── __init__.py         # Main interface and prompt preparation
├── constructor.py      # Prompt construction
├── agent_view.py       # Agent-specific view generation
├── messages.py         # Message management
└── README.md          # This file
```

## Configuration

Prompts are configured in the simulation config directory:

```
config/
└── config_00/
    ├── settings.json   # Main configuration file
    └── prompts/        # Prompt files
        ├── morality/
        │   ├── moral.txt
        │   └── immoral.txt
        ├── rules.txt
        └── strategies.txt
```

## Dependencies

- `scr.models.agent`: Agent model
- `scr.models.checkpoint`: Checkpoint model
- `scr.models.logs`: Logging models
- `scr.utils.prompt_loader`: Utility for loading prompt files
- `scr.utils.logger`: Logging utilities 