# Morality-AI Simulation Module

## Overview
The simulation module is a core component of the Morality-AI project that handles agent-based simulations in a social environment. It manages agent decision-making, environment interactions, and simulation state management.

## Directory Structure
```
simulation/
├── act_manager/         # Manages agent actions and their execution
├── agent_decision.py    # Core agent decision-making logic
├── cli/                 # Command-line interface components
├── env_manager/         # Environment state and management
├── initialization/      # Simulation initialization and setup
└── runner/             # Simulation execution and control
```

## Core Components

### Agent Decision System
The `agent_decision.py` module handles the process of agents making decisions in the simulation:
- Manages AI model interactions for agent decision-making
- Handles action validation and retry logic
- Maintains agent state and memory
- Implements warning capture and error handling

Key features:
- Retry mechanism for invalid actions (configurable max retries)
- Warning capture system for action validation
- State discrepancy detection
- Memory management for agent-specific information

### Action Management
The `act_manager/` directory contains components for:
- Processing and validating agent actions
- Updating simulation state based on actions
- Managing action history and consequences

### Environment Management
The `env_manager/` directory handles:
- Simulation environment state
- Resource management
- Spatial relationships between agents
- Environment constraints and rules

### Initialization
The `initialization/` directory manages:
- Simulation setup and configuration
- Agent creation and placement
- Initial environment state
- Resource distribution

### Runner
The `runner/` directory contains:
- Simulation execution control
- Time step management
- Event scheduling
- Simulation state persistence

## Key Features
- AI-driven agent decision making
- Robust error handling and validation
- State management and consistency checks
- Configurable simulation parameters
- Comprehensive logging system

## Usage
The simulation module is designed to be used as part of the larger Morality-AI project. It integrates with:
- AI model interfaces for agent decision-making
- Checkpoint system for state management
- Logging system for debugging and monitoring

## Development Guidelines
1. Follow the existing error handling patterns
2. Maintain comprehensive logging
3. Ensure state consistency across all operations
4. Add appropriate type hints and documentation
5. Test new features thoroughly before integration

## Dependencies
- Python 3.13.1
- Project-specific models (Checkpoint, Log)
- AI model integration
- Logging utilities

## Error Handling
The module implements several error handling mechanisms:
- `ActionWarningError` for invalid actions
- Warning capture system for action validation
- State discrepancy detection
- Retry mechanisms for recoverable errors

## Logging
The module uses a comprehensive logging system:
- Different log levels for various types of information
- Warning capture for action validation
- Debug information for agent states and actions
- Error tracking for system issues

## Future Development
Areas for potential enhancement:
- Enhanced agent interaction models
- More sophisticated environment rules
- Improved performance optimization
- Extended validation mechanisms
- Additional simulation scenarios 