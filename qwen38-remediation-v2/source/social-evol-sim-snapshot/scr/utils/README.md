# Agent View Extraction Utility

This utility provides functions to extract and parse agent views from message content in the Morality-AI Simulation.

## Overview

In the Morality-AI Simulation, agent views are formatted as structured text within AI model responses, starting with "Current time step" and containing several JSON blocks with information about the agent, observations, and environment.

This utility helps extract and parse this information from message content.

## Functions

### `extract_agent_view(message_content: str) -> str`

Extracts the raw agent view text from message content, starting from "Current time step" to the end of the message.

### `parse_agent_view(agent_view_text: str) -> Optional[Dict[str, Any]]`

Parses the extracted agent view text into a structured dictionary with the following keys:
- `Current time step`: The current simulation time step
- `your_info`: Information about the agent
- `personal_observations`: Observations specific to the agent
- `general_observations`: General observations about the environment
- `observations`: Information about the physical and social environment

### `extract_and_parse_agent_view(message_content: str) -> Optional[Dict[str, Any]]`

Combines the extract and parse functions to directly extract and parse the agent view from message content.

## Usage Example

```python
from scr.utils.extract_agent_view import extract_and_parse_agent_view

# Message content containing agent view
message = """
AI model response text...

"Current time step": 3
```json
{
    "name": "Agent1",
    "state": {
        "location": [10, 20],
        "hp": 100
    }
}
```
Personal Observations:
```json
["Step 3: Agent1 moved north"]
```
...
"""

# Extract and parse agent view
agent_view = extract_and_parse_agent_view(message)

# Use the parsed data
if agent_view:
    print(f"Time step: {agent_view['Current time step']}")
    print(f"Agent name: {agent_view['your_info']['name']}")
    print(f"Agent location: {agent_view['your_info']['state']['location']}")
```

## Notes

- If the message doesn't contain an agent view, the extract functions will return an empty string and the parse functions will return `None`.
- Error handling is included to handle unexpected formats. 