# Action Validator

The Action Validator module is responsible for validating and processing actions in the Morality-AI simulation. It ensures that actions are valid, contextually appropriate, and follow the simulation's rules.

## Overview

The validator performs several key functions:
- Validates LLM-generated actions against simulation rules
- Ensures actions are contextually appropriate
- Checks for proper action formatting and structure
- Maintains simulation state consistency

## Usage

### Basic Usage

```python
from scr.simulation.act_manager.validator.validator import validate_llm_action
from scr.models.simulation.checkpoint import Checkpoint
from scr.models.prompt_manager import Messages, prepare_agent_prompts
from scr.api.llm_api.client import get_client
from scr.api.llm_api.completions import get_completions

# Initialize required components
checkpoint = Checkpoint()
messages = Messages()
agent = checkpoint.social_environment.agents[0]

# Prepare prompts and get LLM response
messages = prepare_agent_prompts(checkpoint, agent)
response = get_completions(client, messages, model)
messages.append("assistant", response)

# Validate the action
raw_output = messages.get_last_message_content()
success, action, errors = validate_llm_action(raw_output, checkpoint)

# Check results
if success:
    print("✅ Action validated successfully")
else:
    print(f"❌ Validation failed: {errors}")
```

### Running Multiple Tests

```python
def run_multiple_tests(num_tests: int = 10):
    results = []
    for i in range(num_tests):
        success = test_validator()
        results.append(success)
    
    # Print summary
    passed = sum(results)
    failed = num_tests - passed
    print(f"\nTest Summary:")
    print(f"Total tests: {num_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Success rate: {(passed/num_tests)*100:.1f}%")
```

## Components

### Validator (`validator.py`)
The main validator class that processes and validates actions. It:
- Parses LLM responses
- Validates action structure
- Ensures contextual appropriateness
- Returns validation results

### Rules (`rules/`)
Contains specific validation rules for different types of actions:
- Contextual constraints
- Action format validation
- State consistency checks

### Utils (`utils/`)
Helper functions and utilities for the validator:
- JSON processing
- Action parsing
- Error handling

## Validation Process

1. **Input Processing**
   - Receives raw LLM output
   - Parses JSON structure
   - Extracts action components

2. **Context Validation**
   - Checks agent existence
   - Validates target existence
   - Ensures proper state

3. **Action Validation**
   - Verifies action format
   - Checks action parameters
   - Validates against rules

4. **State Update**
   - Creates validation checkpoint
   - Attempts action execution
   - Verifies state consistency

## Error Handling

The validator provides detailed error messages for various failure cases:
- Invalid action format
- Contextual violations
- State inconsistencies
- Missing or invalid parameters

## Best Practices

1. Always check the success flag before proceeding with actions
2. Handle error messages appropriately
3. Use the validation checkpoint for testing
4. Save failed validations for debugging
5. Monitor validation success rates

## Example Output

Successful validation:
```
✅ Test passed: Valid action validation succeeded
```

Failed validation:
```
❌ Test failed: [Error message details]
```

## Dependencies

- `scr.models.agent.actions`
- `scr.models.simulation.checkpoint`
- `scr.models.prompt_manager`
- `scr.api.llm_api` 