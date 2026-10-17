import re
import json
import ast

from scr.models.agent.agent import Agent
from scr.models.simulation.checkpoint import Checkpoint

from typing import List, Dict, Union, Optional, Any
from pydantic import BaseModel

class MemoryEncoder(json.JSONEncoder):
    """Custom JSON encoder that can handle Pydantic models for memory objects"""
    def default(self, obj):
        if hasattr(obj, 'model_dump'):
            return obj.model_dump()
        return super().default(obj)

class ActionAndInteractionEntry(BaseModel):
    timestep: int
    what_others_did_to_me: List[str]  # list of observations about other agents did to me
    what_others_said_to_me: List[str]  # list of messages received from other agents
    what_i_did: List[str]  # list of observations did

class PersonalBehaviorHistory(BaseModel):
    action_and_interaction: List[ActionAndInteractionEntry]
    who_hunted_with_me: Dict[str, List[str]]  # prey_id -> list of observation strings

class YourMemory(BaseModel):
    general_observations: List[str]
    family_news: List[str]
    personal_behavior_history: PersonalBehaviorHistory
    long_term_memory: Dict[str, Any]  # agent_id -> description
    short_term_plan: Any  # free-form plan

def get_or_create_timestep_entry(memory: YourMemory, timestep: int) -> ActionAndInteractionEntry:
    """Gets an existing timestep entry or creates a new one if it doesn't exist."""
    for entry in memory.personal_behavior_history.action_and_interaction:
        if entry.timestep == timestep:
            return entry
    
    new_entry = ActionAndInteractionEntry(
        timestep=timestep,
        what_i_did=[],
        what_others_did_to_me=[],
        what_others_said_to_me=[]
    )
    memory.personal_behavior_history.action_and_interaction.append(new_entry)
    memory.personal_behavior_history.action_and_interaction.sort(key=lambda x: x.timestep)
    return new_entry

def process_observations_for_timestep(memory: YourMemory, timestep: int, all_observations_for_timestep: List[str], agent_id: str):
    """Processes observations for a specific timestep and updates the memory."""
    current_timestep_entry = get_or_create_timestep_entry(memory, timestep)
    processed_indices_this_call = set()

    # First pass: collect all hunting observations
    hunting_observations = {}
    for obs_string in all_observations_for_timestep:
        if "prey" in obs_string:
            prey_match = re.search(r"prey_(\w+)", obs_string)
            if prey_match:
                prey_id = prey_match.group(1)
                if prey_id not in hunting_observations:
                    hunting_observations[prey_id] = []
                hunting_observations[prey_id].append(obs_string)

    # Second pass: process regular observations
    for index, obs_string in enumerate(all_observations_for_timestep):
        if obs_string in processed_indices_this_call:
            continue 

        if f"Agent {agent_id} " in obs_string:
            if obs_string not in current_timestep_entry.what_i_did:
                current_timestep_entry.what_i_did.append(obs_string)
            processed_indices_this_call.add(obs_string)
            continue

        if "communicated" in obs_string:
            processed_indices_this_call.add(obs_string)
            continue

        other_actions_match = re.search(f" {agent_id}", obs_string)  

        if other_actions_match:
            if obs_string not in current_timestep_entry.what_others_did_to_me:
                current_timestep_entry.what_others_did_to_me.append(obs_string)
            processed_indices_this_call.add(obs_string)
            continue

    # Process hunting partners
    for prey_id, observations in hunting_observations.items():
        # Check if our agent was involved in this hunt
        agent_involved = any(f"Agent {agent_id}" in obs for obs in observations)
        if agent_involved:
            if prey_id not in memory.personal_behavior_history.who_hunted_with_me:
                memory.personal_behavior_history.who_hunted_with_me[f"prey_{prey_id}"] = []
            # Add all observations for this prey
            for obs in observations:
                if obs not in memory.personal_behavior_history.who_hunted_with_me[f"prey_{prey_id}"]:
                    memory.personal_behavior_history.who_hunted_with_me[f"prey_{prey_id}"].append(obs)
                processed_indices_this_call.add(obs)

    # Process family news
    parts = agent_id.split('_')
    ancestor_id = "_".join(parts[:2])
    for obs in all_observations_for_timestep:
        if ancestor_id in obs and obs not in processed_indices_this_call:
            memory.family_news.append(obs)
            processed_indices_this_call.add(obs)

    # Add any remaining unprocessed observations to general_observations
    for obs in all_observations_for_timestep:
        if obs not in processed_indices_this_call and obs not in memory.general_observations:
            memory.general_observations.append(obs)

def add_structured_message(memory: YourMemory, timestep: int, from_agent_id: str, to_agent_id: str, message_body: str, agent_id: str):
    """Adds a structured message to the memory if it's directed to this agent."""
    if to_agent_id == agent_id:
        current_timestep_entry = get_or_create_timestep_entry(memory, timestep)
        message_text = str(message_body)
        formatted_message_string = f"Step {timestep}: Agent {from_agent_id} communicated to {to_agent_id}: '{message_text}'"
        if formatted_message_string not in current_timestep_entry.what_others_said_to_me:
            current_timestep_entry.what_others_said_to_me.append(formatted_message_string)

def display_personal_history(memory: YourMemory):
    """Displays the agent's personal history in a formatted JSON structure."""
    print(json.dumps(memory.model_dump(), indent=4, sort_keys=False, cls=MemoryEncoder))

def process_data_for_agent(global_observations_raw: List[str], received_messages_str_list: List[str], memory: YourMemory, agent_id: str, visible_steps: int = 5):
    """
    Processes global observations and structured messages for a single agent.

    :param global_observations_raw: List of raw observation strings.
    :param received_messages_str_list: List of stringified message dictionaries for the agent.
    :param memory: YourMemory instance containing the agent's memory structure.
    :param agent_id: The ID of the agent being processed.
    :param visible_steps: Number of recent steps to include in the memory.
    """
    observations_by_timestep = {}
    for obs_string in global_observations_raw:
        timestep_match = re.match(r"Step (\d+):", obs_string)
        if timestep_match:
            timestep = int(timestep_match.group(1))
            observations_by_timestep.setdefault(timestep, []).append(obs_string)

    # Collect all unique timesteps from both observations and messages
    all_timesteps = set(observations_by_timestep.keys())

    # Pre-process received messages string literals into dictionaries
    parsed_messages_by_timestep = {}
    if received_messages_str_list:
        for msg_str in received_messages_str_list:
            try:
                msg_dict = ast.literal_eval(msg_str)
                ts = msg_dict.get('timestamp')
                if isinstance(ts, int):
                    all_timesteps.add(ts)
                    parsed_messages_by_timestep.setdefault(ts, []).append(msg_dict)
                else:
                    print(f"Warning: Message has invalid timestamp '{ts}': {msg_str}")
            except (ValueError, SyntaxError) as e:
                print(f"Warning: Could not parse message string: '{msg_str}'. Error: {e}")
    
    # Sort timesteps and limit to visible_steps
    sorted_timesteps = sorted(list(all_timesteps))
    if len(sorted_timesteps) > visible_steps:
        sorted_timesteps = sorted_timesteps[-visible_steps-1:-1]

    for timestep in sorted_timesteps:
        # print(f"\n--- Processing Timestep {timestep} ---")
        
        # Process structured messages for this timestep
        if timestep in parsed_messages_by_timestep:
            for msg_data in parsed_messages_by_timestep[timestep]:
                add_structured_message(
                    memory,
                    timestep=msg_data['timestamp'],
                    from_agent_id=msg_data['from_agent'],
                    to_agent_id=msg_data['to_agent'],
                    message_body=msg_data['message'],
                    agent_id=agent_id
                )
        
        # Process observations for this timestep
        if timestep in observations_by_timestep:
            base_observations_for_this_timestep = observations_by_timestep[timestep]
            process_observations_for_timestep(memory, timestep, base_observations_for_this_timestep, agent_id)

def process_checkpoint_data(checkpoint: Checkpoint, agent: Agent) -> YourMemory:
    """
    Processes data from a Checkpoint and Agent structure.

    :param checkpoint: Checkpoint object containing observations
    :param agent: Agent object containing messages and other data
    :return: YourMemory object containing processed data
    """
    # Create memory structure
    memory = YourMemory(
        general_observations=[],  # Initialize general_observations as empty list
        family_news=[],
        personal_behavior_history=PersonalBehaviorHistory(
            action_and_interaction=[],
            who_hunted_with_me={}
        ),
        long_term_memory=agent.memory.long_term_memory,
        short_term_plan=agent.memory.short_term_plan,
    )

    # Get visible steps from configuration
    visible_steps = checkpoint.configuration.agent.view.visible_steps

    # Process the data
    process_data_for_agent(checkpoint.observations, agent.memory.received_messages, memory, agent.id, visible_steps)
    
    return memory

# --- Example Usage ---
if __name__ == "__main__":
    # Initialize checkpoint from config
    checkpoint = Checkpoint.initialize_from_config("configA_z4_withStrategy_noGMem")
    
    # Get first agent
    agent = checkpoint.social_environment.agents[0]
    agent_id = agent.id
    
    # Set example observations
    example_observations = [
        "Step 1: Agent agent_1 collected 2 plant",
        "Step 1: Agent agent_1_d48d3bca collected 2 plant",
        "Step 1: Agent agent_3 fighted agent_1 for 3 damage",
        "Step 5: Agent agent_1 allocated 1 plant with agent agent_3",
        "Step 6: Agent agent_1 allocated 1 plant with agent agent_3_d48d3bca",
        "Step 8: Agent agent_1 consumed meat resource for 3 HP gain, current HP: 16",
        "Step 9: Agent agent_1 consumed plant resource for 3 HP gain, current HP: 18",
        "Step 12: Agent agent_1 collected 3 plant",
        "Step 13: Agent agent_1 hunted prey prey_43d2a81b",
        "Step 13: Agent agent_2 hunted prey prey_35bac91b",
        "Step 13: Agent agent_2 hunted prey prey_43d2a81b",
        "Step 13: Agent agent_3 hunted prey prey_35bac91b",
        "Step 19: Agent agent_1 killed prey prey_8b8a205f and obtained 5 meat units. Inventory: [InventoryItem(type='plant', quantity=2, hp_restore=3), InventoryItem(type='meat', quantity=5, hp_restore=3)]",
        "Step 1: Plant plant_1 found with quantity=1/5, nutrition=3 per unit",
        "Step 1: Plant plant_2 found with quantity=2/5, nutrition=3 per unit",
        "Step 1: Prey prey_43d2a81b found with HP=18/18, fight=3, meat_units=6, nutrition=3 per meat unit",
        "Step 1: Prey prey_35bac9a7 found with HP=16/16, fight=3, meat_units=5, nutrition=3 per meat unit",
        "Step 1: Agent agent_3 damaged prey prey_43d2a81b for 6 damage (remaining HP: 12)",
    ]
    checkpoint.observations = example_observations

    # Set example messages
    agent.memory.received_messages = [
        f"{{'type': 'communication', 'from_agent': 'agent_control', 'to_agent': '{agent_id}', 'message': 'Mission objective updated.', 'timestamp': 1}}",
        f"{{'type': 'communication', 'from_agent': 'agent_beta', 'to_agent': '{agent_id}', 'message': 'Thanks for the resource in step 2!', 'timestamp': 3}}"
    ]

    # Process the data
    agent_memory = process_checkpoint_data(checkpoint, agent)
    
    # Display the processed memory
    print("\nProcessed Memory Structure:")
    display_personal_history(agent_memory)