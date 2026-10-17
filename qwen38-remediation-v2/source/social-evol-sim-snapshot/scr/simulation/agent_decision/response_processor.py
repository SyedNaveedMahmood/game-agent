"""
Response processing module for the Morality-AI Simulation.

This module handles the processing and validation of AI responses.
"""

from typing import Tuple, Optional, Dict, Any
from scr.models.simulation.checkpoint import Checkpoint
from scr.models.prompt_manager import Messages
from scr.api.llm_api.completions import get_completions
from scr.simulation.act_manager.update_checkpoint_from_actions import update_checkpoint_from_actions
from scr.simulation.act_manager.validator.validator import validate_llm_response
from scr.utils.logger import get_logger
from scr.models.agent.responses import Response
from scr.simulation.agent_decision.retry_tracker import ValidationResult, classify_llm_exception

# Initialize logger
logger = get_logger(__name__)

def update_agent_memory_from_response(checkpoint: Checkpoint, response: Response, agent_id: str = None) -> None:
    """
    Update the agent's memory based on the response.

    Args:
        checkpoint (Checkpoint): The current simulation checkpoint
        response (Response): The validated response from the agent
        agent_id (str, optional): The agent ID. If None, reads from metadata (legacy).
    """
    if agent_id is None:
        current_agent_id = checkpoint.metadata.get_current_agent_id()
    else:
        current_agent_id = agent_id
    current_agent = checkpoint.social_environment.get_agent_by_id(current_agent_id)
    
    if not current_agent:
        logger.error(f"Agent {current_agent_id} not found in checkpoint")
        return
    
    # Update the agent's memory with the memory string from the response
    # try:
    #     if response.memory:
    #         current_agent.memory.memory = response.memory
    #         logger.info(f"Updated memory for agent {current_agent_id}")
    # except:
    #     logger.warning("Memory field is not present in the response")
    try:
        if response.short_term_plan:
            current_agent.memory.short_term_plan = response.short_term_plan
            logger.info(f"Updated short_term_plan for agent {current_agent_id}")
    except:
        logger.warning("Short_term_plan field is not present in the response")
    
    # Update the agent's memory with the memory from the response
    try:
        if response.long_term_memory:
            # Update the memory
            current_agent.memory.long_term_memory.update(response.long_term_memory)
            logger.info(f"Updated long_term_memory for agent {current_agent_id}")
    except:
        logger.warning("Long_term_memory field is not present in the response")

def process_llm_response(
    client: object,
    messages: Messages,
    checkpoint: Checkpoint,
    llm_config: object,
    agent_id: str,
    use_chat_model: bool = False,
    output_type: str = "json"
) -> ValidationResult:
    """
    Process a response from an LLM model.

    Returns a ValidationResult carrying the validation stage on failure.
    """
    # Get chat and reasoning models
    try:
        use_two_stage = checkpoint.configuration.llm.two_stage_model
        model = checkpoint.configuration.llm.chat_model
        
        # Two-stage approach with reasoning model first, then chat model
        if use_two_stage:
            # Generate reasoning with reasoning model
            # Only reasoning model is called directly here
            first_response = get_completions(
                client=client,
                messages=messages,
                model=model,
                response_format={'type': 'json_object'},
                stream=False
            )


            
            # Add the reasoning to the messages
            messages.append("assistant", first_response.content)

            #TODO
            reflection_messages = f"""
            Reflect: 
            1. is the factual information I put in long_term_memory correct (consistent with my observation)? 
            1.1. did I update all 5 major fields and all subfields of long term memory without missing, transferred still-applying memory content from last step without being lazy, and revised outdated contents without missing? (i understand, once discarded, the content is not included in the memory anymore) 
            1.2. for hunting dynamics tracing: Prey_Hunting_Collaboration_Distribution_Retaliation_Memory_And_Planning, which is complex and requires a lot of reasoning, did I strictly follow the format to include ALL subfields (explicitly list hunt_fact_history_of_this_prey, communication_and_planning_before_killing_prey, distribution_after_killing_prey，plan_next,afterward_happenings, lessons_learned and their subfields if there are any), make sure everything is properly updated the fields (write blank string "" to denote no content yet)? especially did I update  hunting_fact_history field correctly?
            1.3. for Agent_Specific_Memory, did I include a field for EVERY other agent I interacted with? Did i miss any agent in my memory update?
            2. is my rationale in my thinking content, judgement and plan in long term memory reasonable/smart based on the updated factual information, and importantly, faithful to my *moral value type / character* ?
            2.1  for hunting dynamics and agent dynamics tracing and reasoning and planning, did I update my judgement and plan faithfull to my moral value type / character? 
            2.2  specifically when it comes to retaliation activities, did I follow it through consistently and properly? Did I forget about to update my judgement, plan, goal and execution? 
            3. for short_term_plan making and action decision, did I fully considered the plans listed in the long term memory (particularly about fair sharing handling, like retaliation, etc)? 
            Reflect and improve my response in the prescribed format again. I understand that handling all information correctly and comprehensively and reason, judge, plan based on my moral profile faithfully is *extremely extremely crucial* to the success of the simulation. I will spare no effort to make sure I do it perfectly.
            Only this round's response will be preserved.

            """ \
            + \
            """
            Check the long_term_memory response against this format:
            {
            "Prey_Hunting_Collaboration_Distribution_Retaliation_Memory_And_Planning": {
                "<prey_id>": {
                "hunt_fact_history_of_this_prey": {
                    "<agent_id>": {
                    "time_step": "int",
                    "result": "string: 'failed and being damaged by prey' OR 'successfully damaged prey'",
                    "damage": "int",
                    "if_killed": "boolean"
                    }
                },
                "communication_and_planning_before_killing_prey": {
                    "amount_of_reward": "int",
                    "who_communicated_to_hunt_together": ["<agent_id>"],
                    "who_I_want_to_collaborate": ["<agent_id>"],
                    "mutually_confirmed_agents_for_collaboration": ["<agent_id>"],
                    "anyone_wants_me_to_not_hunt_this_prey": {
                    "<agent_id>": {
                        "why": "string",
                        "ignore_or_follow": "string",
                        "if_he_hunted_do_I_share": "boolean"
                    }
                    },
                    "my_own_distribution_plan": {
                    "thinking": "string",
                    "share_method": "string: 'fair_to_all_collaborator', 'only_to_my_allies_in_this_hunt', or 'all_to_self'",
                    "<agent_id>": "int"
                    }
                },
                "distribution_after_killing_prey": {
                    "time_step_killed_prey": "int",
                    "winner": "<agent_id>",
                    "reward_redistributed_yet": "boolean",
                    "time_passed_unallocated": "int",
                    "judge_if_winner_still_planning_to_share": "string",
                    "actual_reward_allocation_by_winner": {
                    "<agent_id>": "int or 'unallocated'"
                    },
                    "evaluating_the_redistribution": "string",
                    "is_fair_allocation_by_winner": "string: 'true', 'false', 'NA'",
                    "free_rider_winner": "string: 'true', 'false', or 'NA'"
                },
                "plan_next": {
                    "thinking": "string",
                    "stage": "string: 'closed_with_fair_share', 'keep_hunting', 'wait_and_ask_for_sharing', 'warn_and_plan_for_retaliation', 'execute_retaliation', 'finished_retaliation', 'give_up_retaliation'",
                    "plan": "string",
                    "retaliation_plan": {
                    "collaboration_plan": ["<agent_id>"],
                    "retaliation_method": "string: 'rob' or 'fight'",
                    "retaliation_goal": "string"
                    }
                },
                "afterward_happenings": {
                    "thinking": "string",
                    "retaliation_events": {
                    "time_step_<int>": "string"
                    },
                    "other_events": "string"
                },
                "lessons_learned": "string"
                }
            },
            "Agent_Specific_Memory": {
                "<agent_id>": {
                "important_interaction_history": {
                    "what_i_did_to_him": {
                    "time_step_<int>": "int",
                    "action_type": "string: 'fight', 'rob', or 'allocate'",
                    "if_success": "boolean",
                    "reason": "string",
                    "target_moral_type": "string"
                    },
                    "what_he_did_to_me": {
                    "time_step_<int>": "int",
                    "action_type": "string: 'fight', 'rob', or 'allocate'",
                    "if_success": "boolean",
                    "reason": "string",
                    "target_moral_type": "string"
                    }
                },
                "thinking": "string",
                "moral_type": "string",
                "relationship": "string: 'family', 'ally', 'enemy', etc.",
                "agreement": "string",
                "plan": "string"
                }
            },
            "Family_Plan": {
                "<agent_id>": {
                "status": "string",
                "plan": "string"
                }
            },
            "Plan_For_Reproduction": {
                "thinking": "string",
                "preconditions_and_subgoals": "string",
                "estimated_time_to_produce_next_child": "int"
            },
            "Strategies": "string"
            }
            """
            
            # Add a system message to constrain the chat model
            messages.append("user", reflection_messages)

            logger.info(f"Reflection messages: {reflection_messages}")
            
            second_response = get_completions(
                client=client,
                messages=messages,
                model=model,
                response_format={'type': 'json_object'},
                stream=False
            )
            
            # Extract content from the response
            if hasattr(second_response, 'content'):
                second_content = second_response.content
            else:
                second_content = str(second_response)
            
            # Store the AI response
            messages.append("assistant", second_content)

            # Validate and process the response
            result = validate_llm_response(second_content, checkpoint, agent_id=agent_id)

            # Log any errors
            if not result.success:
                logger.error(f"Error processing agent {agent_id} response: [{result.error_message}]")
                error_message = f"Your response is invalid. Please try again. Error: {result.error_message}"
                messages.append("user", error_message)

            return result
            
        # Single model approach (chat model only)
        else:            
            # JSON output format
            if output_type == "json" or output_type == "structured_outputs":
                # Get completion with json_object response format
                actions_response = get_completions(
                    client=client,
                    messages=messages,
                    model=model,
                    response_format={'type': 'json_object'},
                    stream=False
                )
                
                # Extract content from the response
                if hasattr(actions_response, 'content'):
                    actions_content = actions_response.content
                else:
                    actions_content = str(actions_response)
                
                # Store the AI response
                messages.append("assistant", actions_content)

                result = validate_llm_response(actions_content, checkpoint, agent_id=agent_id)

                # Log any errors
                if not result.success:
                    logger.error(f"Error processing agent {agent_id} response: [{result.error_message}]")
                    error_message = f"Your response is invalid. Please try again. Error: {result.error_message}"
                    messages.append("user", error_message)

                return result
                
            # Text output format
            else:
                # Get completion as text
                response = get_completions(
                    client=client,
                    messages=messages,
                    model=model,
                    stream=False
                )
                
                # Extract content from the response
                if hasattr(response, 'content'):
                    text_content = response.content
                else:
                    text_content = str(response)
                
                # Store the AI response
                messages.append("assistant", text_content)

                return ValidationResult(success=True, response=text_content)

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        error_msg = str(e) or f"{type(e).__name__}: no details available"
        logger.error(f"Error processing agent {agent_id} response: [{error_msg}]\nTraceback: {error_trace}")
        messages.append("user", f"LLM call failed with error: {error_msg}. Please try again.")
        return ValidationResult(
            success=False,
            errors=[error_msg],
            error_type_override=classify_llm_exception(e),
        )