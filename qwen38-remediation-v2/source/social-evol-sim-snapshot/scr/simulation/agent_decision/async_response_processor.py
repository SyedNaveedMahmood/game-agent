"""
Async response processing module for the Morality-AI Simulation.

Async version of response_processor.py for parallel LLM calls.
"""

import asyncio
from typing import Callable, Optional, Tuple
from scr.models.simulation.checkpoint import Checkpoint
from scr.models.prompt_manager import Messages
from scr.api.llm_api.completions import async_get_completions
from scr.simulation.act_manager.validator.validator import validate_llm_response
from scr.utils.logger import get_logger
from scr.utils import sim_logger
from scr.models.agent.responses import Response
from scr.simulation.agent_decision.retry_tracker import ValidationResult, classify_llm_exception

logger = get_logger(__name__)


async def async_process_llm_response(
    client,
    messages: Messages,
    checkpoint: Checkpoint,
    llm_config,
    agent_id: str,
    use_chat_model: bool = False,
    output_type: str = "json",
    on_token: Optional[Callable] = None,
) -> ValidationResult:
    """
    Async version of process_llm_response.

    Process a response from an LLM model using async I/O.
    """
    try:
        use_two_stage = checkpoint.configuration.llm.two_stage_model
        model = checkpoint.configuration.llm.chat_model

        def _truncate_messages(msgs):
            """Truncate message content to 2000 chars max for logging."""
            truncated = []
            for m in msgs:
                entry = dict(m) if isinstance(m, dict) else {"role": getattr(m, "role", ""), "content": getattr(m, "content", "")}
                if isinstance(entry.get("content"), str) and len(entry["content"]) > 2000:
                    entry["content"] = entry["content"][:2000] + "...[truncated]"
                truncated.append(entry)
            return truncated

        sim_logger.emit("llm_request", type="decision", agent_id=agent_id, model=model, two_stage=use_two_stage, message_count=len(messages.messages), messages=_truncate_messages(messages.messages))

        if use_two_stage:
            if on_token:
                on_token("stage", "first_call")
            first_response = await async_get_completions(
                client=client,
                messages=messages,
                model=model,
                response_format={"type": "json_object"},
                on_token=on_token,
            )

            messages.append("assistant", first_response.content)

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
            {{
            "Prey_Hunting_Collaboration_Distribution_Retaliation_Memory_And_Planning": {{
                "<prey_id>": {{
                "hunt_fact_history_of_this_prey": {{
                    "<agent_id>": {{
                    "time_step": "int",
                    "result": "string: 'failed and being damaged by prey' OR 'successfully damaged prey'",
                    "damage": "int",
                    "if_killed": "boolean"
                    }}
                }},
                "communication_and_planning_before_killing_prey": {{
                    "amount_of_reward": "int",
                    "who_communicated_to_hunt_together": ["<agent_id>"],
                    "who_I_want_to_collaborate": ["<agent_id>"],
                    "mutually_confirmed_agents_for_collaboration": ["<agent_id>"],
                    "anyone_wants_me_to_not_hunt_this_prey": {{
                    "<agent_id>": {{
                        "why": "string",
                        "ignore_or_follow": "string",
                        "if_he_hunted_do_I_share": "boolean"
                    }}
                    }},
                    "my_own_distribution_plan": {{
                    "thinking": "string",
                    "share_method": "string: 'fair_to_all_collaborator', 'only_to_my_allies_in_this_hunt', or 'all_to_self'",
                    "<agent_id>": "int"
                    }}
                }},
                "distribution_after_killing_prey": {{
                    "time_step_killed_prey": "int",
                    "winner": "<agent_id>",
                    "reward_redistributed_yet": "boolean",
                    "time_passed_unallocated": "int",
                    "judge_if_winner_still_planning_to_share": "string",
                    "actual_reward_allocation_by_winner": {{
                    "<agent_id>": "int or 'unallocated'"
                    }},
                    "evaluating_the_redistribution": "string",
                    "is_fair_allocation_by_winner": "string: 'true', 'false', 'NA'",
                    "free_rider_winner": "string: 'true', 'false', or 'NA'"
                }},
                "plan_next": {{
                    "thinking": "string",
                    "stage": "string: 'closed_with_fair_share', 'keep_hunting', 'wait_and_ask_for_sharing', 'warn_and_plan_for_retaliation', 'execute_retaliation', 'finished_retaliation', 'give_up_retaliation'",
                    "plan": "string",
                    "retaliation_plan": {{
                    "collaboration_plan": ["<agent_id>"],
                    "retaliation_method": "string: 'rob' or 'fight'",
                    "retaliation_goal": "string"
                    }}
                }},
                "afterward_happenings": {{
                    "thinking": "string",
                    "retaliation_events": {{
                    "time_step_<int>": "string"
                    }},
                    "other_events": "string"
                }},
                "lessons_learned": "string"
                }}
            }},
            "Agent_Specific_Memory": {{
                "<agent_id>": {{
                "important_interaction_history": {{
                    "what_i_did_to_him": {{
                    "time_step_<int>": "int",
                    "action_type": "string: 'fight', 'rob', or 'allocate'",
                    "if_success": "boolean",
                    "reason": "string",
                    "target_moral_type": "string"
                    }},
                    "what_he_did_to_me": {{
                    "time_step_<int>": "int",
                    "action_type": "string: 'fight', 'rob', or 'allocate'",
                    "if_success": "boolean",
                    "reason": "string",
                    "target_moral_type": "string"
                    }}
                }},
                "thinking": "string",
                "moral_type": "string",
                "relationship": "string: 'family', 'ally', 'enemy', etc.",
                "agreement": "string",
                "plan": "string"
                }}
            }},
            "Family_Plan": {{
                "<agent_id>": {{
                "status": "string",
                "plan": "string"
                }}
            }},
            "Plan_For_Reproduction": {{
                "thinking": "string",
                "preconditions_and_subgoals": "string",
                "estimated_time_to_produce_next_child": "int"
            }},
            "Strategies": "string"
            }}
            """

            messages.append("user", reflection_messages)

            if on_token:
                on_token("stage", "reflection")
            second_response = await async_get_completions(
                client=client,
                messages=messages,
                model=model,
                response_format={"type": "json_object"},
                on_token=on_token,
            )

            second_content = second_response.content if hasattr(second_response, "content") else str(second_response)
            messages.append("assistant", second_content)

            result = validate_llm_response(second_content, checkpoint, agent_id=agent_id)

            sim_logger.emit(
                "llm_response", type="decision", agent_id=agent_id,
                success=result.success,
                content=second_content[:2000] if second_content else "",
                input_tokens=getattr(second_response, "input_tokens", None),
                output_tokens=getattr(second_response, "output_tokens", None),
                model=getattr(second_response, "model", model),
                duration_s=getattr(second_response, "duration_s", None),
            )

            if not result.success:
                logger.error(f"Error processing agent {agent_id} response: [{result.error_message}]")
                error_message = f"Your response is invalid. Please try again. Error: {result.error_message}"
                messages.append("user", error_message)

            return result

        else:
            if on_token:
                on_token("stage", "first_call")
            if output_type == "json" or output_type == "structured_outputs":
                actions_response = await async_get_completions(
                    client=client,
                    messages=messages,
                    model=model,
                    response_format={"type": "json_object"},
                    on_token=on_token,
                )

                actions_content = actions_response.content if hasattr(actions_response, "content") else str(actions_response)
                messages.append("assistant", actions_content)

                result = validate_llm_response(actions_content, checkpoint, agent_id=agent_id)

                sim_logger.emit(
                    "llm_response", type="decision", agent_id=agent_id,
                    success=result.success,
                    content=actions_content[:2000] if actions_content else "",
                    input_tokens=getattr(actions_response, "input_tokens", None),
                    output_tokens=getattr(actions_response, "output_tokens", None),
                    model=getattr(actions_response, "model", model),
                    duration_s=getattr(actions_response, "duration_s", None),
                )

                if not result.success:
                    logger.error(f"Error processing agent {agent_id} response: [{result.error_message}]")
                    error_message = f"Your response is invalid. Please try again. Error: {result.error_message}"
                    messages.append("user", error_message)

                return result

            else:
                response = await async_get_completions(
                    client=client,
                    messages=messages,
                    model=model,
                    on_token=on_token,
                )

                text_content = response.content if hasattr(response, "content") else str(response)
                messages.append("assistant", text_content)

                sim_logger.emit(
                    "llm_response", type="decision", agent_id=agent_id,
                    success=True,
                    content=text_content[:2000] if text_content else "",
                    input_tokens=getattr(response, "input_tokens", None),
                    output_tokens=getattr(response, "output_tokens", None),
                    model=getattr(response, "model", model),
                    duration_s=getattr(response, "duration_s", None),
                )

                return ValidationResult(success=True, response=text_content)

    except asyncio.TimeoutError:
        error_msg = "Timeout waiting for LLM response"
        logger.error(f"Timeout processing agent {agent_id} LLM response")
        messages.append("user", f"{error_msg}. Please respond more concisely.")
        return ValidationResult(success=False, errors=[error_msg], error_type_override="timeout")

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
