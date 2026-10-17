from scr.models.simulation.checkpoint import Checkpoint
from scr.models.agent.actions import Collect, Allocate, Fight, Hunt, Reproduce, Communicate, Rob, DoNothing
from scr.utils.logger import get_logger
from scr.models.agent.agent import Agent
from scr.simulation.act_manager.action_handler.collect import collect
from scr.simulation.act_manager.action_handler.reproduce import reproduce
from scr.simulation.act_manager.action_handler.communicate import communicate
from scr.simulation.act_manager.action_handler.fight import fight
from scr.simulation.act_manager.action_handler.rob import rob
from scr.simulation.act_manager.action_handler.hunt import hunt
from scr.simulation.act_manager.action_handler.doNothing import do_nothing
from scr.simulation.act_manager.action_handler.allocate import allocate


# Initialize logger
logger = get_logger(__name__)

# Update checkpoint based on responses
def update_checkpoint_from_actions(checkpoint: Checkpoint, agent_id: str = None, debug: bool = False):
    """Updates checkpoint based on agent responses.

    Args:
        checkpoint: The current simulation checkpoint.
        agent_id: The agent whose actions to apply. If None, reads from metadata (legacy).
        debug: Whether to print debug info.
    """
    if agent_id is None:
        agent_id = checkpoint.metadata.get_current_agent_id()

    # Find agent in social environment
    agent = next((a for a in checkpoint.social_environment.agents if a.id == agent_id), None)
    if not agent:
        logger.warning(f"Agent {agent_id} not found in checkpoint, skipping action application")
        return

    if not agent.is_alive():
        logger.warning(f"Agent {agent_id} is dead, skipping action application")
        return

    if not agent.response_history:
        logger.warning(f"Agent {agent_id} has no response history, skipping action application")
        return

    # Get the last response and extract the action from it
    response = agent.response_history[-1]
    action = response.action.root

    if isinstance(action, Collect):
        collect(checkpoint, agent, action)

    elif isinstance(action, Allocate):
        allocate(checkpoint, agent, action)

    elif isinstance(action, Fight):
        fight(checkpoint, agent, action)

    elif isinstance(action, Reproduce):
        reproduce(checkpoint, agent, action)

    elif isinstance(action, Communicate):
        communicate(checkpoint, agent, action)

    elif isinstance(action, Rob):
        rob(checkpoint, agent, action)

    elif isinstance(action, Hunt):
        hunt(checkpoint, agent, action)

    elif isinstance(action, DoNothing):
        do_nothing(checkpoint, agent, action)

    # Remove any dead prey animals after the action is handled
    checkpoint.physical_environment.remove_dead_prey()

    # Spawn new prey with a small chance each time step
    checkpoint.physical_environment.spawn_new_prey()
