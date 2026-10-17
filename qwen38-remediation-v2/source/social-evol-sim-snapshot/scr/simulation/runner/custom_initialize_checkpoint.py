from scr.models.core.base_models import InventoryItem
from scr.models.simulation.checkpoint import Checkpoint
from scr.utils.logger import get_logger
import uuid

logger = get_logger(__name__)

def custom_initialize_checkpoint(checkpoint: Checkpoint) -> Checkpoint:
    agents = checkpoint.social_environment.agents
    flag = False
    for agent in agents:
        if agent.type == "kin_focused_moral" and not flag:
            agent.inventory = [InventoryItem.create_plant_item(quantity=10, hp_restore=4)]
            agent.memory.short_term_plan = "agent 2 is my son, I'll share a plant with him now"
            flag = True
        elif agent.type == "kin_focused_moral" and flag:
            agent.inventory = []

    return checkpoint
