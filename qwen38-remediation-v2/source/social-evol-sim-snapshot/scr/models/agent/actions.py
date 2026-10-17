"""
Actions Module.

This module defines the available actions for agents in the simulation.
"""

from enum import Enum
from pydantic import BaseModel, Field, RootModel
from typing import Literal, Optional, Union, List, Dict, Any

class ActionBase(BaseModel):
    action_type: str
    reason: str 

#  ---------------- Collect ----------------
class Collect(ActionBase):
    action_type: Literal["collect"]
    resource_id: str  # ID of the specific resource to collect
    quantity: int # Amount to collect

#  ---------------- Allocation ----------------
class AllocationReasonLabel(str, Enum):
    SHARING_WITH_FAMILY = "sharing_with_family"
    SHARING_WITH_OTHER = "sharing_with_other"
    REDISTRIBUTION = "redistribution"

class Allocate(ActionBase):
    action_type: Literal["allocate"]
    # target_agent_ids: List[str]  # List of target agent IDs
    allocation_reason_label: AllocationReasonLabel
    allocation_plan: Dict[str, int]  # Amount to allocate

#  ---------------- Fight ----------------
class FightReasonLabel(str, Enum):
    RETALIATE_AGAINST_ATTACK = "retaliate_against_attack"
    RETALIATE_AGAINST_ROB = "retaliate_against_rob"
    PUNISH_UNFAIR_SHARING = "punish_for_unfair_sharing"

class Fight(ActionBase):
    action_type: Literal["fight"]
    target_agent_id: str
    fight_reason_label: FightReasonLabel

#  ---------------- Rob ----------------
class RobReasonLabel(str, Enum):
    RETALIATE_AGAINST_ATTACK = "retaliate_against_attack"
    RETALIATE_AGAINST_ROB = "retaliate_against_rob"
    PUNISH_UNFAIR_SHARING = "punish_for_unfair_sharing"

class Rob(ActionBase):
    action_type: Literal["rob"]
    resource_type: Literal["plant", "meat"]
    quantity: int
    target_agent_id: str 
    rob_reason_label: RobReasonLabel

#  ---------------- Hunt ----------------
class Hunt(ActionBase):
    action_type: Literal["hunt"]
    prey_id: str

#  ---------------- Reproduce ----------------
class Reproduce(ActionBase):
    action_type: Literal["reproduce"]

#  ---------------- Communicate ----------------
class Communicate(ActionBase):
    action_type: Literal["communicate"]
    target_agent_ids: List[str]  # List of target agent IDs
    message: str

#  ---------------- Do Nothing ----------------
class DoNothing(ActionBase):
    action_type: Literal["do_nothing"]

# Define the discriminated union with the discriminator set in the base class
class Action(RootModel):
    """Union type for all possible actions."""
    root: Union[Collect, Allocate, Fight, Rob, Hunt, Reproduce, Communicate, DoNothing]

    def dict(self, *args, **kwargs):
        """Convert the action to a dictionary."""
        if isinstance(self.root, dict):
            return self.root
        return self.root.model_dump(*args, **kwargs)

    @classmethod
    def validate(cls, v):
        """Validate the action."""
        if isinstance(v, dict):
            action_type = v.get("action_type")
            if action_type == "collect":
                return cls(root=Collect(**v))
            elif action_type == "allocate":
                return cls(root=Allocate(**v))
            elif action_type == "fight":
                return cls(root=Fight(**v))
            elif action_type == "rob":
                return cls(root=Rob(**v))
            elif action_type == "hunt":
                return cls(root=Hunt(**v))
            elif action_type == "reproduce":
                return cls(root=Reproduce(**v))
            elif action_type == "communicate":
                return cls(root=Communicate(**v))
            elif action_type == "do_nothing":
                return cls(root=DoNothing(**v))
            else:
                raise ValueError(f"Invalid action type: {action_type}")
        return v

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Action":
        """Create an Action instance from a dictionary."""
        if not isinstance(data, dict):
            raise ValueError("Input must be a dictionary")
        
        # If the action is nested under an 'action' key, extract it
        if "action" in data:
            data = data["action"]
        
        action_type = data.get("action_type")
        if not action_type:
            raise ValueError("Missing action_type field")
        
        try:
            if action_type == "collect":
                return cls(root=Collect(**data))
            elif action_type == "allocate":
                return cls(root=Allocate(**data))
            elif action_type == "fight":
                return cls(root=Fight(**data))
            elif action_type == "rob":
                return cls(root=Rob(**data))
            elif action_type == "hunt":
                return cls(root=Hunt(**data))
            elif action_type == "reproduce":
                return cls(root=Reproduce(**data))
            elif action_type == "communicate":
                return cls(root=Communicate(**data))
            elif action_type == "do_nothing":
                return cls(root=DoNothing(**data))
            else:
                raise ValueError(f"Invalid action type: {action_type}")
        except Exception as e:
            raise ValueError(f"Error creating {action_type} action: {str(e)}")
