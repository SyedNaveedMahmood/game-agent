"""
Response models for agent actions in the simulation.
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict, Any
from scr.models.agent.actions import Action


class Response(BaseModel):
    agent_id: str = Field(
        description="The ID of the agent that is responding"
    )
    thinking: str = Field(
        description="The thinking and planing scratchpad. Refer to previous rule part of how to think and plan here.  Limited up to 1000 characters."
    )
    short_term_plan: Any = Field(
        description="Only put what you plan to do in near future. If nothing it can be left blank. Update timely. *Do NOT* put information like numbers and locations about prey or plant, or inventory here. Note down the time step when you make plan and when you want to finish the plan. Limited up to 1000 characters."
    )
    long_term_memory: Dict[str, Any] = Field(
        description="Refer to prevous rule part of how to update the long term memory. Limited up to 1000 characters."
    )
    action: Action