# models/metadata.py
from pydantic import BaseModel
from typing import Dict, Optional, List

class Metadata(BaseModel):
    """Metadata for a simulation run."""
    current_time_step: int = 0
    execution_queue: List[str] = []
    run_id: str
    current_agent_index: int = 0

    def get_current_agent_id(self) -> str:
        return self.execution_queue[self.current_agent_index]
    
    def get_next_agent_id(self) -> str:
        self.current_agent_index += 1
        return self.execution_queue[self.current_agent_index]
    
    def advance_agent_index(self):
        self.current_agent_index += 1
