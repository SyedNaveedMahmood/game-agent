from pydantic import BaseModel
from typing import List, Dict, Any

class Memory(BaseModel):
    """Represents an agent's memory."""
    short_term_plan: Any = ""
    received_messages: List[str] = []
    long_term_memory: Dict[str, Any] = {}