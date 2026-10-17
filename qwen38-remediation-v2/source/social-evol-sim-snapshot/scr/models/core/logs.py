# 
from pydantic import BaseModel
from typing import List, Optional, Dict, Any

# Remove the import of Messages to break the circular dependency
# from scr.models.prompt_manager.messages import Messages

class Log(BaseModel):
    # Use Any instead of Messages to break the circular dependency
    messages: Optional[Any] = None

class Logs(BaseModel):
    error_logs: List[str] = []
    debug_logs: List[str] = []

class Events(BaseModel):
    action_logs: List[str] = []
    global_events: List[str] = []
