from pydantic import BaseModel
from dataclasses import dataclass
from typing import Optional

# Dependencies for tools and the agent
@dataclass
class MyDependencies:
    user_id: int  # User ID for personalized behavior
    user_location: str  # Location for context-specific data

# Result type for structured agent responses
class MyResultType(BaseModel):
    summary: str  # A brief summary of the agent's response
    details: Optional[str] = None  # Optional additional details
