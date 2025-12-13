from datetime import datetime
from typing import List, Optional, Literal
from pydantic import BaseModel, Field
import uuid

class Fact(BaseModel):
    """A semantic fact about the world or user."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    key: str
    value: str
    category: str = "general"
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    source: str = "user"  # user, agent, system
    relevance: float = 1.0  # 0.0 to 1.0 importance score

class Episode(BaseModel):
    """A narrative log of an event."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    outcome: Optional[Literal["success", "failure", "neutral"]] = "neutral"
    tags: List[str] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.now)
    relevance: float = 1.0

class WorkingMemoryItem(BaseModel):
    """A transient thought or active context item."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    ttl: int = 5  # "Turns" to live, or generic time unit

