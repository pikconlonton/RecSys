from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field

class LogCreate(BaseModel):
    user_id: str
    business_id: str
    action: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class Log(LogCreate):
    id: int

    model_config = ConfigDict(from_attributes=True)