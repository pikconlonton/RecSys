from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class FriendsUpsert(BaseModel):
    user_id: str
    friends: list[str] = Field(default_factory=list)
    timestamp: datetime | None = None


class SocialInteractionCreate(BaseModel):
    user_id: str
    business_id: str

    # FE can send multiple behaviors. We'll map them to weights.
    # Allowed (recommended): view, like, rate, visit
    action: str

    # Optional strength override from FE; if None, backend uses action-based defaults.
    weight: float | None = None

    timestamp: datetime | None = None
