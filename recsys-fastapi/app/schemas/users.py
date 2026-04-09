from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class UserBase(BaseModel):
    user_id: str
    name: str | None = None
    email: str | None = None
    avatar_url: str | None = None


class UserCreate(UserBase):
    """Payload for creating a user.

    All fields except user_id are optional and can be filled later.
    """


class UserUpdate(BaseModel):
    """Payload for updating a user.

    Partial update semantics: only provided fields will be changed.
    """

    name: str | None = None
    email: str | None = None
    avatar_url: str | None = None


class User(UserBase):
    created_at: datetime | None = None
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)
