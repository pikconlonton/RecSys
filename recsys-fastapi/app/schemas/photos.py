from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PhotoBase(BaseModel):
    photo_id: str
    path: str


class Photo(PhotoBase):
    created_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)
