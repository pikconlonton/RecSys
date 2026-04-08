from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class BusinessBase(BaseModel):
    business_id: str
    name: str | None = None
    stars: float | None = None
    review_count: int | None = None
    categories: str | None = None
    address: str | None = None
    lat: float | None = None
    lng: float | None = None


class BusinessUpsert(BusinessBase):
    pass


class Business(BusinessBase):
    updated_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)
