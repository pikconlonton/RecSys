from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import crud
from app.db.session import get_db
from app.schemas.social import FriendsUpsert, SocialInteractionCreate


router = APIRouter(prefix="/social", tags=["social"])


@router.post("/friends/upsert")
def upsert_friends(payload: FriendsUpsert, db: Session = Depends(get_db)):
    # timestamp currently not used for storage beyond updated_at.
    processed = crud.replace_friends(db=db, user_id=payload.user_id, friends=payload.friends)
    return {"user_id": payload.user_id, "processed": processed}


@router.post("/interactions")
def create_interaction(payload: SocialInteractionCreate, db: Session = Depends(get_db)):
    action = payload.action.lower().strip()

    # Default weights if FE doesn't provide a weight.
    action_weights = {
        "view": 1.0,
        "visit": 1.0,
        "like": 2.0,
        "rate": 3.0,
    }

    weight = payload.weight
    if weight is None:
        weight = action_weights.get(action, 1.0)

    ts = payload.timestamp or datetime.utcnow()

    obj = crud.create_social_interaction(
        db,
        {
            "user_id": payload.user_id,
            "business_id": payload.business_id,
            "action": action,
            "weight": float(weight),
            "timestamp": ts,
        },
    )

    return {
        "id": obj.id,
        "user_id": obj.user_id,
        "business_id": obj.business_id,
        "action": obj.action,
        "weight": obj.weight,
        "timestamp": obj.timestamp.isoformat() if obj.timestamp else None,
    }
