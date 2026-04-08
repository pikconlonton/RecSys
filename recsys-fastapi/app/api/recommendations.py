from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.services.recommender import recommender_service


router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("/{user_id}")
def get_recommendations(user_id: int, topk: int = 10, db: Session = Depends(get_db)):
    """Return recommendations for FE.

    Current implementation uses the latest logs and returns a ranked list.
    Once logs include item_id/business_id, this endpoint can serve your real
    session-aware + Faiss recommendations.
    """

    return {
        "user_id": user_id,
        "topk": topk,
        "items": recommender_service.recommend(db=db, user_id=user_id, topk=topk),
    }
