from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import crud
from app.db.session import get_db
from app.services.recommender import recommender_service


router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.get("/{user_id}")
def get_recommendations(user_id: str, topk: int = 10, db: Session = Depends(get_db)):
    """Return recommendations for FE.

    Current implementation uses the latest logs and returns a ranked list.
    Once logs include item_id/business_id, this endpoint can serve your real
    session-aware + Faiss recommendations.
    """

    items = recommender_service.recommend(db=db, user_id=user_id, topk=topk)
    business_ids: list[str] = []
    for it in items:
        if not isinstance(it, dict):
            continue
        bid = it.get("business_id")
        if isinstance(bid, str) and bid:
            business_ids.append(bid)

    biz_map = crud.get_businesses_by_ids(db, list(dict.fromkeys(business_ids)))

    enriched_items = []
    for it in items:
        if not isinstance(it, dict):
            enriched_items.append(it)
            continue

        bid = it.get("business_id")
        biz = biz_map.get(bid) if isinstance(bid, str) else None
        metadata = None
        if biz is not None:
            metadata = {
                "name": biz.name,
                "stars": biz.stars,
                "review_count": biz.review_count,
                "categories": biz.categories,
                "address": biz.address,
                "lat": biz.lat,
                "lng": biz.lng,
            }
        enriched_items.append({**it, "metadata": metadata})

    return {
        "user_id": user_id,
        "topk": topk,
        "items": enriched_items,
    }
