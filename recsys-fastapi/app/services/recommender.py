from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.db import crud


class RecommenderService:
    """Recommendation facade for the FastAPI app.

    Contract:
    - Input: DB session + optional user_id
    - Output: list[dict] suitable for JSON response

        Notes:
        - With `business_id` available, we can already return a basic behavioral
            recommendation: most-viewed businesses from the latest logs.
        - This file remains the right place to plug your real session-aware + Faiss
            inference (`srcs/inference/inference.py`) once you provide the offline
            artifacts and mapping logic.
    """

    def __init__(self, recent_log_limit: int = 10):
        self.recent_log_limit = recent_log_limit

    def recommend(self, db: Session, user_id: int | None, topk: int = 10) -> list[dict[str, Any]]:
        logs = crud.get_recent_logs(db=db, limit=self.recent_log_limit, user_id=user_id)

        if not logs:
            return []

        # Basic heuristic: recommend the most frequently viewed businesses.
        viewed = [l.business_id for l in logs if l.action == "view" and l.business_id]
        counts = Counter(viewed)
        now = datetime.utcnow().isoformat()

        recs: list[dict[str, Any]] = []
        for i, (biz_id, score) in enumerate(counts.most_common(topk), start=1):
            recs.append(
                {
                    "rank": i,
                    "type": "business",
                    "business_id": biz_id,
                    "score": float(score),
                    "generated_at": now,
                }
            )
        return recs


recommender_service = RecommenderService()
