from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any

from sqlalchemy.orm import Session

from app.db import crud
from app.services.artefacts import RecSysArtefacts
from app.services.inference_runtime import InferenceConfig, recommend_from_recent_views


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
        self._artefacts: RecSysArtefacts | None = None
        self._cfg = InferenceConfig()

    def set_artefacts(self, artefacts: RecSysArtefacts) -> None:
        self._artefacts = artefacts

    def recommend(self, db: Session, user_id: int | None, topk: int = 10) -> list[dict[str, Any]]:
        # Pull recent logs for that user. If for any reason the filter returns
        # empty but data exists (e.g., legacy rows), fall back to global recent.
        logs = crud.get_recent_logs(db=db, limit=self.recent_log_limit, user_id=user_id)
        if user_id is not None and not logs:
            logs = crud.get_recent_logs(db=db, limit=self.recent_log_limit, user_id=None)

        if not logs:
            return []

        # Prefer real Faiss inference if artefacts are loaded.
        if self._artefacts is not None and user_id is not None:
            recent_views = [l.business_id for l in logs if l.action == "view" and l.business_id]

            # BE user_id is int; artefacts may use Yelp string IDs.
            # We currently store user_id as stringified int in DB, so try that.
            recs = recommend_from_recent_views(
                artefacts=self._artefacts,
                user_id=str(user_id),
                recent_business_ids=recent_views,
                topk=topk,
                config=self._cfg,
            )

            # Keep response contract stable with previous fields.
            now = datetime.utcnow().isoformat()
            for r in recs:
                r.setdefault("generated_at", now)
            return recs

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
