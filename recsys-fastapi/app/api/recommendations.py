from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import crud
from app.db.models import SocialInteraction
from app.db.session import get_db
from app.services.recommender import recommender_service


router = APIRouter(prefix="/recommendations", tags=["recommendations"])


def _compute_friend_weights(
    user_id: str,
    friend_ids: list[str],
    *,
    temperature: float = 0.1,
) -> dict[str, float]:
    """Embedding-based weights (method B) when artefacts + mappings exist.

    Falls back to uniform weights if embeddings/mappings are missing.
    """

    if not friend_ids:
        return {}

    artefacts = getattr(recommender_service, "_artefacts", None)
    if artefacts is None:
        w = 1.0 / len(friend_ids)
        return {fid: w for fid in friend_ids}

    if user_id not in artefacts.user2idx:
        w = 1.0 / len(friend_ids)
        return {fid: w for fid in friend_ids}

    # Keep only friends that exist in mapping
    fids = [fid for fid in friend_ids if fid in artefacts.user2idx]
    if not fids:
        w = 1.0 / len(friend_ids)
        return {fid: w for fid in friend_ids}

    import torch
    import torch.nn.functional as F

    u = artefacts.user_h[artefacts.user2idx[user_id]]
    f_vecs = torch.stack([artefacts.user_h[artefacts.user2idx[fid]] for fid in fids], dim=0)
    sims = F.normalize(f_vecs, dim=1) @ F.normalize(u, dim=0)
    weights = F.softmax(sims / temperature, dim=0)
    return {fid: float(w) for fid, w in zip(fids, weights)}


@router.get("/{user_id}")
def get_recommendations(
    user_id: str,
    topk: int = 10,
    use_social: bool = False,
    gamma: float = 0.2,
    debug: int = 0,
    db: Session = Depends(get_db),
):
    """Return recommendations for FE.

    Current implementation uses the latest logs and returns a ranked list.
    Once logs include item_id/business_id, this endpoint can serve your real
    session-aware + Faiss recommendations.
    """

    items = recommender_service.recommend(db=db, user_id=user_id, topk=topk)

    debug_payload = None
    if debug:
        # Derive which path was used without changing service contracts.
        # NOTE: This is best-effort debug information for FE/dev only.
        artefacts = getattr(recommender_service, "_artefacts", None)
        has_artefacts = artefacts is not None
        has_user_embedding = bool(has_artefacts and user_id in getattr(artefacts, "user2idx", {}))

        # Recompute recent views count cheaply.
        logs = crud.get_recent_logs(db=db, limit=getattr(recommender_service, "recent_log_limit", 10), user_id=user_id)
        recent_views = [l.business_id for l in logs if l.action == "view" and l.business_id]

        if has_user_embedding:
            path = "faiss_session" if recent_views else "faiss_user_only"
        else:
            path = "heuristic_db_fill"

        debug_payload = {
            "path": path,
            "has_artefacts": has_artefacts,
            "has_user_embedding": has_user_embedding,
            "recent_log_limit": int(getattr(recommender_service, "recent_log_limit", 10)),
            "recent_views_count": int(len(recent_views)),
        }

    # Optional: social reranking using FE-provided friends graph + interactions.
    # We keep backward compatibility: if disabled or no social signal, items unchanged.
    if use_social and items:
        friend_ids = crud.get_friends(db=db, user_id=user_id)
        if friend_ids:
            # Candidate set: only the business_ids we already plan to return.
            candidate_bids: list[str] = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                bid = it.get("business_id")
                if isinstance(bid, str) and bid:
                    candidate_bids.append(bid)

            if candidate_bids:
                weights = _compute_friend_weights(user_id, friend_ids)
                # Only compute scores for friends we have weights for
                weighted_friend_ids = list(weights.keys())
                if weighted_friend_ids:
                    # Load per-friend interactions for the candidate set.
                    rows = (
                        db.query(
                            SocialInteraction.user_id,
                            SocialInteraction.business_id,
                            SocialInteraction.weight,
                        )
                        .filter(SocialInteraction.user_id.in_(weighted_friend_ids))
                        .filter(SocialInteraction.business_id.in_(list(dict.fromkeys(candidate_bids))))
                        .all()
                    )

                    social: dict[str, float] = {}
                    for fid, bid, w in rows:
                        social[bid] = social.get(bid, 0.0) + float(weights.get(fid, 0.0)) * float(w or 0.0)

                    # Re-rank items (keep stable fields), attach scoring breakdown
                    ranked = []
                    for it in items:
                        if not isinstance(it, dict):
                            ranked.append((0.0, it))
                            continue
                        bid = it.get("business_id")
                        emb = float(it.get("score") or 0.0)
                        soc = float(social.get(bid, 0.0)) if isinstance(bid, str) else 0.0
                        final = (1.0 - gamma) * emb + gamma * soc
                        new_it = {
                            **it,
                            "score": final,
                            "scoring": {
                                "emb_score": emb,
                                "social_score": soc,
                                "final_score": final,
                                "gamma": gamma,
                            },
                        }
                        ranked.append((final, new_it))

                    ranked.sort(key=lambda x: x[0], reverse=True)
                    # Fix ranks after rerank
                    reranked_items = []
                    for i, (_, it) in enumerate(ranked, start=1):
                        if isinstance(it, dict):
                            it["rank"] = i
                        reranked_items.append(it)
                    items = reranked_items
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

    resp = {
        "user_id": user_id,
        "topk": topk,
        "items": enriched_items,
    }

    if debug_payload is not None:
        resp["debug"] = debug_payload

    return resp
