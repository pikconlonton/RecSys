from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F

from app.services.artefacts import RecSysArtefacts


@dataclass(frozen=True)
class InferenceConfig:
    alpha: float = 0.4
    temperature: float = 0.1


def _unique_preserve_order(xs: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for x in xs:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def attention_weighted_mean(
    item_embeds: torch.Tensor,  # [K, D]
    query: torch.Tensor,  # [D]
    temperature: float = 0.1,
) -> torch.Tensor:
    scores = (item_embeds @ query) / temperature
    weights = F.softmax(scores, dim=0)
    return (weights.unsqueeze(1) * item_embeds).sum(0)


def combine(user_vec: torch.Tensor, session_vec: torch.Tensor, alpha: float) -> torch.Tensor:
    merged = alpha * user_vec + (1 - alpha) * session_vec
    return F.normalize(merged, dim=0)


def recommend_from_recent_views(
    artefacts: RecSysArtefacts,
    user_id: str,
    recent_business_ids: list[str],
    topk: int = 10,
    config: InferenceConfig = InferenceConfig(),
) -> list[dict]:
    """Real inference: user_h + session(recent view biz) -> Faiss topK.

    Returns a list of dicts with keys: rank, type, business_id, score.

    Error modes:
      - unknown user: return []
      - no valid recent items: fallback to user-only recs
    """

    if user_id not in artefacts.user2idx:
        return []

    u_idx = artefacts.user2idx[user_id]
    u_vec = artefacts.user_h[u_idx]

    # Map and dedupe recent business ids (embedding-row indices)
    recent_business_ids = _unique_preserve_order(recent_business_ids)
    recent_row_idxs = [artefacts.biz2idx[b] for b in recent_business_ids if b in artefacts.biz2idx]

    if recent_row_idxs:
        recent_embeds = artefacts.biz_h[recent_row_idxs]
        session_emb = attention_weighted_mean(recent_embeds, u_vec, temperature=config.temperature)
        q = combine(u_vec, session_emb, alpha=config.alpha)
    else:
        # Fallback: no session info, just use user embedding.
        q = F.normalize(u_vec, dim=0)

    query_np = q.detach().cpu().numpy().astype("float32")[None, :]

    # Faiss index is IP over L2-normalized vectors; normalize query defensively.
    import faiss  # local import for faster module load during tooling

    faiss.normalize_L2(query_np)

    # Fetch extra to filter already-seen session items.
    # IMPORTANT: Faiss returns indices into the index (0..ntotal-1). These are
    # not guaranteed to match `biz2idx` (embedding row ids) unless we explicitly
    # add vectors in that exact order.
    #
    # For correctness and to keep this runtime generic, we filter by
    # business_id, not by index integer.
    seen_biz_ids = set(recent_business_ids)
    # Cap fetch_k to index size to avoid requesting more than available.
    # If index is small (tests), fetch all so filtering seen items still leaves some.
    fetch_k = min(
        max(topk + len(seen_biz_ids) + 50, topk + len(seen_biz_ids) + 1),
        int(artefacts.index.ntotal),
    )
    distances, indices = artefacts.index.search(query_np, fetch_k)

    recs: list[dict] = []
    rank = 0
    for score, biz_idx in zip(distances[0], indices[0]):
        biz_idx_int = int(biz_idx)
        bid = artefacts.idx2biz.get(biz_idx_int)
        if not bid:
            continue
        if bid in seen_biz_ids:
            continue
        rank += 1
        recs.append(
            {
                "rank": rank,
                "type": "business",
                "business_id": bid,
                "score": float(score),
            }
        )
        if rank >= topk:
            break

    return recs
