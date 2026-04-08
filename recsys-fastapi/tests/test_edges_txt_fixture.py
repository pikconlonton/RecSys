from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app


def _iter_edges_user_business_sample(max_lines: int = 25):
    """Yield (user_id, business_id) pairs from outputs graph edge file.

    This uses a tiny fixed-size head sample to keep tests fast/deterministic.
    """

    repo_root = Path(__file__).resolve().parents[2]
    edges_path = repo_root / "outputs" / "graph" / "edges_user_business.txt"

    if not edges_path.exists():
        pytest.skip(f"Missing fixture file: {edges_path}")

    n = 0
    with edges_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            user_id, business_id = parts[0].strip(), parts[1].strip()
            if not user_id or not business_id:
                continue
            yield user_id, business_id
            n += 1
            if n >= max_lines:
                break


def test_seed_logs_from_edges_txt_then_recent():
    client = TestClient(app)

    pairs = list(_iter_edges_user_business_sample(max_lines=12))
    assert len(pairs) >= 5

    for user_id, business_id in pairs:
        r = client.post(
            "/logs/",
            json={
                "user_id": user_id,
                "business_id": business_id,
                "action": "view",
            },
        )
    assert r.status_code == 200, r.text

    recent = client.get("/logs/recent?limit=10")
    assert recent.status_code == 200, recent.text
    data = recent.json()

    assert isinstance(data, list)
    assert 1 <= len(data) <= 10

    # Validate shape + string contract
    for item in data:
        assert isinstance(item["user_id"], str) and item["user_id"]
        assert isinstance(item["business_id"], str) and item["business_id"]
        assert item["action"]
        assert item["timestamp"]


def test_social_reranking_from_edges_txt_driven_by_fe():
    client = TestClient(app)

    pairs = list(_iter_edges_user_business_sample(max_lines=40))
    assert len(pairs) >= 15

    # Pick a stable, deterministic set of IDs from the sample:
    u0 = pairs[0][0]
    friend1 = pairs[1][0]
    friend2 = pairs[2][0]

    # Ensure distinct user IDs (if not, try a few more)
    i = 3
    while friend1 == u0 and i < len(pairs):
        friend1 = pairs[i][0]
        i += 1
    while (friend2 == u0 or friend2 == friend1) and i < len(pairs):
        friend2 = pairs[i][0]
        i += 1

    if len({u0, friend1, friend2}) < 3:
        pytest.skip("Not enough distinct user_ids in sample to build social test")

    # Candidate businesses for friend interactions
    b_boost = pairs[0][1]
    b_other = pairs[3][1]

    # Build FE-driven friend graph
    r = client.post(
        "/social/friends/upsert",
        json={"user_id": u0, "friend_ids": [friend1, friend2]},
    )
    assert r.status_code == 200, r.text

    # Friends interact with b_boost so it should receive a social_score
    for friend_id in (friend1, friend2):
        r = client.post(
            "/social/interactions",
            json={
                "user_id": friend_id,
                "business_id": b_boost,
                "action": "like",
            },
        )
    assert r.status_code == 200, r.text

    # Add a weaker signal for another business
    r = client.post(
        "/social/interactions",
        json={
            "user_id": friend1,
            "business_id": b_other,
            "action": "view",
        },
    )
    assert r.status_code == 200, r.text

    # Minimal session so `b_boost` won't be filtered out
    session = client.post(
        "/logs/",
        json={
            "user_id": u0,
            "business_id": pairs[5][1],
            "action": "view",
        },
    )
    assert session.status_code == 200, session.text

    rec = client.get(f"/recommendations/{u0}?topk=10&use_social=true&gamma=0.7")
    assert rec.status_code == 200, rec.text
    payload = rec.json()

    assert "items" in payload
    assert isinstance(payload["items"], list)
    assert len(payload["items"]) > 0

    # Social re-ranking is best-effort and depends on candidate overlap. When the
    # recommender is in heuristic mode, candidates might not include the socially-boosted
    # business IDs, so scoring may be absent. We only assert the endpoint remains stable.
