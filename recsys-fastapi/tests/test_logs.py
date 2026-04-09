from fastapi.testclient import TestClient
from app.main import app
from app.db.crud import create_log, get_recent_logs
from app.schemas.logs import LogCreate
import pytest

from app.db.session import SessionLocal
from app.db.models import Log

@pytest.fixture
def log_data():
    return {
    "user_id": "u1",
        "action": "test_action",
    "business_id": "biz_1",
        "timestamp": "2023-10-01T12:00:00Z"
    }

def test_create_log(log_data):
    with TestClient(app) as client:
        response = client.post("/logs/", json=log_data)
    assert response.status_code == 200
    assert response.json()["user_id"] == log_data["user_id"]
    assert response.json()["action"] == log_data["action"]

def test_get_recent_logs():
    with TestClient(app) as client:
        # Arrange: create one log so recent endpoint returns data.
        response_create = client.post(
            "/logs/",
            json={"user_id": "u1", "action": "view", "business_id": "biz_1"},
        )
        assert response_create.status_code == 200

        response = client.get("/logs/recent/")
        assert response.status_code == 200
        assert isinstance(response.json(), list)
        assert len(response.json()) <= 10

def test_create_log_invalid_data():
    with TestClient(app) as client:
        response = client.post("/logs/", json={"invalid_field": "data"})
        assert response.status_code == 422

def test_get_recent_logs_empty():
    # Ensure no data leaked from previous tests.
    db = SessionLocal()
    try:
        db.query(Log).delete()
        db.commit()
    finally:
        db.close()

    with TestClient(app) as client:
        # With a fresh database, no logs exist.
        response = client.get("/logs/recent/")
    assert response.status_code == 404
    assert response.json() == {"detail": "No logs found"}


def test_recommendations_with_injected_artefacts():
    # This test avoids loading the big real outputs/*.
    # Instead, we inject a tiny artefact set and verify the endpoint returns
    # business_ids from Faiss search (excluding the session item).
    import numpy as np
    import torch
    import faiss

    from app.services.artefacts import RecSysArtefacts
    from app.services.recommender import recommender_service

    user_h = torch.tensor([[1.0, 0.0]], dtype=torch.float32)  # one user
    biz_h = torch.tensor(
        [
            [1.0, 0.0],  # b1 (same as user)
            [0.9, 0.1],  # b2
            [0.0, 1.0],  # b3 (orthogonal)
        ],
        dtype=torch.float32,
    )
    biz_np = biz_h.numpy().astype("float32")
    faiss.normalize_L2(biz_np)
    index = faiss.IndexFlatIP(biz_np.shape[1])
    index.add(biz_np)

    artefacts = RecSysArtefacts(
        user_h=user_h,
        biz_h=biz_h,
        index=index,
        user2idx={"1": 0},
    # In current API, Log.business_id is an app-level string; user_id is int.
    # For this test we keep business IDs simple and consistent.
    biz2idx={"1": 0, "2": 1, "3": 2},
    idx2biz={0: "1", 1: "2", 2: "3"},
    )
    recommender_service.set_artefacts(artefacts)

    with TestClient(app) as client:
        # If the app lifespan preloaded the real outputs/* artefacts, clear them
        # so our injected tiny artefacts are what's used.
        try:
            client.app.state.recommender_artefacts = None
        except Exception:
            pass

    # Create a session view log on business "1" so recs should not include it.
    r = client.post("/logs/", json={"user_id": "1", "action": "view", "business_id": "1"})
    assert r.status_code == 200

    rec = client.get("/recommendations/1?topk=2")
    assert rec.status_code == 200
    payload = rec.json()
    assert payload["user_id"] == "1"
    assert payload["topk"] == 2
    assert isinstance(payload["items"], list)

    # Ensure it returns business ids and excludes the session item.
    returned = [it["business_id"] for it in payload["items"]]
    assert "1" not in returned
    assert len(returned) >= 1
    # Should recommend a non-seen business.
    assert returned[0] in {"2", "3"}


def test_product_like_10_logs_then_recommendations():
    """Product-like flow: FE posts logs continuously, then calls recommendations.

    This test validates the end-to-end contract:
      - POST /logs/ accepts events
      - service stores them
      - GET /recommendations/{user_id} uses the *latest 10 logs* as session buffer
      - returns a stable response shape

    We intentionally force the heuristic path (no artefacts) to keep the test
    fast and self-contained.
    """

    user_id = "u999"
    with TestClient(app) as client:
        # Post 10 logs (views). This matches the product assumption "đủ 10 logs".
        logs = [
            {"user_id": user_id, "action": "view", "business_id": "b1"},
            {"user_id": user_id, "action": "view", "business_id": "b2"},
            {"user_id": user_id, "action": "view", "business_id": "b2"},
            {"user_id": user_id, "action": "view", "business_id": "b3"},
            {"user_id": user_id, "action": "view", "business_id": "b2"},
            {"user_id": user_id, "action": "view", "business_id": "b9"},
            {"user_id": user_id, "action": "view", "business_id": "b9"},
            {"user_id": user_id, "action": "view", "business_id": "b4"},
            {"user_id": user_id, "action": "view", "business_id": "b2"},
            {"user_id": user_id, "action": "view", "business_id": "b2"},
        ]

        for payload in logs:
            r = client.post("/logs/", json=payload)
            assert r.status_code == 200

        # Sanity: recent logs endpoint should now have data.
        recent = client.get("/logs/recent/")
        assert recent.status_code == 200
        assert isinstance(recent.json(), list)
        assert len(recent.json()) >= 1

        rec = client.get(f"/recommendations/{user_id}?topk=5")
        assert rec.status_code == 200
        body = rec.json()

        assert body["user_id"] == user_id
        assert body["topk"] == 5
        assert isinstance(body["items"], list)

        # In product, items can be empty if real-inference mapping doesn't contain this user.
        # If it's non-empty, validate stable item schema.
        if body["items"]:
            first = body["items"][0]
            assert set(first.keys()) >= {"rank", "type", "business_id", "score", "generated_at"}
    
def test_recommendations_enriched_with_business_metadata():
    user_id = "u_meta"

    with TestClient(app) as client:
        # Upsert business metadata
        upsert_res = client.post(
            "/businesses/upsert",
            json=[
                {
                    "business_id": "b_meta_1",
                    "name": "Biz Meta 1",
                    "stars": 4.2,
                    "review_count": 12,
                    "categories": "Cafe",
                    "address": "123 Test St",
                    "lat": 10.5,
                    "lng": 20.5,
                }
            ],
        )
        assert upsert_res.status_code == 200

        # Create a log so heuristic fallback will recommend this business_id
        log_res = client.post(
            "/logs/",
            json={"user_id": user_id, "business_id": "b_meta_1", "action": "view"},
        )
        assert log_res.status_code == 200

        rec_res = client.get(f"/recommendations/{user_id}?topk=5")
        assert rec_res.status_code == 200
        data = rec_res.json()
        assert "items" in data

        if data["items"]:
            first = data["items"][0]
            assert "metadata" in first
            # If the heuristic path returns the same business_id, metadata must be present.
            if first.get("business_id") == "b_meta_1":
                assert first["metadata"]["name"] == "Biz Meta 1"


def test_recommendations_one_log_still_returns_full_topk_via_db_fill():
    """If user has only 1 recent view log, heuristic would normally produce 1 unique
    business_id. The recommender should top-up candidates from `businesses` so FE
    still receives `topk` items (cold-start fill has score=0.0).
    """

    user_id = "u_one_log"
    with TestClient(app) as client:
        # Ensure we have enough businesses in DB to fill.
        payload = []
        for i in range(1, 8):
            payload.append(
                {
                    "business_id": f"b_fill_{i}",
                    "name": f"Fill Biz {i}",
                    "stars": 4.0,
                    "review_count": 10 + i,
                    "categories": "Test",
                    "address": "Test",
                    "lat": 0.0,
                    "lng": 0.0,
                }
            )

        r = client.post("/businesses/upsert", json=payload)
        assert r.status_code == 200

        # Only one log.
        r = client.post(
            "/logs/",
            json={"user_id": user_id, "action": "view", "business_id": "b_fill_1"},
        )
        assert r.status_code == 200

        topk = 5
        rec = client.get(f"/recommendations/{user_id}?topk={topk}")
        assert rec.status_code == 200
        data = rec.json()
        assert len(data["items"]) == topk

        # First item should be the viewed business with score >= 1.
        assert data["items"][0]["business_id"] == "b_fill_1"
        assert float(data["items"][0]["score"]) >= 1.0

        # Remaining are filled from DB and have score 0.0.
        for it in data["items"][1:]:
            assert it["business_id"].startswith("b_fill_")


def test_social_reranking_from_fe_data():
    """Social reranking should boost businesses that friends interacted with.

    We keep it self-contained:
    - inject tiny embeddings so friend weights can be computed (method B)
    - use heuristic recommendation so candidate set is deterministic
    - friend interaction boosts one of the candidates
    """

    import torch

    from app.services.artefacts import RecSysArtefacts
    from app.services.recommender import recommender_service

    # user u1 and friend u2
    user_h = torch.tensor(
        [
            [1.0, 0.0],  # u1
            [1.0, 0.0],  # u2 (very similar)
        ],
        dtype=torch.float32,
    )
    # biz embeddings not used in heuristic rec path, but artefacts require them
    biz_h = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ],
        dtype=torch.float32,
    )

    # Dummy faiss index (won't be used because user u1 isn't in mapping below for Faiss rec)
    import faiss
    import numpy as np

    biz_np = biz_h.numpy().astype("float32")
    faiss.normalize_L2(biz_np)
    index = faiss.IndexFlatIP(biz_np.shape[1])
    index.add(biz_np)

    artefacts = RecSysArtefacts(
        user_h=user_h,
        biz_h=biz_h,
        index=index,
        # IMPORTANT: omit "u1" to force heuristic recommendation path
        # (Faiss inference returns [] if user_id not found).
        # Friend embeddings still exist so weights can be computed.
        user2idx={"u2": 1},
        biz2idx={"b1": 0, "b2": 1},
        idx2biz={0: "b1", 1: "b2"},
    )
    recommender_service.set_artefacts(artefacts)

    with TestClient(app) as client:
        # User session logs so heuristic returns b1 (2 views) and b2 (1 view)
        client.post("/logs/", json={"user_id": "u1", "action": "view", "business_id": "b1"})
        client.post("/logs/", json={"user_id": "u1", "action": "view", "business_id": "b1"})
        client.post("/logs/", json={"user_id": "u1", "action": "view", "business_id": "b2"})

        # FE upserts friend list: u1 -> [u2]
        r = client.post(
            "/social/friends/upsert",
            json={"user_id": "u1", "friends": ["u2"]},
        )
        assert r.status_code == 200

        # FE posts friend interaction on b2 with a strong weight
        r = client.post(
            "/social/interactions",
            json={"user_id": "u2", "business_id": "b2", "action": "like", "weight": 10.0},
        )
        assert r.status_code == 200

        # Without social: b1 should rank first (view-count score 2 > 1)
        base = client.get("/recommendations/u1?topk=2")
        assert base.status_code == 200
        base_items = base.json()["items"]
        assert base_items[0]["business_id"] == "b1"

        # With social: b2 should be boosted and can outrank b1
        rr = client.get("/recommendations/u1?topk=2&use_social=true&gamma=0.2")
        assert rr.status_code == 200
        items = rr.json()["items"]
        assert items[0]["business_id"] == "b2"
        assert "scoring" in items[0]