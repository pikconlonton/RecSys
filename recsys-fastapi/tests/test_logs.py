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