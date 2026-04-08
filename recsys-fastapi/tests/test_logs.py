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
        "user_id": 1,
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
            json={"user_id": 1, "action": "view", "business_id": "biz_1"},
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