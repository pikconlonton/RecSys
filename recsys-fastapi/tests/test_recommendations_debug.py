from fastapi.testclient import TestClient

from app.main import app


def test_recommendations_debug_flag_includes_debug_block():
    client = TestClient(app)
    r = client.get("/recommendations/u_debug_test?topk=5&debug=1")
    assert r.status_code == 200
    data = r.json()
    assert "debug" in data
    assert isinstance(data["debug"], dict)
    assert data["debug"].get("path") in {"faiss_user_only", "faiss_session", "heuristic_db_fill"}


def test_recommendations_default_does_not_include_debug_block():
    client = TestClient(app)
    r = client.get("/recommendations/u_debug_test?topk=5")
    assert r.status_code == 200
    data = r.json()
    assert "debug" not in data
