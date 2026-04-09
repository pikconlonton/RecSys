from fastapi.testclient import TestClient

from app.main import app


def _sample_user(user_id: str = "u_test_1") -> dict:
    return {
        "user_id": user_id,
        "name": "Test User",
        "email": f"{user_id}@example.com",
        "avatar_url": "https://example.com/avatar.png",
    }


def test_create_and_get_user_detail():
    client = TestClient(app)

    payload = _sample_user("u_detail_1")
    create_res = client.post("/users/", json=payload)
    assert create_res.status_code == 200, create_res.text
    body = create_res.json()

    assert body["user_id"] == payload["user_id"]
    assert body["name"] == payload["name"]
    assert body["email"] == payload["email"]

    get_res = client.get(f"/users/{payload['user_id']}")
    assert get_res.status_code == 200, get_res.text
    detail = get_res.json()
    assert detail["user_id"] == payload["user_id"]
    assert detail["name"] == payload["name"]


def test_get_user_not_found():
    client = TestClient(app)

    res = client.get("/users/non_existing_user")
    assert res.status_code == 404
    assert res.json() == {"detail": "User not found"}


def test_list_users_contains_created():
    client = TestClient(app)

    # Create a few users
    ids = ["u_list_1", "u_list_2", "u_list_3"]
    for uid in ids:
        res = client.post("/users/", json=_sample_user(uid))
        assert res.status_code == 200, res.text

    list_res = client.get("/users/?limit=1000")
    assert list_res.status_code == 200, list_res.text
    items = list_res.json()

    assert isinstance(items, list)
    returned_ids = {u["user_id"] for u in items}
    assert set(ids).issubset(returned_ids)


def test_update_user():
    client = TestClient(app)

    uid = "u_update_1"
    create_res = client.post("/users/", json=_sample_user(uid))
    assert create_res.status_code == 200, create_res.text

    update_payload = {"name": "Updated Name", "email": "updated@example.com"}
    update_res = client.put(f"/users/{uid}", json=update_payload)
    assert update_res.status_code == 200, update_res.text
    updated = update_res.json()

    assert updated["user_id"] == uid
    assert updated["name"] == "Updated Name"
    assert updated["email"] == "updated@example.com"


def test_update_user_not_found():
    client = TestClient(app)

    res = client.put("/users/unknown", json={"name": "X"})
    assert res.status_code == 404
    assert res.json() == {"detail": "User not found"}


def test_delete_user():
    client = TestClient(app)

    uid = "u_delete_1"
    create_res = client.post("/users/", json=_sample_user(uid))
    assert create_res.status_code == 200, create_res.text

    del_res = client.delete(f"/users/{uid}")
    assert del_res.status_code == 200, del_res.text
    assert del_res.json() == {"deleted": True}

    # After deletion, fetching should 404
    get_res = client.get(f"/users/{uid}")
    assert get_res.status_code == 404


def test_delete_user_not_found():
    client = TestClient(app)

    res = client.delete("/users/does_not_exist")
    assert res.status_code == 404
    assert res.json() == {"detail": "User not found"}
