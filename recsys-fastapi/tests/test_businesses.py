from fastapi.testclient import TestClient

from app.main import app


def _sample_business(business_id: str = "biz_test_1") -> dict:
    return {
        "business_id": business_id,
        "name": "Test Business",
        "stars": 4.5,
        "review_count": 10,
        "categories": "Test, Category",
        "address": "123 Test St",
        "lat": 10.0,
        "lng": 20.0,
    }


def test_upsert_and_get_business_detail():
    client = TestClient(app)

    payload = [_sample_business("biz_detail_1")]
    upsert_res = client.post("/businesses/upsert", json=payload)
    assert upsert_res.status_code == 200, upsert_res.text
    assert upsert_res.json()["processed"] == 1

    get_res = client.get("/businesses/biz_detail_1")
    assert get_res.status_code == 200, get_res.text
    body = get_res.json()

    assert body["business_id"] == "biz_detail_1"
    assert body["name"] == payload[0]["name"]
    assert body["stars"] == payload[0]["stars"]
    assert body["review_count"] == payload[0]["review_count"]


def test_get_business_not_found():
    client = TestClient(app)

    res = client.get("/businesses/non_existing_id")
    assert res.status_code == 404
    assert res.json() == {"detail": "Business not found"}


def test_list_businesses():
    client = TestClient(app)

    # Upsert a few businesses
    payload = [
        _sample_business("biz_list_1"),
        _sample_business("biz_list_2"),
        _sample_business("biz_list_3"),
    ]
    upsert_res = client.post("/businesses/upsert", json=payload)
    assert upsert_res.status_code == 200, upsert_res.text

    list_res = client.get("/businesses/?limit=1000")
    assert list_res.status_code == 200, list_res.text
    items = list_res.json()

    assert isinstance(items, list)
    # At least the three we just upserted should exist in the result set
    ids = {it["business_id"] for it in items}
    assert {"biz_list_1", "biz_list_2", "biz_list_3"}.issubset(ids)


def test_list_businesses_pagination():
    client = TestClient(app)

    # Ensure there are at least 3 businesses
    payload = [
        _sample_business("biz_page_1"),
        _sample_business("biz_page_2"),
        _sample_business("biz_page_3"),
    ]
    upsert_res = client.post("/businesses/upsert", json=payload)
    assert upsert_res.status_code == 200, upsert_res.text

    # First page
    page1 = client.get("/businesses/?skip=0&limit=2")
    assert page1.status_code == 200, page1.text
    data1 = page1.json()
    assert len(data1) <= 2

    # Second page
    page2 = client.get("/businesses/?skip=2&limit=2")
    assert page2.status_code == 200, page2.text
    data2 = page2.json()

    # Basic sanity: pages should not be identical when enough data exists
    if len(data1) == 2 and len(data2) >= 1:
        assert data1 != data2
