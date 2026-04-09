from fastapi.testclient import TestClient

from app.db.models import Business, BusinessPhoto, Photo
from app.db.session import SessionLocal
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


def _seed_business_with_photos():
    """Tạo 1 business với 2 photos để test cover_photo."""

    db = SessionLocal()
    try:
        biz = Business(business_id="biz_cover_1", name="Biz Cover Test")
        db.add(biz)

        p1 = Photo(photo_id="p_first", path="https://example.com/first.jpg")
        p2 = Photo(photo_id="p_second", path="https://example.com/second.jpg")
        db.add_all([p1, p2])
        db.flush()

        db.add_all(
            [
                BusinessPhoto(business_id=biz.business_id, photo_id=p1.photo_id),
                BusinessPhoto(business_id=biz.business_id, photo_id=p2.photo_id),
            ]
        )
        db.commit()
    finally:
        db.close()


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
    # Khi chưa có ảnh gắn với business_detail_1, cover_photo phải là null
    assert "cover_photo" in body
    assert body["cover_photo"] is None


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
    # cover_photo luôn có key, nhưng có thể null nếu chưa gắn ảnh
    for it in items:
        assert "cover_photo" in it


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


def test_business_detail_includes_cover_photo_from_first_image():
    client = TestClient(app)

    _seed_business_with_photos()

    res = client.get("/businesses/biz_cover_1")
    assert res.status_code == 200, res.text
    body = res.json()

    assert body["business_id"] == "biz_cover_1"
    assert body["cover_photo"] == "https://example.com/first.jpg"


def test_business_list_includes_cover_photo_when_available():
    client = TestClient(app)

    _seed_business_with_photos()

    res = client.get("/businesses/?limit=1000")
    assert res.status_code == 200, res.text
    items = res.json()

    match = next((b for b in items if b["business_id"] == "biz_cover_1"), None)
    assert match is not None
    assert match["cover_photo"] == "https://example.com/first.jpg"
