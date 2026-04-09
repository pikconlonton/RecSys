from fastapi.testclient import TestClient

from app.main import app
from app.db.models import Business, BusinessPhoto, Photo
from app.db.session import SessionLocal


def _create_business_with_photos():
    db = SessionLocal()
    try:
        biz = Business(business_id="biz_photo_1", name="Biz With Photos")
        db.add(biz)

        p1 = Photo(photo_id="p1", path="https://example.com/p1.jpg")
        p2 = Photo(photo_id="p2", path="https://example.com/p2.jpg")
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


def test_get_business_photos_returns_linked_photos():
    client = TestClient(app)

    # Seed DB with a business and two photos
    _create_business_with_photos()

    res = client.get("/businesses/biz_photo_1/photos")
    assert res.status_code == 200, res.text

    items = res.json()
    assert isinstance(items, list)
    ids = {p["photo_id"] for p in items}
    assert {"p1", "p2"}.issubset(ids)


def test_get_business_photos_empty_when_no_photos():
    client = TestClient(app)

    # Create a business without photos
    db = SessionLocal()
    try:
        biz = Business(business_id="biz_no_photos", name="No Photos Biz")
        db.add(biz)
        db.commit()
    finally:
        db.close()

    res = client.get("/businesses/biz_no_photos/photos")
    assert res.status_code == 200, res.text
    items = res.json()
    assert items == []
