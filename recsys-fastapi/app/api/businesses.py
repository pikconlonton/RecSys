from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import crud
from app.db.session import get_db
from app.schemas.businesses import Business, BusinessUpsert
from app.schemas.photos import Photo


router = APIRouter(prefix="/businesses", tags=["businesses"])


@router.post("/upsert")
def upsert_businesses(payload: list[BusinessUpsert], db: Session = Depends(get_db)):
    processed = crud.upsert_businesses(db, [b.model_dump() for b in payload])
    return {"processed": processed}


@router.get("/", response_model=list[Business])
def list_businesses(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """Return a list of businesses (simple pagination via skip/limit)."""

    items = crud.list_businesses(db=db, skip=skip, limit=limit)
    return items


@router.get("/{business_id}", response_model=Business)
def get_business_detail(business_id: str, db: Session = Depends(get_db)):
    """Return details for a single business.

    Mirrors the style of a typical "get user detail" endpoint.
    """

    biz = crud.get_business(db=db, business_id=business_id)
    if biz is None:
        raise HTTPException(status_code=404, detail="Business not found")
    return biz


@router.get("/{business_id}/photos", response_model=list[Photo])
def list_business_photos(business_id: str, db: Session = Depends(get_db)):
    """Lấy danh sách ảnh cho 1 business.

    Nếu business chưa gắn ảnh nào thì trả về list rỗng.
    """

    photos = crud.get_photos_for_business(db=db, business_id=business_id)
    return photos
