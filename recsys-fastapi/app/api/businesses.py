from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db import crud
from app.db.session import get_db
from app.schemas.businesses import BusinessUpsert


router = APIRouter(prefix="/businesses", tags=["businesses"])


@router.post("/upsert")
def upsert_businesses(payload: list[BusinessUpsert], db: Session = Depends(get_db)):
    processed = crud.upsert_businesses(db, [b.model_dump() for b in payload])
    return {"processed": processed}
