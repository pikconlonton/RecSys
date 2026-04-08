from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.db import crud
from app.schemas.logs import Log, LogCreate
from app.db.session import get_db

router = APIRouter()

@router.post("/logs/", response_model=Log)
def create_log(log: LogCreate, db: Session = Depends(get_db)):
    return crud.create_log(db=db, log=log)

@router.get("/logs/recent/", response_model=list[Log])
def get_recent_logs(db: Session = Depends(get_db)):
    logs = crud.get_recent_logs(db=db)
    if not logs:
        raise HTTPException(status_code=404, detail="No logs found")
    return logs