from sqlalchemy.orm import Session

from app.db.models import Log
from app.schemas.logs import LogCreate

def create_log(db: Session, log: LogCreate):
    # pydantic v2: prefer model_dump; v1: fallback to dict
    data = log.model_dump() if hasattr(log, "model_dump") else log.dict()
    db_log = Log(**data)
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log

def get_recent_logs(db: Session, limit: int = 10, user_id: int | None = None):
    q = db.query(Log)
    if user_id is not None:
        q = q.filter(Log.user_id == str(user_id))
    return q.order_by(Log.timestamp.desc()).limit(limit).all()