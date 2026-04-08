from sqlalchemy.orm import Session
from app.db.models import Log
from app.db.crud import create_log as db_create_log, get_recent_logs as db_get_recent_logs

def log_action(db: Session, user_id: int, action: str):
    log_entry = Log(user_id=user_id, action=action)
    db_create_log(db, log_entry)

def retrieve_recent_logs(db: Session):
    return db_get_recent_logs(db)