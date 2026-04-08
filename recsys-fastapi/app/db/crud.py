from sqlalchemy.orm import Session

from app.db.models import Business, Log
from app.schemas.logs import LogCreate

def create_log(db: Session, log: LogCreate):
    # pydantic v2: prefer model_dump; v1: fallback to dict
    data = log.model_dump() if hasattr(log, "model_dump") else log.dict()
    # user_id is a string in the API contract; DB column is String.
    # Keep consistent for filtering.
    if "user_id" in data and data["user_id"] is not None:
        data["user_id"] = str(data["user_id"])

    db_log = Log(**data)
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log

def get_recent_logs(db: Session, limit: int = 10, user_id: str | None = None):
    q = db.query(Log)
    if user_id is not None:
        q = q.filter(Log.user_id == user_id)
    return q.order_by(Log.timestamp.desc()).limit(limit).all()


def upsert_businesses(db: Session, businesses: list[dict]) -> int:
    """Insert/update business metadata.

    Note: kept simple & portable across SQLite/Postgres by using per-row upsert.
    """

    if not businesses:
        return 0

    for payload in businesses:
        business_id = payload.get("business_id")
        if not business_id:
            continue

        obj = db.get(Business, business_id)
        if obj is None:
            obj = Business(business_id=business_id)
            db.add(obj)

        for field in (
            "name",
            "stars",
            "review_count",
            "categories",
            "address",
            "lat",
            "lng",
        ):
            if field in payload:
                setattr(obj, field, payload.get(field))

    db.commit()
    return len(businesses)


def get_businesses_by_ids(db: Session, business_ids: list[str]) -> dict[str, Business]:
    if not business_ids:
        return {}

    rows = db.query(Business).filter(Business.business_id.in_(business_ids)).all()
    return {b.business_id: b for b in rows}