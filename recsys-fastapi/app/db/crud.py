from sqlalchemy.orm import Session

from app.db.models import Business, Log, SocialFriend, SocialInteraction
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
    # Cast key to str to satisfy type checkers (SQLAlchemy Column type vs runtime value).
    return {str(b.business_id): b for b in rows}


def get_business(db: Session, business_id: str) -> Business | None:
    """Fetch a single business by its id.

    Returns None if not found.
    """

    if not business_id:
        return None
    return db.get(Business, business_id)


def list_businesses(db: Session, skip: int = 0, limit: int = 100) -> list[Business]:
    """Return a slice of businesses for simple listing/pagination."""

    q = db.query(Business).order_by(Business.business_id)
    if skip:
        q = q.offset(skip)
    if limit:
        q = q.limit(limit)
    return q.all()


def replace_friends(db: Session, user_id: str, friends: list[str]) -> int:
    """Replace friend list for a user (idempotent upsert model)."""

    # Clear existing edges
    db.query(SocialFriend).filter(SocialFriend.user_id == user_id).delete()

    # Insert new edges
    unique = []
    seen: set[str] = set()
    for f in friends:
        if not f or f == user_id or f in seen:
            continue
        seen.add(f)
        unique.append(f)

    for f in unique:
        db.add(SocialFriend(user_id=user_id, friend_id=f))

    db.commit()
    return len(unique)


def get_friends(db: Session, user_id: str) -> list[str]:
    rows = (
        db.query(SocialFriend.friend_id).filter(SocialFriend.user_id == user_id).all()
    )
    return [r[0] for r in rows]


def create_social_interaction(db: Session, payload: dict) -> SocialInteraction:
    obj = SocialInteraction(**payload)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def get_social_scores_for_candidates(
    db: Session,
    friend_ids: list[str],
    candidate_business_ids: list[str],
) -> dict[str, float]:
    """Aggregate social weight per business over a set of friend_ids.

    Returns mapping business_id -> summed_weight.
    """

    if not friend_ids or not candidate_business_ids:
        return {}

    rows = (
        db.query(SocialInteraction.business_id, SocialInteraction.weight)
        .filter(SocialInteraction.user_id.in_(friend_ids))
        .filter(SocialInteraction.business_id.in_(candidate_business_ids))
        .all()
    )

    out: dict[str, float] = {}
    for biz_id, w in rows:
        out[biz_id] = out.get(biz_id, 0.0) + float(w or 0.0)
    return out
