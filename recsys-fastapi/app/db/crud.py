from sqlalchemy.orm import Session

from app.db.models import (
    Business,
    BusinessPhoto,
    Log,
    Photo,
    SocialFriend,
    SocialInteraction,
    User as UserModel,
)
from app.schemas.logs import LogCreate
from app.schemas.users import UserCreate


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


def create_user(db: Session, payload: UserCreate) -> UserModel:
    """Create a new user profile.

    If you need idempotent semantics on user_id, handle it at API level
    (e.g. check get_user before create).
    """

    data = payload.model_dump() if hasattr(payload, "model_dump") else payload.dict()
    # Keep user_id as string consistently across the system.
    data["user_id"] = str(data["user_id"])

    obj = UserModel(**data)
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def get_recent_logs(db: Session, limit: int = 10, user_id: str | None = None):
    q = db.query(Log)
    if user_id is not None:
        q = q.filter(Log.user_id == user_id)
    return q.order_by(Log.timestamp.desc()).limit(limit).all()


def get_user(db: Session, user_id: str) -> UserModel | None:
    if not user_id:
        return None
    return db.get(UserModel, str(user_id))


def list_users(db: Session, skip: int = 0, limit: int = 100) -> list[UserModel]:
    q = db.query(UserModel).order_by(UserModel.user_id)
    if skip:
        q = q.offset(skip)
    if limit:
        q = q.limit(limit)
    return q.all()


def update_user(db: Session, user_id: str, payload) -> UserModel | None:
    """Update an existing user.

    Expects a Pydantic model or dict-like object with `dict`/`model_dump`.
    Only provided fields are updated.
    """

    obj = get_user(db, user_id)
    if obj is None:
        return None

    if hasattr(payload, "model_dump"):
        data = payload.model_dump(exclude_unset=True)
    else:
        data = (
            payload.dict(exclude_unset=True)
            if hasattr(payload, "dict")
            else dict(payload)
        )

    # Never change primary key via update payload.
    data.pop("user_id", None)

    for field, value in data.items():
        setattr(obj, field, value)

    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def delete_user(db: Session, user_id: str) -> bool:
    obj = get_user(db, user_id)
    if obj is None:
        return False

    db.delete(obj)
    db.commit()
    return True


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

    obj = db.get(Business, business_id)
    if obj is None:
        return None

    # Attach cover_photo (ảnh đầu tiên nếu có) để API trả ra kèm theo business.
    photos = get_photos_for_business(db, business_id=str(obj.business_id))
    if photos:
        # dùng path của ảnh đầu tiên làm cover
        setattr(obj, "cover_photo", photos[0].path)
    else:
        setattr(obj, "cover_photo", None)

    return obj


def list_businesses(db: Session, skip: int = 0, limit: int = 100) -> list[Business]:
    """Return a slice of businesses for simple listing/pagination."""

    q = db.query(Business).order_by(Business.business_id)
    if skip:
        q = q.offset(skip)
    if limit:
        q = q.limit(limit)

    items = q.all()

    # Gắn cover_photo cho từng business (ảnh đầu tiên nếu có).
    for obj in items:
        photos = get_photos_for_business(db, business_id=str(obj.business_id))
        if photos:
            setattr(obj, "cover_photo", photos[0].path)
        else:
            setattr(obj, "cover_photo", None)

    return items


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


def create_or_update_photo(db: Session, photo_id: str, path: str) -> Photo:
    """Create or update a photo record by photo_id.

    Đảm bảo path luôn được cập nhật nếu FE gửi lại cùng photo_id.
    """

    if not photo_id:
        raise ValueError("photo_id is required")

    key = str(photo_id)
    obj = db.get(Photo, key)
    if obj is None:
        obj = Photo(photo_id=key, path=path)
        db.add(obj)
    else:
        # Use setattr để tránh cảnh báo type checker với SQLAlchemy Column type.
        setattr(obj, "path", path)

    db.commit()
    db.refresh(obj)
    return obj


def link_photo_to_business(
    db: Session, business_id: str, photo_id: str
) -> BusinessPhoto:
    """Ensure there is a mapping giữa business và photo.

    Nếu mapping đã tồn tại thì trả về bản ghi cũ, không tạo bản ghi trùng.
    """

    if not business_id or not photo_id:
        raise ValueError("business_id and photo_id are required")

    biz_key = str(business_id)
    photo_key = str(photo_id)

    link = (
        db.query(BusinessPhoto)
        .filter(
            BusinessPhoto.business_id == biz_key,
            BusinessPhoto.photo_id == photo_key,
        )
        .first()
    )

    if link is None:
        link = BusinessPhoto(business_id=biz_key, photo_id=photo_key)
        db.add(link)
        db.commit()
        db.refresh(link)

    return link


def get_photos_for_business(db: Session, business_id: str) -> list[Photo]:
    """Return all photos gắn với 1 business.

    Nếu business chưa có ảnh thì trả về list rỗng.
    """

    if not business_id:
        return []

    biz_key = str(business_id)

    rows = (
        db.query(Photo)
        .join(BusinessPhoto, Photo.photo_id == BusinessPhoto.photo_id)
        .filter(BusinessPhoto.business_id == biz_key)
        .order_by(BusinessPhoto.id.asc())
        .all()
    )

    return rows
