from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String

from app.db.session import Base


class User(Base):
    """Basic user profile.

    This service mainly cares about a stable string user_id that is reused
    across logs, social_friends, social_interactions, and recommendations.
    Additional fields (name/email/avatar) are optional and can be managed by FE/backoffice.
    """

    __tablename__ = "users"

    # Stable external user identifier (string), used in all other tables.
    user_id = Column(String, primary_key=True, index=True)

    name = Column(String, nullable=True)
    email = Column(String, nullable=True)
    avatar_url = Column(String, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Log(Base):
    __tablename__ = "logs"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)
    business_id = Column(String, index=True)
    action = Column(String, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f"<Log(id={self.id}, user_id={self.user_id}, action={self.action}, timestamp={self.timestamp})>"


class Business(Base):
    __tablename__ = "businesses"

    # Yelp business_id is a stable string id
    business_id = Column(String, primary_key=True, index=True)

    name = Column(String, nullable=True)
    stars = Column(Float, nullable=True)
    review_count = Column(Integer, nullable=True)
    categories = Column(String, nullable=True)
    address = Column(String, nullable=True)
    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)

    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<Business(business_id={self.business_id}, name={self.name})>"


class Photo(Base):
    """Photo metadata stored in DB.

    Hiện tại backend chỉ cần id ảnh (string) và đường dẫn (path/url).
    Các trường khác (caption, width/height, v.v.) có thể bổ sung sau nếu cần.
    """

    __tablename__ = "photos"

    # Stable string id cho ảnh (có thể map sang id trong hệ thống khác).
    photo_id = Column(String, primary_key=True, index=True)

    # Đường dẫn ảnh (có thể là URL hoặc path tương đối).
    path = Column(String, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)


class BusinessPhoto(Base):
    """N-N mapping giữa business và photo.

    Mỗi record thể hiện 1 ảnh thuộc về 1 business.
    """

    __tablename__ = "business_photos"

    id = Column(Integer, primary_key=True, index=True)
    business_id = Column(String, index=True, nullable=False)
    photo_id = Column(String, index=True, nullable=False)


class SocialFriend(Base):
    """Directed friend edge from FE: user_id -> friend_id."""

    __tablename__ = "social_friends"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    friend_id = Column(String, index=True, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class SocialInteraction(Base):
    """Social interaction history from FE (any user, incl. friends): user -> business."""

    __tablename__ = "social_interactions"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True, nullable=False)
    business_id = Column(String, index=True, nullable=False)

    action = Column(String, index=True, nullable=False)
    weight = Column(Float, nullable=False, default=1.0)

    timestamp = Column(DateTime, default=datetime.utcnow)
