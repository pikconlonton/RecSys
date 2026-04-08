from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String

from app.db.session import Base

class Log(Base):
    __tablename__ = 'logs'

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