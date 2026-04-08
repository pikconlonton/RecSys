from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String

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