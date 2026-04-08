from pydantic import BaseModel
from datetime import datetime

class LogCreate(BaseModel):
    user_id: int
    business_id: str
    action: str
    timestamp: datetime = datetime.now()

class Log(LogCreate):
    id: int

    class Config:
        from_attributes = True