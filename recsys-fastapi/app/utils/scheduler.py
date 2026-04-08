from apscheduler.schedulers.background import BackgroundScheduler
from datetime import datetime
from app.db import crud
from app.db.session import get_db
from fastapi import Depends

scheduler = BackgroundScheduler()

def fetch_recent_logs(db=Depends(get_db)):
    recent_logs = crud.get_recent_logs(db)
    print(f"Fetched logs at {datetime.now()}: {recent_logs}")

def start_scheduler():
    scheduler.add_job(fetch_recent_logs, 'interval', seconds=2)
    scheduler.start()