from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db import crud
from app.db.session import get_db
from app.schemas.users import User, UserCreate, UserUpdate


router = APIRouter(prefix="/users", tags=["users"])


@router.post("/", response_model=User)
def create_user(payload: UserCreate, db: Session = Depends(get_db)):
    """Create a new user profile.

    If a user with the same id already exists, this will return a 400 at DB level
    (PK conflict) unless the database is empty. For idempotent upsert, build it
    on top of get_user + update_user.
    """

    # We let DB enforce uniqueness of user_id.
    return crud.create_user(db, payload)


@router.get("/", response_model=list[User])
def list_users(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """List users with simple pagination."""

    return crud.list_users(db, skip=skip, limit=limit)


@router.get("/{user_id}", response_model=User)
def get_user_detail(user_id: str, db: Session = Depends(get_db)):
    """Fetch user profile by id."""

    user = crud.get_user(db, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.put("/{user_id}", response_model=User)
def update_user(user_id: str, payload: UserUpdate, db: Session = Depends(get_db)):
    """Update an existing user profile.

    Only fields present in the payload will be updated.
    """

    user = crud.update_user(db, user_id, payload)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@router.delete("/{user_id}")
def delete_user(user_id: str, db: Session = Depends(get_db)):
    """Delete a user profile.

    Returns deleted=True when successful; 404 if user does not exist.
    """

    deleted = crud.delete_user(db, user_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="User not found")
    return {"deleted": True}
