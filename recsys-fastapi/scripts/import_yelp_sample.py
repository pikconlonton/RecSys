#!/usr/bin/env python3
"""Import a small head sample from Yelp JSON into the local SQLite DB.

This script reads the first N records from:
- ../datasets/yelp_academic_dataset_business.json
- ../datasets/yelp_academic_dataset_user.json

and inserts minimal fields into the existing `Business` and `User` tables.

Usage (from recsys-fastapi/ directory):

    # Use the same DATABASE_URL as your API server
    # e.g. fish shell:
    #   set -x DATABASE_URL sqlite:///./test.db
    # then:
    #   python scripts/import_yelp_sample.py --limit 100

The script is idempotent with respect to primary keys: if a row already
exists, it will be updated (for Business) or left as-is (for User).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, Any

from app.db.session import SessionLocal
from app.db.models import Business, User


ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = ROOT / "datasets"
BUSINESS_PATH = DATASETS_DIR / "yelp_academic_dataset_business.json"
USER_PATH = DATASETS_DIR / "yelp_academic_dataset_user.json"


def _iter_json_lines(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a Yelp JSON-lines file.

    Yelp academic dataset is typically JSON per line (not a big array).
    """

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed lines
                continue


def import_businesses(limit: int) -> int:
    db = SessionLocal()
    imported = 0
    try:
        for obj in _iter_json_lines(BUSINESS_PATH):
            bid = obj.get("business_id")
            if not bid:
                continue

            # Map Yelp fields -> our Business model fields
            name = obj.get("name")
            stars = obj.get("stars")
            review_count = obj.get("review_count")
            address = obj.get("address")
            lat = obj.get("latitude")
            lng = obj.get("longitude")

            # categories in Yelp is an array or comma-separated string; we store as comma string
            cats = obj.get("categories")
            if isinstance(cats, list):
                categories = ", ".join(cats)
            else:
                categories = cats

            existing = db.get(Business, bid)
            if existing is None:
                existing = Business(business_id=bid)
                db.add(existing)

            # Assign via setattr to keep type-checkers happy with SQLAlchemy models.
            setattr(existing, "name", name)
            setattr(existing, "stars", float(stars) if stars is not None else None)
            setattr(
                existing,
                "review_count",
                int(review_count) if review_count is not None else None,
            )
            setattr(existing, "address", address)
            setattr(existing, "categories", categories)
            setattr(existing, "lat", float(lat) if lat is not None else None)
            setattr(existing, "lng", float(lng) if lng is not None else None)

            imported += 1
            if imported >= limit:
                break

        db.commit()
        return imported
    finally:
        db.close()


def import_users(limit: int) -> int:
    db = SessionLocal()
    imported = 0
    try:
        for obj in _iter_json_lines(USER_PATH):
            uid = obj.get("user_id")
            if not uid:
                continue

            name = obj.get("name")

            existing = db.get(User, uid)
            if existing is None:
                existing = User(user_id=uid)
                db.add(existing)

            # Only set/overwrite basic profile fields; ignore stats for now.
            setattr(existing, "name", name)

            imported += 1
            if imported >= limit:
                break

        db.commit()
        return imported
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import Yelp head sample into local DB"
    )
    parser.add_argument(
        "--limit", type=int, default=100, help="Max records per file to import"
    )
    args = parser.parse_args()

    if not BUSINESS_PATH.exists():
        raise SystemExit(f"Missing business dataset: {BUSINESS_PATH}")
    if not USER_PATH.exists():
        raise SystemExit(f"Missing user dataset: {USER_PATH}")

    n_biz = import_businesses(args.limit)
    n_users = import_users(args.limit)

    print(f"Imported businesses={n_biz}, users={n_users}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
