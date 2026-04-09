#!/usr/bin/env python3
"""Import a small head sample of Yelp photos into the local SQLite DB.

This script reads the first N records from:
- ../datasets/photos.json (Yelp photo metadata, JSON per line)
- And maps them to actual image files under ../datasets/photos/

For each photo record, it will:
- Insert/update a row into the `photos` table (photo_id, path)
- Insert a row into the `business_photos` mapping table (business_id, photo_id)

The image `path` stored in DB will be a relative filesystem path like:
- "../datasets/photos/<filename>.jpg"

Usage (from recsys-fastapi/ directory):

    # Use the same DATABASE_URL as your API server
    # e.g. fish shell:
    #   set -x DATABASE_URL sqlite:///./test.db
    # then:
    #   python scripts/import_yelp_photos.py --limit 100

The script is mostly idempotent:
- If a Photo with the same photo_id already exists, its `path` is updated.
- If a BusinessPhoto mapping already exists for (business_id, photo_id), it is not duplicated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterator, Any

from app.db.models import BusinessPhoto, Photo
from app.db.session import SessionLocal


ROOT = Path(__file__).resolve().parents[2]
DATASETS_DIR = ROOT / "datasets"
PHOTOS_JSON_PATH = DATASETS_DIR / "photos.json"
PHOTOS_DIR = DATASETS_DIR / "photos"


def _iter_json_lines(path: Path) -> Iterator[dict[str, Any]]:
    """Yield JSON objects from a Yelp JSON-lines file."""

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


def _list_image_files(limit: int | None = None) -> list[Path]:
    """Return a sorted list of image file paths from PHOTOS_DIR.

    We keep it simple: any file with a common image extension is accepted.
    """

    if not PHOTOS_DIR.exists():
        raise SystemExit(f"Missing photos directory: {PHOTOS_DIR}")

    exts = {".jpg", ".jpeg", ".png", ".webp"}
    files = [
        p for p in PHOTOS_DIR.iterdir() if p.is_file() and p.suffix.lower() in exts
    ]
    files.sort()

    if limit is not None:
        files = files[:limit]

    if not files:
        raise SystemExit(f"No image files found under: {PHOTOS_DIR}")

    return files


def import_photos(limit: int) -> int:
    """Import up to `limit` photos from photos.json into DB.

    We pair each JSON record with the next image file from PHOTOS_DIR
    in sorted order, so path will "tăng dần" theo thứ tự file.
    """

    if not PHOTOS_JSON_PATH.exists():
        raise SystemExit(f"Missing photos metadata: {PHOTOS_JSON_PATH}")

    image_files = _list_image_files(limit=limit)
    image_iter = iter(image_files)

    db = SessionLocal()
    imported = 0
    try:
        for obj in _iter_json_lines(PHOTOS_JSON_PATH):
            if imported >= limit:
                break

            try:
                image_path = next(image_iter)
            except StopIteration:
                # Ran out of image files
                break

            photo_id = obj.get("photo_id")
            business_id = obj.get("business_id")
            if not photo_id or not business_id:
                continue

            # Store path as relative path from recsys-fastapi/ to the datasets folder.
            rel_path = f"../datasets/photos/{image_path.name}"

            # Upsert Photo
            key = str(photo_id)
            photo = db.get(Photo, key)
            if photo is None:
                photo = Photo(photo_id=key, path=rel_path)
                db.add(photo)
            else:
                # Ensure path is up to date
                setattr(photo, "path", rel_path)

            # Upsert BusinessPhoto mapping (avoid duplicates)
            link = (
                db.query(BusinessPhoto)
                .filter(
                    BusinessPhoto.business_id == str(business_id),
                    BusinessPhoto.photo_id == key,
                )
                .first()
            )
            if link is None:
                db.add(
                    BusinessPhoto(
                        business_id=str(business_id),
                        photo_id=key,
                    )
                )

            imported += 1

        db.commit()
        return imported
    finally:
        db.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import Yelp photo sample into local DB",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max number of photos to import",
    )
    args = parser.parse_args()

    n_photos = import_photos(args.limit)
    print(f"Imported photos={n_photos}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
