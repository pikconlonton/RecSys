#!/usr/bin/env python3
"""Seed local SQLite DB with sample data so you can demo the API quickly.

Seeds:
- users (basic profiles)
- logs (views)
- businesses (minimal metadata)
- social friends
- social interactions (like/view)

By default we seed into the project's default sqlite file `./app.db`.
Override by setting DATABASE_URL, e.g:
  DATABASE_URL=sqlite:///./test.db python scripts/seed_db.py

This script is safe to re-run: it clears existing rows for these tables first.
"""

from __future__ import annotations

import os
import random
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Ensure the project root (recsys-fastapi) is on sys.path so `app.*` imports work
# when running this script directly (python scripts/seed_db.py).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _repo_root() -> Path:
    # scripts/seed_db.py -> recsys-fastapi/scripts -> recsys-fastapi -> RecSys
    return Path(__file__).resolve().parents[2]


def _iter_edges_user_business_sample(max_lines: int = 200):
    edges_path = _repo_root() / "outputs" / "graph" / "edges_user_business.txt"
    if not edges_path.exists():
        raise FileNotFoundError(f"Missing edges file: {edges_path}")

    n = 0
    with edges_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) != 2:
                continue
            user_id, business_id = parts[0].strip(), parts[1].strip()
            if not user_id or not business_id:
                continue
            yield user_id, business_id
            n += 1
            if n >= max_lines:
                break


def main() -> int:
    random.seed(0)

    # Ensure the project root (recsys-fastapi) is on sys.path so `app.*` imports work
    # when running this script directly (python scripts/seed_db.py).
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Local imports so they see the adjusted sys.path
    from app.db.models import Business, Log, SocialFriend, SocialInteraction, User
    from app.db.session import Base, SessionLocal, engine

    # Ensure schema exists (mirrors app lifespan behavior)
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

    pairs = list(_iter_edges_user_business_sample(max_lines=300))
    if len(pairs) < 30:
        raise RuntimeError(
            "Not enough edges rows to seed. Increase max_lines or check file."
        )

    users = list(dict.fromkeys([u for u, _ in pairs]))[:20]
    businesses = list(dict.fromkeys([b for _, b in pairs]))[:60]

    now = datetime.utcnow()

    db = SessionLocal()
    try:
        # Hard reset tables (already dropped, but keep explicit intent)
        db.query(Log).delete()
        db.query(Business).delete()
        db.query(SocialFriend).delete()
        db.query(SocialInteraction).delete()
        db.query(User).delete()
        db.commit()

        # Seed user profiles (basic fake data for each sampled user_id)
        for i, uid in enumerate(users, start=1):
            db.add(
                User(
                    user_id=uid,
                    name=f"User {i}",
                    email=f"{uid}@example.com",
                    avatar_url="https://example.com/avatar.png",
                )
            )
        db.commit()

        # Seed businesses metadata
        for i, bid in enumerate(businesses, start=1):
            db.add(
                Business(
                    business_id=bid,
                    name=f"Business {i}",
                    stars=float(random.choice([3.5, 4.0, 4.5, 5.0])),
                    review_count=int(random.randint(10, 500)),
                    categories=random.choice(
                        [
                            "Cafe, Coffee & Tea",
                            "Pizza, Italian",
                            "Sushi Bars, Japanese",
                            "Bars, Nightlife",
                            "Vietnamese, Noodles",
                        ]
                    ),
                    address=f"{random.randint(1, 999)} Market St, Philadelphia",
                    lat=39.9526 + random.uniform(-0.02, 0.02),
                    lng=-75.1652 + random.uniform(-0.02, 0.02),
                )
            )
        db.commit()

        # Seed logs: ~120 view logs spread over users/businesses
        for i in range(120):
            u = random.choice(users)
            b = random.choice(businesses)
            ts = now - timedelta(seconds=(120 - i))
            db.add(Log(user_id=u, business_id=b, action="view", timestamp=ts))
        db.commit()

        # Seed social: pick a "main" user and 5 friends
        main_user = users[0]
        friend_ids = users[1:6]
        for fid in friend_ids:
            db.add(SocialFriend(user_id=main_user, friend_id=fid))
        db.commit()

        # Seed interactions from friends to a few businesses
        boost_biz = businesses[0]
        other_biz = businesses[1]
        for fid in friend_ids:
            db.add(
                SocialInteraction(
                    user_id=fid,
                    business_id=boost_biz,
                    action="like",
                    weight=2.0,
                    timestamp=now - timedelta(minutes=random.randint(1, 120)),
                )
            )
        db.add(
            SocialInteraction(
                user_id=friend_ids[0],
                business_id=other_biz,
                action="view",
                weight=1.0,
                timestamp=now - timedelta(minutes=10),
            )
        )
        db.commit()

        print("Seed complete")
        print(f"DATABASE_URL={os.getenv('DATABASE_URL', 'sqlite:///./test.db')}")
        print(f"Seeded users={len(users)} businesses={len(businesses)}")
        print(f"main_user_for_social={main_user}")
        print(f"boost_business_id={boost_biz}")

    finally:
        db.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
