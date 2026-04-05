"""
Tạo embedding 384-dim cho mỗi user bằng SentenceTransformer.

╔════════════════════════════════════════════════════════════════════╗
║  PIPELINE:                                                         ║
║  1. Load Philadelphia user IDs (list từ bước build edge)          ║
║  2. Load user profile metadata (name, avg_stars, elite, fans)    ║
║  3. Thu thập reviews do user viết (tối đa 10 reviews)            ║
║  4. Build text = profile + reviews                                ║
║  5. Encode bằng all-MiniLM-L6-v2 → vector 384-dim               ║
║  6. Lưu vào user_embeddings.pkl                                   ║
║                                                                    ║
║  TẠI SAO EMBED USER TỪ TEXT (không chỉ mean(biz_h)):            ║
║  - Reviews của user chứa thông tin sở thích cụ thể             ║
║  - Profile (elite, fans) cho thấy mức độ active                 ║
║  - Cùng model (MiniLM) với business → cùng embedding space      ║
║                                                                    ║
║  INPUT:  philadelphia_user_ids.txt                                 ║
║          yelp_academic_dataset_user.json                           ║
║          yelp_academic_dataset_review.json                         ║
║  OUTPUT: user_embeddings.pkl                                       ║
║          Format: {user_id_str: np.ndarray(384,)}                  ║
╚════════════════════════════════════════════════════════════════════╝
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════
# ── Paths ────────────────────────────────────────────────────────────────────
USER_IDS_PATH  = Path("ouputs/philadelphia_user_ids.txt")     # Danh sách user_id Philadelphia
USER_JSON_PATH = Path(r"Yelp-JSON\Yelp JSON\yelp_dataset\yelp_academic_dataset_user.json")
REVIEW_PATH    = Path(r"Yelp-JSON\Yelp JSON\yelp_dataset\yelp_academic_dataset_review.json")
OUT_PATH       = Path("outputs/user_embeddings.pkl")

MAX_REVIEWS = 10   # Số review tối đa cho mỗi user (giới hạn text length)

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1: Load danh sách user_id Philadelphia
# File này được tạo từ bước build edge (unique users có interaction)
# ══════════════════════════════════════════════════════════════════════════════
print("Loading Philadelphia user IDs ...")
philly_user_ids = set()
with USER_IDS_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        v = line.strip()
        if v:
            philly_user_ids.add(v)
print(f"  Total users: {len(philly_user_ids):,}")

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2: Load user profile metadata từ Yelp user JSON
# Chỉ giữ user thuộc Philadelphia (filter bằng philly_user_ids set)
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading user profiles ...")
user_meta = {}
with USER_JSON_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        uid = obj.get("user_id")
        if uid in philly_user_ids:
            user_meta[uid] = {
                "name":          obj.get("name", ""),
                "review_count":  obj.get("review_count", 0),
                "average_stars": obj.get("average_stars", 0),
                "elite":         obj.get("elite", ""),
                "fans":          obj.get("fans", 0),
            }
print(f"  Profiles loaded: {len(user_meta):,}")

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 3: Thu thập reviews do mỗi user viết
# Tối đa 10 reviews/user (giới hạn text length cho SentenceTransformer)
# ══════════════════════════════════════════════════════════════════════════════
print("\nCollecting user reviews ...")
user_reviews = defaultdict(list)

with REVIEW_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        uid = obj.get("user_id")
        if uid in philly_user_ids and len(user_reviews[uid]) < MAX_REVIEWS:
            user_reviews[uid].append(obj.get("text", ""))

total_with_reviews = sum(1 for uid in philly_user_ids if user_reviews[uid])
print(f"  Users with at least 1 review: {total_with_reviews:,}")


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 4: Build text representation cho mỗi user
# Kết hợp: profile metadata + reviews text
# ══════════════════════════════════════════════════════════════════════════════
def build_user_text(meta: dict, reviews: list) -> str:
    """
    Tạo text representation cho 1 user:
    - Name: tên người dùng
    - Average rating: mức độ khó tính (rating trung bình)
    - Fans: mức độ ảnh hưởng
    - Elite years: có phải power user không
    - Reviews: nội dung review chứa sở thích cụ thể

    Text này sẽ được SentenceTransformer encode thành vector 384-dim.
    Cùng model với business → cùng embedding space → có thể so sánh.
    """
    name          = meta.get("name", "")
    review_count  = meta.get("review_count", 0)
    avg_stars     = meta.get("average_stars", 0)
    elite         = meta.get("elite", "") or ""
    fans          = meta.get("fans", 0)
    review_text   = " ".join(reviews)

    text = (
        f"Name: {name}\n"
        f"Average rating: {avg_stars} stars over {review_count} reviews\n"
        f"Fans: {fans}\n"
        f"Elite years: {elite}\n"
        f"Reviews: {review_text}"
    )
    return text


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 5: Encode text → 384-dim vector
# Dùng GPU (cuda:0) để tăng tốc, ~211K users
# User không có review sẽ bị skip (thiếu thông tin để embed)
# ══════════════════════════════════════════════════════════════════════════════
print("\nLoading SentenceTransformer model ...")
model = SentenceTransformer("all-MiniLM-L6-v2", device="cuda:0")

user_embeddings = {}
skipped = 0

target_ids = list(philly_user_ids)

print(f"\nEmbedding {len(target_ids):,} users ...")
for uid in tqdm(target_ids):
    reviews = user_reviews.get(uid, [])
    if not reviews:
        skipped += 1
        continue

    meta = user_meta.get(uid, {})
    text = build_user_text(meta, reviews)

    vec = model.encode(text, show_progress_bar=False)
    user_embeddings[uid] = vec

print(f"\nEmbedded  : {len(user_embeddings):,}")
print(f"Skipped (no reviews): {skipped:,}")

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 6: Lưu định dạng pickle
# Format: {user_id_str: np.ndarray(384,)}
# ══════════════════════════════════════════════════════════════════════════════
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with OUT_PATH.open("wb") as f:
    pickle.dump(user_embeddings, f)

size_mb = OUT_PATH.stat().st_size / (1024 ** 2)
print(f"\nSaved → {OUT_PATH}  ({size_mb:.1f} MB)")
