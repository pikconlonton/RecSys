"""
Inference: gợi ý business cho user dựa trên session-aware embedding + Faiss.

╔════════════════════════════════════════════════════════════════════╗
║  PIPELINE:                                                         ║
║  1. Load trained user_h, biz_h (128-dim, L2-norm)                 ║
║  2. Resolve user (by ID hoặc random)                              ║
║  3. Xây session embedding từ recent businesses                    ║
║     - Attention-weighted mean: user_vec attend lên recent items   ║
║  4. Combine: user_final = α·user_h + (1-α)·session_emb           ║
║  5. Faiss search top-K candidates                                 ║
║  6. Filter bỏ businesses đã ở trong session → show results        ║
║                                                                    ║
║  Lưu ý: File này KHÔNG có social score.                             ║
║  Dùng inference_social.py nếu muốn re-rank bằng friend influence.  ║
╚════════════════════════════════════════════════════════════════════╝

Usage:
  # Với session thật:
  python inference.py --user_id="USER_ID" --recent="BIZ1,BIZ2,BIZ3" --topk 20

  # Fake session (tự lấy từ lịch sử user):
  python inference.py --user_id="USER_ID" --fake_session --session_size 5 --topk 20

  # Random user + fake session:
  python inference.py --random_user --fake_session --topk 20
"""

import argparse
import json
import random
import faiss
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════════════
# Paths & Config
# ══════════════════════════════════════════════════════════════════════════════
OUT_DIR   = Path("D:/RecSys/outputs")
UB_PATH   = Path("D:\RecSys\outputs\graph\edges_user_business.txt")  # Lịch sử user-business
BIZ_JSON  = Path("D:/RecSys/Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_business.json")
ALPHA     = 0.4  # Trọng số user_h vs session — 0.4 nghĩa session chiếm 60%


def load_biz_meta(path: Path) -> dict:
    """Load metadata business từ Yelp JSON.
    Return: {business_id: {name, categories}} — dùng để hiển thị kết quả.
    """
    meta = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            meta[obj["business_id"]] = {
                "name":       obj.get("name", ""),
                "categories": obj.get("categories", ""),
            }
    return meta


def load_user_history(ub_path: Path, user2idx: dict, biz2idx: dict) -> dict:
    """Load lịch sử user-business từ edge file.
    Return: {user_idx: [biz_idx, ...]} — dùng cho fake_session và random_user.
    Chỉ giữ các edges mà cả user và biz đều có trong graph.
    """
    user_pos = defaultdict(list)
    with ub_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2 and parts[0] in user2idx and parts[1] in biz2idx:
                user_pos[user2idx[parts[0]]].append(biz2idx[parts[1]])
    return user_pos


def attention_weighted_mean(
    item_embeds: torch.Tensor,   # [K, D]  recent business embeddings
    query: torch.Tensor,         # [D]     user embedding
    temperature: float = 0.1,
) -> torch.Tensor:
    """Scaled dot-product attention: user_vec attend lên recent items.

    Công thức: weights = softmax(item_embeds @ query / temp)
    Temperature thấp (0.1) → phân bố sharp, item gần user nhất chiếm trọng số lớn.
    Kết quả: weighted sum của item embeddings.
    """
    # scores: [K]
    scores = (item_embeds @ query) / temperature
    weights = F.softmax(scores, dim=0)          # [K]
    return (weights.unsqueeze(1) * item_embeds).sum(0)  # [D]


def combine(user_vec: torch.Tensor, session_vec: torch.Tensor, alpha: float = ALPHA):
    """Kết hợp user embedding và session embedding.
    merged = α·user + (1-α)·session, rồi L2-normalize.
    L2-normalize để Faiss IP search = cosine similarity.
    """
    merged = alpha * user_vec + (1 - alpha) * session_vec
    return F.normalize(merged, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Session-aware Faiss recommendation")
    parser.add_argument("--user_id",      type=str, default=None, help="Yelp user_id string")
    parser.add_argument("--random_user",  action="store_true", help="Pick a random user with history")
    parser.add_argument("--recent",       type=str, default=None,
                        help="Comma-separated business_id strings the user recently visited")
    parser.add_argument("--fake_session", action="store_true",
                        help="Auto-pick recent businesses from user's history")
    parser.add_argument("--session_size", type=int, default=5,
                        help="Number of businesses to sample for fake session (default: 5)")
    parser.add_argument("--topk",         type=int, default=20)
    parser.add_argument("--alpha",        type=float, default=ALPHA,
                        help="Weight for trained user embed vs session (0-1)")
    args = parser.parse_args()

    # ── Load artefacts ────────────────────────────────────────────────────────
    print("Loading embeddings & index …")
    user_h   = torch.load(OUT_DIR / "user_h.pt",   weights_only=True)
    biz_h    = torch.load(OUT_DIR / "biz_h.pt",    weights_only=True)
    mappings = torch.load(OUT_DIR / "mappings.pt",  weights_only=False)
    index    = faiss.read_index(str(OUT_DIR / "faiss_biz.index"))

    user2idx = mappings["user2idx"]
    biz2idx  = mappings["biz2idx"]
    idx2biz  = mappings["idx2biz"]
    idx2user = mappings["idx2user"]

    # ══ Load user history (để random_user hoặc fake_session) ═══════════
    user_pos = None
    if args.fake_session or args.random_user:
        print("Loading user history …")
        user_pos = load_user_history(UB_PATH, user2idx, biz2idx)

    # ══ Xác định user (bằng ID hoặc random) ═════════════════════════════────
    if args.random_user:
        # Pick a random user who has enough history
        candidates = [u for u, bizs in user_pos.items() if len(bizs) >= args.session_size + 3]
        if not candidates:
            candidates = [u for u, bizs in user_pos.items() if len(bizs) >= 2]
        u_idx = random.choice(candidates)
        user_id = idx2user[u_idx]
        print(f"  Random user picked: {user_id} ({len(user_pos[u_idx])} interactions)")
    elif args.user_id:
        user_id = args.user_id
        if user_id not in user2idx:
            print(f"ERROR: user_id '{user_id}' not found in graph")
            return
        u_idx = user2idx[user_id]
    else:
        print("ERROR: specify --user_id or --random_user")
        return

    u_vec = user_h[u_idx]  # [D]

    # ══ Xây session từ recent businesses ═══════════════════════════════────
    if args.fake_session:
        if user_pos is None or u_idx not in user_pos or len(user_pos[u_idx]) == 0:
            print(f"ERROR: user has no history to build fake session")
            return
        history = user_pos[u_idx]
        k = min(args.session_size, len(history))
        sampled = random.sample(history, k)
        recent_ids = [idx2biz[i] for i in sampled]
        recent_indices = sampled
    elif args.recent:
        recent_ids = [s.strip() for s in args.recent.split(",") if s.strip()]
        recent_indices = []
        for bid in recent_ids:
            if bid in biz2idx:
                recent_indices.append(biz2idx[bid])
            else:
                print(f"  WARNING: business_id '{bid}' not in graph, skipping")
    else:
        print("ERROR: specify --recent or --fake_session")
        return

    if not recent_indices:
        print("ERROR: no valid businesses in session")
        return

    recent_embeds = biz_h[recent_indices]  # [K, D]

    # ══ Tạo session embedding (attention-weighted mean) ═════════════════────────
    session_emb = attention_weighted_mean(recent_embeds, u_vec)

    # ══ Combine user + session, Faiss search ══════════════════════════
    user_final = combine(u_vec, session_emb, args.alpha)
    query_np   = user_final.unsqueeze(0).numpy().astype("float32")
    faiss.normalize_L2(query_np)  # Đảm bảo L2-norm cho IP search

    # Fetch nhiều hơn topk để bù cho việc filter session businesses
    seen_set = set(recent_indices)
    fetch_k  = args.topk + len(seen_set) + 10
    distances, indices = index.search(query_np, fetch_k)

    # ══ Hiển thị kết quả ══════════════════════════════════════════════
    print("Loading business metadata …")
    biz_meta = load_biz_meta(BIZ_JSON)

    # ── In kết quả top-K (bỏ qua businesses đã trong session) ──────
    print(f"\n{'='*80}")
    print(f"User: {user_id}  (idx={u_idx})")
    total_hist = len(user_pos[u_idx]) if user_pos and u_idx in user_pos else "?"
    print(f"Total history: {total_hist} businesses")
    print(f"\nSession ({'FAKE' if args.fake_session else 'PROVIDED'}, {len(recent_indices)} items):")
    for bid in recent_ids:
        m = biz_meta.get(bid, {})
        print(f"  • {m.get('name', bid):<45s} [{m.get('categories', '')}]")
    print(f"\nAlpha (user vs session): {args.alpha}")
    print(f"{'='*80}")

    print(f"\nTop-{args.topk} Recommendations:")
    print(f"{'Rank':<6}{'Score':<10}{'Name':<45}{'Categories'}")
    print("-" * 110)

    rank = 0
    for score, biz_idx in zip(distances[0], indices[0]):
        if biz_idx in seen_set:
            continue
        rank += 1
        if rank > args.topk:
            break
        bid = idx2biz.get(int(biz_idx), "???")
        m   = biz_meta.get(bid, {})
        name = m.get("name", bid)
        cats = m.get("categories", "")
        print(f"{rank:<6}{score:<10.4f}{name:<45}{cats}")

    print()


if __name__ == "__main__":
    main()
