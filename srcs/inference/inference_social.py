"""
Inference với Social Score — Kết hợp 3 tín hiệu để recommend:

  ┌─────────────────────────────────────────────────────────────────┐
  │  final_score = (1 - γ) × embedding_score + γ × social_score   │
  └─────────────────────────────────────────────────────────────────┘

  Trong đó:
    embedding_score = cosine(user_final, biz_h[b])
        user_final  = α × user_h[u] + (1-α) × session_emb
        session_emb = attention_weighted_mean(biz_h[recent_items], user_h[u])

    social_score(u, b) = Σ_f  w_f × I(f đã interact với b)
        w_f = softmax( cos(h_u, h_f) / τ )   over friends F of u
        → Bạn bè càng giống user → ý kiến càng có trọng lượng
        → Business nào nhiều bạn (giống mình) đã ghé → social score cao

Pipeline:
  1. Load user_h, biz_h, faiss index, friend graph, user history
  2. Tạo session (fake hoặc real) → session embedding
  3. Combine user_h + session → query embedding
  4. Faiss search → top 5×K candidates (lấy nhiều để re-rank)
  5. Tính social_score cho từng candidate
  6. Re-rank: final = (1-γ) × emb + γ × social
  7. In kết quả với cả 3 cột score

Usage:
  # Random user, fake session, social score:
  python inference_social.py --random_user --fake_session --topk 15 --gamma 0.2

  # Specific user + session thật:
  python inference_social.py --user_id="USER_ID" --recent="BIZ1,BIZ2" --gamma 0.3

  # Chỉnh alpha (user vs session) và gamma (social weight):
  python inference_social.py --random_user --fake_session --alpha 0.4 --gamma 0.2
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
# PATHS — Chỉnh lại nếu cấu trúc thư mục khác
# ══════════════════════════════════════════════════════════════════════════════
OUT_DIR  = Path("../../outputs")
UB_PATH  = Path("../../outputs/graph/edges_user_business.txt")   # user→business edges
UU_PATH  = Path("../../outputs/graph/edges_user_user.txt")       # user→user friend edges
BIZ_JSON = Path("../../Yelp-JSON/Yelp JSON/yelp_dataset/yelp_academic_dataset_business.json")

# ══════════════════════════════════════════════════════════════════════════════
# DEFAULT HYPERPARAMETERS
# ══════════════════════════════════════════════════════════════════════════════
ALPHA = 0.4   # α: trọng số user_h vs session_emb  (cao → thiên về profile dài hạn)
GAMMA = 0.2   # γ: trọng số social score           (cao → bạn bè ảnh hưởng nhiều hơn)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_biz_meta(path: Path) -> dict:
    """
    Đọc metadata business từ Yelp JSON.
    Returns: {business_id_str: {"name": ..., "categories": ...}}
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
    """
    Đọc edge file user→business.
    Returns: {user_idx: [biz_idx, ...]}  — danh sách business mỗi user đã tương tác
    """
    user_pos = defaultdict(list)
    with ub_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2 and parts[0] in user2idx and parts[1] in biz2idx:
                user_pos[user2idx[parts[0]]].append(biz2idx[parts[1]])
    return user_pos


def load_friend_map(uu_path: Path, user2idx: dict) -> dict:
    """
    Đọc edge file user→user (friendship / social graph).
    Cạnh undirected: nếu (A, B) thì cả A→B và B→A.
    Returns: {user_idx: [friend_idx, ...]}
    """
    friends = defaultdict(set)
    with uu_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2 and parts[0] in user2idx and parts[1] in user2idx:
                u = user2idx[parts[0]]
                v = user2idx[parts[1]]
                friends[u].add(v)
                friends[v].add(u)     # undirected
    # set → list cho indexing
    return {u: list(fl) for u, fl in friends.items()}


# ══════════════════════════════════════════════════════════════════════════════
# SESSION EMBEDDING — Attention-weighted mean
# ══════════════════════════════════════════════════════════════════════════════

def attention_weighted_mean(
    item_embeds: torch.Tensor,   # [K, D]  — embedding của K business gần đây
    query: torch.Tensor,         # [D]     — embedding của user
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Scaled dot-product attention:
      score_k  = (item_k · query) / τ
      weight_k = softmax(scores)
      output   = Σ weight_k × item_k

    Temperature thấp (0.1) → tập trung vào item giống user nhất trong session.
    Temperature cao (1.0)  → trọng số đều hơn giữa các item.
    """
    scores  = (item_embeds @ query) / temperature   # [K]
    weights = F.softmax(scores, dim=0)               # [K], tổng = 1
    return (weights.unsqueeze(1) * item_embeds).sum(0)  # [D]


def combine(user_vec: torch.Tensor, session_vec: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Kết hợp user embedding dài hạn + session embedding ngắn hạn:
      merged = α × user_vec + (1 - α) × session_vec
    Sau đó L2-normalize để dùng với cosine similarity.

    α cao (0.7–0.9): ưu tiên sở thích lâu dài của user
    α thấp (0.2–0.4): ưu tiên context hiện tại (đang xem gì)
    """
    merged = alpha * user_vec + (1 - alpha) * session_vec
    return F.normalize(merged, dim=0)


# ══════════════════════════════════════════════════════════════════════════════
# SOCIAL SCORE — "Bạn bè ảnh hưởng"
# ══════════════════════════════════════════════════════════════════════════════

def compute_social_scores(
    u_idx: int,
    user_h: torch.Tensor,          # [N_user, D]  — embedding tất cả users
    friend_map: dict,              # {user_idx: [friend_idx, ...]}
    user_pos: dict,                # {user_idx: [biz_idx, ...]}
    candidate_set: set,            # tập candidate business để tính score
    temperature: float = 0.1,
) -> dict:
    """
    Social score: đo mức "bạn bè giống mình đã thích business này".

    Công thức:
      social_score(u, b) = Σ_{f ∈ Friends(u)}  w_f × I(f đã interact với b)

      w_f = softmax( cos(h_u, h_f) / τ )
          → friend f càng giống user u (trong embedding space) → w_f càng lớn
          → bạn bè không giống mình thì ý kiến ít quan trọng

    Ví dụ:
      User A có 3 bạn: B (w=0.5), C (w=0.3), D (w=0.2)
      Business X: B đã đi, D đã đi → social(A, X) = 0.5 + 0.2 = 0.7
      Business Y: C đã đi           → social(A, Y) = 0.3

    Returns: {biz_idx: float}  — social score cho mỗi business trong candidate_set
    """
    friends = friend_map.get(u_idx, [])
    if not friends:
        # User không có bạn → social score = 0 cho tất cả
        return {}

    # ── Tính trọng số cho mỗi friend ─────────────────────────────────────────
    u_vec  = user_h[u_idx]           # [D]
    f_vecs = user_h[friends]         # [|F|, D]

    # Cosine similarity giữa user và từng friend
    sims = F.cosine_similarity(
        u_vec.unsqueeze(0),   # [1, D]
        f_vecs,               # [|F|, D]
        dim=1,
    )  # [|F|]

    # Softmax với temperature → chuẩn hoá thành xác suất
    # τ nhỏ → friend giống nhất chiếm phần lớn trọng số
    # τ lớn → trọng số đều hơn
    weights = F.softmax(sims / temperature, dim=0)  # [|F|], tổng = 1.0

    # ── Tích luỹ social score cho mỗi candidate business ─────────────────────
    scores = defaultdict(float)
    for i, f_idx in enumerate(friends):
        w = weights[i].item()
        # Lấy danh sách business mà friend f_idx đã tương tác
        f_bizs = user_pos.get(f_idx, [])
        for b in f_bizs:
            # Chỉ tính cho candidate (business đã qua Faiss filter)
            if b in candidate_set:
                scores[b] += w

    return dict(scores)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── CLI arguments ─────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Session-aware Faiss recommendation + Social Score re-ranking"
    )
    parser.add_argument("--user_id",      type=str, default=None,
                        help="Yelp user_id string")
    parser.add_argument("--random_user",  action="store_true",
                        help="Chọn random user có đủ history + friends")
    parser.add_argument("--recent",       type=str, default=None,
                        help="Comma-separated business_id (session thật)")
    parser.add_argument("--fake_session", action="store_true",
                        help="Tự sample session từ lịch sử user")
    parser.add_argument("--session_size", type=int, default=5,
                        help="Số business trong fake session (default: 5)")
    parser.add_argument("--topk",         type=int, default=20,
                        help="Số kết quả trả về (default: 20)")
    parser.add_argument("--alpha",        type=float, default=ALPHA,
                        help="α: user_h vs session (0=toàn session, 1=toàn user profile)")
    parser.add_argument("--gamma",        type=float, default=GAMMA,
                        help="γ: social weight (0=không social, 1=toàn social)")
    args = parser.parse_args()

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 1: Load tất cả artefacts
    # ══════════════════════════════════════════════════════════════════════════
    print("Loading embeddings & Faiss index …")
    user_h   = torch.load(OUT_DIR / "user_h.pt",  weights_only=True)   # [N_user, D]
    biz_h    = torch.load(OUT_DIR / "biz_h.pt",   weights_only=True)   # [N_biz, D]
    mappings = torch.load(OUT_DIR / "mappings.pt", weights_only=False)
    index    = faiss.read_index(str(OUT_DIR / "faiss_biz.index"))

    user2idx = mappings["user2idx"]   # {user_id_str → int}
    biz2idx  = mappings["biz2idx"]    # {biz_id_str  → int}
    idx2biz  = mappings["idx2biz"]    # {int → biz_id_str}
    idx2user = mappings["idx2user"]   # {int → user_id_str}

    # Load user history (cần cho fake_session + social score)
    print("Loading user history …")
    user_pos = load_user_history(UB_PATH, user2idx, biz2idx)

    # Load friend graph (cần cho social score)
    print("Loading friend graph …")
    friend_map = load_friend_map(UU_PATH, user2idx)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 2: Chọn user
    # ══════════════════════════════════════════════════════════════════════════
    if args.random_user:
        # Ưu tiên user có đủ history VÀ có bạn (để social score có ý nghĩa)
        candidates = [
            u for u, bizs in user_pos.items()
            if len(bizs) >= args.session_size + 3    # đủ history
            and len(friend_map.get(u, [])) >= 3      # có ít nhất 3 bạn
        ]
        if not candidates:
            # Fallback: bỏ điều kiện friend
            candidates = [u for u, bizs in user_pos.items() if len(bizs) >= args.session_size + 3]
        if not candidates:
            candidates = [u for u, bizs in user_pos.items() if len(bizs) >= 2]

        u_idx   = random.choice(candidates)
        user_id = idx2user[u_idx]
        n_friends = len(friend_map.get(u_idx, []))
        print(f"  Random user: {user_id}")
        print(f"    History: {len(user_pos[u_idx])} businesses  |  Friends: {n_friends}")
    elif args.user_id:
        user_id = args.user_id
        if user_id not in user2idx:
            print(f"ERROR: user_id '{user_id}' not found in graph")
            return
        u_idx = user2idx[user_id]
    else:
        print("ERROR: cần --user_id hoặc --random_user")
        return

    u_vec = user_h[u_idx]  # [D] — embedding đã train của user

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 3: Tạo session (fake hoặc provided)
    # ══════════════════════════════════════════════════════════════════════════
    if args.fake_session:
        # Lấy random K business từ lịch sử user làm "session giả"
        history = user_pos.get(u_idx, [])
        if not history:
            print("ERROR: user không có history để tạo fake session")
            return
        k = min(args.session_size, len(history))
        sampled = random.sample(history, k)
        recent_ids     = [idx2biz[i] for i in sampled]
        recent_indices = sampled
    elif args.recent:
        # Parse danh sách business_id từ CLI
        recent_ids = [s.strip() for s in args.recent.split(",") if s.strip()]
        recent_indices = []
        for bid in recent_ids:
            if bid in biz2idx:
                recent_indices.append(biz2idx[bid])
            else:
                print(f"  WARNING: business '{bid}' not in graph → bỏ qua")
    else:
        print("ERROR: cần --recent hoặc --fake_session")
        return

    if not recent_indices:
        print("ERROR: không có business hợp lệ trong session")
        return

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 4: Build session embedding + combine với user embedding
    # ══════════════════════════════════════════════════════════════════════════
    recent_embeds = biz_h[recent_indices]  # [K, D]

    # Attention-weighted mean: user attend over recent items
    session_emb = attention_weighted_mean(recent_embeds, u_vec)

    # Combine: α × user_profile + (1-α) × session_context
    user_final = combine(u_vec, session_emb, args.alpha)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 5: Faiss search → lấy NHIỀU candidates để re-rank
    # ══════════════════════════════════════════════════════════════════════════
    query_np = user_final.unsqueeze(0).numpy().astype("float32")
    faiss.normalize_L2(query_np)

    seen_set = set(recent_indices)  # filter business đã trong session

    # Lấy 5×topk candidates (nhiều hơn bình thường để social re-rank có ý nghĩa)
    fetch_k = args.topk * 5 + len(seen_set)
    distances, indices = index.search(query_np, fetch_k)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 6: Compute social score cho từng candidate
    # ══════════════════════════════════════════════════════════════════════════
    # Tạo set candidate (loại bỏ business đã xem)
    candidate_set = set(int(i) for i in indices[0] if int(i) not in seen_set)

    social_scores = compute_social_scores(
        u_idx       = u_idx,
        user_h      = user_h,
        friend_map  = friend_map,
        user_pos    = user_pos,
        candidate_set = candidate_set,
    )

    n_friends      = len(friend_map.get(u_idx, []))
    n_social_hits  = sum(1 for v in social_scores.values() if v > 0)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 7: Re-rank — final = (1-γ) × emb_score + γ × social_score
    # ══════════════════════════════════════════════════════════════════════════
    results = []
    for emb_score, biz_idx in zip(distances[0], indices[0]):
        biz_idx = int(biz_idx)
        if biz_idx in seen_set:
            continue
        s_score = social_scores.get(biz_idx, 0.0)
        # Kết hợp: embedding score (Faiss cosine) + social score (friend influence)
        final = (1 - args.gamma) * float(emb_score) + args.gamma * s_score
        results.append((final, float(emb_score), s_score, biz_idx))

    # Sắp xếp theo final score giảm dần
    results.sort(key=lambda x: x[0], reverse=True)

    # ══════════════════════════════════════════════════════════════════════════
    # STEP 8: Load metadata & in kết quả
    # ══════════════════════════════════════════════════════════════════════════
    print("Loading business metadata …")
    biz_meta = load_biz_meta(BIZ_JSON)

    # ── Header ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*100}")
    print(f"  User   : {user_id}  (idx={u_idx})")
    total_hist = len(user_pos.get(u_idx, []))
    print(f"  History: {total_hist} businesses  |  Friends: {n_friends}  |  Social overlap: {n_social_hits}/{len(candidate_set)} candidates")
    print(f"\n  Session ({'FAKE' if args.fake_session else 'PROVIDED'}, {len(recent_indices)} items):")
    for bid in recent_ids:
        m = biz_meta.get(bid, {})
        print(f"    • {m.get('name', bid):<45s} [{m.get('categories', '')}]")
    print(f"\n  Weights: α={args.alpha} (user vs session)  |  γ={args.gamma} (social)")
    print(f"{'═'*100}")

    # ── Table ─────────────────────────────────────────────────────────────────
    # 3 cột score: Final (tổng hợp), Emb (embedding), Social (bạn bè)
    print(f"\n  Top-{args.topk} Recommendations:")
    print(f"  {'Rank':<6}{'Final':<10}{'Emb':<10}{'Social':<10}{'Name':<40}{'Categories'}")
    print(f"  {'-'*126}")

    for rank, (final, emb_s, soc_s, biz_idx) in enumerate(results[:args.topk], 1):
        bid  = idx2biz.get(biz_idx, "???")
        m    = biz_meta.get(bid, {})
        name = m.get("name", bid)
        cats = m.get("categories", "")

        # Hiển thị social score: "-" nếu = 0 (dễ đọc hơn 0.0000)
        soc_str = f"{soc_s:.4f}" if soc_s > 0 else "   -"
        print(f"  {rank:<6}{final:<10.4f}{emb_s:<10.4f}{soc_str:<10}{name:<40}{cats}")

    # ── Summary ───────────────────────────────────────────────────────────────
    if n_friends == 0:
        print(f"\n  ⚠ User không có friend → social score = 0 cho tất cả → kết quả = embedding only")
    elif n_social_hits == 0:
        print(f"\n  ⚠ User có {n_friends} friends nhưng không ai interact với candidates → social = 0")
    else:
        # Đếm bao nhiêu business trong top-K có social boost
        top_k_results = results[:args.topk]
        boosted = sum(1 for _, _, s, _ in top_k_results if s > 0)
        print(f"\n  ✓ {boosted}/{args.topk} kết quả được boost bởi social score")

    print()


if __name__ == "__main__":
    main()
