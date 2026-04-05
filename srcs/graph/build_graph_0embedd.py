"""
Ghép 3 loại edge thành PyG HeteroData (chưa có embedding).

╔════════════════════════════════════════════════════════════════════╗
║  PIPELINE:                                                         ║
║  1. Đọc 3 file edge text:                                          ║
║     - edges_user_business.txt  (user rates business)              ║
║     - edges_user_user.txt      (user friends user)                ║
║     - edges_business_business_similar.txt  (biz similar biz)      ║
║  2. Tạo mapping: string ID → integer index (sorted, reproducible) ║
║  3. Convert edges thành edge_index tensor [2, num_edges]          ║
║  4. Build HeteroData với 5 loại cạnh:                             ║
║     - (user, rates, business)       — forward                     ║
║     - (business, rated_by, user)    — reverse (cho GNN bi-direct) ║
║     - (user, friends, user)         — social                      ║
║     - (business, similar, business) — forward                     ║
║     - (business, similar_rev, biz)  — reverse                     ║
║  5. Lưu {data, user2idx, biz2idx} vào graph.pt                    ║
║                                                                    ║
║  TẠI SAO CẦN REVERSE EDGES:                                       ║
║  - GNN propagate message theo chiều cạnh                        ║
║  - Nếu chỉ có user→business, business không nhận message từ user║
║  - Reverse edges giúp message đi 2 chiều → undirected graph     ║
║                                                                    ║
║  INPUT:  3 file edge text (.txt)                                   ║
║  OUTPUT: graph.pt (HeteroData, chưa có node features)             ║
╚════════════════════════════════════════════════════════════════════╝
"""

import torch
from torch_geometric.data import HeteroData
from pathlib import Path
import time

# ══════════════════════════════════════════════════════════════════════════════
# Paths — 3 file edge và file output
# ══════════════════════════════════════════════════════════════════════════════
UB_PATH  = Path("ouputs/edges_user_business.txt")       # user ─rates→ business
UU_PATH  = Path("ouputs/edges_user_user.txt")           # user ─friends→ user
BB_PATH  = Path("ouputs/edges_business_business_similar.txt")  # biz ─similar→ biz
OUT_PATH = Path("ouputs/graph.pt")

t0 = time.time()
SEP = "=" * 60

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1: Load raw edges từ text files
# Mỗi dòng: src_id \t dst_id
# ══════════════════════════════════════════════════════════════════════════════
print(SEP)
print("STEP 1 — Loading raw edges")
print(SEP)


def load_edges(path):
    """Load edge list từ file TSV. Returns: [(src_id, dst_id), ...]"""
    edges = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                edges.append((parts[0], parts[1]))
    return edges

ub_raw = load_edges(UB_PATH)
uu_raw = load_edges(UU_PATH)
bb_raw = load_edges(BB_PATH)

print(f"  user-business edges : {len(ub_raw):>10,}")
print(f"  user-user edges     : {len(uu_raw):>10,}")
print(f"  biz-biz similar     : {len(bb_raw):>10,}")

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2: Tạo mapping string ID → integer index
# Sorted để đảm bảo reproducibility (cùng data → cùng index)
# ══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("STEP 2 — Building node ID → integer index mappings")
print(SEP)

# Thu thập tất cả unique IDs từ cả 3 loại edge
all_user_ids = set()
all_biz_ids  = set()

for u, b in ub_raw:
    all_user_ids.add(u)
    all_biz_ids.add(b)

for u, v in uu_raw:
    all_user_ids.add(u)
    all_user_ids.add(v)

for b1, b2 in bb_raw:
    all_biz_ids.add(b1)
    all_biz_ids.add(b2)

# Sorted mapping để reproducible (cùng input → cùng index)
user2idx = {uid: i for i, uid in enumerate(sorted(all_user_ids))}
biz2idx  = {bid: i for i, bid in enumerate(sorted(all_biz_ids))}

num_users = len(user2idx)
num_biz   = len(biz2idx)

print(f"  Unique users     : {num_users:>10,}")
print(f"  Unique businesses: {num_biz:>10,}")

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 3: Convert string edges → integer edge_index tensors
# edge_index shape: [2, num_edges] — format PyG chuẩn
# ══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("STEP 3 — Converting edges to index tensors")
print(SEP)


def to_edge_index(raw_edges, src_map, dst_map):
    """
    Convert list[(src_str, dst_str)] → torch.Tensor [2, N].
    Skip edges có ID không tìm thấy trong mapping (shouldn't happen).
    """
    src_list, dst_list = [], []
    skipped = 0
    for s, d in raw_edges:
        if s in src_map and d in dst_map:
            src_list.append(src_map[s])
            dst_list.append(dst_map[d])
        else:
            skipped += 1
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    return edge_index, skipped

ei_ub, skip_ub = to_edge_index(ub_raw, user2idx, biz2idx)
ei_uu, skip_uu = to_edge_index(uu_raw, user2idx, user2idx)
ei_bb, skip_bb = to_edge_index(bb_raw, biz2idx, biz2idx)

print(f"  (user, rates, business)   shape={list(ei_ub.shape)}  skipped={skip_ub}")
print(f"  (user, friends, user)     shape={list(ei_uu.shape)}  skipped={skip_uu}")
print(f"  (business, similar, biz)  shape={list(ei_bb.shape)}  skipped={skip_bb}")

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 4: Build HeteroData (PyG)
# HeteroData = graph với nhiều loại node và edge
# ══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("STEP 4 — Assembling HeteroData")
print(SEP)

data = HeteroData()

# Số nodes cho mỗi loại (chưa có features, sẽ thêm ở embedding_graph.py)
data["user"].num_nodes    = num_users
data["business"].num_nodes = num_biz

# 3 loại cạnh chính (forward direction)
data["user", "rates",   "business"].edge_index = ei_ub
data["user", "friends", "user"    ].edge_index = ei_uu
data["business", "similar", "business"].edge_index = ei_bb

# 2 loại cạnh reverse để GNN có thể truyền message 2 chiều
# .flip(0) = đảo hàng 0 và hàng 1 của edge_index [2, N] → đảo chiều edge
data["business", "rated_by", "user"    ].edge_index = ei_ub.flip(0)
data["business", "similar_rev", "business"].edge_index = ei_bb.flip(0)

print(data)

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 5: Lưu graph (chưa có features)
# Sẽ được embedding_graph.py thêm features và lưu lại
# ══════════════════════════════════════════════════════════════════════════════
print()
print(SEP)
print("STEP 5 — Saving graph")
print(SEP)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
# Lưu cả data + mappings để các bước sau có thể map ngược index → string ID
torch.save(
    {
        "data":     data,
        "user2idx": user2idx,
        "biz2idx":  biz2idx,
    },
    OUT_PATH,
)
size_mb = OUT_PATH.stat().st_size / (1024 ** 2)
elapsed = time.time() - t0

print(f"  Saved  → {OUT_PATH}")
print(f"  Size   : {size_mb:.2f} MB")
print(f"  Elapsed: {elapsed:.1f}s")
print()
print("✓ Graph build complete.")
