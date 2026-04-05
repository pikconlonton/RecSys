"""
Ghép pretrained embeddings (384-dim) vào HeteroData graph.

╔══════════════════════════════════════════════════════════════════════╗
║  PIPELINE:                                                          ║
║  1. Load graph.pt (chứa HeteroData + user2idx + biz2idx)           ║
║  2. Load business_embeddings.pkl và user_embeddings.pkl            ║
║  3. Align embeddings theo node index (idx → id → embedding)       ║
║     - Node thiếu embedding → zero vector                           ║
║  4. Inject vào data['user'].x và data['business'].x               ║
║  5. Save → embedded_graph.pt                                       ║
║                                                                      ║
║  TẠI SAO CẦN FILE NÀY:                                              ║
║  - graph.pt chỉ có topology (edges), chưa có features              ║
║  - Embeddings từ SentenceTransformer lưu dạng dict {id: vector}    ║
║  - GNN cần features dạng tensor [N_nodes, 384] align theo index    ║
║                                                                      ║
║  INPUT:  outputs/graph.pt, outputs/*_embeddings.pkl                 ║
║  OUTPUT: outputs/embedded_graph.pt (~366MB)                         ║
╚══════════════════════════════════════════════════════════════════════╝
"""
import pickle
import numpy as np
import torch
from pathlib import Path

SEP = "=" * 60

# ══════════════════════════════════════════════════════════════════════════════
# Paths & Config
# ══════════════════════════════════════════════════════════════════════════════
GRAPH_PATH   = Path("outputs/graph.pt")              # Graph topology (no features)
BIZ_EMB_PATH = Path("outputs/business_embeddings.pkl")  # {biz_id: np.array(384,)}
USR_EMB_PATH = Path("outputs/user_embeddings.pkl")      # {user_id: np.array(384,)}
OUT_PATH     = Path("outputs/embedded_graph.pt")        # Graph + features

EMB_DIM = 384  # all-MiniLM-L6-v2 output dimension

# ══ 1. Load graph ════════════════════════════════════════════════════════════
# graph.pt chứa: HeteroData (topology) + user2idx + biz2idx mappings
print(SEP)
print("STEP 1 — Loading graph")
print(SEP)

checkpoint = torch.load(GRAPH_PATH, weights_only=False)
data     = checkpoint["data"]
user2idx = checkpoint["user2idx"]   # {user_id_str: int}
biz2idx  = checkpoint["biz2idx"]    # {biz_id_str:  int}

num_users = data["user"].num_nodes
num_biz   = data["business"].num_nodes

print(f"  Users     : {num_users:,}")
print(f"  Businesses: {num_biz:,}")

# Edge stats before embedding
print()
print("  Edge relations:")
for edge_type in data.edge_types:
    ei = data[edge_type].edge_index
    src_type, rel, dst_type = edge_type
    print(f"    ({src_type}, {rel}, {dst_type})"
          f"  |  edges={ei.shape[1]:>10,}"
          f"  src_range=[0, {ei[0].max().item()}]"
          f"  dst_range=[0, {ei[1].max().item()}]")

# ══ 2. Load embeddings (.pkl) ════════════════════════════════════════════════
# Mỗi file pkl là dict {string_id: numpy_array(384,)}
print()
print(SEP)
print("STEP 2 — Loading embeddings")
print(SEP)

with BIZ_EMB_PATH.open("rb") as f:
    biz_emb: dict = pickle.load(f)   # {biz_id: np.ndarray(384,)}

with USR_EMB_PATH.open("rb") as f:
    usr_emb: dict = pickle.load(f)   # {user_id: np.ndarray(384,)}

print(f"  Business embeddings loaded: {len(biz_emb):,}")
print(f"  User embeddings loaded    : {len(usr_emb):,}")

# ══ 3. Build aligned feature matrices ════════════════════════════════════════
# Duyệt từng node index → tra ngược ra string ID → lấy embedding
# Nếu không tìm thấy → giữ zero vector (GNN sẽ học từ neighbors)
print()
print(SEP)
print("STEP 3 — Aligning embeddings to node indices")
print(SEP)

# Reverse mappings: idx → id_str  (for fast lookup)
idx2biz  = {v: k for k, v in biz2idx.items()}
idx2user = {v: k for k, v in user2idx.items()}

# Business feature matrix
biz_matrix  = np.zeros((num_biz,  EMB_DIM), dtype=np.float32)
biz_missing = 0
for idx in range(num_biz):
    bid = idx2biz.get(idx)
    vec = biz_emb.get(bid) if bid else None
    if vec is not None:
        biz_matrix[idx] = vec
    else:
        biz_missing += 1

# User feature matrix
usr_matrix  = np.zeros((num_users, EMB_DIM), dtype=np.float32)
usr_missing = 0
for idx in range(num_users):
    uid = idx2user.get(idx)
    vec = usr_emb.get(uid) if uid else None
    if vec is not None:
        usr_matrix[idx] = vec
    else:
        usr_missing += 1

biz_covered  = num_biz   - biz_missing
usr_covered  = num_users - usr_missing
print(f"  Business nodes : {num_biz:,}")
print(f"    ✓ covered : {biz_covered:,}  ({biz_covered/num_biz*100:.1f}%)")
print(f"    ✗ missing : {biz_missing:,}  ({biz_missing/num_biz*100:.1f}%)  → zero vector")
print()
print(f"  User nodes     : {num_users:,}")
print(f"    ✓ covered : {usr_covered:,}  ({usr_covered/num_users*100:.1f}%)")
print(f"    ✗ missing : {usr_missing:,}  ({usr_missing/num_users*100:.1f}%)  → zero vector")

# ══ 4. Inject features vào HeteroData ════════════════════════════════════════
# Gán tensor features cho mỗi node type
print()
print(SEP)
print("STEP 4 — Injecting features into graph")
print(SEP)

data["user"].x     = torch.tensor(usr_matrix, dtype=torch.float)
data["business"].x = torch.tensor(biz_matrix, dtype=torch.float)

# ── Thống kê features (norm, coverage) ─────────────────────────────────────
for node_type, matrix in [("user", usr_matrix), ("business", biz_matrix)]:
    norms = np.linalg.norm(matrix, axis=1)
    nonzero = int((norms > 0).sum())
    print(f"  {node_type}.x")
    print(f"    shape      : {list(matrix.shape)}")
    print(f"    dtype      : {matrix.dtype}")
    print(f"    non-zero   : {nonzero:,} / {len(norms):,}  ({nonzero/len(norms)*100:.1f}%)")
    print(f"    norm stats : min={norms.min():.4f}  max={norms.max():.4f}"
          f"  mean={norms.mean():.4f}  std={norms.std():.4f}")
    print()

# ── Thống kê edges ─────────────────────────────────────────────────────────
total_edges = sum(data[et].edge_index.shape[1] for et in data.edge_types)
print(f"  Total edge types : {len(data.edge_types)}")
print(f"  Total edges      : {total_edges:,}")
print()
for edge_type in data.edge_types:
    src_type, rel, dst_type = edge_type
    n = data[edge_type].edge_index.shape[1]
    print(f"    ({src_type}, {rel}, {dst_type})  →  {n:,}")

# ══ 5. Save graph kèm features ═══════════════════════════════════════════════
# Lưu lại cả data, user2idx, biz2idx để inference dùng
print()
print(SEP)
print("STEP 5 — Saving graph with features")
print(SEP)

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "data":     data,
        "user2idx": user2idx,
        "biz2idx":  biz2idx,
    },
    OUT_PATH,
)

size_mb = OUT_PATH.stat().st_size / (1024 ** 2)
print(f"  Saved  → {OUT_PATH}")
print(f"  Size   : {size_mb:.2f} MB")

# ── Final summary ──────────────────────────────────────────────────────────
print()
print(SEP)
print("  FINAL GRAPH SUMMARY")
print(SEP)
print(f"  Node types       : {len(data.node_types)}")
for nt in data.node_types:
    x = data[nt].get("x")
    n = data[nt].num_nodes
    feat = f"x=({n}, {x.shape[1]})" if x is not None else f"num_nodes={n}"
    print(f"    {nt:<12}  {feat}")
print()
print(f"  Edge types       : {len(data.edge_types)}")
for edge_type in data.edge_types:
    src_type, rel, dst_type = edge_type
    n = data[edge_type].edge_index.shape[1]
    print(f"    ({src_type}, {rel}, {dst_type})  →  {n:,} edges")
print()
print("Done.!!!")
