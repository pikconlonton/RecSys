"""
Export trained embeddings từ HeteroLightGCN checkpoint.

╔════════════════════════════════════════════════════════════════════╗
║  PIPELINE:                                                         ║
║  1. Load embedded_graph.pt (graph + raw 384-dim features)         ║
║  2. Build 4 normalized adj matrices (giống training)              ║
║  3. Load best.pt checkpoint → khôi phục model weights             ║
║  4. Forward pass (eval mode, no grad) → user_h, biz_h            ║
║  5. Lưu user_h.pt, biz_h.pt, mappings.pt                         ║
║                                                                    ║
║  TẠI SAO CẦN FILE NÀY (không dùng thẳng checkpoint):              ║
║  - Checkpoint lưu model weights, không lưu embedding output       ║
║  - Cần forward pass qua GNN để tạo final embeddings              ║
║  - Embeddings đã L2-normalize, sẵn sàng cho Faiss search         ║
║                                                                    ║
║  OUTPUT:                                                           ║
║  - user_h.pt : [211,115 × 128] L2-normalized user embeddings     ║
║  - biz_h.pt  : [14,567 × 128]  L2-normalized biz embeddings      ║
║  - mappings.pt: {user2idx, biz2idx, idx2user, idx2biz}            ║
╚════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
from pathlib import Path
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════════════════
# Config — phải khớp với training config
# ══════════════════════════════════════════════════════════════════════════════
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GRAPH_PATH = Path("../../outputs/embedded_graph.pt")  # Graph với 384-dim features
UB_PATH = Path("../../outputs/edges_user_business.txt")
BB_PATH = Path("../../outputs/edges_business_business_similar.txt")
CKPT_PATH = Path("../../outputs/ckpt1/best.pt")  # Best checkpoint từ training
OUT_DIR = Path("../../outputs")

# Model hyperparameters (PHẢI GIỐNG training notebook)
IN_DIM = 384  # SentenceTransformer output dimension
EMBED_DIM = 128  # GNN embedding dimension
NUM_LAYERS = 3  # Số lớp LightGCN propagation
DROPOUT = 0.1


# ══════════════════════════════════════════════════════════════════════════════
# Model — PHẢI GIỐNG EXACTLY với class trong training notebook
# Nếu sửa model training thì phải sửa ở đây nữa
# ══════════════════════════════════════════════════════════════════════════════
class HeteroLightGCN(nn.Module):
    """
    HeteroLightGCN: LightGCN trên đồ thị heterogeneous.

    Kiến trúc:
    - Linear projection: 384 (SentenceTransformer) → 128 (embed_dim)
    - num_layers lớp propagation KHÔNG CÓ THAM SỐ (chỉ là mean aggregation)
    - Output = mean pooling tất cả các lớp (kể cả lớp 0)
    - L2-normalize → cosine similarity = dot product

    LightGCN propagation (mỗi layer):
      u_new = adj_bu @ b + adj_uu @ u   (user nhận message từ biz + friends)
      b_new = adj_ub @ u + adj_bb @ b   (biz nhận message từ users + similar biz)

    Sparse matmul luôn cast về fp32 vì CUDA sparse không hỗ trợ fp16.
    """

    def __init__(self, in_dim, embed_dim, num_layers, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.user_proj = nn.Linear(in_dim, embed_dim, bias=False)
        self.biz_proj = nn.Linear(in_dim, embed_dim, bias=False)
        self.sim_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, user_feat, biz_feat, adj_ub, adj_bu, adj_uu, adj_bb):
        u0 = self.dropout(self.user_proj(user_feat))
        b0 = self.dropout(self.biz_proj(biz_feat))
        u_layers, b_layers = [u0], [b0]
        u_cur, b_cur = u0, b0
        for _ in range(self.num_layers):
            u_f, b_f = u_cur.float(), b_cur.float()
            u_new = (adj_bu @ b_f + adj_uu @ u_f).to(u_cur.dtype)
            b_new = (adj_ub @ u_f + adj_bb @ b_f).to(b_cur.dtype)
            u_cur, b_cur = u_new, b_new
            u_layers.append(u_cur)
            b_layers.append(b_cur)
        user_emb = torch.stack(u_layers, dim=0).mean(dim=0)
        biz_emb = torch.stack(b_layers, dim=0).mean(dim=0)
        user_h = F.normalize(user_emb, dim=-1)
        biz_h = F.normalize(biz_emb, dim=-1)
        return user_h, biz_h


# ══════════════════════════════════════════════════════════════════════════════
# Helpers: Symmetric normalized adjacency matrix
# D_src^{-1/2} * A * D_dst^{-1/2}  (giống training notebook)
# ══════════════════════════════════════════════════════════════════════════════
def make_norm_adj(edge_index, n_src, n_dst):
    """
    Build symmetric-normalized sparse adj (n_dst × n_src).
    Normalize: val = D_dst^{-1/2} * D_src^{-1/2}
    Clamp max=1e5 để tránh inf khi node có degree = 0.
    """
    row, col = edge_index[0].cpu(), edge_index[1].cpu()
    deg_dst = torch.zeros(n_dst).scatter_add_(0, col, torch.ones(col.size(0)))
    deg_src = torch.zeros(n_src).scatter_add_(0, row, torch.ones(row.size(0)))
    d_dst = deg_dst.pow(-0.5).clamp(max=1e5)
    d_src = deg_src.pow(-0.5).clamp(max=1e5)
    val = d_dst[col] * d_src[row]
    indices = torch.stack([col, row], dim=0)
    return torch.sparse_coo_tensor(indices, val, (n_dst, n_src)).coalesce()


def main():
    print(f"Device: {DEVICE}")

    # ── 1. Load graph ─────────────────────────────────────────────────────────
    print("Loading graph …")
    ckpt_graph = torch.load(GRAPH_PATH, weights_only=False)
    data = ckpt_graph["data"]
    user2idx = ckpt_graph["user2idx"]
    biz2idx = ckpt_graph["biz2idx"]

    NUM_USERS = data["user"].num_nodes
    NUM_BIZ = data["business"].num_nodes
    print(f"  Users={NUM_USERS:,}  Businesses={NUM_BIZ:,}")

    # ══ 2. Build adjacency matrices (giống training notebook) ═══════════════
    # adj_ub: (N_biz, N_user)  — user → business
    # adj_bu: (N_user, N_biz)  — business → user (reverse)
    # adj_uu: (N_user, N_user) — user friends
    # adj_bb: (N_biz, N_biz)   — similar + similar_rev (merged)────
    print("Building adjacency matrices …")
    t0 = time.time()

    adj_ub = make_norm_adj(
        data[("user", "rates", "business")].edge_index, NUM_USERS, NUM_BIZ
    ).to(DEVICE)
    adj_bu = make_norm_adj(
        data[("business", "rated_by", "user")].edge_index, NUM_BIZ, NUM_USERS
    ).to(DEVICE)
    adj_uu = make_norm_adj(
        data[("user", "friends", "user")].edge_index, NUM_USERS, NUM_USERS
    ).to(DEVICE)

    bb_ei1 = data[("business", "similar", "business")].edge_index
    bb_ei2 = data[("business", "similar_rev", "business")].edge_index
    _idx = torch.cat(
        [
            torch.stack([bb_ei1[1].cpu(), bb_ei1[0].cpu()]),
            torch.stack([bb_ei2[1].cpu(), bb_ei2[0].cpu()]),
        ],
        dim=1,
    )
    _val = torch.ones(_idx.size(1))
    _adj = torch.sparse_coo_tensor(_idx, _val, (NUM_BIZ, NUM_BIZ)).coalesce()
    _row, _col = _adj.indices()[1], _adj.indices()[0]
    _dd = torch.zeros(NUM_BIZ).scatter_add_(0, _col, torch.ones(_col.size(0)))
    _ds = torch.zeros(NUM_BIZ).scatter_add_(0, _row, torch.ones(_row.size(0)))
    _vn = _dd[_col].pow(-0.5).clamp(max=1e5) * _ds[_row].pow(-0.5).clamp(max=1e5)
    adj_bb = (
        torch.sparse_coo_tensor(torch.stack([_col, _row]), _vn, (NUM_BIZ, NUM_BIZ))
        .coalesce()
        .to(DEVICE)
    )

    print(f"  Done in {time.time() - t0:.1f}s")

    # ══ 3. Load model weights từ checkpoint ═════════════════════════════
    # Checkpoint chứa: model state_dict, optimizer, epoch, best_loss────────
    print(f"Loading checkpoint: {CKPT_PATH}")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)

    model = HeteroLightGCN(IN_DIM, EMBED_DIM, NUM_LAYERS, DROPOUT).to(DEVICE)
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(
        f"  Loaded epoch {ckpt.get('epoch', '?')}, best_loss={ckpt.get('best_loss', '?')}"
    )

    # ══ 4. Forward pass (eval mode, no grad) ═══════════════════════════
    # model.eval() tắt dropout → embeddings deterministic
    # no_grad() tiết kiệm VRAM (không cần backward)────────
    print("Running forward pass …")
    user_feat = data["user"].x.to(DEVICE)
    biz_feat = data["business"].x.to(DEVICE)

    with torch.no_grad():
        user_h, biz_h = model(user_feat, biz_feat, adj_ub, adj_bu, adj_uu, adj_bb)

    user_h = user_h.cpu()
    biz_h = biz_h.cpu()
    print(f"  user_h: {user_h.shape}  biz_h: {biz_h.shape}")

    # ══ 5. Lưu embeddings + mappings ════════════════════════════════════
    # user_h.pt: [N_user, 128] — dùng cho inference + social score
    # biz_h.pt:  [N_biz, 128]  — dùng cho Faiss index
    # mappings.pt: dict các mapping để convert giữa index ↔ string ID────
    torch.save(user_h, OUT_DIR / "user_h.pt")
    torch.save(biz_h, OUT_DIR / "biz_h.pt")

    # Also save mappings for inference convenience
    torch.save(
        {
            "user2idx": user2idx,
            "biz2idx": biz2idx,
            "idx2user": {v: k for k, v in user2idx.items()},
            "idx2biz": {v: k for k, v in biz2idx.items()},
        },
        OUT_DIR / "mappings.pt",
    )

    print(f"Saved to {OUT_DIR}/{{user_h, biz_h, mappings}}.pt")


if __name__ == "__main__":
    main()
