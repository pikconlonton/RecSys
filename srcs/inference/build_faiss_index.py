"""
Build Faiss IndexFlatIP cho business embeddings — fast top-K retrieval.

╔════════════════════════════════════════════════════════════════════╗
║  INPUT  : outputs/biz_h.pt (L2-normalised, [14567 × 128])          ║
║  OUTPUT : outputs/faiss_biz.index (IndexFlatIP, brute-force)      ║
║                                                                    ║
║  TẠI SAO IndexFlatIP (không dùng IVF/HNSW):                       ║
║  - Chỉ 14,567 businesses → brute-force đủ nhanh (< 1ms)           ║
║  - Không cần training index, không mất recall                     ║
║  - IP (Inner Product) = cosine similarity vì vectors đã L2-norm  ║
╚════════════════════════════════════════════════════════════════════╝
"""

import faiss
import torch
import numpy as np
from pathlib import Path

# ══════════════════════════════════════════════════════════════════════════════
# Paths
# ══════════════════════════════════════════════════════════════════════════════
OUT_DIR  = Path("../../outputs")
BIZ_PATH = OUT_DIR / "biz_h.pt"       # Embedding đã export từ GNN
IDX_PATH = OUT_DIR / "faiss_biz.index"  # Sẽ lưu Faiss index tại đây


def main():
    print("Loading business embeddings …")
    biz_h = torch.load(BIZ_PATH, weights_only=True)  # [N_biz, 128]
    biz_np = biz_h.numpy().astype("float32")
    n, d = biz_np.shape
    print(f"  {n:,} businesses, dim={d}")

    # L2-normalize lại (biz_h.pt đã normalize rồi, nhưng để chắc chắn)
    # Sau khi normalize, Inner Product = cosine similarity
    faiss.normalize_L2(biz_np)

    # IndexFlatIP: brute-force inner product search
    # Không cần train, exact search, phù hợp với ~15K vectors
    print("Building IndexFlatIP \u2026")
    index = faiss.IndexFlatIP(d)
    index.add(biz_np)  # Thêm tất cả business vectors vào index
    print(f"  Index total: {index.ntotal:,}")

    faiss.write_index(index, str(IDX_PATH))
    print(f"Saved → {IDX_PATH}")


if __name__ == "__main__":
    main()
