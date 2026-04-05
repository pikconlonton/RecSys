"""
Tạo cạnh Business → Business (similar edges) bằng Faiss KNN.

╔════════════════════════════════════════════════════════════════════╗
║  PIPELINE:                                                         ║
║  1. Load business embeddings (384-dim, SentenceTransformer)       ║
║  2. L2-normalize → cosine similarity = inner product              ║
║  3. Build Faiss IndexFlatIP → KNN search top-11                  ║
║     (11 vì kết quả đầu tiên là chính nó, bỏ qua)                ║
║  4. Filter: chỉ giữ cạnh có cosine > 0.6 (loại noise)            ║
║  5. Ghi ra file: biz_id1 \t biz_id2 \t score                     ║
║                                                                    ║
║  TẠI SAO FAISS (không brute-force Python):                        ║
║  - 14,567 biz × 14,567 = 212M cặp → Faiss có SIMD optimize       ║
║  - IndexFlatIP = exact search, phù hợp dataset nhỏ              ║
║                                                                    ║
║  TẠI SAO THRESHOLD 0.6:                                          ║
║  - Quá thấp → edge noise (business không liên quan)               ║
║  - Quá cao → graph sparse, GNN thiếu tín hiệu                   ║
║  - 0.6 là trade-off hợp lý cho text similarity                   ║
║                                                                    ║
║  INPUT:  business_embeddings.pkl  (từ embedding_business.py)      ║
║  OUTPUT: similar_business_edges2.txt                               ║
╚════════════════════════════════════════════════════════════════════╝
"""

import pickle
import numpy as np
import faiss

# ══ Bước 1: Load embedding đã tạo từ embedding_business.py ═════════════════════
# Format: {business_id_str: np.ndarray(384,)}
with open('business_embeddings.pkl', 'rb') as f:
    business_embeddings = pickle.load(f)
print("Loaded embeddings:", len(business_embeddings))

# ══ Bước 2: Convert dict → matrix cho Faiss ══════════════════════════════
# Giữ thứ tự business_ids để map từ index → id sau khi search
business_ids = list(business_embeddings.keys())
vectors = np.array([business_embeddings[b] for b in business_ids]).astype('float32')

# L2-normalize: sau khi normalize, Inner Product = Cosine Similarity
# Đây là trick quan trọng: Faiss IndexFlatIP tính dot product,
# nhưng nếu vector đã normalize thì dot product = cosine similarity
faiss.normalize_L2(vectors)

# ══ Bước 3: Build Faiss index ═══════════════════════════════════════════
# IndexFlatIP = exact inner product search (brute-force nhưng Faiss tối ưu SIMD)
index = faiss.IndexFlatIP(vectors.shape[1])  # dim = 384
index.add(vectors)                           # thêm toàn bộ vectors vào index

# ══ Bước 4: KNN search ═════════════════════════════════════════════════
# D[i] = top-11 cosine scores cho business i  (giảm dần)
# I[i] = top-11 indices cho business i
# Lấy 11 vì kết quả đầu tiên (index 0) luôn là chính nó (score ≈ 1.0)
D, I = index.search(vectors, 11)
print("Done FAISS search")

# ══ Bước 5: Filter theo threshold và ghi edge ═══════════════════════════
with open('similar_business_edges2.txt', 'w') as f:
    for i, neighbors in enumerate(I):
        for k, j in enumerate(neighbors[1:]):  # Bỏ index 0 (chính nó)
            score = D[i][k+1]

            # Chỉ giữ cạnh có cosine similarity > 0.6
            # Đảm bảo 2 business thực sự tương tự về mặt nội dung
            if score > 0.6:
                f.write(f"{business_ids[i]}\t{business_ids[j]}\t{score:.4f}\n")

print("Saved similar edges")