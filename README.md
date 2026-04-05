# RecSys — Yelp Philadelphia Recommendation System

> **Graph Neural Network (HeteroLightGCN)** kết hợp **session-aware inference** và **social re-ranking** trên dữ liệu Yelp Philadelphia.

```
Raw Yelp JSON ──► Text Embedding (384-d) ──► Graph Construction ──► GNN Training
     ──► Export Embeddings (128-d) ──► Faiss Index ──► Inference + Social Re-ranking
```

---

## Mục lục

1. [Tổng quan kiến trúc](#1-tổng-quan-kiến-trúc)
2. [Phase 1 — Text Embedding](#2-phase-1--text-embedding-sentencetransformer)
3. [Phase 2 — Xây dựng đồ thị](#3-phase-2--xây-dựng-đồ-thị-heterogeneous-graph)
4. [Phase 3 — HeteroLightGCN](#4-phase-3--heterolightgcn)
5. [Phase 4 — Faiss Index](#5-phase-4--faiss-index)
6. [Phase 5 — Inference (Session-aware)](#6-phase-5--inference-session-aware)
7. [Phase 6 — Social Re-ranking](#7-phase-6--social-re-ranking)
8. [Tổng hợp Hyperparameters](#8-tổng-hợp-hyperparameters)
9. [Cấu trúc dự án](#9-cấu-trúc-dự-án)
10. [Cách chạy](#10-cách-chạy)

---

## 1. Tổng quan kiến trúc

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PREPARATION                            │
│                                                                     │
│  Yelp JSON ──► embedding_business.py ──► business_embeddings.pkl   │
│            ──► embedding_users.py    ──► user_embeddings.pkl       │
│            ──► build_user_busines_edges.py ──► edges_user_business  │
│            ──► build_user_user_edges.py    ──► edges_user_user     │
│            ──► build_similar_edges2.py     ──► similar_biz_edges   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       GRAPH ASSEMBLY                                │
│                                                                     │
│  build_graph_0embedd.py ──► graph.pt (HeteroData, 5 edge types)    │
│  embedding_graph.py     ──► embedded_graph.pt (+384-dim features)  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       GNN TRAINING                                  │
│                                                                     │
│  train_graph.ipynb ──► ckpt1/best.pt (HeteroLightGCN weights)     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       INFERENCE PIPELINE                            │
│                                                                     │
│  export_embeddings.py  ──► user_h.pt, biz_h.pt, mappings.pt       │
│  build_faiss_index.py  ──► faiss_biz.index                        │
│  inference.py          ──► Top-K recommendations                   │
│  inference_social.py   ──► Top-K + social re-ranking               │
└─────────────────────────────────────────────────────────────────────┘
```

Hệ thống xử lý **211,115 users** và **14,567 businesses** tại Philadelphia.

---

## 2. Phase 1 — Text Embedding (SentenceTransformer)

**Model:** `all-MiniLM-L6-v2` — pre-trained trên 1B+ sentence pairs, output **384 dimensions**.

Cả user và business đều dùng **cùng model** → nằm trong **cùng embedding space**, cho phép so sánh cross-type.

### 2.1 Business Embedding (`srcs/graph/embedding_business.py`)

Mỗi business được biểu diễn bằng một chuỗi text kết hợp metadata + reviews:

```
text(b) = "Name: {name} | Categories: {categories}
           | Attributes: {key1:val1, key2:val2, ...}
           | Rating: {stars} stars with {review_count} reviews
           | Reviews: {review_1} {review_2} ... {review_20}"
```

- **Tối đa 20 reviews** mỗi business (truncate nếu nhiều hơn)
- **Bỏ qua** business không có review nào
- **Max sequence length:** 256 tokens (giới hạn của MiniLM)
- **Chỉ lấy** businesses ở Philadelphia → **14,567 businesses**

**Output:** `business_embeddings.pkl` — `{business_id: np.ndarray(384,)}`

### 2.2 User Embedding (`srcs/graph/embedding_users.py`)

```
text(u) = "Name: {name} | Average rating: {avg_stars} stars over {review_count} reviews
           | Fans: {fans} | Elite years: {elite}
           | Reviews: {review_1} {review_2} ... {review_10}"
```

- **Tối đa 10 reviews** mỗi user
- Chỉ lấy users có interaction tại Philadelphia → **~211,115 users**

**Output:** `user_embeddings.pkl` — `{user_id: np.ndarray(384,)}`

---

## 3. Phase 2 — Xây dựng đồ thị (Heterogeneous Graph)

### 3.1 User → Business Edges (`srcs/graph/build_user_busines_edges.py`)

Chỉ giữ **positive interactions** (signal rõ ràng rằng user thích business):

$$\text{edge}(u, b) \iff b \in \text{Philadelphia} \;\wedge\; \text{stars}(u, b) \geq 4$$

**Lý do threshold ≥ 4:** Rating 1-3 là negative/neutral, không nên propagate qua GNN vì sẽ làm user embedding gần với business mà user không thích.

### 3.2 User → User Edges (`srcs/graph/build_user_user_edges.py`)

Friendship edge phải qua **3 bộ lọc**:

$$\text{edge}(u, v) \iff \underbrace{u \in \text{Valid}}_{\text{(1) user hợp lệ}} \;\wedge\; \underbrace{v \in \text{Valid}}_{\text{(2) friend hợp lệ}} \;\wedge\; \underbrace{|\text{Items}(u) \cap \text{Items}(v)| > 0}_{\text{(3) có chung ≥1 business}}$$

| Bộ lọc | Mục đích |
|---|---|
| Valid set | User phải có ≥1 interaction ở Philadelphia |
| Co-interaction | Đảm bảo friendship có ý nghĩa (không phải noise) |
| Top-K = 10 | Giới hạn tối đa 10 friends/user để tránh hub nodes |

### 3.3 Business → Business Similar Edges (`srcs/graph/build_similar_edges2.py`)

Dùng **Faiss KNN** trên 384-dim business embeddings:

1. L2-normalize tất cả vectors:

$$\hat{\mathbf{e}}_b = \frac{\mathbf{e}_b}{\|\mathbf{e}_b\|_2}$$

2. Faiss `IndexFlatIP` search top-11 neighbors (bỏ vị trí 0 = chính nó)
3. Chỉ giữ cạnh có cosine similarity > threshold:

$$\text{edge}(b_i, b_j) \iff \cos(\mathbf{e}_{b_i}, \mathbf{e}_{b_j}) > 0.6 \;\wedge\; j \in \text{KNN}_{10}(i)$$

**Threshold 0.6:** Đủ nghiêm để chỉ nối business thực sự tương tự (cùng loại, cùng khu vực, cùng style).

### 3.4 Graph Assembly (`srcs/graph/build_graph_0embedd.py`)

Lắp ráp thành PyG `HeteroData` với **5 loại cạnh**:

| Edge Type | Hướng | Ý nghĩa |
|---|---|---|
| `(user, rates, business)` | user → biz | User đánh giá ≥4 sao |
| `(business, rated_by, user)` | biz → user | **Reverse edge** — cho GNN bi-directional |
| `(user, friends, user)` | user ↔ user | Social connection |
| `(business, similar, business)` | biz → biz | Content similarity |
| `(business, similar_rev, business)` | biz → biz | **Reverse edge** |

**Tại sao cần reverse edges:** LightGCN propagation yêu cầu message đi cả 2 chiều. Nếu chỉ có `user → biz`, business không thể gửi information ngược lại cho user.

**Output:** `graph.pt` — `{data: HeteroData, user2idx: dict, biz2idx: dict}`

### 3.5 Embedding Alignment (`srcs/graph/embedding_graph.py`)

Ghép 384-dim pretrained embeddings vào graph nodes:

$$\mathbf{X}_{\text{user}} \in \mathbb{R}^{211{,}115 \times 384}, \quad \mathbf{X}_{\text{biz}} \in \mathbb{R}^{14{,}567 \times 384}$$

- Duyệt từng node index → tra ngược ra string ID → lấy embedding từ pkl
- **Node thiếu embedding → zero vector** (GNN sẽ học representations từ neighbors)

**Output:** `embedded_graph.pt` (~366MB)

---

## 4. Phase 3 — HeteroLightGCN

### 4.1 Kiến trúc tổng quan

```
Input Features       Linear Projection       K Layers LightGCN        Mean Pool + L2 Norm
[N × 384]      ──►   [N × 128]         ──►   [N × 128] × (K+1)  ──►  [N × 128]
                      (có tham số)            (KHÔNG có tham số)        (final embeddings)
```

### 4.2 Linear Projection (có tham số)

Mỗi node type có projection matrix riêng (không share weights):

$$\mathbf{u}^{(0)} = \text{Dropout}\!\left(\mathbf{W}_u \cdot \mathbf{x}_u\right), \quad \mathbf{W}_u \in \mathbb{R}^{128 \times 384}$$

$$\mathbf{b}^{(0)} = \text{Dropout}\!\left(\mathbf{W}_b \cdot \mathbf{x}_b\right), \quad \mathbf{W}_b \in \mathbb{R}^{128 \times 384}$$

- **Bias = False** (tiêu chuẩn trong LightGCN)
- **Dropout = 0.1** (chỉ khi training)

### 4.3 Symmetric Normalized Adjacency

Hàm `make_norm_adj` xây dựng sparse adjacency matrix chuẩn hóa:

$$\tilde{\mathbf{A}} = \mathbf{D}_{\text{dst}}^{-1/2} \cdot \mathbf{A} \cdot \mathbf{D}_{\text{src}}^{-1/2}$$

Trong đó:
- $\mathbf{A}$ — adjacency matrix (sparse COO format)
- $\mathbf{D}_{\text{src}}[j] = \sum_i \mathbf{A}[i,j]$ — degree của source nodes
- $\mathbf{D}_{\text{dst}}[i] = \sum_j \mathbf{A}[i,j]$ — degree của destination nodes
- Clamp: $d^{-1/2} \leq 10^5$ (tránh division by zero khi degree = 0)

**4 ma trận adjacency được xây dựng:**

| Ma trận | Shape | Ý nghĩa |
|---|---|---|
| $\tilde{\mathbf{A}}_{ub}$ | $(N_b, N_u)$ | Message: user → business |
| $\tilde{\mathbf{A}}_{bu}$ | $(N_u, N_b)$ | Message: business → user |
| $\tilde{\mathbf{A}}_{uu}$ | $(N_u, N_u)$ | Message: user ↔ user (friends) |
| $\tilde{\mathbf{A}}_{bb}$ | $(N_b, N_b)$ | Message: business ↔ business (similar + reverse, merged) |

### 4.4 LightGCN Propagation (KHÔNG CÓ THAM SỐ)

Tại mỗi layer $l = 1, \ldots, L$:

$$\mathbf{u}^{(l)} = \tilde{\mathbf{A}}_{bu} \cdot \mathbf{b}^{(l-1)} + \tilde{\mathbf{A}}_{uu} \cdot \mathbf{u}^{(l-1)}$$

$$\mathbf{b}^{(l)} = \tilde{\mathbf{A}}_{ub} \cdot \mathbf{u}^{(l-1)} + \tilde{\mathbf{A}}_{bb} \cdot \mathbf{b}^{(l-1)}$$

**Giải thích từng thành phần:**
- $\tilde{\mathbf{A}}_{bu} \cdot \mathbf{b}^{(l-1)}$: User nhận thông tin từ businesses đã rate
- $\tilde{\mathbf{A}}_{uu} \cdot \mathbf{u}^{(l-1)}$: User nhận thông tin từ friends
- $\tilde{\mathbf{A}}_{ub} \cdot \mathbf{u}^{(l-1)}$: Business nhận thông tin từ users đã rate nó
- $\tilde{\mathbf{A}}_{bb} \cdot \mathbf{b}^{(l-1)}$: Business nhận thông tin từ similar businesses

> **Đặc điểm LightGCN:** Không có non-linear activation, không có weight matrix tại mỗi layer. Chỉ là message passing qua normalized adjacency → đơn giản nhưng hiệu quả cho collaborative filtering.

### 4.5 Layer Mean Pooling + L2 Normalization

Kết hợp representations từ tất cả layers (kể cả layer 0):

$$\mathbf{h}_u = \text{L2Norm}\!\left(\frac{1}{L+1} \sum_{l=0}^{L} \mathbf{u}^{(l)}\right)$$

$$\mathbf{h}_b = \text{L2Norm}\!\left(\frac{1}{L+1} \sum_{l=0}^{L} \mathbf{b}^{(l)}\right)$$

$$\text{L2Norm}(\mathbf{x}) = \frac{\mathbf{x}}{\|\mathbf{x}\|_2}$$

- **Mean pooling thay vì chỉ lấy layer cuối:** Giữ lại thông tin từ cả local (layer 0) và global (layer L) perspectives.
- **L2-normalize:** Sau khi normalize, inner product = cosine similarity → thuận tiện cho Faiss search.

### 4.6 Training

| Hyperparameter | Giá trị |
|---|---|
| Embedding dimension | 128 |
| Số layers | 3 |
| Dropout | 0.1 |
| Matmul precision | fp32 (CUDA sparse không hỗ trợ fp16) |
| Best epoch | 30 |
| Best loss | L\_user = 0.4361 |

**Output:**
- `user_h.pt` — $[211{,}115 \times 128]$ L2-normalized
- `biz_h.pt` — $[14{,}567 \times 128]$ L2-normalized
- `mappings.pt` — `{user2idx, biz2idx, idx2user, idx2biz}`

---

## 5. Phase 4 — Faiss Index (`srcs/inference/build_faiss_index.py`)

Build `IndexFlatIP` trên business embeddings 128-dim:

| Thông số | Giá trị |
|---|---|
| Index type | `IndexFlatIP` (brute-force, exact) |
| Dimension | 128 |
| Số vectors | 14,567 |
| Search time | < 1ms |

**Tại sao `IndexFlatIP` (không dùng IVF/HNSW):**
- Chỉ 14,567 vectors → brute-force search đã rất nhanh
- Exact search, **không mất recall**
- Không cần training phase cho index
- Inner Product = cosine similarity vì vectors đã L2-normalize

**Output:** `faiss_biz.index`

---

## 6. Phase 5 — Inference Session-aware (`srcs/inference/inference.py`)

### 6.1 Session Embedding — Scaled Dot-Product Attention

Cho user embedding $\mathbf{h}_u$ và $K$ business gần đây $\{\mathbf{h}_{b_1}, \ldots, \mathbf{h}_{b_K}\}$:

**Bước 1:** Tính attention scores

$$\text{score}_k = \frac{\mathbf{h}_{b_k} \cdot \mathbf{h}_u}{\tau}, \quad \tau = 0.1$$

**Bước 2:** Softmax để được attention weights

$$w_k = \frac{\exp(\text{score}_k)}{\sum_{j=1}^{K} \exp(\text{score}_j)}$$

**Bước 3:** Weighted sum tạo session embedding

$$\mathbf{s} = \sum_{k=1}^{K} w_k \cdot \mathbf{h}_{b_k}$$

> **Temperature $\tau = 0.1$** rất thấp → phân bố **sharp**: item nào gần user nhất sẽ chiếm gần hết trọng số. Điều này hợp lý vì trong session, item user vừa xem gần đây nhất thường phản ánh intent hiện tại.

### 6.2 Combine User Profile + Session Context

$$\mathbf{q} = \text{L2Norm}\!\left(\alpha \cdot \mathbf{h}_u + (1 - \alpha) \cdot \mathbf{s}\right)$$

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| $\alpha$ | 0.4 | User profile chiếm 40%, session context chiếm **60%** |

**Ý nghĩa:** Session context được ưu tiên hơn user profile dài hạn — phù hợp với scenario người dùng đang tìm kiếm cụ thể.

### 6.3 Faiss Top-K Search

$$\text{candidates} = \text{argTopK}_{b}\!\left(\mathbf{q} \cdot \mathbf{h}_b\right)$$

- Fetch `topk + |session| + 10` candidates (bù cho filtering)
- **Filter bỏ** businesses đã nằm trong session (tránh recommend lại cái user vừa xem)
- Trả về top-K kèm score, name, categories

### 6.4 Modes

| Mode | Command |
|---|---|
| Session thật | `--user_id="..." --recent="BIZ1,BIZ2"` |
| Fake session | `--user_id="..." --fake_session --session_size 5` |
| Random user | `--random_user --fake_session` |

---

## 7. Phase 6 — Social Re-ranking (`srcs/inference/inference_social.py`)

Bổ sung **friend influence** vào ranking — nếu bạn bè (giống user) đã ghé một business, business đó nên được ưu tiên hơn.

### 7.1 Friend Embeddings & Weights

Cho user $u$ với tập bạn $F(u) = \{f_1, \ldots, f_{|F|}\}$:

**Cosine similarity giữa user và mỗi friend:**

$$\text{sim}(u, f) = \cos(\mathbf{h}_u, \mathbf{h}_f) = \mathbf{h}_u \cdot \mathbf{h}_f$$

(Vì $\mathbf{h}$ đã L2-normalize nên dot product = cosine)

**Softmax để tạo friend weights:**

$$w_f = \frac{\exp\!\left(\text{sim}(u,f) / \tau\right)}{\sum_{f' \in F(u)} \exp\!\left(\text{sim}(u,f') / \tau\right)}, \quad \tau = 0.1$$

> Friend nào có embedding **gần user nhất** (cùng sở thích) sẽ nhận trọng số lớn nhất.

### 7.2 Social Score

$$\text{social}(u, b) = \sum_{f \in F(u)} w_f \cdot \mathbb{I}\!\left[f \text{ rated } b\right]$$

Trong đó $\mathbb{I}[\cdot]$ là indicator function (1 nếu friend $f$ đã rate business $b$, 0 ngược lại).

**Ý nghĩa:** Nếu nhiều bạn bè (đặc biệt là bạn giống user) đã ghé business $b$ → social score cao → $b$ được đẩy lên trong ranking.

### 7.3 Final Re-ranking Formula

$$\boxed{\text{final}(u, b) = (1 - \gamma) \cdot \underbrace{\text{emb\_score}(u, b)}_{\text{cosine similarity}} + \gamma \cdot \underbrace{\text{social}(u, b)}_{\text{friend influence}}}$$

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| $\gamma$ | 0.2 | 80% embedding score + 20% social score |
| $\tau$ (social attention) | 0.1 | Sharp attention trên friends |
| `fetch_k` | $5 \times \text{topk} + |\text{session}|$ | Lấy nhiều candidates hơn cho re-ranking |

### 7.4 Pipeline hoàn chỉnh

```
1. Load user_h, biz_h, Faiss index, friend graph, user history
2. Resolve user → u_vec
3. Build session embedding (attention-weighted mean)
4. Combine: q = L2Norm(α·u_vec + (1-α)·session)
5. Faiss search → 5×topk candidates
6. Compute friend weights (softmax cosine)
7. Compute social_score cho mỗi candidate
8. Re-rank: final = (1-γ)·emb + γ·social
9. Sort by final score → return top-K
```

---

## 8. Tổng hợp Hyperparameters

| Tham số | Giá trị | Phase |
|---|---|---|
| SentenceTransformer | `all-MiniLM-L6-v2` | Text Embedding |
| Text embedding dim | 384 | Text Embedding |
| Max reviews/business | 20 | Business Embedding |
| Max reviews/user | 10 | User Embedding |
| City filter | Philadelphia | Data Filtering |
| Star threshold (U→B edge) | ≥ 4 | Graph Construction |
| Friend top-K | 10 | User-User Edges |
| Co-interaction (U↔U) | ≥ 1 business chung | User-User Edges |
| KNN K (B↔B edges) | 10 | Similar Business Edges |
| Cosine threshold (B↔B) | 0.6 | Similar Business Edges |
| GNN IN\_DIM | 384 | HeteroLightGCN |
| GNN EMBED\_DIM | 128 | HeteroLightGCN |
| GNN NUM\_LAYERS | 3 | HeteroLightGCN |
| GNN DROPOUT | 0.1 | HeteroLightGCN |
| Temperature $\tau$ | 0.1 | Session Attention + Social |
| $\alpha$ (user vs session) | 0.4 | Inference |
| $\gamma$ (social weight) | 0.2 | Social Re-ranking |
| Faiss index type | `IndexFlatIP` | Inference |

### Data Dimensions

| Artifact | Shape | Mô tả |
|---|---|---|
| `business_embeddings.pkl` | {14,567: $\mathbb{R}^{384}$} | Pretrained text embeddings |
| `user_embeddings.pkl` | {211,115: $\mathbb{R}^{384}$} | Pretrained text embeddings |
| `embedded_graph.pt` | ~366MB | HeteroData + 384-dim features |
| `user_h.pt` | [211,115 × 128] | GNN output, L2-normalized |
| `biz_h.pt` | [14,567 × 128] | GNN output, L2-normalized |
| `faiss_biz.index` | 14,567 × 128-d | IndexFlatIP |

---

## 9. Cấu trúc dự án

```
RecSys/
├── README.md
├── srcs/
│   ├── graph/                          # Phase 1-2: Embedding + Graph
│   │   ├── embedding_business.py       # Text → 384-dim business embeddings
│   │   ├── embedding_users.py          # Text → 384-dim user embeddings
│   │   ├── build_user_busines_edges.py # User-Business edges (stars ≥ 4)
│   │   ├── build_user_user_edges.py    # User-User edges (friends + co-interaction)
│   │   ├── build_similar_edges2.py     # Business-Business edges (Faiss KNN)
│   │   ├── build_graph_0embedd.py      # Assemble HeteroData (5 edge types)
│   │   ├── embedding_graph.py          # Align embeddings → embedded_graph.pt
│   │   └── train_graph.ipynb           # HeteroLightGCN training notebook
│   │
│   ├── inference/                      # Phase 4-6: Inference pipeline
│   │   ├── export_embeddings.py        # Checkpoint → user_h.pt, biz_h.pt
│   │   ├── build_faiss_index.py        # biz_h.pt → faiss_biz.index
│   │   ├── inference.py                # Session-aware recommendation
│   │   └── inference_social.py         # + Social score re-ranking
│   │
│   ├── checks/                         # Validation scripts
│   │   ├── check_business_edges.py     # Verify similar edge quality
│   │   └── check_embedding.py          # Verify embedding stats
│   │
│   └── RecSysProject.ipynb             # Main experiment notebook
│
├── outputs/                            # Generated artifacts
│   ├── graph.pt                        # Graph topology
│   ├── embedded_graph.pt               # Graph + features (~366MB)
│   ├── user_h.pt                       # [211K × 128] user embeddings
│   ├── biz_h.pt                        # [14.5K × 128] business embeddings
│   ├── mappings.pt                     # user2idx, biz2idx, idx2user, idx2biz
│   ├── faiss_biz.index                 # Faiss IndexFlatIP
│   ├── ckpt1/best.pt                   # Best model checkpoint
│   └── graph/                          # Edge files (.txt)
│
├── Yelp-JSON/                          # Raw Yelp dataset
│   └── Yelp JSON/yelp_dataset/
│       ├── yelp_academic_dataset_business.json
│       ├── yelp_academic_dataset_review.json
│       ├── yelp_academic_dataset_user.json
│       ├── yelp_academic_dataset_tip.json
│       └── yelp_academic_dataset_checkin.json
│
└── Yelp-Photos/                        # Yelp photos (unused)
```

---

## 10. Cách chạy

### Thứ tự chạy pipeline

```bash
# 1. Text Embedding
python srcs/graph/embedding_business.py
python srcs/graph/embedding_users.py

# 2. Build Edges
python srcs/graph/build_user_busines_edges.py
python srcs/graph/build_user_user_edges.py
python srcs/graph/build_similar_edges2.py

# 3. Assemble Graph
python srcs/graph/build_graph_0embedd.py
python srcs/graph/embedding_graph.py

# 4. Train (Jupyter notebook)
# Mở và chạy srcs/graph/train_graph.ipynb

# 5. Export Embeddings
python srcs/inference/export_embeddings.py

# 6. Build Faiss Index
python srcs/inference/build_faiss_index.py

# 7. Inference
python srcs/inference/inference.py --random_user --fake_session --topk 20
python srcs/inference/inference_social.py --random_user --fake_session --topk 20
```

### Inference options

```bash
# Session thật (cung cấp business IDs):
python srcs/inference/inference.py \
  --user_id="USER_ID" \
  --recent="BIZ_ID1,BIZ_ID2,BIZ_ID3" \
  --topk 20

# Fake session (tự sample từ lịch sử user):
python srcs/inference/inference.py \
  --user_id="USER_ID" \
  --fake_session --session_size 5

# Random user + fake session:
python srcs/inference/inference.py \
  --random_user --fake_session

# Với social re-ranking:
python srcs/inference/inference_social.py \
  --random_user --fake_session \
  --gamma 0.2 --topk 20
```
