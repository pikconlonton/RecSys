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

#### 4.6.1 Dataset Design

Hai `Dataset` class chạy song song mỗi epoch:

| Dataset | Mô tả | Sample format |
|---|---|---|
| `SimDataset` | Positive pairs business–business từ `edges_business_business_similar.txt` | `(anchor_idx, pos_idx)` |
| `UserDataset` | Triplet user–business từ `edges_user_business.txt` + negative sampling | `(user_idx, biz_pos, biz_neg)` |

**SimDataset** — đơn giản: mỗi cặp $(b_i, b_j)$ trong file similar edges trở thành một positive pair. Không cần negative sampling riêng vì InfoNCE dùng **in-batch negatives**.

**UserDataset** — phức tạp hơn, negative sampling phụ thuộc `hard_prob` (float $\in [0, 1]$):

```
Với mỗi (user, biz_pos):
  if random() < hard_prob:
      # Hard negative: chọn biz tương tự biz_pos mà user chưa rate
      candidates = hard_neg_map[biz_pos] \ user_pos[user]
      biz_neg = random.choice(candidates)   # fallback random nếu rỗng
  else:
      # Random negative: chọn biz bất kỳ ∉ user_pos[user]
      biz_neg = random_biz ∉ user_pos[user]
```

`hard_neg_map` ban đầu được xây từ text-similarity tĩnh (`edges_business_business_similar.txt`), sau đó được **rebuild động** từ embedding hiện tại mỗi `REBUILD_NEG_EVERY` epoch (xem mục 4.6.3).

Cả hai dataset dùng chung `DataLoader` với `batch_size=4096`, `shuffle=True`, `drop_last=True`.

#### 4.6.2 Loss Functions

**1. InfoNCE Loss (Similarity Head)**

Dùng cho `SimDataset` — kéo embedding business tương tự lại gần nhau:

$$
\mathcal{L}_{\text{sim}} = -\frac{1}{B} \sum_{i=1}^{B} \log \frac{\exp(\text{sim}(\mathbf{z}_i^a, \mathbf{z}_i^p) / \tau)}{\sum_{j=1}^{B} \exp(\text{sim}(\mathbf{z}_i^a, \mathbf{z}_j^p) / \tau)}
$$

- $\mathbf{z}^a, \mathbf{z}^p$: output của `sim_head` (MLP 128→128→128, ReLU), đã L2-normalize
- $\tau = 0.07$ — temperature thấp → phân phối sắc nét, buộc model phân biệt rõ
- **In-batch negatives**: tất cả $B-1$ samples khác trong batch đóng vai trò negative → không cần sample negative riêng
- Labels = diagonal: `labels = torch.arange(B)`, dùng `F.cross_entropy`

**2. BPR Loss (User–Business)**

Dùng cho `UserDataset` — buộc $\text{score}(u, b^+) > \text{score}(u, b^-)$:

$$
\mathcal{L}_{\text{user}} = -\frac{1}{B} \sum_{i=1}^{B} \ln \sigma\!\left(\mathbf{h}_{u_i}^\top \mathbf{h}_{b_i^+} - \mathbf{h}_{u_i}^\top \mathbf{h}_{b_i^-}\right)
$$

- $\mathbf{h}_u, \mathbf{h}_b$: output chính của HeteroLightGCN (mean pooling + L2-normalize)
- Dot product = cosine similarity (vì đã normalize)
- `F.logsigmoid` cho numerical stability

**Tổng loss mỗi step:**

$$
\mathcal{L} = \lambda_{\text{sim}} \cdot \mathcal{L}_{\text{sim}} + \lambda_{\text{user}} \cdot \mathcal{L}_{\text{user}}
$$

| Hệ số | Giá trị | Ý nghĩa |
|---|---|---|
| $\lambda_{\text{sim}}$ | 0.5 | Trọng số InfoNCE (auxiliary) |
| $\lambda_{\text{user}}$ | 1.0 | Trọng số BPR (primary) |

#### 4.6.3 Training Strategy — Curriculum Learning + Dynamic Hard Negatives

**Curriculum Schedule (3 giai đoạn):**

```
Epoch  1–10  │ random      │ hard_prob = 0.0  → chỉ random negatives
Epoch 11–30  │ mixed       │ hard_prob tăng tuyến tính 0.0 → 1.0
Epoch 31–80  │ hard        │ hard_prob = 1.0  → chỉ hard negatives
```

Giai đoạn **mixed** tránh oscillation bằng cách flip **per-sample** (mỗi sample tự coin-flip theo `hard_prob`) thay vì flip toàn bộ epoch.

**Dynamic Hard Negative Rebuild:**

Mỗi `REBUILD_NEG_EVERY = 10` epoch (từ epoch 11 trở đi):

1. Forward pass lấy `biz_h` hiện tại (detach, no grad)
2. Build `faiss.IndexFlatIP` trên `biz_h` (đã L2-normalize)
3. Tìm `top_k = 15` nearest neighbors cho mỗi business
4. Cập nhật `hard_neg_map` → `UserDataset` dùng negative mới

→ Hard negatives luôn **adaptive** theo embedding hiện tại, không bị stale từ text-similarity tĩnh ban đầu.

**LR Schedule:**

| Sự kiện | LR |
|---|---|
| Khởi tạo | $10^{-3}$ |
| StepLR decay | $\times 0.8$ mỗi 15 epoch |
| Warm restart tại epoch 31 (hard phase) | $3 \times 10^{-4}$ |

**Gradient clipping:** `max_norm = 1.0` — ngăn gradient explosion khi chuyển sang hard negatives.

**Training loop mỗi epoch:**

```
1. Single forward pass toàn graph → user_h, biz_h, user_proj, biz_proj
2. Lặp N = min(len(sim_loader), len(user_loader), 128) steps:
     a. Lấy batch từ sim_loader  → tính L_sim  (InfoNCE)
     b. Lấy batch từ user_loader → tính L_user (BPR)
     c. Cộng dồn: accum_loss += λ_sim × L_sim + λ_user × L_user
3. Backward trên accum_loss / N (1 lần backward duy nhất)
4. Gradient clipping → optimizer step → scheduler step
5. Nếu epoch % 10 == 0 và đã qua mixed phase → rebuild hard_neg_map
6. Save best.pt nếu L_user giảm; save epoch checkpoint mỗi 5 epoch
```

> **Lưu ý kiến trúc:** Forward pass chạy **1 lần / epoch** trên toàn graph (full-batch GNN propagation), sau đó loss được tính trên mini-batches bằng cách index vào embedding table. Điều này tiết kiệm rất nhiều computation so với forward lại mỗi step.

#### 4.6.4 Hyperparameters

| Hyperparameter | Giá trị |
|---|---|
| Embedding dimension | 128 |
| Số layers | 3 |
| Dropout | 0.1 |
| Batch size | 4,096 |
| Max steps / epoch | 128 |
| Epochs | 80 |
| Optimizer | Adam |
| Initial LR | $10^{-3}$ |
| LR decay | $\times 0.8$ / 15 epochs (StepLR) |
| LR hard restart | $3 \times 10^{-4}$ tại epoch 31 |
| Temperature $\tau$ | 0.07 |
| $\lambda_{\text{sim}}$ | 0.5 |
| $\lambda_{\text{user}}$ | 1.0 |
| Grad clip max\_norm | 1.0 |
| AMP | ON (CUDA), OFF (CPU) |
| Matmul precision | fp32 (CUDA sparse không hỗ trợ fp16) |
| Hard neg rebuild interval | 10 epochs |
| Hard neg top\_k | 15 |
| Best epoch | 30 |
| Best loss | $\mathcal{L}_{\text{user}} = 0.4361$ |

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

### 7.1 Động lực

Embedding score từ Phase 5 chỉ dựa trên **sở thích cá nhân** (user profile + session). Tuy nhiên, hành vi thực tế cho thấy người dùng thường bị ảnh hưởng bởi **bạn bè có cùng gu**. Social Re-ranking khai thác tín hiệu này: nếu những người bạn giống mình đã ghé một business, business đó đáng được ưu tiên hơn.

Ý tưởng:
- Không phải mọi bạn bè đều có ảnh hưởng ngang nhau — bạn **giống mình** (trong embedding space) thì ý kiến **quan trọng hơn**
- Social score là tín hiệu **bổ sung**, không thay thế embedding score → kết hợp bằng trọng số $\gamma$

### 7.2 Tổng quan pipeline

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  ① Load artefacts: user_h, biz_h, Faiss index, friend graph, history   │
│                         │                                                │
│  ② Resolve user ──► u_vec = user_h[u]                                   │
│                         │                                                │
│  ③ Session embedding ──► s = attention_weighted_mean(recent, u_vec)     │
│                         │                                                │
│  ④ Combine ──► q = L2Norm(α·u_vec + (1-α)·s)                          │
│                         │                                                │
│  ⑤ Faiss search(q, 5×K) ──► ~100 candidates (nhiều hơn để re-rank)    │
│                         │                                                │
│  ⑥ Social score ──► với mỗi candidate b:                               │
│     │  Lấy friends F(u) từ social graph                                 │
│     │  Tính w_f = softmax(cos(h_u, h_f) / τ) cho mỗi friend           │
│     │  social(u,b) = Σ w_f × I(f đã ghé b)                            │
│                         │                                                │
│  ⑦ Re-rank ──► final = (1-γ)·emb_score + γ·social_score               │
│                         │                                                │
│  ⑧ Sort by final ──► return Top-K                                      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Bước 1-4: Session-aware query (giống Phase 5)

Các bước 1-4 **hoàn toàn giống** `inference.py` (xem [Phase 5](#6-phase-5--inference-session-aware)):

$$\mathbf{q} = \text{L2Norm}\!\left(\alpha \cdot \mathbf{h}_u + (1 - \alpha) \cdot \mathbf{s}\right), \quad \alpha = 0.4$$

### 7.4 Bước 5: Faiss search — lấy nhiều candidates

Thay vì lấy đúng top-K như Phase 5, ở đây lấy **5×K candidates**:

$$\text{candidates} = \text{argTop}_{5K}\!\left(\mathbf{q} \cdot \mathbf{h}_b\right)$$

**Lý do:** Social re-ranking có thể đẩy business từ vị trí #50 lên top-10 nếu nhiều bạn bè đã ghé. Cần pool candidates đủ lớn để re-ranking có ý nghĩa.

### 7.5 Bước 6: Tính Social Score

#### 7.5.1 Friend Weights — ai quan trọng hơn ai?

Cho user $u$ với tập bạn $F(u) = \{f_1, \ldots, f_{|F|}\}$. Trước tiên tính **mức độ tương đồng** giữa user và từng friend:

$$\text{sim}(u, f) = \cos(\mathbf{h}_u, \mathbf{h}_f) = \mathbf{h}_u \cdot \mathbf{h}_f$$

(Dot product = cosine vì $\mathbf{h}$ đã L2-normalize)

Sau đó **softmax với temperature** để chuyển thành trọng số xác suất:

$$w_f = \frac{\exp\!\left(\text{sim}(u,f) \;/\; \tau\right)}{\sum_{f' \in F(u)} \exp\!\left(\text{sim}(u,f') \;/\; \tau\right)}, \quad \tau = 0.1$$

**Ý nghĩa temperature $\tau = 0.1$:** Phân bố rất **sharp** — friend giống user nhất chiếm phần lớn trọng số, friend xa trong embedding space gần như bị bỏ qua. Nếu $\tau = 1.0$ thì trọng số sẽ đều hơn giữa tất cả friends.

#### 7.5.2 Social Score — "bạn giống mình đã thích gì?"

$$\text{social}(u, b) = \sum_{f \in F(u)} w_f \cdot \mathbb{I}\!\left[f \text{ rated } b\right]$$

Trong đó $\mathbb{I}[\cdot]$ là **indicator function**: bằng 1 nếu friend $f$ đã tương tác với business $b$ (rating ≥ 4), bằng 0 ngược lại.

**Ví dụ minh họa:**

```
User A có 3 bạn với trọng số:
  Bạn B: w = 0.50  (rất giống A)
  Bạn C: w = 0.30  (khá giống)
  Bạn D: w = 0.20  (ít giống)

Business X: B đã ghé ✓, D đã ghé ✓  → social(A, X) = 0.50 + 0.20 = 0.70
Business Y: C đã ghé ✓              → social(A, Y) = 0.30
Business Z: không ai ghé             → social(A, Z) = 0.00
```

→ Business X được boost mạnh nhất vì bạn B (giống user nhất) đã ghé.

#### 7.5.3 Edge cases

| Trường hợp | Xử lý |
|---|---|
| User không có friend | social = 0 cho tất cả → kết quả = embedding only |
| Có friends nhưng không ai ghé candidates | social = 0 → không ảnh hưởng ranking |
| Random user selection | Ưu tiên user có ≥ 3 friends để social score có ý nghĩa |

### 7.6 Bước 7: Final Re-ranking

Kết hợp 2 tín hiệu bằng **weighted linear combination**:

$$\boxed{\text{final}(u, b) = (1 - \gamma) \cdot \underbrace{\cos(\mathbf{q},\; \mathbf{h}_b)}_{\text{embedding score}} \;+\; \gamma \cdot \underbrace{\text{social}(u, b)}_{\text{friend influence}}}$$

| Tham số | Giá trị | Ý nghĩa |
|---|---|---|
| $\gamma$ | 0.2 | **80% embedding + 20% social** — social là tín hiệu phụ |
| $\tau$ | 0.1 | Sharp attention: chỉ bạn giống nhất mới có ảnh hưởng |
| `fetch_k` | $5K + |\text{session}|$ | Pool candidates lớn cho re-ranking |

**Tại sao $\gamma = 0.2$ (không cao hơn):**
- Embedding score đã capture collaborative filtering qua GNN → tín hiệu chính
- Social score là **complementary signal** — bổ sung nhưng không override
- $\gamma$ quá cao → hệ thống trở thành "bạn bè quyết định" thay vì "sở thích cá nhân"
- Trong thực nghiệm, $\gamma = 0.2$ đã giúp spread score từ ~0.006 (embedding only) lên ~0.06, tạo ra sự phân biệt rõ ràng hơn giữa các candidates

### 7.7 Output

Kết quả hiển thị **3 cột score** cho mỗi business:

```
  Rank  Final     Emb       Social    Name                                    Categories
  ──────────────────────────────────────────────────────────────────────────────────────
  1     0.8547    0.9684    0.4070    The Belgian Café                        Bars, Belgian
  2     0.8234    0.9672    0.2450    Green Eggs Café                         Breakfast
  3     0.7801    0.9751       -      Reading Terminal Market                 Food Hall
  ...
```

- **Final**: Score cuối cùng (đã kết hợp embedding + social)
- **Emb**: Cosine similarity thuần từ Faiss
- **Social**: Mức "friend influence" ( `-` = 0, không có bạn nào ghé)

Summary cuối output cho biết bao nhiêu kết quả trong top-K được boost bởi social score.

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

Pipeline đi qua 3 giai đoạn biến đổi kích thước:

```
Raw text ──► SentenceTransformer ──► 384-dim ──► GNN projection ──► 128-dim ──► Faiss search
```

#### Pretrained embeddings (384-dim) — output Phase 1

| Artifact | Type | Entries | Vector dim | Mô tả |
|---|---|---|---|---|
| `business_embeddings.pkl` | dict | 14,567 | 384 | Text embedding mỗi business (name + categories + reviews) |
| `user_embeddings.pkl` | dict | ~211,115 | 384 | Text embedding mỗi user (profile + reviews) |

#### Graph data — output Phase 2

| Artifact | Size | Nội dung |
|---|---|---|
| `graph.pt` | ~50MB | HeteroData topology (5 edge types) + user2idx + biz2idx |
| `embedded_graph.pt` | ~366MB | graph.pt + user.x $[211{,}115 \times 384]$ + business.x $[14{,}567 \times 384]$ |

#### Trained embeddings (128-dim) — output Phase 3-4

| Artifact | Shape | Dtype | Mô tả |
|---|---|---|---|
| `user_h.pt` | $[211{,}115 \times 128]$ | float32, L2-norm | GNN-trained user embeddings |
| `biz_h.pt` | $[14{,}567 \times 128]$ | float32, L2-norm | GNN-trained business embeddings |
| `mappings.pt` | dict (4 keys) | — | user2idx, biz2idx, idx2user, idx2biz |
| `faiss_biz.index` | 14,567 vectors × 128-d | IndexFlatIP | Brute-force cosine search index |
| `ckpt1/best.pt` | — | — | Model checkpoint (epoch 30, L\_user=0.4361) |

#### Edge files — output Phase 2

| File | Format | Mô tả |
|---|---|---|
| `edges_user_business.txt` | `user_id \t biz_id` | Positive interactions (stars ≥ 4) |
| `edges_user_user.txt` | `user_id \t user_id` | Filtered friendships (co-interaction) |
| `edges_business_business_similar.txt` | `biz_id \t biz_id \t score` | KNN similar (cosine > 0.6) |

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
