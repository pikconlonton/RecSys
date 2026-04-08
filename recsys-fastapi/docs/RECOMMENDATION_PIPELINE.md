# Recommendation Pipeline (FE → Logs → Session Buffer → RecSys)

Tài liệu này mô tả chi tiết pipeline khi FE gọi API để lấy recommendation: hệ thống đọc log hành vi, dựng session ngắn hạn, chạy thuật toán (Faiss/embeddings nếu có) và trả dữ liệu cho FE.

## Mục tiêu

- FE biết **khi nào cần gửi log** và **gọi recommendation**.
- Đội dự án thống nhất **contract dữ liệu** (input/output).
- Giải thích rõ 2 chế độ chạy:
  - **Real inference** (embeddings + Faiss) khi đã load được artefacts trong `outputs/`
  - **Fallback heuristic** khi thiếu artefacts hoặc mapping không khớp

---

## Tổng quan luồng chạy

### 1) FE gửi log hành vi

Endpoint:
- `POST /logs/`

Payload (chuẩn cho recommender):

```json
{
  "user_id": "123",
  "action": "view",
  "business_id": "<yelp_business_id>",
  "timestamp": "2026-04-08T12:30:00Z"
}
```

Điểm quan trọng:
- **`action="view"`** là event chính được dùng để build session.
- **`business_id`** phải là id ổn định (string), lý tưởng là Yelp `business_id` trong dataset và trong artefacts `mappings.pt`.
- `timestamp` có thể omit, BE sẽ tự set thời gian hiện tại (UTC).

BE sẽ lưu vào DB qua `app/db/crud.py::create_log`.

### 2) FE gọi recommendation

Endpoint:
- `GET /recommendations/{user_id}?topk=10`

BE gọi `recommender_service.recommend(...)` để tạo danh sách `items`.

---

## Session buffer là gì?

Session buffer là một danh sách ngắn các item (business) mà user vừa tương tác gần đây.

Implementation hiện tại:
- `RecommenderService(recent_log_limit=10)` (mặc định)
- Khi recommend cho một user, BE đọc **tối đa 10 log mới nhất** của user đó:
  - dùng `crud.get_recent_logs(db, limit=10, user_id=user_id)`
  - nếu vì lý do nào đó query theo `user_id` rỗng nhưng DB vẫn có data, BE fallback lấy global recent logs (để service không “trắng” hẳn)

Dựng session views:
- `recent_views = [log.business_id for log in logs if log.action == "view"]`

Nghĩa là: chỉ những log có `action == "view"` mới tham gia vào session.

---

## Chế độ 1: Real inference (embeddings + Faiss)

### Điều kiện

Real inference chạy khi:
- Service load được artefacts (embeddings + index) lúc startup
- và `user_id` lookup được trong `mappings.pt` (đúng mapping)

Artefacts được load **một lần lúc startup** trong `app/main.py`:
- `load_artefacts()` đọc:
  - `outputs/user_h.pt`
  - `outputs/biz_h.pt`
  - `outputs/mappings.pt`
  - `outputs/faiss_biz.index`

Cấu hình đường dẫn:
- Env var: `RECSYS_OUTPUTS_DIR=/path/to/RecSys/outputs`
- Nếu không set env var, code fallback theo monorepo: `RecSys/outputs`

### User ID mapping (rất quan trọng)

Trong code hiện tại:
- API nhận `user_id` là **string** (path param và ở log payload)
- Khi lookup trong artefacts, BE dùng **`user_id`** (string) để tra `artefacts.user2idx`

Vì vậy để production chạy “real inference” đúng:
- `user2idx` trong artefacts phải chứa key trùng với `user_id` mà FE gửi.

### Thuật toán thực thi (tóm tắt)

Nằm trong `app/services/inference_runtime.py::recommend_from_recent_views`.

Inputs:
- `user_id` (string key để lookup user embedding)
- `recent_business_ids` từ session buffer
- `topk`

Steps:
1) Lookup user embedding `u_vec` từ `artefacts.user_h`
2) Map `recent_business_ids` → indices bằng `artefacts.biz2idx`
3) Nếu session có item hợp lệ:
   - Lấy embedding các item gần đây `recent_embeds`
   - Tính **attention-weighted mean** với query là `u_vec`
   - Kết hợp user embedding và session embedding:
     - `q = alpha * user + (1-alpha) * session`
     - normalize
4) Nếu session trống / business_id không map được:
   - fallback query = normalized user embedding
5) Chạy Faiss search (IndexFlatIP) để lấy candidates
6) Lọc bỏ các item đã xuất hiện trong session (filter theo `business_id`)
7) Trả ra danh sách topK

Các hyperparams hiện tại:
- `alpha = 0.4`
- `temperature = 0.1`

### Output format (items)

Mỗi item trả về có dạng:

```json
{
  "rank": 1,
  "type": "business",
  "business_id": "<yelp_business_id>",
  "score": 0.8123,
  "generated_at": "2026-04-08T12:31:00.000000",
  "metadata": {
    "name": "Pizza Palace",
    "stars": 4.5,
    "review_count": 624,
    "categories": "Pizza, Italian",
    "address": "123 Market St",
    "lat": 39.9526,
    "lng": -75.1652
  }
}
```

Notes:
- `metadata` được join từ bảng `businesses` trong DB.
- Nếu chưa có metadata cho `business_id` thì `metadata` sẽ là `null`.

### Upsert business metadata

Endpoint:
- `POST /businesses/upsert`

Payload:

```json
[
  {
    "business_id": "biz_abc",
    "name": "Pizza Palace",
    "stars": 4.5,
    "review_count": 624,
    "categories": "Pizza, Italian",
    "address": "123 Market St",
    "lat": 39.9526,
    "lng": -75.1652
  }
]
```

Giải thích:
- `score` là inner product (cosine-like nếu embeddings đã normalize) từ Faiss.
- `generated_at` được gắn thêm ở layer `RecommenderService` để FE debug.

---

## Chế độ 2: Fallback heuristic (khi thiếu artefacts hoặc mapping fail)

Nếu service không load được artefacts (thiếu file/dependency) hoặc chưa tích hợp mapping chuẩn, hệ thống sẽ fallback.

Heuristic hiện tại:
- Đếm tần suất `business_id` trong recent logs có `action=="view"`
- Sort giảm dần
- `score` = số lần xuất hiện

Format item vẫn giống (rank/type/business_id/score/generated_at) để FE không cần đổi.

---

## Response contract của `GET /recommendations/{user_id}`

Response 200:

```json
{
  "user_id": "123",
  "topk": 5,
  "items": [
    {
      "rank": 1,
      "type": "business",
      "business_id": "biz_abc",
      "score": 0.81,
  "generated_at": "2026-04-08T12:31:00.000000",
  "metadata": null
    }
  ]
}
```

Notes:
- `items` có thể rỗng nếu:
  - user chưa có log nào
  - hoặc user_id không có trong mapping (real inference) và recent logs không đủ để heuristic tạo recs

---

## FE Best Practices

### Khi nào gửi log?

- Gửi `action="view"` khi user mở trang detail business.
- Có thể gửi thêm `click`, `like`, `bookmark`… nhưng hiện recommender chỉ dùng `view` làm session buffer.

### Tần suất gọi recommendation

- Thường dùng theo page load / refresh UI.
- Nếu muốn “real-time” hơn, có thể gọi lại sau khi gửi log view (nhưng nên debounce).

### Đảm bảo ID thống nhất

- `business_id` FE gửi phải trùng với id mà recommender hiểu (trong `mappings.pt`).
- Nếu FE dùng id nội bộ khác, cần một lớp mapping trước khi gửi lên BE.

---

## Troubleshooting nhanh

### 1) API trả items rỗng

Kiểm tra:
- DB có log `view` cho user chưa?
- `business_id` có map được trong `mappings.pt` không?
- `user_id` có trong `user2idx` không? (hiện lookup theo `str(user_id)`)
- `user_id` có trong `user2idx` không? (hiện lookup theo string `user_id`)

### 2) Service chạy nhưng không có real inference

- Kiểm tra env `RECSYS_OUTPUTS_DIR`
- Kiểm tra file trong outputs có đủ: `user_h.pt`, `biz_h.pt`, `mappings.pt`, `faiss_biz.index`
- Nếu thiếu dependency (`faiss`, `torch`) thì service vẫn chạy nhưng rec chỉ là heuristic.
