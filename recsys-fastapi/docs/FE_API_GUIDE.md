# FE API Guide (RecSys FastAPI)

Tài liệu này dành cho team FE để gọi API của service `recsys-fastapi`.

## Recommendation pipeline (chi tiết)

Nếu bạn cần hiểu "call API vào thì hệ thống tính recommendation như nào" (logs → session buffer → Faiss/heuristic → response), xem thêm:

- `docs/RECOMMENDATION_PIPELINE.md`

## Base URL

- Local: `http://localhost:8000`
- Nếu chạy qua Docker Compose / deploy: dùng domain/host tương ứng của môi trường.

> Lưu ý: Service đang bật CORS `allow_origins=["*"]` nên FE gọi trực tiếp được (dev).

---

## 0) (Dev) Seed dữ liệu SQLite để demo nhanh

Nếu bạn muốn vừa chạy API lên là có sẵn dữ liệu để thử (users/logs/business/social), backend có script seed SQLite.

### Yêu cầu

- Đã tạo venv và cài deps trong `recsys-fastapi/`.
- Có file dataset edges: `../outputs/graph/edges_user_business.txt` (đã có sẵn trong monorepo này).

### Seed DB

Chạy trong thư mục `recsys-fastapi/`:

```bash
source .venv/bin/activate

# Seed vào file sqlite ./test.db (khuyến nghị để không đè app.db nếu bạn đang dùng)
DATABASE_URL=sqlite:///./test.db python scripts/seed_db.py
```

Script sẽ:

- recreate schema (drop/create)
- seed `users` (basic profile), `logs` (view), `businesses` (metadata tối thiểu), `social_friends`, `social_interactions`
- in ra `main_user_for_social` và `boost_business_id` để bạn demo social nhanh

### Chạy API dùng DB vừa seed

```bash
source .venv/bin/activate
DATABASE_URL=sqlite:///./test.db uvicorn app.main:app --reload
```

Mở swagger:

- `http://127.0.0.1:8000/docs`

### Gọi nhanh để kiểm tra

- `GET /logs/recent/` (phải trả về list 10 logs)
- `GET /recommendations/{main_user_for_social}?topk=10&use_social=true&gamma=0.7`

> Lưu ý: social re-ranking là best-effort, chỉ có tác dụng khi candidate list có overlap với các business mà friends đã tương tác.

---

## 1) Health check

### `GET /`

**Mục đích**: kiểm tra service up.

**Response 200**

```json
{ "message": "Welcome to the User Action Logging API!" }
```

---

## 2) Gửi log hành vi user

### `POST /logs/`

**Mục đích**: FE gửi event (ví dụ user view business) để BE lưu DB.

### Request body

```json
{
  "user_id": "123",
  "action": "view",
  "business_id": "biz_abc",
  "timestamp": "2026-04-08T12:30:00Z"
}
```

#### Field rules

- `user_id` (string): id user.
- `action` (string): tên hành vi. Hiện recommender dùng mạnh nhất là `"view"`.
- `business_id` (string): id business/item.
- `timestamp` (string ISO-8601, optional): nếu FE không gửi thì BE tự set thời gian hiện tại.

### Response 200

Trả về log đã lưu (có thêm `id`).

```json
{
  "id": 1,
  "user_id": "123",
  "business_id": "biz_abc",
  "action": "view",
  "timestamp": "2026-04-08T12:30:00Z"
}
```

### Response 422 (Validation error)

Khi thiếu field bắt buộc hoặc sai type.

---

## 3) Lấy 10 log gần nhất

### `GET /logs/recent/`

**Mục đích**: debug/monitor hoặc BE side dùng để phân tích nhanh.

### Response 200

```json
[
  {
    "id": 10,
    "user_id": 123,
    "business_id": "biz_abc",
    "action": "view",
    "timestamp": "2026-04-08T12:30:00Z"
  }
]
```

### Response 404

Khi chưa có log.

```json
{ "detail": "No logs found" }
```

---

## 4) Lấy recommendations cho 1 user

### `GET /recommendations/{user_id}?topk=10`

**Mục đích**: FE gọi để lấy danh sách gợi ý theo user.

- `user_id` (path param): id user.
- `topk` (query param, optional, default = 10): số lượng item muốn lấy.

### Example

`GET /recommendations/123?topk=5`

### Response 200

```json
{
  "user_id": "123",
  "topk": 5,
  "items": [
    {
      "rank": 1,
      "type": "business",
      "business_id": "biz_abc",
      "score": 3.0,
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
  ]
}
```

> `metadata` có thể là `null` nếu hệ thống chưa có thông tin business trong DB.

---

## 5) Business metadata & APIs

### 5.1 Cấu trúc bảng `businesses` trong DB

Trong DB, backend có bảng `businesses` để lưu metadata của từng business:

- `business_id` (string, PK)
- `name` (string, nullable)
- `stars` (float, nullable)
- `review_count` (int, nullable)
- `categories` (string, nullable)
- `address` (string, nullable)
- `lat` (float, nullable)
- `lng` (float, nullable)
- `updated_at` (datetime)

> Lưu ý: thông tin user chi tiết (name/email/avatar) được lưu ở bảng `users` (xem thêm mục **7)**).
> Các bảng `logs`, `social_friends`, `social_interactions` chỉ lưu `user_id` dạng string để tham chiếu tới user.

### 5.2 Upsert business metadata

`POST /businesses/upsert`

**Mục đích**: FE/backoffice đẩy thông tin business vào DB để BE join/enrich output recommendations.

Request body – danh sách business (array):

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

Response 200:

```json
{ "processed": 1 }
```

### 5.3 Lấy danh sách business

`GET /businesses/`

**Mục đích**: FE/backoffice liệt kê business (đơn giản, phục vụ list/selector, debug, v.v.).

Query params:

- `skip` (int, default = 0) – offset bản ghi
- `limit` (int, default = 100) – số bản ghi tối đa trả về

Response 200:

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
    "lng": -75.1652,
    "updated_at": "2026-04-09T10:00:00Z"
  }
]
```

### 5.4 Lấy chi tiết 1 business

`GET /businesses/{business_id}`

**Mục đích**: FE/backoffice lấy chi tiết metadata cho 1 business (thường dùng ở trang detail).

Path param:

- `business_id` (string)

Response 200:

```json
{
  "business_id": "biz_abc",
  "name": "Pizza Palace",
  "stars": 4.5,
  "review_count": 624,
  "categories": "Pizza, Italian",
  "address": "123 Market St",
  "lat": 39.9526,
  "lng": -75.1652,
  "updated_at": "2026-04-09T10:00:00Z"
}
```

Response 404:

```json
{ "detail": "Business not found" }
```

---

## 6) Social re-ranking (FE gửi social graph + social interactions)

Backend hỗ trợ social re-ranking dựa trên dữ liệu FE gửi lên.

### 6.1 Upsert friend list

`POST /social/friends/upsert`

```json
{
  "user_id": "u1",
  "friends": ["u2", "u3"],
  "timestamp": "2026-04-09T10:00:00Z"
}
```

Response:

```json
{ "user_id": "u1", "processed": 2 }
```

### 6.2 Gửi social interactions (friends → business)

`POST /social/interactions`

```json
{
  "user_id": "u2",
  "business_id": "biz_abc",
  "action": "visit",
  "weight": 1.0,
  "timestamp": "2026-04-09T10:02:00Z"
}
```

`action` khuyến nghị: `view`, `visit`, `like`, `rate` (BE sẽ map default weight nếu FE không gửi `weight`).

### 6.3 Bật social re-ranking khi lấy recommendation

`GET /recommendations/{user_id}?topk=10&use_social=true&gamma=0.2`

Khi bật social:
- `score` sẽ trở thành `final_score`
- response có thêm trường `scoring` để debug:

```json
{
  "rank": 1,
  "type": "business",
  "business_id": "biz_abc",
  "score": 0.8547,
  "generated_at": "2026-04-09T10:03:00.000000",
  "metadata": null,
  "scoring": {
    "emb_score": 0.9684,
    "social_score": 0.4070,
    "final_score": 0.8547,
    "gamma": 0.2
  }
}
```

#### Ý nghĩa `items[]`

- Hiện tại BE dùng heuristic đơn giản: đếm `business_id` xuất hiện nhiều nhất trong các log gần đây có `action == "view"`.
- `score` hiện là số lần xuất hiện (float).

> Khi bạn tích hợp thuật toán recsys thật (Faiss/embeddings) thì format vẫn nên giữ ổn định để FE không phải đổi nhiều.

---

## 7) User profiles & APIs

Backend cung cấp bảng `users` và một bộ API đơn giản để quản lý thông tin cơ bản của user.

### 7.1 Cấu trúc bảng `users` trong DB

- `user_id` (string, PK)
- `name` (string, nullable)
- `email` (string, nullable)
- `avatar_url` (string, nullable)
- `created_at` (datetime)
- `updated_at` (datetime)

Các bảng khác (`logs`, `social_friends`, `social_interactions`) reference user thông qua field `user_id` (string) này.

### 7.2 Tạo user mới

`POST /users/`

Request body:

```json
{
  "user_id": "u123",
  "name": "Nguyen Van A",
  "email": "u123@example.com",
  "avatar_url": "https://example.com/avatar.png"
}
```

Response 200:

```json
{
  "user_id": "u123",
  "name": "Nguyen Van A",
  "email": "u123@example.com",
  "avatar_url": "https://example.com/avatar.png",
  "created_at": "2026-04-09T10:00:00Z",
  "updated_at": "2026-04-09T10:00:00Z"
}
```

> Lưu ý: service hiện không enforce unique email; uniqueness được đảm bảo bởi `user_id`.

### 7.3 Lấy danh sách user

`GET /users/`

Query params:

- `skip` (int, default = 0)
- `limit` (int, default = 100)

Response 200:

```json
[
  {
    "user_id": "u123",
    "name": "Nguyen Van A",
    "email": "u123@example.com",
    "avatar_url": "https://example.com/avatar.png",
    "created_at": "2026-04-09T10:00:00Z",
    "updated_at": "2026-04-09T10:00:00Z"
  }
]
```

### 7.4 Lấy chi tiết 1 user

`GET /users/{user_id}`

Response 200:

```json
{
  "user_id": "u123",
  "name": "Nguyen Van A",
  "email": "u123@example.com",
  "avatar_url": "https://example.com/avatar.png",
  "created_at": "2026-04-09T10:00:00Z",
  "updated_at": "2026-04-09T10:00:00Z"
}
```

Response 404:

```json
{ "detail": "User not found" }
```

### 7.5 Cập nhật user

`PUT /users/{user_id}`

Request body (partial update – chỉ field có gửi mới được cập nhật):

```json
{
  "name": "Nguyen Van B",
  "email": "new_email@example.com"
}
```

Response 200: giống schema user detail, với giá trị mới.

Response 404 khi `user_id` không tồn tại.

### 7.6 Xoá user

`DELETE /users/{user_id}`

Response 200:

```json
{ "deleted": true }
```

Response 404 khi user không tồn tại:

```json
{ "detail": "User not found" }
```

---

## FE Integration notes

### Nên gửi log khi nào?

- Khi user **view** business detail → gửi `action="view"` + `business_id=<id>`
- (Tuỳ bạn) click, add_to_cart, like… cũng có thể gửi, nhưng recommender hiện ưu tiên `view`.

### Debounce / batching

- Nếu FE bắn log quá dày (scroll/list view), nên debounce/throttle hoặc batch để giảm load.

### Id mapping

- `business_id` nên là id ổn định (string) thống nhất với data/embeddings phía recsys.

---

## Optional: ví dụ gọi bằng JavaScript (fetch)

```js
// POST /logs/
await fetch('http://localhost:8000/logs/', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
  user_id: '123',
    action: 'view',
    business_id: 'biz_abc',
    timestamp: new Date().toISOString(),
  }),
});

// GET /recommendations/{user_id}
const res = await fetch('http://localhost:8000/recommendations/123?topk=10');
const data = await res.json();
console.log(data.items);
```
