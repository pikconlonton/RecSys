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
  "user_id": 123,
  "action": "view",
  "business_id": "biz_abc",
  "timestamp": "2026-04-08T12:30:00Z"
}
```

#### Field rules

- `user_id` (int): id user.
- `action` (string): tên hành vi. Hiện recommender dùng mạnh nhất là `"view"`.
- `business_id` (string): id business/item.
- `timestamp` (string ISO-8601, optional): nếu FE không gửi thì BE tự set thời gian hiện tại.

### Response 200

Trả về log đã lưu (có thêm `id`).

```json
{
  "id": 1,
  "user_id": 123,
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
  "user_id": 123,
  "topk": 5,
  "items": [
    {
      "rank": 1,
      "type": "business",
      "business_id": "biz_abc",
      "score": 3.0,
      "generated_at": "2026-04-08T12:31:00.000000"
    }
  ]
}
```

#### Ý nghĩa `items[]`

- Hiện tại BE dùng heuristic đơn giản: đếm `business_id` xuất hiện nhiều nhất trong các log gần đây có `action == "view"`.
- `score` hiện là số lần xuất hiện (float).

> Khi bạn tích hợp thuật toán recsys thật (Faiss/embeddings) thì format vẫn nên giữ ổn định để FE không phải đổi nhiều.

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
    user_id: 123,
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
