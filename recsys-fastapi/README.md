# RecSys FastAPI

FastAPI backend cho dự án Recommendation System.

Service cung cấp:
- Gửi/đọc logs hành vi user (lưu DB).
- Gọi recommendations theo `user_id`.
- (Mới) Upsert metadata cho business để **enrich** output recommendations với `items[].metadata`.

## Docs cho FE

- FE calling guide: [`docs/FE_API_GUIDE.md`](docs/FE_API_GUIDE.md)
- Recommendation pipeline: [`docs/RECOMMENDATION_PIPELINE.md`](docs/RECOMMENDATION_PIPELINE.md)

## API tóm tắt

> `user_id` hiện là **string** end-to-end.

- `GET /` health check
- `POST /logs/` ghi log
- `GET /logs/recent/` đọc 10 logs gần nhất
- `GET /recommendations/{user_id}?topk=10` lấy recommendations (mỗi item có thêm `metadata` nếu có)
- `POST /businesses/upsert` upsert business metadata

## Chạy local

```bash
uvicorn app.main:app --reload
```

Mở swagger:
- `http://127.0.0.1:8000/docs`

## Env vars quan trọng

- `DATABASE_URL` (ví dụ Postgres):
   - `postgresql+psycopg2://user:pass@host:5432/dbname`
- `RECSYS_OUTPUTS_DIR` (optional): trỏ tới thư mục `outputs/` để bật real inference (torch + faiss).

## Testing

```bash
pytest -q
```