"""
Tạo embedding 384-dim cho mỗi business bằng SentenceTransformer.

╔════════════════════════════════════════════════════════════════════╗
║  PIPELINE:                                                         ║
║  1. Load business JSON → lọc Philadelphia                         ║
║  2. Thu thập reviews cho mỗi business (tối đa 20 reviews)         ║
║  3. Tổng hợp text = name + categories + attributes + reviews     ║
║  4. Encode bằng all-MiniLM-L6-v2 → vector 384-dim               ║
║  5. Lưu vào business_embeddings.pkl                                ║
║                                                                    ║
║  TẠI SAO DÙNG all-MiniLM-L6-v2:                                   ║
║  - Nhỏ (80MB), nhanh, phù hợp T4 16GB                           ║
║  - Output 384-dim: đủ expressive mà không quá lớn               ║
║  - Đã pre-train trên 1B+ sentence pairs                          ║
║                                                                    ║
║  TẠI SAO TỔNG HỢP TEXT (không chỉ review):                      ║
║  - Reviews: nội dung chi tiết, cảm nhận thực tế               ║
║  - Categories: loại hình kinh doanh (Restaurant, Bar, ...)       ║
║  - Attributes: đặc điểm cụ thể (WiFi, parking, ...)              ║
║  - Name: tên thương hiệu                                          ║
║  → Kết hợp tất cả cho embedding toàn diện nhất                 ║
║                                                                    ║
║  INPUT:  yelp_academic_dataset_business.json                       ║
║          yelp_academic_dataset_review.json                         ║
║  OUTPUT: business_embeddings.pkl                                   ║
║          Format: {business_id_str: np.ndarray(384,)}              ║
╚════════════════════════════════════════════════════════════════════╝
"""

import json
import pandas as pd
from collections import defaultdict
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: Đọc Yelp JSONL
# ══════════════════════════════════════════════════════════════════════════════
def load_json_lines(path):
    """Load file JSON line-delimited (JSONL) → DataFrame."""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

business_df = load_json_lines('Yelp-JSON\Yelp JSON\yelp_dataset\yelp_academic_dataset_business.json')

# Lọc chỉ Philadelphia để giảm khối lượng (14,567 business)
philly_business = business_df[business_df['city'] == 'Philadelphia']
business_ids = set(philly_business['business_id'])

del business_df  # Giải phóng RAM (DataFrame gốc rất lớn)


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2: Thu thập reviews cho mỗi business
# Tối đa 20 reviews/business để giới hạn độ dài text
# ══════════════════════════════════════════════════════════════════════════════
def collect_reviews(review_path, business_ids, max_reviews=20):
    """
    Stream qua toàn bộ file review, giữ lại text của reviews
    thuộc Philadelphia businesses.

    max_reviews=20: giới hạn số review/business để:
      - Text không quá dài (SentenceTransformer có max_seq_length=256 tokens)
      - Tiết kiệm RAM
      - 20 reviews đủ đại diện cho sentiment và thông tin business

    Returns: {business_id: [review_text_1, review_text_2, ...]}
    """
    reviews = defaultdict(list)

    with open(review_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            b = data['business_id']

            if b in business_ids:
                if len(reviews[b]) < max_reviews:
                    reviews[b].append(data['text'])

    return reviews

reviews_dict = collect_reviews(
    'Yelp-JSON\Yelp JSON\yelp_dataset\yelp_academic_dataset_review.json',
    business_ids
)


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 3: Build text representation cho mỗi business
# Kết hợp: name + categories + attributes + reviews → 1 đoạn text
# ══════════════════════════════════════════════════════════════════════════════
def build_business_text(row, reviews):
    """
    Tạo text representation cho 1 business bằng cách kết hợp:
    - Name: tên thương hiệu
    - Categories: loại hình (Restaurant, Bar, Nightlife, ...)
    - Attributes: đặc điểm (WiFi, parking, price range, ...)
    - Rating + review count: mức độ phổ biến
    - Reviews: nội dung review thực tế của khách hàng

    Text này sẽ được SentenceTransformer encode thành vector 384-dim.
    Lưu ý: model có max_seq_length=256 tokens, text dài quá sẽ bị truncate.
    Đây là trade-off: nhiều thông tin hơn nhưng có thể mất phần cuối.
    """
    name = row.get('name', '')
    categories = row.get('categories', '') or ''
    stars = row.get('stars', 0)
    review_count = row.get('review_count', 0)

    # attributes là dict trong Yelp JSON, ví dụ:
    # {"WiFi": "free", "BikeParking": "True", "RestaurantsPriceRange2": "2"}
    # Convert sang string để SentenceTransformer hiểu được
    attrs = row.get('attributes')
    if isinstance(attrs, dict):
        attr_text = ", ".join([f"{k}:{v}" for k, v in attrs.items()])
    else:
        attr_text = ""

    # Nối tất cả reviews thành 1 đoạn text dài
    review_text = " ".join(reviews)

    text = f"""
    Name: {name}
    Categories: {categories}
    Attributes: {attr_text}
    Rating: {stars} stars with {review_count} reviews
    Reviews: {review_text}
    """

    return text

# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 4: Encode text → 384-dim vector bằng SentenceTransformer
# all-MiniLM-L6-v2: nhỏ (80MB), nhanh, output 384-dim
# Dùng GPU (cuda:0) để tăng tốc (T4 16GB dư dả)
# ══════════════════════════════════════════════════════════════════════════════
model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda:0')

business_embeddings = {}

# Duyệt từng business, build text, encode
# Business không có review sẽ bị skip (không đủ thông tin để embed)
for _, row in tqdm(philly_business.iterrows(), total=len(philly_business)):
    b_id = row['business_id']
    reviews = reviews_dict.get(b_id, [])

    if len(reviews) == 0:
        continue  # Bỏ qua business không có review

    text = build_business_text(row, reviews)

    # Encode: text → np.ndarray(384,)
    # show_progress_bar=False vì đã có tqdm bên ngoài
    vec = model.encode(
        text,
        show_progress_bar=False
    )

    business_embeddings[b_id] = vec

print("Total embedded:", len(business_embeddings))


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 5: Lưu định dạng pickle
# Format: {business_id_str: np.ndarray(384,)}
# ══════════════════════════════════════════════════════════════════════════════
with open('business_embeddings.pkl', 'wb') as f:
    pickle.dump(business_embeddings, f)

print("Saved business embeddings")