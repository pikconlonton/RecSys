"""
Tạo cạnh User → Business (interaction edges) cho hệ thống RecSys.

╔════════════════════════════════════════════════════════════════════╗
║  MỤC TIÊU:                                                       ║
║  Từ Yelp review data → tạo file edge (user_id, business_id)      ║
║  chỉ giữ lại positive interactions (stars ≥ 4)                    ║
║                                                                    ║
║  PIPELINE:                                                         ║
║  1. Load business JSON → lọc chỉ Philadelphia                    ║
║  2. Duyệt toàn bộ reviews (~7M dòng), giữ review nếu:           ║
║     - business_id thuộc Philadelphia                               ║
║     - stars ≥ 4 (positive preference signal)                      ║
║  3. Ghi edge: user_id \t business_id                              ║
║                                                                    ║
║  TẠI SAO STARS ≥ 4:                                               ║
║  - Rating 4-5 = user thực sự thích → positive signal cho GNN     ║
║  - Rating 1-3 = implicit negative → không tạo edge               ║
║  - Nếu giữ tất cả rating, model bị confuse giữa like/dislike     ║
║                                                                    ║
║  INPUT:  yelp_academic_dataset_business.json                       ║
║          yelp_academic_dataset_review.json                         ║
║  OUTPUT: edges_user_business.txt  (format: user_id\tbusiness_id)  ║
╚════════════════════════════════════════════════════════════════════╝
"""

import pandas as pd
import json


def load_json_lines(path):
    """
    Đọc file JSON dạng line-delimited (JSONL).
    Yelp dataset lưu mỗi dòng là 1 JSON object riêng biệt.
    Returns: pd.DataFrame
    """
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1: Lọc business thuộc Philadelphia
# ══════════════════════════════════════════════════════════════════════════════
# Load toàn bộ business → filter theo city
# Chỉ giữ Philadelphia để giảm graph size (phù hợp T4 16GB)
business = load_json_lines('/content/yelp_academic_dataset_business.json')

philly_business = business[business['city'] == 'Philadelphia']

# Tạo set để lookup O(1) khi duyệt review
philly_business_ids = set(philly_business['business_id'])

num_business = len(philly_business_ids)
print("Số business Philadelphia:", num_business)


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2: Duyệt review → tạo edge (user, business)
# ══════════════════════════════════════════════════════════════════════════════
def build_user_business_edges(review_path, business_ids, output_path):
    """
    Quét toàn bộ review file (single-pass, stream processing).

    Với mỗi review:
      - Kiểm tra business_id có thuộc Philadelphia không (set lookup O(1))
      - Kiểm tra stars ≥ 4 (positive interaction)
      - Nếu cả 2 OK → ghi edge (user_id, business_id)

    Lưu ý:
      - Một user có thể review nhiều business → nhiều edges
      - Một user có thể review 1 business nhiều lần → duplicate edges
        (sẽ được xử lý ở bước build graph nếu cần)
      - File output dạng TSV (tab-separated values)
    """
    count = 0

    with open(review_path, 'r', encoding='utf-8') as f, \
         open(output_path, 'w') as out:

        for i, line in enumerate(f):
            data = json.loads(line)

            # Điều kiện: business ở Philadelphia VÀ rating ≥ 4
            if data['business_id'] in business_ids and data['stars'] >= 4:
                u = data['user_id']
                b = data['business_id']

                out.write(f"{u}\t{b}\n")
                count += 1

            # Progress log mỗi 500K dòng (file review rất lớn ~7M dòng)
            if i % 500000 == 0:
                print("Processed:", i)

    print("Total edges:", count)


build_user_business_edges(
    review_path='/content/yelp_academic_dataset_review.json',
    business_ids=philly_business_ids,
    output_path='/content/edges_user_business.txt'
)