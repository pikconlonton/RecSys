"""
Kiểm tra chất lượng similar business edges.

Chọn random 1 edge từ similar_business_edges2.txt,
in thông tin chi tiết 2 businesses để so sánh thủ công
(tên, city, categories, stars, review_count, attributes).

Dùng để verify: 2 businesses được nối edge có thực sự tương tự không.
"""
import json
import pandas as pd
#check 2 Business Embedding Vector 
def load_json_lines(path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return pd.DataFrame(data)

business_df = load_json_lines('Yelp-JSON\Yelp JSON\yelp_dataset\yelp_academic_dataset_business.json')

# convert thành dict cho lookup nhanh
business_dict = {
    row['business_id']: row
    for _, row in business_df.iterrows()
}
import random

edges = []
with open('similar_business_edges2.txt', 'r') as f:
    for line in f:
        parts = line.strip().split('\t')
        b1, b2 = parts[0], parts[1]
        score = float(parts[2]) if len(parts) > 2 else None
        edges.append((b1, b2, score))

# chọn random 1 edge
b1, b2, score = random.choice(edges)

print("Selected edge:")
print(b1, "<->", b2, "| score:", score)
def print_business_info(b):
    info = business_dict.get(b)

    if info is None:
        print("Not found:", b)
        return

    print("="*50)
    print("Name:", info.get('name'))
    print("City:", info.get('city'))
    print("Categories:", info.get('categories'))
    print("Stars:", info.get('stars'))
    print("Review count:", info.get('review_count'))
    print("Attributes:", info.get('attributes'))

print("\nBusiness 1:")
print_business_info(b1)

print("\nBusiness 2:")
print_business_info(b2)