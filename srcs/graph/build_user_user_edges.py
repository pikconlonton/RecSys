"""
Tạo cạnh User → User (social / friendship edges).

╔════════════════════════════════════════════════════════════════════╗
║  MỤC TIÊU:                                                       ║
║  Từ Yelp user data → tạo social edges giữa users                ║
║  Có 3 điều kiện lọc để đảm bảo chất lượng edge:               ║
║                                                                    ║
║  LỌC 1: User u phải có interaction ở Philadelphia                ║
║     → loại user không liên quan đến dataset                     ║
║  LỌC 2: Friend v cũng phải có interaction ở Philadelphia         ║
║     → loại friend "chết" không có dữ liệu                       ║
║  LỌC 3: Co-interaction: u và v phải có ≥ 1 business chung      ║
║     → đảm bảo friendship có ý nghĩa (cùng sở thích)             ║
║                                                                    ║
║  Top-k=10: giới hạn số bạn mỗi user để tránh graph quá dense   ║
║                                                                    ║
║  INPUT:  edges_user_business.txt (đã tạo từ bước trước)          ║
║          yelp_academic_dataset_user.json                           ║
║  OUTPUT: edges_user_user.txt  (format: user_id\tuser_id)          ║
╚════════════════════════════════════════════════════════════════════╝
"""

from collections import defaultdict
import json


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 1: Load user → business interactions
# Dùng để biết user nào đã tương tác và kiểm tra co-interaction
# ══════════════════════════════════════════════════════════════════════════════
def build_user2items(edge_path):
    """
    Đọc file edge user→business → dict {user_id: set(business_ids)}.
    Set dùng để check co-interaction nhanh: items_u & items_v (O(min(|A|,|B|))).
    """
    user2items = defaultdict(set)

    with open(edge_path, 'r') as f:
        for line in f:
            u, b = line.strip().split('\t')
            user2items[u].add(b)

    print("Num users with interactions:", len(user2items))
    return user2items


user2items = build_user2items('/content/edges_user_business.txt')


# ══════════════════════════════════════════════════════════════════════════════
# BƯỚC 2: Build social edges với 3 điều kiện lọc
# ══════════════════════════════════════════════════════════════════════════════
def build_user_user_edges_from_edgefile(user_path, user2items, output_path, top_k=10):
    """
    Tạo social edges giữa users với 3 điều kiện lọc:

    Điều kiện 1: User u phải có interaction ở Philadelphia (u ∈ valid_users)
    Điều kiện 2: Friend v cũng phải có interaction (v ∈ valid_users)
    Điều kiện 3: Co-interaction: |items_u ∩ items_v| > 0
                User và friend phải có ít nhất 1 business chung

    Top-k: chỉ lấy 10 friends đầu tiên từ friend list của Yelp.
    Lý do: giới hạn degree của mỗi node → tránh graph quá dense,
    và ưu tiên friends được Yelp xếp trước (có thể là thân hơn).

    Lưu ý: Yelp lưu friends dưới dạng comma-separated string,
    ví dụ: "abc123, def456, ghi789"
    """
    count = 0
    valid_users = set(user2items.keys())  # Chỉ user đã có interaction

    with open(user_path, 'r', encoding='utf-8') as f, \
         open(output_path, 'w') as out:

        for i, line in enumerate(f):
            data = json.loads(line)
            u = data['user_id']

            # Điều kiện 1: user phải có interaction ở Philadelphia
            if u not in valid_users:
                continue

            friends = data.get('friends', '')
            if not friends:
                continue

            items_u = user2items[u]

            # Lấy top-k friends (giới hạn degree để graph không quá dày)
            friend_list = friends.split(',')[:top_k]

            for v in friend_list:
                v = v.strip()

                # Điều kiện 2: friend cũng phải có interaction ở Philadelphia
                if v not in valid_users:
                    continue

                # Điều kiện 3: co-interaction — có ít nhất 1 business chung
                # Góp phần đảm bảo edge có tín hiệu: 2 user này thực sự
                # có sở thích chung, không chỉ là bạn trên Yelp
                if len(items_u & user2items[v]) > 0:
                    out.write(f"{u}\t{v}\n")
                    count += 1

            # Progress log mỗi 500K dòng (file user rất lớn)
            if i % 500000 == 0:
                print("Processed:", i, "| edges:", count)

    print("Total user-user edges:", count)


build_user_user_edges_from_edgefile(
    user_path='/content/yelp_academic_dataset_user.json',
    user2items=user2items,
    output_path='/content/edges_user_user.txt',
    top_k=10
)