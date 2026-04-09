# Yelp RecSys UI (Frontend)

Đây là giao diện người dùng (Frontend) cho hệ thống Gợi ý địa điểm Yelp RecSys, được xây dựng bằng ReactJS. Giao diện này cho phép người dùng duyệt các địa điểm, xem chi tiết, và nhận các gợi ý cá nhân hóa từ hệ thống Recommendation System ở Backend.

## Tech Stack

Dự án Frontend được phát triển với các công nghệ chính sau:

- **React 19**: Thư viện JavaScript hàng đầu để xây dựng giao diện người dùng.
- **React Router 7**: Quản lý định tuyến (routing) trong ứng dụng Single Page Application (SPA).
- **Tailwind CSS 4**: Một framework CSS utility-first giúp xây dựng giao diện nhanh chóng và linh hoạt.
- **Leaflet Map & React-Leaflet**: Thư viện bản đồ tương tác, được sử dụng để hiển thị vị trí các địa điểm.
- **Sass**: Bộ tiền xử lý CSS, giúp viết CSS có tổ chức và mạnh mẽ hơn.
- **Axios**: HTTP client dùng để giao tiếp với Backend API.

## Hướng dẫn cài đặt (Installation)

Để chạy dự án Frontend trên môi tính của bạn, làm theo các bước sau:

1.  **Cài đặt Dependencies**:
    Mở terminal trong thư mục gốc của dự án `yelp-recsys-fe` và chạy lệnh:

```bash
npm install
```

2.  **Kiểm tra kết nối Backend**:
    Đảm bảo rằng `baseURL` trong file `src/api/axiosClient.js` trỏ đúng đến địa chỉ của Backend API. Mặc định đang là `http://localhost:8000`.

````javascript
 // c:\Recomen\yelp-recsys-fe\src\api\axiosClient.js
 const axiosClient = axios.create({
   baseURL: 'http://localhost:8000', // Đảm bảo URL này đúng với Backend của bạn
   headers: {
     'Content-Type': 'application/json',
   },
 });
 ```

3.  **Khởi chạy ứng dụng**:
Sau khi cài đặt xong và kiểm tra cấu hình Backend, bạn có thể khởi chạy ứng dụng Frontend bằng lệnh:
```bash
npm start
````

Ứng dụng sẽ mở trên trình duyệt tại `http://localhost:3000`.

## Thông tin hệ thống API (Backend)

Frontend giao tiếp với Backend FastAPI tại `http://localhost:8000`. Dưới đây là các endpoint chính mà Frontend sử dụng:

- **Tham số Query**:
  - `topk`: Số lượng gợi ý muốn lấy (ví dụ: `topk=20`).
  - `use_social`: `true` để bật tính năng re-ranking dựa trên tương tác xã hội.
  - `gamma`: Trọng số cho social re-ranking (ví dụ: `gamma=0.2`).
- **Mô tả**: Endpoint này trả về một danh sách các `item`, mỗi `item` bao gồm `business_id`, `score`, và `metadata` chứa các thông tin chi tiết của quán ăn (Name, Address, Stars, Lat, Lng). Frontend sẽ trích xuất thông tin chi tiết quán từ `metadata` này.

- **Body (JSON)**:
  ```json
  {
    "user_id": "string",
    "business_id": "string",
    "action": "view", // Hoặc các hành vi khác như 'click', 'like'
    "timestamp": "ISO-8601 string (optional)"
  }
  ```
- **Mô tả**: Frontend gửi log `action: 'view'` mỗi khi người dùng truy cập trang chi tiết của một quán ăn. Dữ liệu này được Backend sử dụng để huấn luyện và cải thiện mô hình gợi ý.

## Luồng dữ liệu (Data Flow)

### Trang chi tiết quán ăn (`src/pages/BusinessDetail.jsx`)

Khi người dùng truy cập một trang chi tiết quán ăn (ví dụ: `/business/some_business_id`), component `BusinessDetail.jsx` sẽ thực hiện các bước sau:

1.  **Lấy ID từ URL**: Component sử dụng `useParams()` từ `react-router-dom` để trích xuất `business_id` từ URL.

2.  **Gửi View Log**: Ngay lập tức, một `POST` request được gửi đến endpoint `/logs/` của Backend. Request này chứa `user_id` của người dùng hiện tại, `business_id` của quán vừa xem, `action: 'view'`, và `timestamp`. Đây là cơ chế quan trọng để thu thập dữ liệu hành vi người dùng, phục vụ cho việc huấn luyện và cập nhật mô hình Recommendation System.

3.  **Lấy dữ liệu chi tiết và gợi ý**: Frontend gọi `GET /recommendations/{user_id}` với `topk` đủ lớn (ví dụ `topk=20`) và `use_social=true`, `gamma=0.2`.

- Từ danh sách gợi ý trả về, Frontend tìm kiếm quán ăn có `business_id` trùng với ID từ URL để hiển thị thông tin chi tiết của quán đó.
- Các quán ăn còn lại trong danh sách gợi ý được sử dụng để hiển thị mục "Có thể bạn cũng thích".

## Lưu ý kỹ thuật

````javascript
import L from 'leaflet';
import icon from 'leaflet/dist/images/marker-icon.png';
import iconShadow from 'leaflet/dist/images/marker-shadow.png';

 let DefaultIcon = L.icon({
   iconUrl: icon,
   shadowUrl: iconShadow,
   iconSize:,
   iconAnchor:
 });
 L.Marker.prototype.options.icon = DefaultIcon;
 ```

-   **Cấu hình Tailwind CSS 4**: Với Tailwind CSS phiên bản 4, quá trình cấu hình đã được đơn giản hóa đáng kể. Không cần file `tailwind.config.js` truyền thống trong nhiều trường hợp, hoặc nó sẽ có cấu trúc tối giản hơn. Tailwind 4 tập trung vào JIT (Just-In-Time) mode mặc định, giúp tối ưu hóa CSS đầu ra. Đảm bảo rằng bạn đang sử dụng cú pháp và quy trình build phù hợp với Tailwind 4.

---


````

<!--
[PROMPT_SUGGESTION]Could you review the `BusinessDetail.jsx` file for any potential performance improvements or best practices?[/PROMPT_SUGGESTION]
[PROMPT_SUGGESTION]I need to add a new feature to allow users to "like" a business. How should I modify the frontend and what backend API calls would be involved?[/PROMPT_SUGGESTION]
