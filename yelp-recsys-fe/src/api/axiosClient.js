import axios from 'axios';

const axiosClient = axios.create({
  baseURL: 'http://localhost:8000', // Khi nào dùng ngrok thì đổi ở đây duy nhất 1 chỗ
  headers: {
    'Content-Type': 'application/json',
  },
});

// Tự động lấy data từ response của axios để FE dùng cho gọn
axiosClient.interceptors.response.use(
  (response) => response.data,
  (error) => Promise.reject(error)
);

export default axiosClient;