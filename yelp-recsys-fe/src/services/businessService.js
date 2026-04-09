import axiosClient from "../api/axiosClient";

export const businessService = {
  // Lấy chi tiết 1 quán theo ID
  getById: (id) => {
    const url = `/businesses/${id}`;
    return axiosClient.get(url);
  },
  getAll: (params = { skip: 0, limit: 100 }) => {
    const url = `/businesses/`;
    return axiosClient.get(url, { params });
  },
};

export const recService = {
  // GET /recommendations/{user_id}
  getForUser: (userId, params = { topk: 10, use_social: true, gamma: 0.2 }) => {
    const url = `/recommendations/${userId}`;
    return axiosClient.get(url, { params });
  },
};
