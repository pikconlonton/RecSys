import React, { useContext, useEffect, useState } from 'react';
import { AppContext } from '../context/AppContext';
import axiosClient from '../api/axiosClient'; // Hoặc đường dẫn tới file axios của bạn

const UserSelector = () => {
    const { selectedUser, setSelectedUser } = useContext(AppContext);
    // LUÔN khởi tạo là mảng rỗng để không bị lỗi .map()
    const [users, setUsers] = useState([]); 
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchUsers = async () => {
            try {
                const response = await axiosClient.get('/users/?skip=0&limit=100');
                
                // Kiểm tra xem dữ liệu nằm ở response hay response.data
                const data = response?.data || response;

                // CHỈ setUsers nếu data thực sự là một mảng
                if (Array.isArray(data)) {
                    setUsers(data);
                    if (!selectedUser && data.length > 0) {
                        setSelectedUser(data[0]);
                    }
                } else {
                    console.error("Dữ liệu API trả về không phải mảng:", data);
                    setUsers([]); 
                }
            } catch (err) {
                console.error("Lỗi API:", err);
                setUsers([]); 
            } finally {
                setLoading(false);
            }
        };
        fetchUsers();
    }, []);

    if (loading) return <div className="md-select-pill">Loading...</div>;

    return (
        <div className="user-selector-wrapper">
            <select
                value={selectedUser?.user_id || ""}
                onChange={(e) => {
                    const user = users.find(u => u.user_id === e.target.value);
                    if (user) setSelectedUser(user);
                }}
                className="md-select-pill"
            >
                <option value="" disabled>-- Chọn người dùng --</option>
                {/* Sử dụng optional chaining để an toàn tuyệt đối */}
                {users?.map(u => (
                    <option key={u.user_id} value={u.user_id}>
                        {u.name}
                    </option>
                ))}
            </select>
        </div>
    );
};
export default UserSelector;