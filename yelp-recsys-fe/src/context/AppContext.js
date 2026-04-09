import React, { createContext, useState, useEffect } from "react";

export const AppContext = createContext();

export const AppProvider = ({ children }) => {
  const [selectedUser, setSelectedUser] = useState(null);
  const [history, setHistory] = useState([]);

  // 1. Logic khi đổi người dùng: Load lịch sử riêng
  useEffect(() => {
    if (selectedUser?.user_id) {
      console.log(`🔄 Hệ thống: Đang nạp dữ liệu cho User ${selectedUser.name}`);
      
      const saved = localStorage.getItem(`history_${selectedUser.user_id}`);
      
      // FIX: Chốt chặn lỗi "undefined" hoặc dữ liệu hỏng
      if (saved && saved !== "undefined") {
        try {
          setHistory(JSON.parse(saved));
        } catch (e) {
          console.error("Dữ liệu history bị hỏng, reset về mảng rỗng");
          setHistory([]);
        }
      } else {
        setHistory([]);
      }
    }
  }, [selectedUser?.user_id]);

  // 2. Lưu lịch sử theo từng User ID
  useEffect(() => {
    // Chỉ lưu nếu có user và history có dữ liệu hoặc vừa được dọn dẹp (mảng rỗng)
    if (selectedUser?.user_id) {
      localStorage.setItem(`history_${selectedUser.user_id}`, JSON.stringify(history));
    }
  }, [history, selectedUser?.user_id]);

  return (
    <AppContext.Provider
      value={{ 
        selectedUser, 
        setSelectedUser, 
        history, 
        setHistory 
      }}
    >
      {children}
    </AppContext.Provider>
  );
};