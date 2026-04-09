import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { AppProvider } from "./context/AppContext";

// Import Pages
import Header from "./components/Header";
import Dashboard from "./pages/Dashboard";
import BusinessDetail from "./pages/BusinessDetail";
import Profile from "./pages/Profile";


function App() {
  return (
    <AppProvider>
      <Router>
        <Header />
        
        

        <main>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/business/:id" element={<BusinessDetail />} />
            <Route path="/profile" element={<Profile />} />
            {/* Vẫn giữ route này để vào bằng link trực tiếp */}
          </Routes>
        </main>
      </Router>
    </AppProvider>
  );
}

export default App;
