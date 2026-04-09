import React, { useContext, useEffect, useState } from 'react';
import { AppContext } from '../context/AppContext';
import { recService } from '../services/businessService'; // ← Import service xịn
import BusinessCard from '../components/BusinessCard';
import HeroSection from '../components/HeroSection';
import SectionContainer from '../components/SectionContainer';
import './Dashboard.scss';

// --- Constants (Giữ nguyên của Huy) ---
const CATEGORY_ITEMS = [
    { id: 1, name: 'Cà phê', icon: 'fa-mug-hot' },
    { id: 2, name: 'Trà sữa', icon: 'fa-leaf' },
    { id: 3, name: 'Ăn vặt', icon: 'fa-ice-cream' },
    { id: 4, name: 'Quán nhậu', icon: 'fa-beer-mug-empty' },
    { id: 5, name: 'Hẹn hò', icon: 'fa-martini-glass-citrus' },
    { id: 6, name: 'Mua sắm', icon: 'fa-bag-shopping' },
    { id: 7, name: 'Làm đẹp', icon: 'fa-spa' },
    { id: 8, name: 'Giải trí', icon: 'fa-gamepad' },
];

// --- Sub-components (Giữ nguyên giao diện đẹp của Huy) ---
const CategoryGrid = () => (
    <section className="dashboard__categories" aria-label="Danh mục">
        <div className="dashboard__categories-grid">
            <div className="dashboard__category-feature">
                <i className="fa-solid fa-utensils dashboard__category-feature-icon" aria-hidden="true" />
                <h3 className="dashboard__category-feature-title">Nhà hàng &<br />Ẩm thực</h3>
            </div>
            <div className="dashboard__category-items">
                {CATEGORY_ITEMS.map(({ id, name, icon }) => (
                    <button key={id} className="dashboard__category-item" aria-label={name}>
                        <i className={`fa-solid ${icon} dashboard__category-item-icon`} aria-hidden="true" />
                        <span className="dashboard__category-item-label">{name}</span>
                    </button>
                ))}
            </div>
        </div>
    </section>
);

const PromoBanner = () => (
    <div className="dashboard__promo" role="banner" aria-label="Ưu đãi đặc quyền">
        <img
            src="https://images.unsplash.com/photo-1555396273-367ea4eb4db5?auto=format&fit=crop&w=1200&q=80"
            alt="Không gian nhà hàng sang trọng"
            className="dashboard__promo-img"
            loading="lazy"
        />
        <div className="dashboard__promo-overlay" aria-hidden="true" />
        <div className="dashboard__promo-content">
            <span className="dashboard__promo-eyebrow">Ưu đãi đặc quyền</span>
            <h2 className="dashboard__promo-title">Trở thành hội viên được ưu đãi</h2>
        </div>
    </div>
);

// --- Main Component ---
const Dashboard = () => {
    const { selectedUser } = useContext(AppContext);
    const [trendingList, setTrendingList] = useState([]);
    const [socialRecs, setSocialRecs] = useState([]);
    const [personalRecs, setPersonalRecs] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchDashboardData = async () => {
            setLoading(true);
            // FIX 1: Dùng user_id cho đúng với Object từ API /users/
            const userId = selectedUser?.user_id || "---r61b7EpVPkb4UVme5tA";

            try {
                // FIX 2: Tăng topk lên 20 để đủ chia cho 3 Section
                const res = await recService.getForUser(userId, {
                    topk: 10, 
                    use_social: false,
                    gamma: 0.2
                });

                // Axios trả về dữ liệu trong .data, nếu đã có interceptor thì dùng trực tiếp res
                const data = res?.data || res;

                if (data && data.items) {
                    const allItems = data.items;

                    // 1. Dành riêng cho bạn (Top 8 AI)
                    setPersonalRecs(allItems.slice(0, 8));

                    // 2. Bạn bè đề xuất (Lọc score social > 0)
                    const socialOnly = allItems.filter(item => item.scoring?.social_score > 0);
                    // Nếu không có ai trong bạn bè đi đâu, lấy 4 thằng tiếp theo trong list để tránh trống UI
                    setSocialRecs(socialOnly.length > 0 ? socialOnly.slice(0, 4) : allItems.slice(8, 12));

                    // 3. Khám phá xu hướng (Lấy phần còn lại)
                    setTrendingList(allItems.slice(12, 16));
                }
            } catch (error) {
                console.error("Lỗi Dashboard:", error);
            } finally {
                setLoading(false);
            }
        };

        fetchDashboardData();
        // Trigger lại mỗi khi user_id thay đổi
    }, [selectedUser?.user_id]);

    if (loading) return <div className="dashboard__loading">Đang chuẩn bị menu cho Huy...</div>;

    return (
        <div className="dashboard">
            <div className="dashboard__inner">
                <HeroSection />

                <CategoryGrid />

                {/* --- SECTION 1: DÀNH RIÊNG CHO BẠN (QUAN TRỌNG NHẤT) --- */}
                <SectionContainer title="Dành riêng cho bạn">
                    <div className="dashboard__card-grid">
                        {personalRecs.map(item => (
                            <BusinessCard key={item.business_id} item={item} />
                        ))}
                    </div>
                </SectionContainer>

                {/* --- SECTION 2: BẠN BÈ ĐỀ XUẤT --- */}
                <SectionContainer title="Bạn bè của bạn đề xuất" isInverted>
                    <div className="dashboard__card-grid">
                        {socialRecs.map(item => (
                            <BusinessCard key={item.business_id} item={item} isSocial />
                        ))}
                    </div>
                </SectionContainer>

                <PromoBanner />

                {/* --- SECTION 3: ĐANG HOT (XUỐNG CUỐI) --- */}
                <SectionContainer title="Khám phá xu hướng">
                    <div className="dashboard__card-grid">
                        {trendingList.map(item => (
                            <BusinessCard key={item.business_id} item={item} />
                        ))}
                    </div>
                </SectionContainer>
            </div>
        </div>
    );
};

export default Dashboard;