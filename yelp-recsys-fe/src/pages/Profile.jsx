import React, { useContext } from 'react';
import { useNavigate } from 'react-router-dom';
import { AppContext } from '../context/AppContext';
import SectionContainer from '../components/SectionContainer';
import BusinessCard from '../components/BusinessCard';
import './Profile.scss';

const Profile = () => {
    const { selectedUser, history } = useContext(AppContext);
    const navigate = useNavigate();

    // Mock dữ liệu "Đã ghé thăm" cực xịn
    const visitedPlaces = [
        { id: 'v1', name: 'The Belgian Café', stars: 4.8, review_count: 1200, photo_url: 'https://picsum.photos/seed/cafe1/400/300', match: 0.95, friends_who_liked: ['Chung', 'An'] },
        { id: 'v2', name: 'Philadelphia Pizza Co', stars: 4.2, review_count: 540, photo_url: 'https://picsum.photos/seed/pizza1/400/300', match: 0.88 },
        { id: 'v3', name: 'South Philly Barbacoa', stars: 4.9, review_count: 2100, photo_url: 'https://picsum.photos/seed/mex/400/300', match: 0.99, friends_who_liked: ['Đức'] }
    ];

    return (
        <div className="profile-page">
            {/* Header: Thông tin User làm to, tràn viền nhẹ */}
            <header className="profile-header">
                <div className="profile-header__inner">
                    <div className="profile-header__avatar">
                        {selectedUser?.charAt(0) || 'U'}
                    </div>
                    <div className="profile-header__info">
                        <h1>{selectedUser}</h1>
                        <div className="profile-header__badges">
                            <span className="badge-item"><i className="fa-solid fa-crown"></i> Thành viên Vàng</span>
                            <span className="badge-item"><i className="fa-solid fa-location-dot"></i> Philadelphia, PA</span>
                        </div>
                    </div>
                </div>
            </header>

            <div className="profile-content container">

                {/* Section 1: Đã ghé thăm (Dùng Card to, Grid thoáng) */}
                <SectionContainer title="Địa điểm bạn đã ghé thăm">
                    <div className="profile-grid-visited">
                        {visitedPlaces.map(biz => (
                            <BusinessCard
                                key={biz.id}
                                business={biz}
                                isSocial={!!biz.friends_who_liked}
                                score={biz.match}
                            />
                        ))}
                    </div>
                </SectionContainer>

                {/* Section 2: Vừa xem gần đây (Dùng Horizontal Scroll cho gọn) */}
                <SectionContainer title="Lịch sử xem gần đây">
                    <div className="profile-history-scroll">
                        {history.length > 0 ? history.map((id, index) => (
                            <div key={index} className="history-mini-card" onClick={() => navigate(`/detail/${id}`)}>
                                <img src={`https://picsum.photos/seed/${id}/200/150`} alt="" />
                                <div className="history-mini-card__info">
                                    <span className="name">Nhà hàng #{id}</span>
                                    <span className="date">Xem lúc: 10:30</span>
                                </div>
                            </div>
                        )) : (
                            <div className="empty-state">Bạn chưa xem địa điểm nào gần đây.</div>
                        )}
                    </div>
                </SectionContainer>

            </div>
        </div>
    );
};

export default Profile;