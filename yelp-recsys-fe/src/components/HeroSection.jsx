import React, { useContext } from 'react';
import { AppContext } from '../context/AppContext';
import UserSelector from './UserSelector';
import './HeroSection.scss';

const HeroSection = () => {
    const { selectedUser } = useContext(AppContext);

    return (
        <header className="hero">
            {/* Left: Heading + subtitle */}
            <div className="hero__content">
                <h1 className="hero__title">
                    Recommendation
                    <br />
                    <span className="hero__title-accent">System</span>
                </h1>
                <p className="hero__subtitle">
                    Xin chào,{' '}
                    <strong className="hero__username">
                        {/* FIX: Hiển thị thuộc tính name của Object */}
                        {selectedUser?.name || "Khách"}
                    </strong>.
                    Khám phá những địa điểm được tinh chỉnh riêng cho gu thưởng thức của bạn.
                </p>
            </div>

            {/* Right: User selector card */}
            <aside className="hero__visual" aria-label="Chọn người dùng">
                <div className="hero__card">
                    <div className="hero__card-label">
                        <span className="hero__pulse" aria-hidden="true" />
                        <span>Đang Đóng Vai</span>
                    </div>

                    <UserSelector />

                    <p className="hero__card-hint">
                        Chọn một user để xem hệ thống cá nhân hóa thay đổi.
                    </p>
                </div>
            </aside>
        </header>
    );
};

export default HeroSection;