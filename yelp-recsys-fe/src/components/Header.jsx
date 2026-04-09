import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import './Header.scss';

const NAV_ITEMS = [
    { to: '/', label: 'Trang chủ', icon: 'fa-solid fa-house', exact: true },
    { to: '/khuyen-mai', label: 'Khuyến mãi Hot', icon: 'fa-solid fa-fire', iconClass: 'text-orange' },
    { to: '/top-ban-chay', label: 'Top Nhà hàng', icon: 'fa-solid fa-trophy', iconClass: 'text-gold' },
    { to: '/voucher', label: 'Voucher', icon: 'fa-solid fa-tag', iconClass: 'text-primary' },
];

const ACTION_ITEMS = [
    { to: '/profile', label: 'Hồ sơ', icon: 'fa-regular fa-user' },
    { to: '/saved', label: 'Đã lưu', icon: 'fa-regular fa-heart', badge: 3, badgeVariant: 'danger' },
    { to: '/cart', label: 'Giỏ hàng', icon: 'fa-solid fa-bag-shopping', badge: 0, badgeVariant: 'primary' },
];

const Header = () => {
    const location = useLocation();
    const [searchValue, setSearchValue] = useState('');

    const isActive = (item) =>
        item.exact
            ? location.pathname === item.to
            : location.pathname.startsWith(item.to);

    return (
        <header className="site-header">
            {/* Tier 1: Promo Bar */}
            <div className="site-header__promo">
                <div className="site-header__container site-header__container--flex">
                    <span className="site-header__promo-text">
                        <i className="fa-solid fa-truck-fast site-header__promo-icon" />
                        Hệ Thống Gợi Ý Nhà Hàng, Khu Vui Chơi Dành cho bạn
                    </span>
                    <nav className="site-header__promo-links" aria-label="Hỗ trợ">
                        <Link to="/support">Hỗ trợ khách hàng</Link>
                        <Link to="/stores">Hệ thống cửa hàng</Link>
                    </nav>
                </div>
            </div>

            {/* Tier 2: Main Header */}
            <div className="site-header__main">
                <div className="site-header__container site-header__container--flex">
                    {/* Logo */}
                    <Link to="/" className="site-header__logo" aria-label="RecSys - Trang chủ">
                        <i className="fa-brands fa-yelp site-header__logo-icon" aria-hidden="true" />
                        <span className="site-header__logo-text">Recomend</span>
                    </Link>

                    {/* Search */}
                    <div className="site-header__search" role="search">
                        <input
                            type="search"
                            className="site-header__search-input"
                            placeholder="Tìm nhà hàng, món ăn, đồ uống..."
                            value={searchValue}
                            onChange={(e) => setSearchValue(e.target.value)}
                            aria-label="Tìm kiếm"
                        />
                        <button
                            className="site-header__search-btn"
                            aria-label="Tìm kiếm"
                            type="button"
                        >
                            <i className="fa-solid fa-magnifying-glass" aria-hidden="true" />
                        </button>
                    </div>

                    {/* Actions */}
                    <div className="site-header__actions" role="navigation" aria-label="Tài khoản">
                        {ACTION_ITEMS.map(({ to, label, icon, badge, badgeVariant }) => (
                            <Link key={to} to={to} className="site-header__action" aria-label={label}>
                                <span className="site-header__action-icon-wrap">
                                    <i className={icon} aria-hidden="true" />
                                    {badge !== undefined && (
                                        <span
                                            className={`site-header__badge site-header__badge--${badgeVariant}`}
                                            aria-label={`${badge} mục`}
                                        >
                                            {badge}
                                        </span>
                                    )}
                                </span>
                                <span className="site-header__action-label">{label}</span>
                            </Link>
                        ))}
                    </div>
                </div>
            </div>

            {/* Tier 3: Bottom Nav */}
            <nav className="site-header__nav" aria-label="Điều hướng chính">
                <div className="site-header__container site-header__container--flex">
                    {NAV_ITEMS.map((item) => (
                        <Link
                            key={item.to}
                            to={item.to}
                            className={`site-header__nav-item${isActive(item) ? ' site-header__nav-item--active' : ''}`}
                        >
                            <i className={`${item.icon}${item.iconClass ? ` ${item.iconClass}` : ''}`} aria-hidden="true" />
                            {item.label}
                        </Link>
                    ))}
                </div>
            </nav>
        </header>
    );
};

export default Header;