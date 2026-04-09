import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './BusinessCard.scss';

const FALLBACK_IMAGE = 'https://via.placeholder.com/400x300?text=No+Image';

// --- Sub-components ---
const MatchBadge = ({ score, scoring }) => {
    // Dùng final_score nếu có bật social, không thì dùng score gốc
    const finalScore = scoring?.final_score || score || 0;
    const percent = finalScore <= 1 ? Math.round(finalScore * 100) : Math.min(Math.round(finalScore * 10), 100);
    const tier = percent >= 0 ? 'high' : percent >= 60 ? 'mid' : 'low';

    return (
        <div className={`biz-card__badge-wrapper biz-card__badge-wrapper--${tier}`}>
            <div className="biz-card__badge-main">Match</div>

            {/* Nếu API trả về scoring (có dùng Social), hiện chi tiết AI & Social */}
            {scoring && (
                <div className="biz-card__badge-details">
                    <span title="AI Recommender Score"> {(scoring.emb_score * 100).toFixed(0)}%</span>
                    {scoring.social_score > 0 && (
                        <span title="Social Boost" className="boost"> +{(scoring.social_score * 100).toFixed(0)}%</span>
                    )}
                </div>
            )}
        </div>
    );
};

const StarRating = ({ stars }) => {
    const safeStars = parseFloat(stars) || 0;
    const fullStars = Math.floor(safeStars);
    const hasHalf = safeStars - fullStars >= 0.5;

    return (
        <div className="biz-card__stars">
            {Array.from({ length: 5 }, (_, i) => {
                if (i < fullStars) return <i key={i} className="fa-solid fa-star" />;
                if (i === fullStars && hasHalf) return <i key={i} className="fa-solid fa-star-half-stroke" />;
                return <i key={i} className="fa-regular fa-star" />;
            })}
        </div>
    );
};

// --- Main Component ---
const BusinessCard = ({ item }) => {
    const navigate = useNavigate();
    const [imgError, setImgError] = useState(false);

    if (!item) return null;

    const { business_id, score, scoring, metadata } = item;

    // Xử lý fallback nếu metadata bị null (Như API doc cảnh báo)
    const name = metadata?.name || "Quán mới (Chưa có tên)";
    const stars = metadata?.stars || 0;
    const reviewCount = metadata?.review_count || 0;
    const categories = metadata?.categories?.split(',')[0] || "Nhà hàng";
    const address = metadata?.address?.split(',').slice(-2, -1)[0] || "Đang cập nhật";

    const handleClick = () => navigate(`/business/${item.business_id}`);
    const photoUrl = `https://picsum.photos/seed/${business_id}/400/300`;

    return (
        <article className="biz-card" onClick={handleClick} tabIndex={0} role="button">
            <div className="biz-card__image-wrap">
                <img
                    className="biz-card__image"
                    src={imgError ? FALLBACK_IMAGE : photoUrl}
                    alt={name}
                    onError={() => setImgError(true)}
                    loading="lazy"
                />
                <div className="biz-card__image-overlay" />
                <MatchBadge score={score} scoring={scoring} />
            </div>

            <div className="biz-card__content">
                <h3 className="biz-card__name" title={name}>{name}</h3>

                <div className="biz-card__rating-row">
                    <StarRating stars={stars} />
                    <span className="biz-card__stars-value">{stars.toFixed(1)}</span>
                    <span className="biz-card__review-count">({Number(reviewCount).toLocaleString()})</span>
                </div>

                <p className="biz-card__meta">
                    <i className="fa-solid fa-utensils" /> {categories}
                    <span className="biz-card__meta-sep" />
                    <i className="fa-solid fa-location-dot" /> {address}
                </p>

                {scoring?.social_score > 0 && (
                    <div className="biz-card__social-tag">
                        <i className="fa-solid fa-user-group"></i> Bạn bè đã ghé thăm
                    </div>
                )}
            </div>
        </article>
    );
};

export default BusinessCard;