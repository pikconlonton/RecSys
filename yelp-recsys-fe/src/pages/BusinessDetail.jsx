import  { useEffect, useState, useContext } from 'react';
import { useParams } from 'react-router-dom';
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet';
import L from 'leaflet';

// Components & Context
import BusinessCard from '../components/BusinessCard';
import { AppContext } from '../context/AppContext';

// API Services
import { businessService, recService } from '../services/businessService';
import logService from '../services/logService';

// Assets & Styles
import 'leaflet/dist/leaflet.css';
import "./BusinessDetail.scss";

// Fix icon cho Leaflet (tránh lỗi marker không hiển thị)
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: require('leaflet/dist/images/marker-icon-2x.png'),
    iconUrl: require('leaflet/dist/images/marker-icon.png'),
    shadowUrl: require('leaflet/dist/images/marker-shadow.png'),
});

const BusinessDetail = () => {
    const { id } = useParams();
    const { selectedUser } = useContext(AppContext); 
    const [loading, setLoading] = useState(true);
    const [business, setBusiness] = useState(null);
    const [topKList, setTopKList] = useState([]);
    useEffect(() => {
        const initPage = async () => {
            setLoading(true);
            try {
                // Lấy ID user từ context hoặc dùng fallback từ kết quả Seed
                const userId = selectedUser?.user_id || "---r61b7EpVPkb4UVme5tA";

                // 1) Fetch business detail trước
                const bizRes = await businessService.getById(id);
                const bizData = bizRes?.data || bizRes;

                if (bizData) {
                    setBusiness({
                        ...bizData,
                        latitude: bizData.lat ?? bizData.latitude,
                        longitude: bizData.lng ?? bizData.longitude,
                    });

                    // 2) Ghi nhận hành vi View lên Backend (await) để recs có thể phản ánh session vừa xem
                    try {
                        await logService.sendViewLog(userId, id);
                    } catch (e) {
                        // Nếu log fail vẫn cho FE xem recs (fallback)
                        console.error("Log error:", e);
                    }
                }

                // 3) Fetch recommendations sau khi log view (session-aware)
                const recRes = await recService.getForUser(userId, { topk: 8, use_social: true, gamma: 0.2 });
                const recData = recRes?.data || recRes;

                // Map dữ liệu gợi ý từ trường .items hoặc .results
                setTopKList(recData.items || recData.results || []);

            } catch (error) {
                console.error("Lỗi fetch API:", error);
            } finally {
                // Delay nhẹ 300ms để transition mượt hơn
                setTimeout(() => setLoading(false), 300);
            }
        };

        if (id) initPage();
        
        // Cuộn lên đầu trang khi vào quán mới
        window.scrollTo({ top: 0, behavior: 'smooth' });

    }, [id, selectedUser?.user_id]); // Cập nhật khi đổi quán HOẶC đổi người dùng

    const handleOpenGoogleMaps = () => {
        if (business) {
            // FIX: Link chuẩn cho Google Maps
            window.open(`https://www.google.com/maps/search/?api=1&query=${business.latitude},${business.longitude}`, '_blank');
        }
    };

    const handleOpenStreetMap = () => {
        if (business) {
            window.open(`https://www.openstreetmap.org/?mlat=${business.latitude}&mlon=${business.longitude}#map=15/${business.latitude}/${business.longitude}`, '_blank');
        }
    };

    if (loading) return <div className="loading-state">LỌC DỮ LIỆU RIÊNG CHO BẠN...</div>;
    if (!business) return <div className="loading-state">KHÔNG TÌM THẤY DỮ LIỆU QUÁN ĂN</div>;

    const position = [business.latitude, business.longitude];

    return (
        <div className="atino-detail">
            <div className="container">
                <section className="product-layout">
                    {/* GALLERY */}
                    <div className="gallery-col">
                        <div className="main-image-wrapper">
                            <img
                                src={business.cover_photo || `https://picsum.photos/seed/${id}/800/1000`}
                                alt={business.name}
                            />
                        </div>
                    </div>

                    {/* INFO */}
                    <div className="info-col">
                        <div className="sticky-content">
                            <nav className="breadcrumb">HOME / {business.city?.toUpperCase() || "PHILLY"}</nav>
                            <h1 className="biz-name">{business.name}</h1>

                            <div className="rating-row">
                                <span className="stars">{"★".repeat(Math.round(business.stars || 0))}</span>
                                <span className="review-text">
                                    {(business.stars || 0).toFixed(1)} / 5.0 ({business.review_count} Reviews)
                                </span>
                            </div>

                            <div className="price-status">
                                <span className="status-open">
                                    {business.attributes?.ByAppointmentOnly === "True" ? "• BY APPOINTMENT" : "• OPEN NOW"}
                                </span>
                            </div>

                            <div className="details-list">
                                <div className="detail-item">
                                    <label>ADDRESS</label>
                                    <p>{business.address}, {business.city}, {business.state}</p>
                                </div>
                                <div className="detail-item">
                                    <label>CATEGORIES</label>
                                    <p>{business.categories}</p>
                                </div>
                            </div>

                            <div className="action-btns">
                                <button className="btn-primary">ĐẶT CHỖ NGAY</button>
                                <button className="btn-secondary">LƯU VÀO YÊU THÍCH</button>
                            </div>

                            {/* BẢN ĐỒ */}
                            {business.latitude && business.longitude ? (
                                <div className="map-placeholder">
                                    <div className="map-head">LOCATION MAP</div>
                                    <div className="map-container">
                                        <MapContainer center={position} zoom={15} scrollWheelZoom={false} className="leaflet-map">
                                            <TileLayer url="https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png" />
                                            <Marker position={position}>
                                                <Popup><strong>{business.name}</strong></Popup>
                                            </Marker>
                                        </MapContainer>
                                    </div>
                                    <div className="map-actions">
                                        <button className="map-btn google-btn" onClick={handleOpenGoogleMaps}>GOOGLE MAPS</button>
                                        <button className="map-btn osm-btn" onClick={handleOpenStreetMap}>OPEN STREET MAP</button>
                                    </div>
                                </div>
                            ) : (
                                <div className="map-placeholder">Tọa độ đang được cập nhật...</div>
                            )}
                        </div>
                    </div>
                </section>

                {/* GỢI Ý */}
                <section className="recommendations-area">
                    <h2 className="area-title">DÀNH RIÊNG CHO {selectedUser?.name?.toUpperCase() || "BẠN"}</h2>
                    <div className="atino-grid">
                        {topKList.map((item) => (
                            <div key={item.business_id} className="grid-item">
                                <BusinessCard item={item} />
                            </div>
                        ))}
                    </div>
                </section>
            </div>
        </div>
    );
};

export default BusinessDetail;