import React from 'react';

const SectionContainer = ({ badgeText, title, isInverted = false, children }) => {
    return (
        <section className={`recs-section ${isInverted ? 'social-area' : ''}`}>
            <div className={`section-badge ${isInverted ? 'relative z-10' : ''}`}>
                <span className="pulse-dot"></span>
                <span className="badge-text">{badgeText}</span>
            </div>
            <h2 className={isInverted ? 'relative z-10' : ''}>{title}</h2>
            <div className={`section-grid ${isInverted ? 'relative z-10' : ''}`}>
                {children}
            </div>
        </section>
    );
};

export default SectionContainer;