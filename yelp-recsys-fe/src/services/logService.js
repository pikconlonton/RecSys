import axiosClient from '../api/axiosClient';

const logService = {
  sendViewLog: (userId, businessId) => {
    return axiosClient.post('/logs/', {
      user_id: userId,
      business_id: businessId,
      action: 'view',
      timestamp: new Date().toISOString()
    });
  },
  getRecentLogs: () => axiosClient.get('/logs/recent/')
};

export default logService;