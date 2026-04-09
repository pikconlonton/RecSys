import axiosClient from '../api/axiosClient';

export const socialService = {
  // POST /social/friends/upsert
  upsertFriends: (userId, friends) => {
    return axiosClient.post('/social/friends/upsert', {
      user_id: userId,
      friends: friends,
      timestamp: new Date().toISOString()
    });
  },

  // POST /social/interactions
  sendSocialInteraction: (interactionData) => {
    return axiosClient.post('/social/interactions', {
      ...interactionData,
      timestamp: new Date().toISOString()
    });
  }
};