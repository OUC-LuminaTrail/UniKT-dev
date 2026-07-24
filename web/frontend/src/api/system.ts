import api from './index'

export const refreshRegistry = () => api.post('/registry/refresh')
