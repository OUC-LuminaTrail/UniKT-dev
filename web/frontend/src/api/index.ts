import axios from 'axios'
import { ElNotification } from 'element-plus'

import i18n from '@/plugins/i18n'

const api = axios.create({
  baseURL: '/api',
  timeout: 120000,
})

api.interceptors.response.use(
  (response) => {
    const raw = response.headers['x-messages']
    if (raw) {
      try {
        const messages = JSON.parse(raw) as { level: string; text: string }[]
        for (const m of messages) {
          ElNotification({
            type: m.level as 'error' | 'warning' | 'info' | 'success',
            message: m.text,
            duration: m.level === 'error' ? 0 : 4500,
          })
        }
      } catch { /* ignore malformed header */ }
    }
    return response
  },
  (error) => {
    const d = error.response?.data
    const code = d?.type
    const translated =
      code && i18n.global.te(`error.${code}`) ? i18n.global.t(`error.${code}`) : null
    if (translated) {
      ElNotification.error({ message: translated, duration: 5000 })
    } else if (d?.type && d?.title) {
      ElNotification.error({
        title: d.title,
        message: typeof d.detail === 'string' ? d.detail : d.title,
        duration: 5000,
      })
    } else if (d?.detail) {
      ElNotification.error({
        message: typeof d.detail === 'string' ? d.detail : JSON.stringify(d.detail),
        duration: 5000,
      })
    }
    return Promise.reject(error)
  },
)

export default api