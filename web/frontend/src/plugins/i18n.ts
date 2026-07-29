import { createI18n } from 'vue-i18n'

import enUS from '@/locales/en-US'
import zhCN from '@/locales/zh-CN'

const i18n = createI18n({
  legacy: false,
  locale: 'zh',
  fallbackLocale: 'en',
  messages: {
    zh: zhCN,
    en: enUS,
  },
})

export default i18n
