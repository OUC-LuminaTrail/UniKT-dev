import { computed, watch } from 'vue'
import Cookies from 'universal-cookie'
import zhCn from 'element-plus/es/locale/lang/zh-cn'
import en from 'element-plus/es/locale/lang/en'

import i18n from '@/plugins/i18n'

export type Locale = 'zh' | 'en'

const COOKIE_KEY = 'kt-locale'
const EP_LOCALES = { zh: zhCn, en: en }
const LANG_TAG: Record<Locale, string> = { zh: 'zh-CN', en: 'en' }

const cookies = new Cookies()
const stored = cookies.get<Locale>(COOKIE_KEY)
if (stored === 'zh' || stored === 'en') {
  i18n.global.locale.value = stored
}

watch(
  i18n.global.locale,
  (v) => {
    cookies.set(COOKIE_KEY, v, { maxAge: 365 * 86400, path: '/' })
    document.documentElement.lang = LANG_TAG[v as Locale] ?? 'zh-CN'
  },
  { immediate: true },
)

export function useLocale() {
  const locale = i18n.global.locale
  const setLocale = (v: Locale) => {
    locale.value = v
  }
  const epLocale = computed(() => EP_LOCALES[locale.value as Locale] ?? zhCn)
  return { locale, setLocale, epLocale }
}
