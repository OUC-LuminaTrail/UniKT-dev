import dayjs from 'dayjs'
import 'dayjs/locale/zh-cn'

dayjs.locale('zh-cn')

export function formatDateTime(t: string | null): string {
  if (!t) return '-'
  return dayjs(t).format('YYYY/MM/DD HH:mm:ss')
}
