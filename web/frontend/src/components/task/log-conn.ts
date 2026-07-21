export type ConnState = 'connecting' | 'connected' | 'reconnecting' | 'ended'

export const CONN_MAP: Record<ConnState, { label: string; title: string }> = {
  connecting: { label: '连接中', title: '正在建立日志流连接' },
  connected: { label: '实时', title: '日志流已连接，实时接收输出' },
  reconnecting: { label: '重连中', title: '连接中断，正在尝试重新连接' },
  ended: { label: '已结束', title: '日志流已关闭' },
}
