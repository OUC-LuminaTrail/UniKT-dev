export type ConnState = 'connecting' | 'connected' | 'reconnecting' | 'ended'

export const CONN_MAP: Record<ConnState, { label: string; title: string }> = {
  connecting: { label: 'log.conn.connecting', title: 'log.conn.connectingTitle' },
  connected: { label: 'log.conn.connected', title: 'log.conn.connectedTitle' },
  reconnecting: { label: 'log.conn.reconnecting', title: 'log.conn.reconnectingTitle' },
  ended: { label: 'log.conn.ended', title: 'log.conn.endedTitle' },
}
