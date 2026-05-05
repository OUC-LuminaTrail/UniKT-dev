import { ref, onUnmounted } from 'vue'

export function useWebSocket(url: string) {
  const messages = ref<string[]>([])
  const connected = ref(false)
  const done = ref(false)
  let ws: WebSocket | null = null
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null

  const connect = () => {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
    const wsUrl = url.startsWith('/') ? `${protocol}//${location.host}${url}` : url
    ws = new WebSocket(wsUrl)

    ws.onopen = () => { connected.value = true }
    ws.onclose = () => {
      connected.value = false
      if (!done.value) {
        reconnectTimer = setTimeout(connect, 3000)
      }
    }
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'data') {
        messages.value.push(data.content)
      } else if (data.type === 'done') {
        done.value = true
      }
    }
    ws.onerror = () => { ws?.close() }
  }

  const disconnect = () => {
    done.value = true
    if (reconnectTimer) clearTimeout(reconnectTimer)
    ws?.close()
  }

  connect()
  onUnmounted(disconnect)

  return { messages, connected, done, disconnect }
}
