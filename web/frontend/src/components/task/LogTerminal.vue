<template>
  <div ref="terminalContainer" class="log-terminal"></div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { Terminal } from '@xterm/xterm'
import { FitAddon } from '@xterm/addon-fit'
import { WebLinksAddon } from '@xterm/addon-web-links'
import { SearchAddon } from '@xterm/addon-search'
import '@xterm/xterm/css/xterm.css'
import { resizeTerminal } from '@/api/tasks'
import api from '@/api/index'
import type { ConnState } from './log-conn'

const props = defineProps<{
  wsUrl: string
  taskStatus: string
  taskId: number
  resizePrefix?: string
}>()
const emit = defineEmits<{
  (e: 'ready', terminal: Terminal): void
  (e: 'state', state: ConnState): void
}>()

const terminalContainer = ref<HTMLElement>()
let terminal: Terminal | null = null
let fitAddon: FitAddon | null = null
let ws: WebSocket | null = null
let reconnectTimer: ReturnType<typeof setTimeout> | null = null
let resizeObserver: ResizeObserver | null = null
let byteOffset = 0
let streamEnded = false
let lastCols = 0
let lastRows = 0
let resizeDebounce: ReturnType<typeof setTimeout> | null = null

const isTaskRunning = () => props.taskStatus === 'running' || props.taskStatus === 'stopping'
// A queued task has no PTY to resize yet, but its stream is still expected to
// produce output, so a drop while pending must reconnect.
const mayStillStream = () => isTaskRunning() || props.taskStatus === 'pending'

const sendResize = (cols: number, rows: number) => {
  if (!isTaskRunning()) return
  if (cols === lastCols && rows === lastRows) return
  lastCols = cols
  lastRows = rows
  if (props.resizePrefix) {
    api.post(`${props.resizePrefix}/${props.taskId}/resize`, { cols, rows }).catch(() => {})
  } else {
    resizeTerminal(props.taskId, cols, rows).catch(() => {})
  }
}

const debouncedResize = () => {
  if (resizeDebounce) clearTimeout(resizeDebounce)
  resizeDebounce = setTimeout(() => {
    fitAddon?.fit()
    if (terminal) {
      sendResize(terminal.cols, terminal.rows)
    }
  }, 150)
}

const connect = () => {
  if (streamEnded) return

  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
  let url = props.wsUrl.startsWith('/')
    ? `${protocol}//${location.host}${props.wsUrl}`
    : props.wsUrl

  if (byteOffset > 0) {
    const sep = url.includes('?') ? '&' : '?'
    url += `${sep}from_offset=${byteOffset}`
  }

  ws = new WebSocket(url)

  ws.onopen = () => { emit('state', 'connected') }

  ws.onmessage = (event) => {
    const data = JSON.parse(event.data)
    if (data.type === 'data') {
      terminal?.write(data.content)
      if (data.offset !== undefined) {
        byteOffset = data.offset
      }
    } else if (data.type === 'done') {
      streamEnded = true
      terminal?.write('\r\n\x1b[2m——— Process exited ———\x1b[0m\r\n')
    } else if (data.type === 'error') {
      terminal?.write(`\x1b[31m${data.content}\x1b[0m\r\n`)
    }
  }

  ws.onclose = () => {
    if (streamEnded || !mayStillStream()) {
      emit('state', 'ended')
      return
    }
    emit('state', 'reconnecting')
    reconnectTimer = setTimeout(connect, 2000)
  }

  ws.onerror = () => { ws?.close() }
}

onMounted(() => {
  if (!terminalContainer.value) return

  terminal = new Terminal({
    theme: {
      background: '#1a1b26',
      foreground: '#a9b1d6',
      cursor: '#c0caf5',
      selectionBackground: '#33467c',
      black: '#15161e',
      red: '#f7768e',
      green: '#9ece6a',
      yellow: '#e0af68',
      blue: '#7aa2f7',
      magenta: '#bb9af7',
      cyan: '#7dcfff',
      white: '#a9b1d6',
      brightBlack: '#414868',
      brightRed: '#f7768e',
      brightGreen: '#9ece6a',
      brightYellow: '#e0af68',
      brightBlue: '#7aa2f7',
      brightMagenta: '#bb9af7',
      brightCyan: '#7dcfff',
      brightWhite: '#c0caf5',
    },
    fontSize: 13,
    fontFamily: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", monospace',
    convertEol: false,
    scrollback: 50000,
    cursorBlink: false,
    cursorStyle: 'bar',
    allowProposedApi: true,
  })

  fitAddon = new FitAddon()
  const searchAddon = new SearchAddon()
  const webLinksAddon = new WebLinksAddon()

  terminal.loadAddon(fitAddon)
  terminal.loadAddon(searchAddon)
  terminal.loadAddon(webLinksAddon)

  terminal.open(terminalContainer.value)
  fitAddon.fit()

  emit('ready', terminal)

  sendResize(terminal.cols, terminal.rows)

  terminal.onResize(({ cols, rows }) => {
    sendResize(cols, rows)
  })

  resizeObserver = new ResizeObserver(() => {
    debouncedResize()
  })
  resizeObserver.observe(terminalContainer.value)

  connect()
})

onUnmounted(() => {
  streamEnded = true
  if (reconnectTimer) clearTimeout(reconnectTimer)
  if (resizeDebounce) clearTimeout(resizeDebounce)
  ws?.close()
  resizeObserver?.disconnect()
  terminal?.dispose()
})
</script>

<style scoped>
.log-terminal {
  width: 100%;
  height: 100%;
}

.log-terminal :deep(.xterm) {
  padding: 8px;
}
</style>
