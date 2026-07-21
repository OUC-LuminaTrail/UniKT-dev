<template>
  <section class="log-card" :style="cardStyle">
    <div class="log-card-header">
      <div class="log-card-title">
        <el-icon :size="16"><Tickets /></el-icon>
        <span>运行日志</span>
      </div>
      <div class="header-right">
        <span class="conn-indicator" :class="`conn-${connState}`" :title="connTitle">
          <span class="conn-dot"></span>
          <span class="conn-label">{{ connLabel }}</span>
        </span>
        <button class="scroll-btn" @click="scrollToBottom">
          <el-icon :size="14"><Bottom /></el-icon>
          跳到底部
        </button>
      </div>
    </div>
    <div class="terminal-wrapper">
      <LogTerminal :ws-url="wsUrl" :task-status="taskStatus" :task-id="taskId" :resize-prefix="resizePrefix" @ready="onTerminalReady" @state="onState" />
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { Terminal } from '@xterm/xterm'
import { Tickets, Bottom } from '@element-plus/icons-vue'
import LogTerminal from './LogTerminal.vue'
import { CONN_MAP, type ConnState } from './log-conn'

const props = defineProps<{
  wsUrl: string
  taskStatus: string
  taskId: number
  resizePrefix?: string
  fill?: boolean
}>()

let terminal: Terminal | null = null

const cardStyle = computed(() =>
  props.fill ? { flex: '1 1 0', margin: '0 20px 20px' } : {}
)

const connState = ref<ConnState>('connecting')
const onState = (s: ConnState) => { connState.value = s }

const connLabel = computed(() => CONN_MAP[connState.value].label)
const connTitle = computed(() => CONN_MAP[connState.value].title)

const onTerminalReady = (term: Terminal) => { terminal = term }

const scrollToBottom = () => { terminal?.scrollToBottom() }
</script>

<style scoped>
.log-card {
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  overflow: hidden;
  background: var(--bg-surface);
  display: flex;
  flex-direction: column;
  flex: 1 1 0;
  min-height: 0;
}

.log-card-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 8px 14px;
  background: var(--bg-elevated);
  border-bottom: 1px solid var(--border-muted);
  flex-shrink: 0;
}

.log-card-title {
  display: flex;
  align-items: center;
  gap: 8px;
  font-size: 13px;
  font-weight: 500;
  color: var(--text-primary);
}

.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.conn-indicator {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  color: var(--text-tertiary);
  font-family: var(--font-mono);
  letter-spacing: 0.3px;
}

.conn-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--text-tertiary);
  flex-shrink: 0;
}

.conn-connecting .conn-dot {
  background: var(--accent-orange);
  animation: conn-pulse 1s infinite;
}

.conn-connected .conn-dot {
  background: var(--accent-green);
  box-shadow: 0 0 5px var(--accent-green);
}

.conn-connected .conn-label {
  color: var(--accent-green);
}

.conn-reconnecting .conn-dot {
  background: var(--accent-orange);
  animation: conn-pulse 0.8s infinite;
}

.conn-reconnecting .conn-label {
  color: var(--accent-orange);
}

.conn-ended .conn-dot {
  background: var(--text-tertiary);
}

@keyframes conn-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

.scroll-btn {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 12px;
  color: var(--text-tertiary);
  background: none;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-sm);
  padding: 4px 10px;
  cursor: pointer;
  font-family: var(--font-sans);
  transition: all 0.15s ease;
}

.scroll-btn:hover {
  color: var(--accent-blue);
  border-color: var(--accent-blue);
}

.terminal-wrapper {
  flex: 1 1 0;
  min-height: 0;
  background: var(--term-bg);
}

@media (prefers-reduced-motion: reduce) {
  .conn-connecting .conn-dot,
  .conn-reconnecting .conn-dot { animation: none; }
}
</style>
