<template>
  <section class="log-card" :style="cardStyle">
    <div class="log-card-header">
      <div class="log-card-title">
        <el-icon :size="16"><Tickets /></el-icon>
        <span>运行日志</span>
      </div>
      <button class="scroll-btn" @click="scrollToBottom">
        <el-icon :size="14"><Bottom /></el-icon>
        跳到底部
      </button>
    </div>
    <div class="terminal-wrapper">
      <LogTerminal :ws-url="wsUrl" :task-status="taskStatus" :task-id="taskId" :resize-prefix="resizePrefix" @ready="onTerminalReady" />
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { Terminal } from '@xterm/xterm'
import { Tickets, Bottom } from '@element-plus/icons-vue'
import LogTerminal from './LogTerminal.vue'

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
  background: #1a1b26;
}

.log-card:not([style*="flex: 1"]) .terminal-wrapper {
  height: calc(100vh - 320px);
  min-height: 500px;
  flex: none;
}
</style>
