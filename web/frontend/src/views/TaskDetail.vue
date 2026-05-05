<template>
  <div class="task-detail" v-if="task">
    <header class="detail-header">
      <button class="back-btn" @click="$router.push('/tasks')">
        <svg width="18" height="18" viewBox="0 0 16 16" fill="none">
          <path d="M10 3L5 8L10 13" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"/>
        </svg>
      </button>

      <h1 class="task-name">{{ task.name }}</h1>

      <span class="status-badge" :style="{ '--dot-color': statusMap[task.status]?.color }">
        <span class="status-dot" :class="{ pulse: task.status === 'running' }"></span>
        <span class="status-label">{{ statusMap[task.status]?.label ?? task.status }}</span>
      </span>

      <div class="header-actions" v-if="task.status === 'running'">
        <button class="action-btn stop" :disabled="stopping" @click="handleStop">
          <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><rect x="3" y="3" width="10" height="10" rx="1.5"/></svg>
          <span>{{ stopping ? '停止中…' : '停止' }}</span>
        </button>
        <button class="action-btn kill" :disabled="killing" @click="handleKill">
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
            <path d="M4 4L12 12M12 4L4 12"/>
          </svg>
          <span>{{ killing ? '终止中…' : '强制终止' }}</span>
        </button>
      </div>
    </header>

    <section class="meta-grid">
      <div class="meta-cell">
        <span class="meta-key">模型</span>
        <span class="meta-val">{{ task.model_name }}</span>
      </div>
      <div class="meta-cell">
        <span class="meta-key">数据集</span>
        <span class="meta-val">{{ task.dataset_name }}</span>
      </div>
      <div class="meta-cell">
        <span class="meta-key">运行环境</span>
        <span class="meta-val">{{ task.env_type }}:{{ task.env_name }}</span>
      </div>
      <div class="meta-cell">
        <span class="meta-key">进程 ID</span>
        <span class="meta-val mono">{{ task.pid || '—' }}</span>
      </div>
      <div class="meta-cell">
        <span class="meta-key">开始时间</span>
        <span class="meta-val mono">{{ formatTime(task.started_at) }}</span>
      </div>
      <div class="meta-cell">
        <span class="meta-key">退出码</span>
        <span class="meta-val mono" :class="exitCodeClass">{{ task.exit_code ?? '—' }}</span>
      </div>
    </section>

    <div class="command-block">
      <div class="command-bar">
        <span class="command-label">命令</span>
        <button class="copy-btn" @click="copyCommand">复制</button>
      </div>
      <pre class="command-text"><code>{{ task.command }}</code></pre>
    </div>

    <section class="log-card">
      <div class="log-card-header">
        <div class="log-card-title">
          <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="var(--text-secondary)" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round">
            <polyline points="2 4 2 12 14 12 14 4"/>
            <line x1="5" y1="7" x2="11" y2="7"/>
            <line x1="5" y1="9.5" x2="9" y2="9.5"/>
          </svg>
          <span>运行日志</span>
        </div>
        <button class="scroll-btn" @click="scrollToBottom">
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M3 6L8 11L13 6"/>
          </svg>
          跳到底部
        </button>
      </div>
      <div class="terminal-wrapper">
        <LogTerminal :ws-url="`/api/tasks/${taskId}/logs/stream`" :task-status="task?.status || 'pending'" :task-id="taskId" @ready="onTerminalReady" />
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Terminal } from '@xterm/xterm'
import { getTask, stopTask, killTask, type TaskInfo } from '@/api/tasks'
import LogTerminal from '@/components/task/LogTerminal.vue'

const route = useRoute()
const taskId = Number(route.params.id)
const task = ref<TaskInfo | null>(null)
const stopping = ref(false)
const killing = ref(false)
let pollTimer: ReturnType<typeof setInterval> | null = null
let terminal: Terminal | null = null

const statusMap: Record<string, { color: string; label: string }> = {
  running: { color: 'var(--accent-blue)', label: '运行中' },
  completed: { color: 'var(--accent-green)', label: '已完成' },
  failed: { color: 'var(--accent-red)', label: '已失败' },
  stopped: { color: 'var(--accent-orange)', label: '已停止' },
  pending: { color: 'var(--text-tertiary)', label: '等待中' },
}

const formatTime = (t: string | null) => t ? new Date(t).toLocaleString('zh-CN') : '-'

const exitCodeClass = computed(() => {
  if (task.value?.exit_code == null) return ''
  return task.value.exit_code === 0 ? 'exit-ok' : 'exit-err'
})

const copyCommand = () => {
  if (task.value) navigator.clipboard.writeText(task.value.command)
}

const onTerminalReady = (term: Terminal) => { terminal = term }
const scrollToBottom = () => { terminal?.scrollToBottom() }

const loadTask = async () => { task.value = await getTask(taskId) }

const handleStop = async () => {
  stopping.value = true
  try {
    await stopTask(taskId)
    ElMessage.success('已发送停止信号')
    await pollUntilDone()
  } finally { stopping.value = false }
}

const handleKill = async () => {
  killing.value = true
  try {
    await killTask(taskId)
    ElMessage.success('已强制终止')
    await pollUntilDone()
  } finally { killing.value = false }
}

const pollUntilDone = async () => {
  for (let i = 0; i < 20; i++) {
    await new Promise(r => setTimeout(r, 500))
    await loadTask()
    if (task.value && task.value.status !== 'running' && task.value.status !== 'stopping') break
  }
}

onMounted(() => {
  loadTask()
  pollTimer = setInterval(async () => {
    if (task.value?.status === 'running') await loadTask()
  }, 5000)
})

onUnmounted(() => { if (pollTimer) clearInterval(pollTimer) })
</script>

<style scoped>
.task-detail {
  display: flex;
  flex-direction: column;
  gap: 16px;
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px;
  min-height: 100vh;
  color: var(--text-primary);
}

.detail-header {
  display: flex;
  align-items: center;
  gap: 12px;
  min-height: 36px;
}

.back-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-sm);
  background: var(--bg-surface);
  color: var(--text-secondary);
  cursor: pointer;
  transition: all 0.15s ease;
  flex-shrink: 0;
}

.back-btn:hover {
  background: var(--bg-elevated);
  color: var(--text-primary);
  border-color: var(--accent-blue);
}

.task-name {
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  letter-spacing: -0.01em;
}

.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 20px;
  background: color-mix(in srgb, var(--dot-color, var(--text-tertiary)) 12%, transparent);
  border: 1px solid color-mix(in srgb, var(--dot-color, var(--text-tertiary)) 20%, transparent);
  flex-shrink: 0;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--dot-color, var(--text-tertiary));
  flex-shrink: 0;
}

.status-dot.pulse {
  animation: pulse-glow 2s ease-in-out infinite;
}

@keyframes pulse-glow {
  0%, 100% {
    box-shadow: 0 0 0 0 color-mix(in srgb, var(--accent-blue) 50%, transparent);
    opacity: 1;
  }
  50% {
    box-shadow: 0 0 0 6px color-mix(in srgb, var(--accent-blue) 0%, transparent);
    opacity: 0.7;
  }
}

.status-label {
  font-size: 12px;
  font-weight: 500;
  color: var(--dot-color, var(--text-secondary));
  line-height: 1;
}

.header-actions {
  display: flex;
  gap: 8px;
  margin-left: auto;
  flex-shrink: 0;
}

.action-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-default);
  background: var(--bg-surface);
  color: var(--text-secondary);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s ease;
  font-family: var(--font-sans);
}

.action-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.action-btn.stop:hover:not(:disabled) {
  border-color: var(--accent-orange);
  color: var(--accent-orange);
  background: color-mix(in srgb, var(--accent-orange) 8%, var(--bg-surface));
}

.action-btn.kill:hover:not(:disabled) {
  border-color: var(--accent-red);
  color: var(--accent-red);
  background: color-mix(in srgb, var(--accent-red) 8%, var(--bg-surface));
}

.meta-grid {
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 1px;
  background: var(--border-muted);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  overflow: hidden;
}

.meta-cell {
  display: flex;
  flex-direction: column;
  gap: 2px;
  padding: 10px 14px;
  background: var(--bg-surface);
}

.meta-key {
  font-size: 11px;
  font-weight: 500;
  color: var(--text-tertiary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.meta-val {
  font-size: 13px;
  color: var(--text-primary);
  font-family: var(--font-sans);
  line-height: 1.4;
  word-break: break-all;
}

.meta-val.mono {
  font-family: var(--font-mono);
  font-size: 12.5px;
}

.meta-val.exit-ok {
  color: var(--accent-green);
}

.meta-val.exit-err {
  color: var(--accent-red);
}

.command-block {
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  overflow: hidden;
  background: var(--bg-surface);
}

.command-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 12px;
  background: var(--bg-elevated);
  border-bottom: 1px solid var(--border-muted);
}

.command-label {
  font-size: 11px;
  font-weight: 500;
  color: var(--text-tertiary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.copy-btn {
  font-size: 11px;
  color: var(--text-tertiary);
  background: none;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-sm);
  padding: 2px 8px;
  cursor: pointer;
  font-family: var(--font-sans);
  transition: all 0.15s ease;
}

.copy-btn:hover {
  color: var(--accent-blue);
  border-color: var(--accent-blue);
}

.command-text {
  margin: 0;
  padding: 12px 14px;
  overflow-x: auto;
}

.command-text code {
  font-family: var(--font-mono);
  font-size: 12.5px;
  color: var(--accent-cyan);
  line-height: 1.6;
  word-break: break-all;
  white-space: pre-wrap;
}

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
  height: calc(100vh - 320px);
  min-height: 500px;
  background: #1a1b26;
}

@media (max-width: 900px) {
  .meta-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (max-width: 600px) {
  .meta-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .task-detail {
    padding: 12px;
  }
}
</style>
