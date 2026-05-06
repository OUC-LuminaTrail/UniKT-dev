<template>
  <div class="task-detail task-detail--skeleton" v-if="loading && !task">
    <div class="skeleton-header">
      <div class="skeleton-line skeleton-shimmer" style="width:32px;height:32px;border-radius:var(--radius-sm)"></div>
      <div class="skeleton-line skeleton-shimmer" style="width:180px;height:24px;border-radius:4px"></div>
      <div class="skeleton-line skeleton-shimmer" style="width:80px;height:24px;border-radius:20px"></div>
    </div>
    <SkeletonTable :cols="4" :rows="3" />
    <div class="skeleton-line skeleton-shimmer" style="height:80px;border-radius:var(--radius-md)"></div>
  </div>

  <div class="task-detail" v-else-if="task">
    <header class="detail-header">
      <button class="back-btn" @click="$router.back()">
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

    <LogCard :ws-url="`/api/tasks/${taskId}/logs/stream`" :task-status="task?.status || 'pending'" :task-id="taskId" />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { getTask, stopTask, killTask, type TaskInfo } from '@/api/tasks'
import LogCard from '@/components/task/LogCard.vue'
import SkeletonTable from '@/components/common/SkeletonTable.vue'

const route = useRoute()
const taskId = Number(route.params.id)
const task = ref<TaskInfo | null>(null)
const loading = ref(true)
const stopping = ref(false)
const killing = ref(false)
let pollTimer: ReturnType<typeof setInterval> | null = null

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

const loadTask = async () => {
  task.value = await getTask(taskId)
  loading.value = false
}

const handleStop = async () => {
  try {
    await ElMessageBox.confirm('确定要停止该任务吗？任务将收到 Ctrl+C 信号，SwanLab 会正确记录任务停止信息。', '停止任务', { confirmButtonText: '停止', cancelButtonText: '取消', type: 'warning' })
  } catch { return }
  stopping.value = true
  try {
    await stopTask(taskId)
    ElMessage.success('已发送停止信号')
    await pollUntilDone()
  } finally { stopping.value = false }
}

const handleKill = async () => {
  try {
    await ElMessageBox.confirm('确定要强制终止该任务吗？任务将收到 SIGKILL 信号并立即退出，可能导致数据丢失。', '强制终止任务', { confirmButtonText: '强制终止', cancelButtonText: '取消', type: 'error' })
  } catch { return }
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

.skeleton-header {
  display: flex;
  align-items: center;
  gap: 12px;
  min-height: 36px;
}

.skeleton-line {
  background: var(--bg-elevated, #e0e0e0);
}

.skeleton-shimmer {
  position: relative;
  overflow: hidden;
  background: var(--border-muted, #e8e8e8);
}

.skeleton-shimmer::after {
  content: '';
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, transparent 25%, rgba(255,255,255,0.4) 50%, transparent 75%);
  animation: shimmer 1.5s ease-in-out infinite;
}

@keyframes shimmer {
  0% { transform: translateX(-100%); }
  100% { transform: translateX(100%); }
}

.task-detail--skeleton {
  display: flex;
  flex-direction: column;
  gap: 16px;
  max-width: 1200px;
  margin: 0 auto;
  padding: 24px;
  min-height: 100vh;
}
</style>
