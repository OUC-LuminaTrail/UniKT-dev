<template>
  <el-skeleton :loading="loading && !task" animated>
    <template #template>
      <div class="task-detail">
        <header class="detail-header">
          <el-skeleton-item variant="rect" style="width:32px;height:32px;border-radius:var(--radius-sm);flex-shrink:0" />
          <el-skeleton-item variant="text" style="width:180px;height:20px" />
          <el-skeleton-item variant="rect" style="width:72px;height:24px;border-radius:20px" />
        </header>

        <section class="meta-grid">
          <div class="meta-cell" v-for="i in 6" :key="i">
            <el-skeleton-item variant="text" style="width:40px;height:10px" />
            <el-skeleton-item variant="text" style="width:70%;height:13px;margin-top:2px" />
          </div>
        </section>

        <div class="command-block">
          <div class="command-bar">
            <span class="command-label">命令</span>
          </div>
          <div style="padding:12px 14px">
            <el-skeleton-item variant="text" style="width:90%;height:14px" />
          </div>
        </div>

        <div style="min-height:200px;border:1px solid var(--border-default);border-radius:var(--radius-md);background:var(--bg-surface)">
          <el-skeleton-item variant="text" style="margin:16px;width:60%;height:14px" />
        </div>
      </div>
    </template>
    <template #default>
  <div class="task-detail" v-if="task">
    <header class="detail-header">
      <button class="back-btn" @click="goBack">
        <el-icon :size="18"><ArrowLeft /></el-icon>
      </button>

      <h1 class="task-name">{{ task.name }}</h1>

      <span class="status-badge" :style="{ '--dot-color': statusMap[task.status]?.color }">
        <span class="status-dot" :class="{ pulse: task.status === 'running' }"></span>
        <span class="status-label">{{ statusMap[task.status]?.label ?? task.status }}</span>
      </span>

      <div class="header-actions" v-if="task.status === 'running'">
        <button class="action-btn stop" :disabled="stopping" @click="handleStop">
          <el-icon :size="14"><SwitchButton /></el-icon>
          <span>{{ stopping ? '停止中…' : '停止' }}</span>
        </button>
        <button class="action-btn kill" :disabled="killing" @click="handleKill">
          <el-icon :size="14"><Bottom /></el-icon>
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
      <div class="meta-cell" v-if="hasGpu">
        <span class="meta-key">GPU</span>
        <span class="meta-val">{{ gpuDisplay }}</span>
      </div>
      <div class="meta-cell">
        <span class="meta-key">进程 ID</span>
        <span class="meta-val mono">{{ task.pid || '—' }}</span>
      </div>
      <div class="meta-cell">
        <span class="meta-key">开始时间</span>
        <span class="meta-val mono">{{ formatDateTime(task.started_at) }}</span>
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
  </el-skeleton>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useQuery, useQueryClient } from '@tanstack/vue-query'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { ArrowLeft, SwitchButton, Bottom } from '@element-plus/icons-vue'
import { formatDateTime } from '@/utils/date'
import { getTask, stopTask, killTask, type TaskInfo } from '@/api/tasks'
import LogCard from '@/components/task/LogCard.vue'
import { useSystemCapabilities } from '@/composables/useSystemCapabilities'

const { hasGpu } = useSystemCapabilities()

const route = useRoute()
const router = useRouter()
const queryClient = useQueryClient()
const taskId = Number(route.params.id)

const goBack = () => {
  if (window.history.length > 1) {
    router.back()
  } else {
    router.replace({ name: 'tasks' })
  }
}

const { data: task, isPending: loading } = useQuery({
  queryKey: ['task', taskId],
  queryFn: () => getTask(taskId),
  refetchInterval: (query) => {
    const status = (query.state.data as TaskInfo | undefined)?.status
    return status && ['running', 'pending', 'stopping'].includes(status)
      ? 3000
      : false
  },
})

const stopping = ref(false)
const killing = ref(false)

const statusMap: Record<string, { color: string; label: string }> = {
  running: { color: 'var(--accent-blue)', label: '运行中' },
  completed: { color: 'var(--accent-green)', label: '已完成' },
  failed: { color: 'var(--accent-red)', label: '已失败' },
  stopped: { color: 'var(--accent-orange)', label: '已停止' },
  stopping: { color: 'var(--accent-orange)', label: '停止中' },
  pending: { color: 'var(--text-tertiary)', label: '等待中' },
}

const exitCodeClass = computed(() => {
  if (task.value?.exit_code == null) return ''
  return task.value.exit_code === 0 ? 'exit-ok' : 'exit-err'
})

const gpuDisplay = computed(() => {
  const t = task.value
  if (!t) return '—'
  const val = t.gpu_assigned ?? t.gpu_request
  if (val === null || val === undefined) {
    return t.status === 'pending' ? '自动' : '—'
  }
  return `GPU ${val}`
})

const copyCommand = () => {
  if (task.value) navigator.clipboard.writeText(task.value.command)
}

const handleStop = async () => {
  try {
    await ElMessageBox.confirm('确定要停止该任务吗？任务将收到 Ctrl+C 信号，SwanLab 会正确记录任务停止信息。', '停止任务', { confirmButtonText: '停止', cancelButtonText: '取消', type: 'warning' })
  } catch { return }
  stopping.value = true
  try {
    await stopTask(taskId)
    ElMessage.success('已发送停止信号')
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
  } finally { killing.value = false }
}
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
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
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
</style>
