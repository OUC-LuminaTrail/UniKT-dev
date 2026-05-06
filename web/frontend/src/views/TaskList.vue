<template>
  <div class="task-list">
    <div class="page-header">
      <h1 class="page-title">训练任务</h1>
      <router-link :to="{ name: 'task-new' }" class="btn-primary">
        <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
          <path d="M8 2a.75.75 0 0 1 .75.75v4.5h4.5a.75.75 0 0 1 0 1.5h-4.5v4.5a.75.75 0 0 1-1.5 0v-4.5h-4.5a.75.75 0 0 1 0-1.5h4.5v-4.5A.75.75 0 0 1 8 2Z" />
        </svg>
        新建任务
      </router-link>
    </div>

    <div class="tab-bar">
      <button
        v-for="tab in tabs"
        :key="tab.value"
        :class="['tab-item', { active: activeTab === tab.value }]"
        @click="switchTab(tab.value)"
      >
        {{ tab.label }}
        <span v-if="tab.value === 'running' && activeCount > 0" class="tab-badge">{{ activeCount }}</span>
      </button>
    </div>

    <!-- Loading skeleton -->
    <template v-if="loading">
      <SkeletonTable :rows="5" :cols="8" />
    </template>

    <!-- Active tab: running tasks + queued tasks -->
    <template v-else-if="activeTab === 'running'">
      <div v-if="runningTasks.length === 0 && queueItems.length === 0" class="empty-state">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <path d="M9 9h6M9 13h6M9 17h4" />
        </svg>
        <p>暂无活跃任务</p>
        <span>新建的训练任务会显示在这里</span>
      </div>

      <template v-if="runningTasks.length > 0">
        <div class="section-divider">
          <span class="divider-label">运行中</span>
          <span class="divider-count">{{ runningTasks.length }}</span>
        </div>
        <div class="table-wrapper">
          <table class="data-table">
            <thead>
              <tr>
                <th class="col-id">ID</th>
                <th class="col-name">名称</th>
                <th class="col-model">模型</th>
                <th class="col-dataset">数据集</th>
                <th class="col-env">环境</th>
                <th class="col-status">状态</th>
                <th class="col-time">创建时间</th>
                <th class="col-actions">操作</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="task in runningTasks" :key="task.id">
                <td class="col-id">{{ task.id }}</td>
                <td class="col-name">
                  <router-link :to="{ name: 'task-detail', params: { id: task.id } }" class="task-name-link">{{ task.name }}</router-link>
                </td>
                <td class="col-model">{{ task.model_name }}</td>
                <td class="col-dataset">{{ task.dataset_name }}</td>
                <td class="col-env">
                  <span class="env-tag">{{ task.env_name }}</span>
                </td>
                <td class="col-status">
                  <span class="status-cell">
                    <span class="status-dot status-running" />
                    <span class="status-text">运行中</span>
                  </span>
                </td>
                <td class="col-time">
                  <span class="mono-time">{{ formatTime(task.created_at) }}</span>
                </td>
                <td class="col-actions">
                  <div class="action-group">
                    <router-link :to="{ name: 'task-detail', params: { id: task.id } }" class="action-btn" title="查看详情">
                      <svg width="15" height="15" viewBox="0 0 16 16" fill="currentColor"><path d="M8 3.5c-3.8 0-6.5 3.07-7.35 4.24a.5.5 0 0 0 0 .52C1.5 9.43 4.2 12.5 8 12.5s6.5-3.07 7.35-4.24a.5.5 0 0 0 0-.52C14.5 6.57 11.8 3.5 8 3.5ZM8 11c-2.76 0-5-2.24-5-5h1c0 2.21 1.79 4 4 4s4-1.79 4-4h1c0 2.76-2.24 5-5 5ZM8 6.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Z"/></svg>
                    </router-link>
                    <button class="action-btn action-stop" title="停止任务" @click="handleStop(task.id)">
                      <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><rect x="3" y="3" width="10" height="10" rx="1.5"/></svg>
                    </button>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </template>

      <template v-if="queueItems.length > 0">
        <div class="section-divider">
          <span class="divider-label">排队中</span>
          <span class="divider-count">{{ queueItems.length }}</span>
        </div>
        <div class="table-wrapper">
          <table class="data-table">
            <thead>
              <tr>
                <th class="col-pos">位置</th>
                <th class="col-id">ID</th>
                <th class="col-name">名称</th>
                <th class="col-model">模型</th>
                <th class="col-dataset">数据集</th>
                <th class="col-time">创建时间</th>
                <th class="col-actions">操作</th>
              </tr>
            </thead>
            <tbody>
              <tr v-for="(task, idx) in queueItems" :key="task.id">
                <td class="col-pos">
                  <span class="pos-badge">#{{ idx + 1 }}</span>
                </td>
                <td class="col-id">{{ task.id }}</td>
                <td class="col-name">
                  <span class="task-name-text">{{ task.name }}</span>
                </td>
                <td class="col-model">{{ task.model_name }}</td>
                <td class="col-dataset">{{ task.dataset_name }}</td>
                <td class="col-time">
                  <span class="mono-time">{{ formatTime(task.created_at) }}</span>
                </td>
                <td class="col-actions">
                  <div class="action-group">
                    <button v-if="idx > 0" class="action-btn" title="上移" @click="moveUp(idx)">
                      <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 3.5a.5.5 0 0 1 .354.146l4 4a.5.5 0 0 1-.708.708L8 4.707 4.354 8.354a.5.5 0 1 1-.708-.708l4-4A.5.5 0 0 1 8 3.5z"/></svg>
                    </button>
                    <button v-if="idx < queueItems.length - 1" class="action-btn" title="下移" @click="moveDown(idx)">
                      <svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M8 12.5a.5.5 0 0 1-.354-.146l-4-4a.5.5 0 1 1 .708-.708L8 11.293l3.646-3.647a.5.5 0 0 1 .708.708l-4 4A.5.5 0 0 1 8 12.5z"/></svg>
                    </button>
                    <button class="action-btn action-delete" title="取消任务" @click="handleCancelQueue(task.id)">
                      <svg width="15" height="15" viewBox="0 0 16 16" fill="currentColor"><path d="M6.5 1.75a.25.25 0 0 1 .25-.25h2.5a.25.25 0 0 1 .25.25V3h3.75a.75.75 0 0 1 0 1.5h-.75l-.62 8.97A1.75 1.75 0 0 1 10.14 15H5.86a1.75 1.75 0 0 1-1.74-1.53L3.5 4.5H2.75a.75.75 0 0 1 0-1.5H6.5V1.75ZM5.08 4.5l.59 8.81a.25.25 0 0 0 .25.19h4.16a.25.25 0 0 0 .25-.19l.59-8.81H5.08ZM8 7a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-1.5 0v-3.5A.75.75 0 0 1 8 7Z"/></svg>
                    </button>
                  </div>
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </template>
    </template>

    <!-- Other tabs -->
    <template v-else>
      <div v-if="tasks.length === 0" class="empty-state">
        <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round">
          <rect x="3" y="3" width="18" height="18" rx="2" />
          <path d="M9 9h6M9 13h6M9 17h4" />
        </svg>
        <p>暂无任务</p>
        <span>开始训练后任务将显示在这里</span>
      </div>

      <div v-else class="table-wrapper">
        <table class="data-table">
          <thead>
            <tr>
              <th class="col-id">ID</th>
              <th class="col-name">名称</th>
              <th class="col-model">模型</th>
              <th class="col-dataset">数据集</th>
              <th class="col-env">环境</th>
              <th class="col-status">状态</th>
              <th class="col-time">创建时间</th>
              <th class="col-actions">操作</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="task in tasks" :key="task.id">
              <td class="col-id">{{ task.id }}</td>
              <td class="col-name">
                <router-link :to="{ name: 'task-detail', params: { id: task.id } }" class="task-name-link">{{ task.name }}</router-link>
              </td>
              <td class="col-model">{{ task.model_name }}</td>
              <td class="col-dataset">{{ task.dataset_name }}</td>
              <td class="col-env">
                <span class="env-tag">{{ task.env_name }}</span>
              </td>
              <td class="col-status">
                <span class="status-cell">
                  <span :class="['status-dot', `status-${task.status}`]" />
                  <span class="status-text">{{ statusLabel(task.status) }}</span>
                </span>
              </td>
              <td class="col-time">
                <span class="mono-time">{{ formatTime(task.created_at) }}</span>
              </td>
              <td class="col-actions">
                <div class="action-group">
                  <router-link :to="{ name: 'task-detail', params: { id: task.id } }" class="action-btn" title="查看详情">
                    <svg width="15" height="15" viewBox="0 0 16 16" fill="currentColor"><path d="M8 3.5c-3.8 0-6.5 3.07-7.35 4.24a.5.5 0 0 0 0 .52C1.5 9.43 4.2 12.5 8 12.5s6.5-3.07 7.35-4.24a.5.5 0 0 0 0-.52C14.5 6.57 11.8 3.5 8 3.5ZM8 11c-2.76 0-5-2.24-5-5h1c0 2.21 1.79 4 4 4s4-1.79 4-4h1c0 2.76-2.24 5-5 5ZM8 6.5a1.5 1.5 0 1 0 0 3 1.5 1.5 0 0 0 0-3Z"/></svg>
                  </router-link>
                  <button
                    v-if="task.status !== 'running'"
                    class="action-btn action-delete"
                    title="删除任务"
                    @click="handleDelete(task.id)"
                  >
                    <svg width="15" height="15" viewBox="0 0 16 16" fill="currentColor"><path d="M6.5 1.75a.25.25 0 0 1 .25-.25h2.5a.25.25 0 0 1 .25.25V3h3.75a.75.75 0 0 1 0 1.5h-.75l-.62 8.97A1.75 1.75 0 0 1 10.14 15H5.86a1.75 1.75 0 0 1-1.74-1.53L3.5 4.5H2.75a.75.75 0 0 1 0-1.5H6.5V1.75ZM5.08 4.5l.59 8.81a.25.25 0 0 0 .25.19h4.16a.25.25 0 0 0 .25-.19l.59-8.81H5.08ZM8 7a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-1.5 0v-3.5A.75.75 0 0 1 8 7Z"/></svg>
                  </button>
                </div>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { listTasks, stopTask, deleteTask, type TaskInfo } from '@/api/tasks'
import { getQueue, reorderQueue, type QueueItem } from '@/api/settings'
import SkeletonTable from '@/components/common/SkeletonTable.vue'

const route = useRoute()
const router = useRouter()

const tasks = ref<TaskInfo[]>([])
const runningTasks = ref<TaskInfo[]>([])
const queueItems = ref<QueueItem[]>([])
const activeTab = ref(route.query.tab?.toString() || sessionStorage.getItem('taskListTab') || 'running')

const loading = ref(true)
const initialLoad = ref(true)

const activeCount = computed(() => runningTasks.value.length + queueItems.value.length)

const tabs = [
  { label: '活跃', value: 'running' },
  { label: '已完成', value: 'completed' },
  { label: '已失败', value: 'failed' },
  { label: '已停止', value: 'stopped' },
  { label: '全部', value: 'all' },
]

const statusLabels: Record<string, string> = {
  running: '运行中',
  completed: '已完成',
  failed: '已失败',
  stopped: '已停止',
  pending: '等待中',
}

const statusLabel = (s: string) => statusLabels[s] || s

const switchTab = (tab: string) => {
  activeTab.value = tab
  sessionStorage.setItem('taskListTab', tab)
  router.replace({ query: tab !== 'running' ? { tab } : {} })
  loadAll()
}

const formatTime = (t: string | null) => {
  if (!t) return '-'
  return new Date(t).toLocaleString('zh-CN')
}

const loadAll = async () => {
  if (initialLoad.value) {
    loading.value = true
  }
  try {
    if (activeTab.value === 'running') {
      const [running, queue] = await Promise.all([
        listTasks({ status: 'running' }),
        getQueue().catch(() => []),
      ])
      runningTasks.value = running
      queueItems.value = queue
    } else {
      const filtered = await listTasks({ status: activeTab.value !== 'all' ? activeTab.value : undefined })
      tasks.value = filtered
    }
  } finally {
    loading.value = false
    initialLoad.value = false
  }
}

const moveUp = async (idx: number) => {
  const items = [...queueItems.value]
  const tmp = items[idx]
  items[idx] = items[idx - 1]
  items[idx - 1] = tmp
  queueItems.value = items
  await reorderQueue(items.map(t => t.id))
}

const moveDown = async (idx: number) => {
  const items = [...queueItems.value]
  const tmp = items[idx]
  items[idx] = items[idx + 1]
  items[idx + 1] = tmp
  queueItems.value = items
  await reorderQueue(items.map(t => t.id))
}

const handleStop = async (id: number) => {
  await ElMessageBox.confirm('确定停止该任务？正在进行的训练进度将丢失。', '确认停止', {
    confirmButtonText: '停止',
    cancelButtonText: '取消',
    type: 'warning',
  })
  await stopTask(id)
  ElMessage.success('已发送停止信号')
  setTimeout(loadAll, 2000)
}

const handleCancelQueue = async (id: number) => {
  await ElMessageBox.confirm('确定取消该排队任务？', '确认取消', {
    confirmButtonText: '取消任务',
    cancelButtonText: '返回',
    type: 'warning',
  })
  await stopTask(id)
  ElMessage.success('已取消')
  loadAll()
}

const handleDelete = async (id: number) => {
  await ElMessageBox.confirm('确定删除该任务？此操作不可撤销。', '确认删除', {
    confirmButtonText: '删除',
    cancelButtonText: '取消',
    type: 'warning',
  })
  await deleteTask(id)
  ElMessage.success('已删除')
  loadAll()
}

let pollTimer: ReturnType<typeof setInterval> | null = null

onMounted(() => {
  loadAll()
  pollTimer = setInterval(() => {
    loadAll()
  }, 5000)
})

onUnmounted(() => {
  if (pollTimer) clearInterval(pollTimer)
})
</script>

<style scoped>
.task-list {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.page-title {
  font-size: 20px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
  letter-spacing: -0.01em;
}

.btn-primary {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 16px;
  background: #1f6feb;
  color: #fff;
  border-radius: var(--radius-md);
  font-size: 13px;
  font-weight: 500;
  text-decoration: none;
  border: 1px solid #1f6feb;
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s;
}

.btn-primary:hover {
  background: #388bfd;
  border-color: #388bfd;
}

.tab-bar {
  display: flex;
  gap: 4px;
  padding: 4px;
  background: var(--bg-surface);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-muted);
  width: fit-content;
}

.tab-item {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  background: transparent;
  border: none;
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: color 0.15s, background 0.15s;
  font-family: var(--font-sans);
}

.tab-item:hover {
  color: var(--text-primary);
  background: var(--bg-elevated);
}

.tab-item.active {
  color: var(--accent-blue);
  background: var(--bg-elevated);
  box-shadow: inset 0 -2px 0 var(--accent-blue);
}

.tab-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 18px;
  height: 18px;
  padding: 0 5px;
  border-radius: 9px;
  background: var(--accent-blue);
  color: #fff;
  font-size: 11px;
  font-weight: 600;
  line-height: 1;
}

.section-divider {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-top: 4px;
}

.divider-label {
  font-size: 13px;
  font-weight: 600;
  color: var(--text-secondary);
}

.divider-count {
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--text-tertiary);
  background: var(--bg-overlay);
  padding: 1px 7px;
  border-radius: 10px;
  line-height: 1.6;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 64px 24px;
  color: var(--text-tertiary);
  text-align: center;
}

.empty-state svg {
  margin-bottom: 16px;
  opacity: 0.4;
}

.empty-state p {
  margin: 0 0 4px;
  font-size: 15px;
  font-weight: 500;
  color: var(--text-secondary);
}

.empty-state span {
  font-size: 13px;
}

.table-wrapper {
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  overflow: hidden;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.data-table thead {
  background: var(--bg-elevated);
}

.data-table th {
  padding: 8px 12px;
  text-align: left;
  font-weight: 500;
  font-size: 12px;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.04em;
  border-bottom: 1px solid var(--border-default);
  white-space: nowrap;
}

.data-table td {
  padding: 8px 12px;
  color: var(--text-primary);
  border-bottom: 1px solid var(--border-muted);
  white-space: nowrap;
}

.data-table tbody tr {
  transition: background 0.1s;
}

.data-table tbody tr:hover {
  background: var(--bg-overlay);
}

.data-table tbody tr:last-child td {
  border-bottom: none;
}

.col-pos {
  width: 60px;
}

.col-id {
  width: 56px;
  color: var(--text-tertiary);
  font-family: var(--font-mono);
  font-size: 12px;
}

.col-name {
  min-width: 180px;
}

.col-model,
.col-dataset {
  width: 120px;
}

.col-env {
  width: 100px;
}

.col-status {
  width: 110px;
}

.col-time {
  width: 170px;
}

.col-actions {
  width: 100px;
  text-align: right;
}

.pos-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  padding: 2px 8px;
  border-radius: 10px;
  background: var(--bg-overlay);
  color: var(--text-secondary);
  font-size: 11px;
  font-family: var(--font-mono);
  font-weight: 500;
}

.task-name-link {
  color: var(--text-primary);
  text-decoration: none;
  font-weight: 500;
  transition: color 0.15s;
}

.task-name-link:hover {
  color: var(--accent-blue);
  text-decoration: underline;
  text-underline-offset: 2px;
}

.task-name-text {
  color: var(--text-primary);
  font-weight: 500;
}

.env-tag {
  display: inline-block;
  padding: 2px 8px;
  background: var(--bg-elevated);
  border-radius: 20px;
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--text-secondary);
  border: 1px solid var(--border-muted);
}

.status-cell {
  display: inline-flex;
  align-items: center;
  gap: 7px;
}

.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.status-dot.status-running {
  background: var(--accent-blue);
  box-shadow: 0 0 6px var(--accent-blue);
}

.status-dot.status-completed {
  background: var(--accent-green);
}

.status-dot.status-failed {
  background: var(--accent-red);
}

.status-dot.status-stopped {
  background: var(--accent-orange);
}

.status-dot.status-pending {
  background: var(--text-tertiary);
}

.status-text {
  font-size: 12px;
  font-family: var(--font-mono);
  color: var(--text-secondary);
}

.mono-time {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-secondary);
}

.action-group {
  display: flex;
  align-items: center;
  justify-content: flex-end;
  gap: 4px;
}

.action-btn {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 30px;
  height: 28px;
  border-radius: var(--radius-sm);
  border: none;
  background: transparent;
  color: var(--text-tertiary);
  cursor: pointer;
  transition: color 0.15s, background 0.15s;
  text-decoration: none;
}

.action-btn:hover {
  color: var(--text-primary);
  background: var(--bg-elevated);
}

.action-stop:hover {
  color: var(--accent-orange);
  background: rgba(210, 153, 34, 0.12);
}

.action-delete:hover {
  color: var(--accent-red);
  background: rgba(248, 81, 73, 0.12);
}
</style>
