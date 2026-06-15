<template>
  <div class="task-list">
    <div class="page-header">
      <h1 class="page-title">训练任务</h1>
      <router-link :to="{ name: 'task-new' }" class="btn-primary">
        <el-icon :size="14"><Plus /></el-icon>
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

    <!-- Active tab: running tasks + queued tasks -->
    <template v-if="activeTab === 'running'">
      <div v-if="runningTasks.length === 0 && queueItems.length === 0" class="empty-state">
        <el-icon :size="48"><Document /></el-icon>
        <p>暂无活跃任务</p>
        <span>新建的训练任务会显示在这里</span>
      </div>

      <template v-if="runningTasks.length > 0">
        <div class="section-divider">
          <span class="divider-label">运行中</span>
          <span class="divider-count">{{ runningTasks.length }}</span>
        </div>
        <transition name="batch-bar">
          <div v-if="selectedRunningTasks.length > 0" class="batch-bar">
            <span class="batch-info">已选择 <strong>{{ selectedRunningTasks.length }}</strong> 项</span>
            <button class="batch-stop-btn" @click="handleBatchStop">
              <el-icon :size="14"><SwitchButton /></el-icon>
              批量停止
            </button>
            <button class="batch-clear-btn" @click="clearRunningSelection">取消选择</button>
          </div>
        </transition>
        <el-table
          ref="runningTableRef"
          :data="runningTasks"
          row-key="id"
          size="small"
          class="task-table"
          :default-sort="{ prop: 'id', order: 'ascending' }"
          @selection-change="handleRunningSelectionChange"
        >
          <el-table-column type="selection" width="45" reserve-selection />
          <el-table-column prop="id" label="ID" width="70" sortable />
          <el-table-column label="名称" min-width="160">
            <template #default="{ row }">
              <router-link :to="{ name: 'task-detail', params: { id: row.id } }" class="task-name-link">{{ row.name }}</router-link>
            </template>
          </el-table-column>
          <el-table-column prop="model_name" label="模型" width="110" sortable />
          <el-table-column prop="dataset_name" label="数据集" width="110" sortable />
          <el-table-column prop="env_name" label="环境" width="100" sortable>
            <template #default="{ row }">
              <span class="env-tag">{{ row.env_name }}</span>
            </template>
          </el-table-column>
          <el-table-column label="状态" width="110">
            <template #default>
              <span class="status-cell">
                <span class="status-dot status-running" />
                <span class="status-text">运行中</span>
              </span>
            </template>
          </el-table-column>
          <el-table-column prop="created_at" label="创建时间" width="170" sortable>
            <template #default="{ row }">
              <span class="mono-time">{{ formatDateTime(row.created_at) }}</span>
            </template>
          </el-table-column>
          <el-table-column label="操作" width="90" align="right">
            <template #default="{ row }">
              <div class="action-group">
                <router-link :to="{ name: 'task-detail', params: { id: row.id } }" class="action-btn" title="查看详情">
                  <el-icon :size="15"><View /></el-icon>
                </router-link>
                <button class="action-btn action-stop" title="停止任务" @click="handleStop(row.id)">
                  <el-icon :size="14"><SwitchButton /></el-icon>
                </button>
              </div>
            </template>
          </el-table-column>
        </el-table>
        <el-pagination
          v-if="runningTotal > runningPageSize"
          class="table-pagination"
          v-model:current-page="runningPage"
          v-model:page-size="runningPageSize"
          :total="runningTotal"
          :page-sizes="[10, 20, 50]"
          layout="total, sizes, prev, pager, next"
          size="small"
          @current-change="handleRunningPageChange"
          @size-change="handleRunningPageSizeChange"
        />
      </template>

      <template v-if="queueItems.length > 0">
        <div class="section-divider">
          <span class="divider-label">排队中</span>
          <span class="divider-count">{{ queueItems.length }}</span>
        </div>
        <transition name="batch-bar">
          <div v-if="selectedQueueItems.length > 0" class="batch-bar">
            <span class="batch-info">已选择 <strong>{{ selectedQueueItems.length }}</strong> 项</span>
            <button class="batch-cancel-btn" @click="handleBatchCancelQueue">
              <el-icon :size="14"><Delete /></el-icon>
              批量取消
            </button>
            <button class="batch-clear-btn" @click="clearQueueSelection">取消选择</button>
          </div>
        </transition>
        <el-table
          ref="queueTableRef"
          :data="paginatedQueueItems"
          row-key="id"
          size="small"
          class="task-table"
          @selection-change="handleQueueSelectionChange"
        >
          <el-table-column type="selection" width="45" reserve-selection />
          <el-table-column label="位置" width="70">
            <template #default="{ $index }">
              <span class="pos-badge">#{{ (queuePage - 1) * queuePageSize + $index + 1 }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="id" label="ID" width="70" sortable />
          <el-table-column label="名称" min-width="160">
            <template #default="{ row }">
              <span class="task-name-text">{{ row.name }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="model_name" label="模型" width="110" sortable />
          <el-table-column prop="dataset_name" label="数据集" width="110" sortable />
          <el-table-column prop="created_at" label="创建时间" width="170" sortable>
            <template #default="{ row }">
              <span class="mono-time">{{ formatDateTime(row.created_at) }}</span>
            </template>
          </el-table-column>
          <el-table-column label="操作" width="110" align="right">
            <template #default="{ row, $index }">
              <div class="action-group">
                <button v-if="$index > 0" class="action-btn" title="上移" @click="moveUp($index)">
                  <el-icon :size="14"><ArrowUp /></el-icon>
                </button>
                <button v-if="$index < queueItems.length - 1" class="action-btn" title="下移" @click="moveDown($index)">
                  <el-icon :size="14"><ArrowDown /></el-icon>
                </button>
                <button class="action-btn action-delete" title="取消任务" @click="handleCancelQueue(row.id)">
                  <el-icon :size="15"><Delete /></el-icon>
                </button>
              </div>
            </template>
          </el-table-column>
        </el-table>
        <el-pagination
          v-if="queueItems.length > queuePageSize"
          class="table-pagination"
          v-model:current-page="queuePage"
          v-model:page-size="queuePageSize"
          :total="queueItems.length"
          :page-sizes="[10, 20, 50]"
          layout="total, sizes, prev, pager, next"
          size="small"
        />
      </template>
    </template>

    <!-- Other tabs -->
    <template v-else>
      <div v-if="tasks.length === 0" class="empty-state">
        <el-icon :size="48"><Document /></el-icon>
        <p>暂无任务</p>
        <span>开始训练后任务将显示在这里</span>
      </div>

      <template v-else>
        <transition name="batch-bar">
          <div v-if="selectedTasks.length > 0" class="batch-bar">
            <span class="batch-info">已选择 <strong>{{ selectedTasks.length }}</strong> 项</span>
            <button class="batch-delete-btn" @click="handleBatchDelete">
              <el-icon :size="14"><Delete /></el-icon>
              批量删除
            </button>
            <button class="batch-clear-btn" @click="clearSelection">取消选择</button>
          </div>
        </transition>

        <el-table
          ref="otherTableRef"
          :data="tasks"
          row-key="id"
          size="small"
          class="task-table"
          v-loading="loading"
          :default-sort="{ prop: 'id', order: 'ascending' }"
          @selection-change="handleSelectionChange"
        >
          <el-table-column type="selection" width="45" reserve-selection />
          <el-table-column prop="id" label="ID" width="70" sortable />
        <el-table-column label="名称" min-width="160">
          <template #default="{ row }">
            <router-link :to="{ name: 'task-detail', params: { id: row.id } }" class="task-name-link">{{ row.name }}</router-link>
          </template>
        </el-table-column>
        <el-table-column prop="model_name" label="模型" width="110" sortable />
        <el-table-column prop="dataset_name" label="数据集" width="110" sortable />
        <el-table-column prop="env_name" label="环境" width="100" sortable>
          <template #default="{ row }">
            <span class="env-tag">{{ row.env_name }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="status" label="状态" width="110" sortable>
          <template #default="{ row }">
            <span class="status-cell">
              <span :class="['status-dot', `status-${row.status}`]" />
              <span class="status-text">{{ statusLabel(row.status) }}</span>
            </span>
          </template>
        </el-table-column>
        <el-table-column prop="created_at" label="创建时间" width="170" sortable>
          <template #default="{ row }">
            <span class="mono-time">{{ formatDateTime(row.created_at) }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="finished_at" label="完成时间" width="170" sortable>
          <template #default="{ row }">
            <span class="mono-time">{{ formatDateTime(row.finished_at) }}</span>
          </template>
        </el-table-column>
        <el-table-column label="操作" width="90" align="right">
          <template #default="{ row }">
            <div class="action-group">
              <router-link :to="{ name: 'task-detail', params: { id: row.id } }" class="action-btn" title="查看详情">
                <el-icon :size="15"><View /></el-icon>
              </router-link>
              <button
                v-if="row.status !== 'running'"
                class="action-btn action-delete"
                title="删除任务"
                @click="handleDelete(row.id)"
              >
                <el-icon :size="15"><Delete /></el-icon>
              </button>
            </div>
          </template>
        </el-table-column>
        </el-table>
        <el-pagination
          v-if="total > 0"
          class="table-pagination"
          v-model:current-page="page"
          v-model:page-size="pageSize"
          :total="total"
          :page-sizes="[10, 20, 50, 100]"
          layout="total, sizes, prev, pager, next"
          size="small"
          @current-change="handlePageChange"
          @size-change="handlePageSizeChange"
        />
      </template>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useQuery, useQueryClient } from '@tanstack/vue-query'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { ElTable } from 'element-plus'
import { Plus, View, SwitchButton, ArrowUp, ArrowDown, Delete, Document } from '@element-plus/icons-vue'
import { formatDateTime } from '@/utils/date'
import { listTasks, stopTask, deleteTask, type TaskInfo } from '@/api/tasks'
import { getQueue, reorderQueue, type QueueItem } from '@/api/settings'

const route = useRoute()
const router = useRouter()
const queryClient = useQueryClient()

const activeTab = ref(route.query.tab?.toString() || sessionStorage.getItem('taskListTab') || 'running')

const page = ref(1)
const pageSize = ref(20)
const runningPage = ref(1)
const runningPageSize = ref(20)
const queuePage = ref(1)
const queuePageSize = ref(20)

const selectedTasks = ref<TaskInfo[]>([])
const selectedRunningTasks = ref<TaskInfo[]>([])
const selectedQueueItems = ref<QueueItem[]>([])
const otherTableRef = ref<InstanceType<typeof ElTable>>()
const runningTableRef = ref<InstanceType<typeof ElTable>>()
const queueTableRef = ref<InstanceType<typeof ElTable>>()

const tabs = [
  { label: '全部', value: 'all' },
  { label: '活跃', value: 'running' },
  { label: '已完成', value: 'completed' },
  { label: '已失败', value: 'failed' },
  { label: '已停止', value: 'stopped' },
]

const statusLabels: Record<string, string> = {
  running: '运行中',
  completed: '已完成',
  failed: '已失败',
  stopped: '已停止',
  pending: '等待中',
}

const statusLabel = (s: string) => statusLabels[s] || s

interface TasksData {
  tasks: TaskInfo[]
  total: number
  runningTasks: TaskInfo[]
  runningTotal: number
  queueItems: QueueItem[]
}

const allDataQuery = useQuery({
  queryKey: computed(() => [
    'tasks-list',
    activeTab.value,
    page.value,
    pageSize.value,
    runningPage.value,
    runningPageSize.value,
  ]),
  queryFn: async (): Promise<TasksData> => {
    if (activeTab.value === 'running') {
      const [running, queue] = await Promise.all([
        listTasks({ status: 'running', page: runningPage.value, page_size: runningPageSize.value }),
        getQueue().catch(() => []),
      ])
      return {
        tasks: [],
        total: 0,
        runningTasks: running.items,
        runningTotal: running.total,
        queueItems: queue,
      }
    } else {
      const status = activeTab.value !== 'all' ? activeTab.value : undefined
      const [filtered, running, queue] = await Promise.all([
        listTasks({ status, page: page.value, page_size: pageSize.value }),
        listTasks({ status: 'running', page: 1, page_size: 100 }),
        getQueue().catch(() => []),
      ])
      return {
        tasks: filtered.items,
        total: filtered.total,
        runningTasks: running.items,
        runningTotal: running.total,
        queueItems: queue,
      }
    }
  },
  refetchInterval: 5000,
})

const tasks = computed(() => allDataQuery.data.value?.tasks ?? [])
const runningTasks = computed(() => allDataQuery.data.value?.runningTasks ?? [])
const queueItems = computed(() => allDataQuery.data.value?.queueItems ?? [])
const paginatedQueueItems = computed(() => {
  const start = (queuePage.value - 1) * queuePageSize.value
  return queueItems.value.slice(start, start + queuePageSize.value)
})
const total = computed(() => allDataQuery.data.value?.total ?? 0)
const runningTotal = computed(() => allDataQuery.data.value?.runningTotal ?? 0)
const loading = computed(() => allDataQuery.isPending.value)

const activeCount = computed(() => runningTotal.value + queueItems.value.length)

const invalidateTasks = () => queryClient.invalidateQueries({ queryKey: ['tasks-list'] })

const switchTab = (tab: string) => {
  activeTab.value = tab
  page.value = 1
  runningPage.value = 1
  queuePage.value = 1
  selectedTasks.value = []
  selectedRunningTasks.value = []
  selectedQueueItems.value = []
  sessionStorage.setItem('taskListTab', tab)
  router.replace({ query: tab !== 'running' ? { tab } : {} })
}

const handlePageChange = (p: number) => {
  page.value = p
}

const handlePageSizeChange = (s: number) => {
  pageSize.value = s
  page.value = 1
}

const handleRunningPageChange = (p: number) => {
  runningPage.value = p
}

const handleRunningPageSizeChange = (s: number) => {
  runningPageSize.value = s
  runningPage.value = 1
}

const moveUp = async (idx: number) => {
  const items = [...queueItems.value]
  const tmp = items[idx]
  items[idx] = items[idx - 1]
  items[idx - 1] = tmp
  await reorderQueue(items.map(t => t.id))
  invalidateTasks()
}

const moveDown = async (idx: number) => {
  const items = [...queueItems.value]
  const tmp = items[idx]
  items[idx] = items[idx + 1]
  items[idx + 1] = tmp
  await reorderQueue(items.map(t => t.id))
  invalidateTasks()
}

const handleStop = async (id: number) => {
  await ElMessageBox.confirm('确定停止该任务？正在进行的训练进度将丢失。', '确认停止', {
    confirmButtonText: '停止',
    cancelButtonText: '取消',
    type: 'warning',
  })
  ElMessage.success('已发送停止信号')
  await stopTask(id)
  ElMessage.success('任务已停止')
  setTimeout(invalidateTasks, 2000)
}

const handleCancelQueue = async (id: number) => {
  await ElMessageBox.confirm('确定取消该排队任务？', '确认取消', {
    confirmButtonText: '取消任务',
    cancelButtonText: '返回',
    type: 'warning',
  })
  await stopTask(id)
  ElMessage.success('已取消')
  invalidateTasks()
}

const handleDelete = async (id: number) => {
  await ElMessageBox.confirm('确定删除该任务？此操作不可撤销。', '确认删除', {
    confirmButtonText: '删除',
    cancelButtonText: '取消',
    type: 'warning',
  })
  await deleteTask(id)
  ElMessage.success('已删除')
  invalidateTasks()
}

const handleSelectionChange = (rows: TaskInfo[]) => {
  selectedTasks.value = rows
}

const clearSelection = () => {
  otherTableRef.value?.clearSelection()
}

const handleBatchDelete = async () => {
  const count = selectedTasks.value.length
  const runningSelected = selectedTasks.value.filter(t => t.status === 'running')
  if (runningSelected.length > 0) {
    ElMessage.warning(`选中项包含 ${runningSelected.length} 个运行中的任务，无法删除`)
    return
  }
  await ElMessageBox.confirm(`确定删除选中的 ${count} 个任务？此操作不可撤销。`, '批量删除', {
    confirmButtonText: '删除',
    cancelButtonText: '取消',
    type: 'warning',
  })
  await Promise.all(selectedTasks.value.map(t => deleteTask(t.id)))
  ElMessage.success(`已删除 ${count} 个任务`)
  clearSelection()
  invalidateTasks()
}

// --- Running tasks multi-select ---
const handleRunningSelectionChange = (rows: TaskInfo[]) => {
  selectedRunningTasks.value = rows
}

const clearRunningSelection = () => {
  runningTableRef.value?.clearSelection()
}

const handleBatchStop = async () => {
  const count = selectedRunningTasks.value.length
  await ElMessageBox.confirm(
    `确定停止选中的 ${count} 个运行中任务？正在进行的训练进度将丢失。`,
    '批量停止',
    {
      confirmButtonText: '停止',
      cancelButtonText: '取消',
      type: 'warning',
    }
  )
  ElMessage.success('已发送停止信号')
  await Promise.all(selectedRunningTasks.value.map(t => stopTask(t.id)))
  ElMessage.success(`已停止 ${count} 个任务`)
  clearRunningSelection()
  setTimeout(invalidateTasks, 2000)
}

// --- Queue multi-select ---
const handleQueueSelectionChange = (rows: QueueItem[]) => {
  selectedQueueItems.value = rows
}

const clearQueueSelection = () => {
  queueTableRef.value?.clearSelection()
}

const handleBatchCancelQueue = async () => {
  const count = selectedQueueItems.value.length
  await ElMessageBox.confirm(
    `确定取消选中的 ${count} 个排队任务？`,
    '批量取消',
    {
      confirmButtonText: '取消任务',
      cancelButtonText: '返回',
      type: 'warning',
    }
  )
  await Promise.all(selectedQueueItems.value.map(t => stopTask(t.id)))
  ElMessage.success(`已取消 ${count} 个排队任务`)
  clearQueueSelection()
  invalidateTasks()
}
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
  background: var(--accent-blue);
  color: #fff;
  border-radius: var(--radius-md);
  font-size: 13px;
  font-weight: 500;
  text-decoration: none;
  border: 1px solid var(--accent-blue);
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s;
}

.btn-primary:hover {
  background: #388bfd;
  border-color: #388bfd;
}

html.dark .btn-primary:hover {
  background: #1f6feb;
  border-color: #1f6feb;
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

.task-table {
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  overflow: hidden;
}

.table-pagination {
  margin-top: 12px;
  justify-content: flex-end;
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

.batch-bar {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 4px 0;
}

.batch-info {
  font-size: 13px;
  color: var(--text-secondary);
}

.batch-info strong {
  color: var(--accent-blue);
  font-weight: 600;
}

.batch-delete-btn {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 14px;
  background: rgba(248, 81, 73, 0.08);
  color: var(--accent-red);
  border: 1px solid rgba(248, 81, 73, 0.2);
  border-radius: var(--radius-sm);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s;
  font-family: var(--font-sans);
}

.batch-delete-btn:hover {
  background: rgba(248, 81, 73, 0.15);
  border-color: rgba(248, 81, 73, 0.4);
}

.batch-stop-btn {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 14px;
  background: rgba(210, 153, 34, 0.08);
  color: var(--accent-orange);
  border: 1px solid rgba(210, 153, 34, 0.2);
  border-radius: var(--radius-sm);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s;
  font-family: var(--font-sans);
}

.batch-stop-btn:hover {
  background: rgba(210, 153, 34, 0.15);
  border-color: rgba(210, 153, 34, 0.4);
}

.batch-cancel-btn {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 14px;
  background: rgba(248, 81, 73, 0.08);
  color: var(--accent-red);
  border: 1px solid rgba(248, 81, 73, 0.2);
  border-radius: var(--radius-sm);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s;
  font-family: var(--font-sans);
}

.batch-cancel-btn:hover {
  background: rgba(248, 81, 73, 0.15);
  border-color: rgba(248, 81, 73, 0.4);
}

.batch-clear-btn {
  padding: 4px 12px;
  background: transparent;
  color: var(--text-tertiary);
  border: 1px solid var(--border-muted);
  border-radius: var(--radius-sm);
  font-size: 13px;
  cursor: pointer;
  transition: all 0.15s;
  font-family: var(--font-sans);
}

.batch-clear-btn:hover {
  color: var(--text-secondary);
  border-color: var(--border-default);
}

.batch-bar-enter-active,
.batch-bar-leave-active {
  transition: all 0.2s ease;
}

.batch-bar-enter-from,
.batch-bar-leave-to {
  opacity: 0;
  transform: translateY(-8px);
}
</style>
