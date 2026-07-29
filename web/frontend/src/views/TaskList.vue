<template>
  <div class="task-list">
    <div class="page-header">
      <h1 class="page-title">训练任务</h1>
      <router-link :to="{ name: 'task-new' }" class="btn-primary">
        <el-icon :size="14"><Plus /></el-icon>
        新建任务
      </router-link>
    </div>

    <div class="tab-bar" role="tablist" aria-label="任务状态分组">
      <button
        v-for="tab in tabs"
        :key="tab.value"
        role="tab"
        :aria-selected="activeTab === tab.value"
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
        <div class="section-divider divider-running">
          <span class="divider-pulse" aria-hidden="true"></span>
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
          :default-sort="currentSort"
          @sort-change="onSortChange"
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
          <el-table-column v-if="hasGpu" label="GPU" width="80">
            <template #default="{ row }">
              <span class="gpu-tag">{{ formatGpu(row.gpu_assigned) }}</span>
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
          <el-table-column prop="started_at" label="开始时间" width="170" sortable>
            <template #default="{ row }">
              <span class="mono-time">{{ formatDateTime(row.started_at) }}</span>
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
        <div class="section-divider divider-queued">
          <el-icon :size="13" class="divider-icon" aria-hidden="true"><Clock /></el-icon>
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
          </div>
        </transition>
        <el-table
          ref="queueTableRef"
          :data="paginatedQueueItems"
          row-key="id"
          size="small"
          class="task-table queue-table"
        >
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
          <el-table-column v-if="hasGpu" label="GPU" width="80">
            <template #default="{ row }">
              <span class="gpu-tag">{{ formatGpu(row.gpu_request) }}</span>
            </template>
          </el-table-column>
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
          :key="activeTab"
          :data="tasks"
          row-key="id"
          size="small"
          class="task-table"
          v-loading="loading"
          :default-sort="currentSort"
          @sort-change="onSortChange"
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
        <el-table-column prop="started_at" label="开始时间" width="170" sortable>
          <template #default="{ row }">
            <span class="mono-time">{{ formatDateTime(row.started_at) }}</span>
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
import { ref, computed, watch, nextTick, onBeforeUnmount } from 'vue'
import { useQuery, useQueryClient } from '@tanstack/vue-query'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { ElTable } from 'element-plus'
import { Plus, View, SwitchButton, ArrowUp, ArrowDown, Delete, Document, Clock } from '@element-plus/icons-vue'
import Sortable from '@/utils/sortable'
import type { SortableEvent } from 'sortablejs'
import { formatDateTime } from '@/utils/date'
import { listTasks, stopTask, deleteTask, type TaskInfo } from '@/api/tasks'
import { getQueue, reorderQueue, type QueueItem } from '@/api/settings'
import { useSystemCapabilities } from '@/composables/useSystemCapabilities'

const { hasGpu } = useSystemCapabilities()

const route = useRoute()
const router = useRouter()
const queryClient = useQueryClient()

const activeTab = ref(route.query.tab?.toString() || sessionStorage.getItem('taskListTab') || 'running')

type SortOrder = 'ascending' | 'descending'
interface SortState {
  prop: string
  order: SortOrder
}

// Default sort per tab. Finished-task tabs default to newest-finished-first;
// the running tab has no finished_at column so it keeps id order.
const DEFAULT_SORT: Record<string, SortState> = {
  running: { prop: 'id', order: 'ascending' },
  all: { prop: 'finished_at', order: 'descending' },
  completed: { prop: 'finished_at', order: 'descending' },
  failed: { prop: 'finished_at', order: 'descending' },
  stopped: { prop: 'finished_at', order: 'descending' },
}

// Per-tab sort overrides, persisted across tab switches.
const tabSortOverrides = ref<Record<string, SortState>>({})

const currentSort = computed<SortState>(
  () => tabSortOverrides.value[activeTab.value] ?? DEFAULT_SORT[activeTab.value] ?? DEFAULT_SORT.all
)

const onSortChange = ({ prop, order }: { prop: string | null; order: SortOrder | null }) => {
  if (prop && order) {
    tabSortOverrides.value[activeTab.value] = { prop, order }
  } else {
    delete tabSortOverrides.value[activeTab.value]
  }
}

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

const formatGpu = (val: number | null) =>
  val === null || val === undefined ? '自动' : `GPU ${val}`

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
  refetchInterval: 30000,
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

// --- Queue reordering (drag-and-drop + up/down buttons) ---
let sortableInstance: Sortable | undefined
let dragSnapshot: Element[] | undefined

const pageOffset = () => (queuePage.value - 1) * queuePageSize.value

const setQueueItemsData = (items: QueueItem[]) =>
  queryClient.setQueriesData(
    { queryKey: ['tasks-list'] },
    (old: TasksData | undefined): TasksData | undefined => (old ? { ...old, queueItems: items } : old),
  )

// Optimistic write to the cache, then persist; roll back on failure.
const applyQueueOrder = async (next: QueueItem[], prev: QueueItem[]) => {
  const nextIds = next.map(q => q.id)
  const prevIds = prev.map(q => q.id)
  if (nextIds.length === prevIds.length && nextIds.every((id, i) => id === prevIds[i])) return
  await queryClient.cancelQueries({ queryKey: ['tasks-list'] })
  setQueueItemsData(next)
  try {
    await reorderQueue(nextIds)
    invalidateTasks()
  } catch {
    setQueueItemsData(prev)
    ElMessage.error('调整排队顺序失败，已恢复原顺序')
  }
}

const moveUp = async (idx: number) => {
  const actualIdx = pageOffset() + idx
  if (actualIdx <= 0) return
  const items = [...queueItems.value]
  ;[items[actualIdx - 1], items[actualIdx]] = [items[actualIdx], items[actualIdx - 1]]
  await applyQueueOrder(items, queueItems.value)
}

const moveDown = async (idx: number) => {
  const actualIdx = pageOffset() + idx
  if (actualIdx >= queueItems.value.length - 1) return
  const items = [...queueItems.value]
  ;[items[actualIdx + 1], items[actualIdx]] = [items[actualIdx], items[actualIdx + 1]]
  await applyQueueOrder(items, queueItems.value)
}

const getQueueTbody = (): HTMLElement | null =>
  (queueTableRef.value?.$el as HTMLElement | undefined)?.querySelector('.el-table__body-wrapper tbody') ?? null

// Reflect MultiDrag's selection (rows carrying selectedClass) into reactive state,
// so the batch-cancel bar and count stay in sync after clicks.
const syncSelectionFromDom = () => {
  const tbody = getQueueTbody()
  if (!tbody) {
    selectedQueueItems.value = []
    return
  }
  const selected: QueueItem[] = []
  Array.from(tbody.querySelectorAll('tr')).forEach((tr, i) => {
    if (tr.classList.contains('queue-row-selected')) {
      const item = paginatedQueueItems.value[i]
      if (item) selected.push(item)
    }
  })
  selectedQueueItems.value = selected
}

const onQueueClick = () => {
  void nextTick(syncSelectionFromDom)
}

const onDragStart = (evt: SortableEvent) => {
  const tbody = evt.item.parentNode as HTMLElement | null
  if (!tbody) return
  dragSnapshot = Array.from(tbody.children)
  // Stamp each row's id so the post-drag order can be read straight from the DOM.
  Array.from(tbody.querySelectorAll('tr')).forEach((tr, i) => {
    const item = paginatedQueueItems.value[i]
    if (item) (tr as HTMLElement).dataset.queueId = String(item.id)
  })
}

const onDragEnd = (evt: SortableEvent) => {
  const tbody = evt.item.parentNode as HTMLElement | null
  // MultiDrag has already rearranged the rows; read the new page order from the
  // DOM before restoring the pre-drag order (so Vue owns the final render).
  let next: QueueItem[] | null = null
  if (tbody) {
    const map = new Map(paginatedQueueItems.value.map(q => [q.id, q]))
    const newPageItems: QueueItem[] = []
    for (const tr of Array.from(tbody.querySelectorAll('tr'))) {
      const item = map.get(Number((tr as HTMLElement).dataset.queueId))
      if (item) newPageItems.push(item)
    }
    if (newPageItems.length === paginatedQueueItems.value.length && newPageItems.length > 0) {
      const start = pageOffset()
      next = [
        ...queueItems.value.slice(0, start),
        ...newPageItems,
        ...queueItems.value.slice(start + newPageItems.length),
      ]
    }
  }

  if (dragSnapshot && tbody) dragSnapshot.forEach(node => tbody.appendChild(node))
  dragSnapshot = undefined

  if (next) void applyQueueOrder(next, queueItems.value)
}

const initSortable = () => {
  if (sortableInstance) return
  const tbody = getQueueTbody()
  if (!tbody || tbody.children.length === 0) return
  tbody.addEventListener('click', onQueueClick)
  sortableInstance = Sortable.create(tbody, {
    multiDrag: true,
    selectedClass: 'queue-row-selected',
    animation: 150,
    ghostClass: 'queue-drag-ghost',
    chosenClass: 'queue-drag-chosen',
    dragClass: 'queue-drag-drag',
    filter: '.action-group',
    preventOnFilter: false,
    onStart: onDragStart,
    onEnd: onDragEnd,
  })
}

const teardownSortable = () => {
  if (sortableInstance) {
    sortableInstance.el.removeEventListener('click', onQueueClick)
    sortableInstance.destroy()
  }
  sortableInstance = undefined
}

// Re-init only when the <tbody> node is replaced (tab switch / page-size change),
// not on every 30s refetch (same node → no-op), so an in-flight drag is never interrupted.
watch(
  () => [activeTab.value, queuePage.value, queuePageSize.value, queueItems.value] as const,
  async () => {
    if (activeTab.value !== 'running' || paginatedQueueItems.value.length === 0) {
      teardownSortable()
      return
    }
    await nextTick()
    const tbody = getQueueTbody()
    if (tbody && sortableInstance?.el === tbody) return
    teardownSortable()
    initSortable()
  },
  { immediate: true },
)

onBeforeUnmount(teardownSortable)

const handleStop = async (id: number) => {
  await ElMessageBox.confirm('确定停止该任务？正在进行的训练进度将丢失。', '确认停止', {
    confirmButtonText: '停止',
    cancelButtonText: '取消',
    type: 'warning',
  })
  ElMessage.success('已发送停止信号')
  await stopTask(id)
  ElMessage.success('任务已停止')
  invalidateTasks()
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
  invalidateTasks()
}

// --- Queue batch cancel (selection is driven by MultiDrag clicks) ---
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
  selectedQueueItems.value = []
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

.divider-pulse {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--accent-blue);
  box-shadow: 0 0 0 0 var(--soft-blue);
  animation: divider-pulse 2s infinite;
}

.divider-queued .divider-icon {
  color: var(--text-tertiary);
}

@keyframes divider-pulse {
  0% { box-shadow: 0 0 0 0 var(--soft-blue); }
  70% { box-shadow: 0 0 0 5px transparent; }
  100% { box-shadow: 0 0 0 0 transparent; }
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

.gpu-tag {
  display: inline-block;
  padding: 2px 8px;
  background: var(--soft-green);
  border-radius: 20px;
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--accent-green);
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
  background: var(--soft-orange);
}

.action-delete:hover {
  color: var(--accent-red);
  background: var(--soft-red);
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
  background: var(--soft-red);
  color: var(--accent-red);
  border: 1px solid var(--border-muted);
  border-radius: var(--radius-sm);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s;
  font-family: var(--font-sans);
}

.batch-delete-btn:hover {
  background: var(--accent-red);
  border-color: var(--accent-red);
  color: #fff;
}

.batch-stop-btn {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 14px;
  background: var(--soft-orange);
  color: var(--accent-orange);
  border: 1px solid var(--border-muted);
  border-radius: var(--radius-sm);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s;
  font-family: var(--font-sans);
}

.batch-stop-btn:hover {
  background: var(--accent-orange);
  border-color: var(--accent-orange);
  color: #fff;
}

.batch-cancel-btn {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 4px 14px;
  background: var(--soft-red);
  color: var(--accent-red);
  border: 1px solid var(--border-muted);
  border-radius: var(--radius-sm);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s;
  font-family: var(--font-sans);
}

.batch-cancel-btn:hover {
  background: var(--accent-red);
  border-color: var(--accent-red);
  color: #fff;
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

@media (prefers-reduced-motion: reduce) {
  .divider-pulse { animation: none; }
  .queue-drag-ghost, .queue-drag-chosen, .queue-drag-drag, .queue-row-selected { transition: none; }
}

/* Queue row states: default / hover / selected / selected-hover, day & dark.
   Backgrounds are applied to the cell (td) — el-table paints row bg on td,
   which would otherwise cover a tr-level background and the selected tint. */
.queue-table {
  --el-table-row-hover-bg-color: #eaeef2;
  --queue-selected-bg: rgba(9, 105, 218, 0.12);
  --queue-selected-hover-bg: rgba(9, 105, 218, 0.20);
}

html.dark .queue-table {
  --el-table-row-hover-bg-color: #222a3a;
  --queue-selected-bg: rgba(56, 139, 253, 0.18);
  --queue-selected-hover-bg: rgba(56, 139, 253, 0.28);
}

:deep(.queue-row-selected .el-table__cell),
:deep(.queue-drag-chosen .el-table__cell) {
  background-color: var(--queue-selected-bg) !important;
}

:deep(.queue-row-selected:hover .el-table__cell) {
  background-color: var(--queue-selected-hover-bg) !important;
}

:deep(.queue-drag-ghost) {
  opacity: 0.4;
}

:deep(.queue-drag-drag) {
  opacity: 0.95;
  box-shadow: 0 4px 14px rgba(0, 0, 0, 0.18);
}
</style>
