<template>
  <div class="task-list list-view">
    <div class="page-header">
      <h1 class="page-title">{{ t('route.title.tasks') }}</h1>
      <router-link :to="{ name: 'task-new' }" class="btn-primary">
        <el-icon :size="14"><Plus /></el-icon>
        {{ t('task.list.newTask') }}
      </router-link>
    </div>

    <ListTabBar
      v-model="activeTab"
      :tabs="tabItems"
      :aria-label="t('task.list.tabGroupAria')"
      @update:model-value="switchTab"
    />

    <!-- Active tab: running tasks + queued tasks -->
    <template v-if="activeTab === 'running'">
      <EmptyState
        v-if="runningTasks.length === 0 && queueItems.length === 0"
        :title="t('task.list.emptyActive')"
        :hint="t('task.list.emptyActiveHint')"
      />

      <template v-if="runningTasks.length > 0">
        <div class="section-divider divider-running">
          <span class="divider-pulse" aria-hidden="true"></span>
          <span class="divider-label">{{ t('common.status.running') }}</span>
          <span class="divider-count">{{ runningTasks.length }}</span>
        </div>
        <transition name="batch-bar">
          <div v-if="selectedRunningTasks.length > 0" class="batch-bar">
            <span class="batch-info">{{ t('task.list.selected') }} <strong>{{ selectedRunningTasks.length }}</strong> {{ t('task.list.itemsUnit') }}</span>
            <button class="batch-stop-btn" @click="handleBatchStop">
              <el-icon :size="14"><SwitchButton /></el-icon>
              {{ t('task.list.batchStop') }}
            </button>
            <button class="batch-clear-btn" @click="clearRunningSelection">{{ t('task.list.clearSelection') }}</button>
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
          <el-table-column :label="t('task.list.colName')" min-width="160">
            <template #default="{ row }">
              <router-link :to="{ name: 'task-detail', params: { id: row.id } }" class="task-name-link">{{ row.name }}</router-link>
            </template>
          </el-table-column>
          <el-table-column prop="model_name" :label="t('task.list.colModel')" width="110" sortable />
          <el-table-column prop="dataset_name" :label="t('task.list.colDataset')" width="110" sortable />
          <el-table-column prop="env_name" :label="t('task.list.colEnv')" width="100" sortable>
            <template #default="{ row }">
              <span class="env-tag">{{ row.env_name }}</span>
            </template>
          </el-table-column>
          <el-table-column v-if="hasGpu" label="GPU" width="80">
            <template #default="{ row }">
              <span class="gpu-tag">{{ formatGpu(row.gpu_assigned, t('common.auto')) }}</span>
            </template>
          </el-table-column>
          <el-table-column :label="t('task.list.colStatus')" width="110">
            <template #default>
              <StatusBadge status="running" />
            </template>
          </el-table-column>
          <el-table-column prop="created_at" :label="t('task.list.colCreatedAt')" width="170" sortable>
            <template #default="{ row }">
              <span class="mono-time">{{ formatDateTime(row.created_at) }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="started_at" :label="t('task.list.colStartedAt')" width="170" sortable>
            <template #default="{ row }">
              <span class="mono-time">{{ formatDateTime(row.started_at) }}</span>
            </template>
          </el-table-column>
          <el-table-column :label="t('task.list.colActions')" width="90" align="right">
            <template #default="{ row }">
              <div class="action-group">
                <router-link :to="{ name: 'task-detail', params: { id: row.id } }" class="action-btn" :title="t('task.list.viewDetail')">
                  <el-icon :size="15"><View /></el-icon>
                </router-link>
                <button class="action-btn action-stop" :title="t('task.list.stopTask')" @click="handleStop(row.id)">
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
          <span class="divider-label">{{ t('task.list.queued') }}</span>
          <span class="divider-count">{{ queueItems.length }}</span>
        </div>
        <transition name="batch-bar">
          <div v-if="selectedQueueItems.length > 0" class="batch-bar">
            <span class="batch-info">{{ t('task.list.selected') }} <strong>{{ selectedQueueItems.length }}</strong> {{ t('task.list.itemsUnit') }}</span>
            <button class="batch-cancel-btn" @click="handleBatchCancelQueue">
              <el-icon :size="14"><Delete /></el-icon>
              {{ t('task.list.batchCancel') }}
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
          <el-table-column :label="t('task.list.colPosition')" width="70">
            <template #default="{ $index }">
              <span class="pos-badge">#{{ (queuePage - 1) * queuePageSize + $index + 1 }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="id" label="ID" width="70" sortable />
          <el-table-column :label="t('task.list.colName')" min-width="160">
            <template #default="{ row }">
              <span class="task-name-text">{{ row.name }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="model_name" :label="t('task.list.colModel')" width="110" sortable />
          <el-table-column prop="dataset_name" :label="t('task.list.colDataset')" width="110" sortable />
          <el-table-column v-if="hasGpu" label="GPU" width="80">
            <template #default="{ row }">
              <span class="gpu-tag">{{ formatGpu(row.gpu_request, t('common.auto')) }}</span>
            </template>
          </el-table-column>
          <el-table-column prop="created_at" :label="t('task.list.colCreatedAt')" width="170" sortable>
            <template #default="{ row }">
              <span class="mono-time">{{ formatDateTime(row.created_at) }}</span>
            </template>
          </el-table-column>
          <el-table-column :label="t('task.list.colActions')" width="110" align="right">
            <template #default="{ row, $index }">
              <div class="action-group">
                <button v-if="$index > 0" class="action-btn" :title="t('task.list.moveUp')" @click="moveUp($index)">
                  <el-icon :size="14"><ArrowUp /></el-icon>
                </button>
                <button v-if="$index < queueItems.length - 1" class="action-btn" :title="t('task.list.moveDown')" @click="moveDown($index)">
                  <el-icon :size="14"><ArrowDown /></el-icon>
                </button>
                <button class="action-btn action-delete" :title="t('task.list.cancelTask')" @click="handleCancelQueue(row.id)">
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
      <EmptyState
        v-if="tasks.length === 0"
        :title="t('task.list.empty')"
        :hint="t('task.list.emptyHint')"
      />

      <template v-else>
        <transition name="batch-bar">
          <div v-if="selectedTasks.length > 0" class="batch-bar">
            <span class="batch-info">{{ t('task.list.selected') }} <strong>{{ selectedTasks.length }}</strong> {{ t('task.list.itemsUnit') }}</span>
            <button class="batch-delete-btn" @click="handleBatchDelete">
              <el-icon :size="14"><Delete /></el-icon>
              {{ t('task.list.batchDelete') }}
            </button>
            <button class="batch-clear-btn" @click="clearSelection">{{ t('task.list.clearSelection') }}</button>
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
        <el-table-column :label="t('task.list.colName')" min-width="160">
          <template #default="{ row }">
            <router-link :to="{ name: 'task-detail', params: { id: row.id } }" class="task-name-link">{{ row.name }}</router-link>
          </template>
        </el-table-column>
        <el-table-column prop="model_name" :label="t('task.list.colModel')" width="110" sortable />
        <el-table-column prop="dataset_name" :label="t('task.list.colDataset')" width="110" sortable />
        <el-table-column prop="env_name" :label="t('task.list.colEnv')" width="100" sortable>
          <template #default="{ row }">
            <span class="env-tag">{{ row.env_name }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="status" :label="t('task.list.colStatus')" width="110" sortable>
          <template #default="{ row }">
            <StatusBadge :status="row.status" />
          </template>
        </el-table-column>
        <el-table-column prop="created_at" :label="t('task.list.colCreatedAt')" width="170" sortable>
          <template #default="{ row }">
            <span class="mono-time">{{ formatDateTime(row.created_at) }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="started_at" :label="t('task.list.colStartedAt')" width="170" sortable>
          <template #default="{ row }">
            <span class="mono-time">{{ formatDateTime(row.started_at) }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="finished_at" :label="t('task.list.colFinishedAt')" width="170" sortable>
          <template #default="{ row }">
            <span class="mono-time">{{ formatDateTime(row.finished_at) }}</span>
          </template>
        </el-table-column>
        <el-table-column :label="t('task.list.colActions')" width="90" align="right">
          <template #default="{ row }">
            <div class="action-group">
              <router-link :to="{ name: 'task-detail', params: { id: row.id } }" class="action-btn" :title="t('task.list.viewDetail')">
                <el-icon :size="15"><View /></el-icon>
              </router-link>
              <button
                v-if="row.status !== 'running'"
                class="action-btn action-delete"
                :title="t('task.list.deleteTask')"
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
import '@/styles/list-shared.css'
import { ref, computed, watch, nextTick, onBeforeUnmount } from 'vue'
import { useQuery, useQueryClient } from '@tanstack/vue-query'
import { useRoute, useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { ElTable } from 'element-plus'
import { Plus, View, SwitchButton, ArrowUp, ArrowDown, Delete, Clock } from '@element-plus/icons-vue'
import Sortable from '@/utils/sortable'
import type { SortableEvent } from 'sortablejs'
import { formatDateTime } from '@/utils/date'
import { formatGpu } from '@/utils/format'
import { listTasks, stopTask, deleteTask, type TaskInfo } from '@/api/tasks'
import { getQueue, reorderQueue, type QueueItem } from '@/api/settings'
import { useSystemCapabilities } from '@/composables/useSystemCapabilities'
import { useTabSort } from '@/composables/useTabSort'
import ListTabBar from '@/components/common/ListTabBar.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import StatusBadge from '@/components/common/StatusBadge.vue'

const { t } = useI18n()

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

const { currentSort, onSortChange } = useTabSort(DEFAULT_SORT, activeTab)

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

const tabItems = computed(() => [
  { label: 'task.list.tabAll', value: 'all' },
  { label: 'task.list.tabActive', value: 'running', badge: activeCount.value || undefined },
  { label: 'common.status.completed', value: 'completed' },
  { label: 'common.status.failed', value: 'failed' },
  { label: 'common.status.stopped', value: 'stopped' },
])

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
    ElMessage.error(t('task.list.reorderFailedRestored'))
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
  await ElMessageBox.confirm(t('task.list.stopConfirmMsg'), t('task.list.stopConfirmTitle'), {
    confirmButtonText: t('common.stop'),
    cancelButtonText: t('common.cancel'),
    type: 'warning',
  })
  ElMessage.success(t('task.detail.stopSignalSent'))
  await stopTask(id)
  ElMessage.success(t('task.list.stopped'))
  invalidateTasks()
}

const handleCancelQueue = async (id: number) => {
  await ElMessageBox.confirm(t('task.list.cancelQueueConfirmMsg'), t('task.list.cancelQueueConfirmTitle'), {
    confirmButtonText: t('task.list.cancelTask'),
    cancelButtonText: t('common.back'),
    type: 'warning',
  })
  await stopTask(id)
  ElMessage.success(t('task.list.canceled'))
  invalidateTasks()
}

const handleDelete = async (id: number) => {
  await ElMessageBox.confirm(t('task.list.deleteConfirmMsg'), t('task.list.deleteConfirmTitle'), {
    confirmButtonText: t('common.delete'),
    cancelButtonText: t('common.cancel'),
    type: 'warning',
  })
  await deleteTask(id)
  ElMessage.success(t('task.list.deleted'))
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
  const runningSelected = selectedTasks.value.filter(taskVal => taskVal.status === 'running')
  if (runningSelected.length > 0) {
    ElMessage.warning(t('task.list.batchDeleteRunningBlocked', { n: runningSelected.length }))
    return
  }
  await ElMessageBox.confirm(t('task.list.batchDeleteConfirmMsg', { n: count }), t('task.list.batchDelete'), {
    confirmButtonText: t('common.delete'),
    cancelButtonText: t('common.cancel'),
    type: 'warning',
  })
  await Promise.all(selectedTasks.value.map(taskVal => deleteTask(taskVal.id)))
  ElMessage.success(t('task.list.deletedN', { n: count }))
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
    t('task.list.batchStopConfirmMsg', { n: count }),
    t('task.list.batchStop'),
    {
      confirmButtonText: t('common.stop'),
      cancelButtonText: t('common.cancel'),
      type: 'warning',
    }
  )
  ElMessage.success(t('task.detail.stopSignalSent'))
  await Promise.all(selectedRunningTasks.value.map(taskVal => stopTask(taskVal.id)))
  ElMessage.success(t('task.list.stoppedN', { n: count }))
  clearRunningSelection()
  invalidateTasks()
}

// --- Queue batch cancel (selection is driven by MultiDrag clicks) ---
const handleBatchCancelQueue = async () => {
  const count = selectedQueueItems.value.length
  await ElMessageBox.confirm(
    t('task.list.batchCancelConfirmMsg', { n: count }),
    t('task.list.batchCancel'),
    {
      confirmButtonText: t('task.list.cancelTask'),
      cancelButtonText: t('common.back'),
      type: 'warning',
    }
  )
  await Promise.all(selectedQueueItems.value.map(taskVal => stopTask(taskVal.id)))
  ElMessage.success(t('task.list.canceledN', { n: count }))
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
