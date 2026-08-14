<template>
  <div class="search-list list-view">
    <div class="page-header">
      <h1 class="page-title">{{ t('search.listTitle') }}</h1>
      <router-link :to="{ name: 'search-new' }" class="btn-primary">
        <el-icon :size="14"><Plus /></el-icon>
        {{ t('search.newSearch') }}
      </router-link>
    </div>

    <ListTabBar
      v-model="activeTab"
      :tabs="tabItems"
      :aria-label="t('search.tabGroupAria')"
      @update:model-value="switchTab"
    />

    <EmptyState
      v-if="!loading && tasks.length === 0"
      :title="t('search.emptyTitle')"
      :hint="t('search.emptyHint')"
    />

    <template v-else>
      <transition name="batch-bar">
        <div v-if="selectedTasks.length > 0" class="batch-bar">
          <span class="batch-info">{{ t('search.selected') }} <strong>{{ selectedTasks.length }}</strong> {{ t('search.itemsUnit') }}</span>
          <button class="batch-delete-btn" @click="handleBatchDelete">
            <el-icon :size="14"><Delete /></el-icon>
            {{ t('search.batchDelete') }}
          </button>
          <button class="batch-clear-btn" @click="clearSelection">{{ t('search.clearSelection') }}</button>
        </div>
      </transition>

      <el-table
        ref="tableRef"
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
        <el-table-column :label="t('search.colName')" min-width="160">
          <template #default="{ row }">
            <router-link :to="{ name: 'search-detail', params: { id: row.id } }" class="task-name-link">{{ row.name }}</router-link>
          </template>
        </el-table-column>
        <el-table-column prop="model_name" :label="t('search.colModel')" width="110" sortable />
        <el-table-column prop="dataset_name" :label="t('search.colDataset')" width="110" sortable />
        <el-table-column prop="env_name" :label="t('search.colEnv')" width="100" sortable>
          <template #default="{ row }">
            <span class="env-tag">{{ row.env_name }}</span>
          </template>
        </el-table-column>
        <el-table-column v-if="hasGpu" label="GPU" width="80">
          <template #default="{ row }">
            <span class="gpu-tag">{{ formatGpu(row.gpu_assigned ?? row.gpu_request, t('common.auto')) }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="status" :label="t('search.colStatus')" width="110" sortable>
          <template #default="{ row }">
            <StatusBadge :status="row.status" />
          </template>
        </el-table-column>
        <el-table-column prop="created_at" :label="t('search.colCreatedAt')" width="170" sortable>
          <template #default="{ row }">
            <span class="mono-time">{{ formatDateTime(row.created_at) }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="started_at" :label="t('search.colStartedAt')" width="170" sortable>
          <template #default="{ row }">
            <span class="mono-time">{{ formatDateTime(row.started_at) }}</span>
          </template>
        </el-table-column>
        <el-table-column prop="finished_at" :label="t('search.colFinishedAt')" width="170" sortable>
          <template #default="{ row }">
            <span class="mono-time">{{ formatDateTime(row.finished_at) }}</span>
          </template>
        </el-table-column>
        <el-table-column :label="t('search.colActions')" width="120" align="right">
          <template #default="{ row }">
            <div class="action-group">
              <router-link :to="{ name: 'search-detail', params: { id: row.id } }" class="action-btn" :title="t('search.viewDetail')">
                <el-icon :size="15"><View /></el-icon>
              </router-link>
              <button
                v-if="['running', 'stopping', 'pending'].includes(row.status)"
                class="action-btn action-stop"
                :title="t('search.stopTask')"
                @click="handleStop(row.id)"
              >
                <el-icon :size="14"><SwitchButton /></el-icon>
              </button>
              <button
                v-if="!['running', 'stopping'].includes(row.status)"
                class="action-btn action-delete"
                :title="t('search.deleteTask')"
                @click="handleDelete(row)"
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
  </div>
</template>

<script setup lang="ts">
import '@/styles/list-shared.css'
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute, useRouter } from 'vue-router'
import { useQuery, useQueryClient } from '@tanstack/vue-query'
import { ElMessage, ElMessageBox } from 'element-plus'
import type { ElTable } from 'element-plus'
import { Plus, View, Delete, SwitchButton } from '@element-plus/icons-vue'
import { formatDateTime } from '@/utils/date'
import { formatGpu } from '@/utils/format'
import { listSearches, stopSearch, deleteSearch, type SearchTaskInfo } from '@/api/search'
import { useSystemCapabilities } from '@/composables/useSystemCapabilities'
import { useTabSort } from '@/composables/useTabSort'
import ListTabBar from '@/components/common/ListTabBar.vue'
import EmptyState from '@/components/common/EmptyState.vue'
import StatusBadge from '@/components/common/StatusBadge.vue'

const { t } = useI18n()
const route = useRoute()
const router = useRouter()
const queryClient = useQueryClient()
const { hasGpu } = useSystemCapabilities()

const activeTab = ref(route.query.tab?.toString() || sessionStorage.getItem('searchListTab') || 'all')
const page = ref(1)
const pageSize = ref(20)

type SortOrder = 'ascending' | 'descending'
interface SortState {
  prop: string
  order: SortOrder
}

// Same default-sort policy as the training list: newest-finished-first on
// terminal tabs, id order on the active tab.
const DEFAULT_SORT: Record<string, SortState> = {
  running: { prop: 'id', order: 'ascending' },
  all: { prop: 'finished_at', order: 'descending' },
  completed: { prop: 'finished_at', order: 'descending' },
  failed: { prop: 'finished_at', order: 'descending' },
  stopped: { prop: 'finished_at', order: 'descending' },
}

const { currentSort, onSortChange } = useTabSort(DEFAULT_SORT, activeTab)

const selectedTasks = ref<SearchTaskInfo[]>([])
const tableRef = ref<InstanceType<typeof ElTable>>()

// Running count drives the tab badge (running searches only; searches never
// sit in the reorderable queue UI).
const countsQuery = useQuery({
  queryKey: ['searches-counts'],
  queryFn: async () => {
    const r = await listSearches({ status: 'running', page: 1, page_size: 1 })
    return r.total
  },
  refetchInterval: 30000,
})

const tabItems = computed(() => [
  { label: 'search.tabAll', value: 'all' },
  { label: 'search.tabActive', value: 'running', badge: countsQuery.data.value || undefined },
  { label: 'common.status.completed', value: 'completed' },
  { label: 'common.status.failed', value: 'failed' },
  { label: 'common.status.stopped', value: 'stopped' },
])

const statusParam = computed(() => (activeTab.value === 'all' ? undefined : activeTab.value))

const { data, isPending } = useQuery({
  queryKey: computed(() => ['searches-list', statusParam.value, page.value, pageSize.value]),
  queryFn: () => listSearches({ status: statusParam.value, page: page.value, page_size: pageSize.value }),
  refetchInterval: 30000,
})

const tasks = computed(() => data.value?.items ?? [])
const total = computed(() => data.value?.total ?? 0)
const loading = computed(() => isPending.value)

const invalidate = () => {
  queryClient.invalidateQueries({ queryKey: ['searches-list'] })
  queryClient.invalidateQueries({ queryKey: ['searches-counts'] })
}

const switchTab = (tab: string) => {
  activeTab.value = tab
  page.value = 1
  selectedTasks.value = []
  sessionStorage.setItem('searchListTab', tab)
  router.replace({ query: tab !== 'all' ? { tab } : {} })
}

const handlePageChange = (p: number) => {
  page.value = p
}

const handlePageSizeChange = (s: number) => {
  pageSize.value = s
  page.value = 1
}

const handleSelectionChange = (rows: SearchTaskInfo[]) => {
  selectedTasks.value = rows
}

const clearSelection = () => {
  tableRef.value?.clearSelection()
}

const handleStop = async (id: number) => {
  try {
    await ElMessageBox.confirm(t('search.stopConfirmMsg'), t('search.stopConfirmTitle'), {
      confirmButtonText: t('common.stop'),
      cancelButtonText: t('common.cancel'),
      type: 'warning',
    })
  } catch {
    return
  }
  await stopSearch(id)
  ElMessage.success(t('search.stopSent'))
  invalidate()
}

const handleDelete = async (row: SearchTaskInfo) => {
  await ElMessageBox.confirm(t('search.deleteConfirm', { name: row.name }), t('common.delete'), {
    confirmButtonText: t('common.delete'),
    cancelButtonText: t('common.cancel'),
    type: 'warning',
  })
  await deleteSearch(row.id)
  ElMessage.success(t('search.deleted'))
  invalidate()
}

const handleBatchDelete = async () => {
  const count = selectedTasks.value.length
  const runningSelected = selectedTasks.value.filter(taskVal => ['running', 'stopping', 'pending'].includes(taskVal.status))
  if (runningSelected.length > 0) {
    ElMessage.warning(t('search.batchDeleteRunningBlocked', { n: runningSelected.length }))
    return
  }
  try {
    await ElMessageBox.confirm(t('search.batchDeleteConfirmMsg', { n: count }), t('search.batchDelete'), {
      confirmButtonText: t('common.delete'),
      cancelButtonText: t('common.cancel'),
      type: 'warning',
    })
  } catch {
    return
  }
  try {
    await Promise.all(selectedTasks.value.map(taskVal => deleteSearch(taskVal.id)))
    ElMessage.success(t('search.deletedN', { n: count }))
    clearSelection()
    invalidate()
  } catch (err: any) {
    ElMessage.error(err?.message || t('search.deleteFailed'))
    invalidate()
  }
}
</script>

<style scoped>
.search-list {
  display: flex;
  flex-direction: column;
  gap: 20px;
}
</style>
