<template>
  <div class="search-list">
    <header class="list-header">
      <h1 class="page-title">{{ t('search.listTitle') }}</h1>
      <el-button type="primary" @click="router.push({ name: 'search-new' })">
        <el-icon style="margin-right: 4px"><Plus /></el-icon>
        {{ t('search.newSearch') }}
      </el-button>
    </header>

    <div class="tabs">
      <button
        v-for="tab in tabs"
        :key="tab.key"
        class="tab-btn"
        :class="{ active: activeTab === tab.key }"
        @click="activeTab = tab.key"
      >
        {{ t(tab.label) }}
        <span class="tab-count">{{ counts[tab.key] ?? '' }}</span>
      </button>
    </div>

    <el-skeleton :loading="isPending && !data" animated>
      <template #default>
        <el-table
          :data="data?.items ?? []"
          stripe
          @row-click="(row: SearchTaskInfo) => router.push({ name: 'search-detail', params: { id: row.id } })"
          class="search-table"
          empty-text="—"
        >
          <el-table-column prop="name" :label="t('search.colName')" min-width="200" show-overflow-tooltip />
          <el-table-column prop="model_name" :label="t('search.colModel')" width="120" />
          <el-table-column prop="dataset_name" :label="t('search.colDataset')" width="140" />
          <el-table-column :label="t('search.colStatus')" width="130">
            <template #default="{ row }">
              <span class="status-badge" :style="{ '--dot-color': statusMap[row.status]?.color }">
                <span class="status-dot" :class="{ pulse: row.status === 'running' }"></span>
                <span>{{ statusMap[row.status] ? t(statusMap[row.status].label) : row.status }}</span>
              </span>
            </template>
          </el-table-column>
          <el-table-column v-if="hasGpu" :label="t('search.colGpu')" width="90">
            <template #default="{ row }">{{ gpuText(row) }}</template>
          </el-table-column>
          <el-table-column :label="t('search.colCreated')" width="170">
            <template #default="{ row }">
              <span class="mono">{{ formatDateTime(row.created_at) }}</span>
            </template>
          </el-table-column>
          <el-table-column :label="t('search.colActions')" width="120" align="right" fixed="right">
            <template #default="{ row }">
              <div class="row-actions" @click.stop>
                <el-button
                  v-if="row.status === 'running'"
                  link
                  type="warning"
                  @click="handleStop(row.id)"
                >{{ t('common.stop') }}</el-button>
                <el-button
                  v-if="!['running', 'stopping', 'pending'].includes(row.status)"
                  link
                  type="danger"
                  @click="handleDelete(row)"
                >{{ t('common.delete') }}</el-button>
              </div>
            </template>
          </el-table-column>
        </el-table>

        <el-pagination
          v-if="(data?.total ?? 0) > pageSize"
          class="pager"
          layout="prev, pager, next, total"
          :total="data?.total ?? 0"
          :page-size="pageSize"
          :current-page="page"
          @current-change="page = $event"
        />
      </template>
    </el-skeleton>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { useQuery, useQueryClient } from '@tanstack/vue-query'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus } from '@element-plus/icons-vue'
import { formatDateTime } from '@/utils/date'
import { listSearches, stopSearch, deleteSearch, type SearchTaskInfo } from '@/api/search'
import { useSystemCapabilities } from '@/composables/useSystemCapabilities'
import { statusMap } from '@/composables/useStatusMap'

const { t } = useI18n()
const router = useRouter()
const queryClient = useQueryClient()
const { hasGpu } = useSystemCapabilities()

const tabs = [
  { key: '', label: 'common.status.all' },
  { key: 'running', label: 'common.status.running' },
  { key: 'completed', label: 'common.status.completed' },
  { key: 'failed', label: 'common.status.failed' },
  { key: 'stopped', label: 'common.status.stopped' },
] as const

const activeTab = ref<string>('')
const page = ref(1)
const pageSize = 20

// Tab counts: an unfiltered query drives the per-status badges cheaply.
const allQuery = useQuery({
  queryKey: ['searches-list', ''],
  queryFn: () => listSearches({ page: 1, page_size: 1 }),
  refetchInterval: 30000,
})
const countsQuery = useQuery({
  queryKey: ['searches-counts'],
  queryFn: async () => {
    const entries = await Promise.all(
      (['running', 'completed', 'failed', 'stopped'] as const).map(async (s) => {
        const r = await listSearches({ status: s, page: 1, page_size: 1 })
        return [s, r.total] as const
      }),
    )
    return Object.fromEntries(entries) as Record<string, number>
  },
  refetchInterval: 30000,
})
const counts = computed<Record<string, number>>(() => ({
  '': (allQuery.data.value?.total ?? 0),
  ...(countsQuery.data.value ?? {}),
}))

const statusParam = computed(() => (activeTab.value === '' ? undefined : activeTab.value))

const { data, isPending } = useQuery({
  queryKey: computed(() => ['searches-list', statusParam.value, page.value]),
  queryFn: () => listSearches({ status: statusParam.value, page: page.value, page_size: pageSize }),
  refetchInterval: 30000,
})

const gpuText = (row: SearchTaskInfo) => {
  const val = row.gpu_assigned ?? row.gpu_request
  if (val === null || val === undefined) return row.status === 'pending' ? t('search.gpuAuto') : '—'
  return `GPU ${val}`
}

async function handleStop(id: number) {
  try {
    await stopSearch(id)
    ElMessage.success(t('search.stopSent'))
  } finally {
    queryClient.invalidateQueries({ queryKey: ['searches-list'] })
  }
}

async function handleDelete(row: SearchTaskInfo) {
  try {
    await ElMessageBox.confirm(t('search.deleteConfirm', { name: row.name }), t('common.delete'), {
      confirmButtonText: t('common.delete'),
      cancelButtonText: t('common.cancel'),
      type: 'warning',
    })
  } catch {
    return
  }
  await deleteSearch(row.id)
  ElMessage.success(t('search.deleted'))
  queryClient.invalidateQueries({ queryKey: ['searches-list'] })
  queryClient.invalidateQueries({ queryKey: ['searches-counts'] })
}
</script>

<style scoped>
.search-list {
  display: flex;
  flex-direction: column;
  gap: 16px;
  height: 100%;
  min-height: 0;
  color: var(--text-primary);
}

.list-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.page-title {
  font-size: 22px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
  letter-spacing: -0.3px;
}

.tabs {
  display: flex;
  gap: 4px;
}

.tab-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-sm);
  background: var(--bg-surface);
  color: var(--text-secondary);
  font-size: 13px;
  cursor: pointer;
  transition: all 0.15s ease;
}

.tab-btn:hover {
  border-color: var(--accent-blue);
  color: var(--accent-blue);
}

.tab-btn.active {
  background: color-mix(in srgb, var(--accent-blue) 12%, transparent);
  border-color: var(--accent-blue);
  color: var(--accent-blue);
}

.tab-count {
  font-size: 11px;
  color: var(--text-tertiary);
  font-family: var(--font-mono);
}

.search-table {
  cursor: pointer;
}

.row-actions {
  display: flex;
  justify-content: flex-end;
  gap: 4px;
}

.mono {
  font-family: var(--font-mono);
  font-size: 12.5px;
}

.status-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
}

.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--dot-color, var(--text-tertiary));
}

.status-dot.pulse {
  animation: pulse-glow 2s ease-in-out infinite;
}

@keyframes pulse-glow {
  0%, 100% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--accent-blue) 50%, transparent); opacity: 1; }
  50% { box-shadow: 0 0 0 5px color-mix(in srgb, var(--accent-blue) 0%, transparent); opacity: 0.7; }
}

.pager {
  justify-content: center;
  margin-top: 4px;
}
</style>
