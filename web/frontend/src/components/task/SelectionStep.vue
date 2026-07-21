<template>
  <div class="selection-step">
    <div class="section">
      <div class="section-label">运行环境</div>
      <EnvSelect
        :model-value="envId"
        @update:model-value="emit('update:envId', $event ?? '')"
        :custom-path="customPythonPath"
        @update:custom-path="emit('update:customPythonPath', $event)"
        :environments="environments"
      />
    </div>

    <div class="section" v-if="hasGpu && gpuCount > 0">
      <div class="section-label">GPU 分配</div>
      <el-radio-group
        :model-value="gpuChoice"
        @update:model-value="gpuChoice = $event"
        class="gpu-radio-group"
      >
        <el-radio value="auto" class="gpu-radio">自动分配</el-radio>
        <el-radio
          v-for="i in gpuCount"
          :key="i - 1"
          :value="String(i - 1)"
          class="gpu-radio"
        >
          GPU {{ i - 1 }}
          <span class="gpu-occ" v-if="gpuOccupancy[i - 1]">
            · 占用 {{ gpuOccupancy[i - 1] }}
          </span>
        </el-radio>
      </el-radio-group>
    </div>

    <div class="section">
      <div class="section-label-row">
        <div class="section-label">模型</div>
        <div class="section-actions">
          <el-radio-group v-model="viewMode" class="view-toggle">
            <el-radio-button value="grid"><el-icon><Grid /></el-icon></el-radio-button>
            <el-radio-button value="list"><el-icon><Menu /></el-icon></el-radio-button>
          </el-radio-group>
          <el-button
            class="refresh-btn"
            :loading="refreshing"
            @click="emit('refresh')"
          >
            <el-icon><Refresh /></el-icon>
            刷新
          </el-button>
        </div>
      </div>
      <div class="card-grid" v-if="viewMode === 'grid'">
        <div
          v-for="name in models"
          :key="name"
          class="select-card"
          :class="{ active: modelName === name }"
          role="button"
          tabindex="0"
          :aria-pressed="modelName === name"
          @click="emit('update:modelName', name)"
          @keydown.enter.prevent="emit('update:modelName', name)"
          @keydown.space.prevent="emit('update:modelName', name)"
        >
          <div class="card-icon" :style="{ background: getGradient(name) }">
            {{ name.charAt(0) }}
          </div>
          <div class="card-name" :title="name">{{ name }}</div>
        </div>
      </div>
      <div class="select-list" v-else>
        <div
          v-for="name in models"
          :key="name"
          class="list-item"
          :class="{ active: modelName === name }"
          role="button"
          tabindex="0"
          :aria-pressed="modelName === name"
          @click="emit('update:modelName', name)"
          @keydown.enter.prevent="emit('update:modelName', name)"
          @keydown.space.prevent="emit('update:modelName', name)"
        >
          <span class="list-badge" :style="{ background: getGradient(name) }">{{ name.charAt(0) }}</span>
          <span class="list-name" :title="name">{{ name }}</span>
        </div>
      </div>
    </div>

    <div class="section">
      <div class="section-label">数据集</div>
      <div class="dataset-layout">
        <div class="dataset-cards" v-if="datasets.length && viewMode === 'grid'">
          <div
            v-for="ds in datasets"
            :key="ds.name"
            class="select-card"
            :class="{ active: dataset === ds.name }"
            role="button"
            tabindex="0"
            :aria-pressed="dataset === ds.name"
            @click="onDatasetClick(ds.name)"
            @keydown.enter.prevent="onDatasetClick(ds.name)"
            @keydown.space.prevent="onDatasetClick(ds.name)"
          >
            <div class="card-icon icon-dataset" :style="{ background: getGradient('ds-' + ds.name) }">
              <el-icon :size="18"><Coin /></el-icon>
            </div>
            <div class="card-name" :title="ds.name">{{ ds.name }}</div>
          </div>
        </div>
        <div class="select-list" v-else-if="datasets.length">
          <div
            v-for="ds in datasets"
            :key="ds.name"
            class="list-item"
            :class="{ active: dataset === ds.name }"
            role="button"
            tabindex="0"
            :aria-pressed="dataset === ds.name"
            @click="onDatasetClick(ds.name)"
            @keydown.enter.prevent="onDatasetClick(ds.name)"
            @keydown.space.prevent="onDatasetClick(ds.name)"
          >
            <span class="list-badge icon-dataset" :style="{ background: getGradient('ds-' + ds.name) }">
              <el-icon :size="14"><Coin /></el-icon>
            </span>
            <span class="list-name" :title="ds.name">{{ ds.name }}</span>
          </div>
        </div>

        <DatasetMetadataPanel
          :dataset="dataset"
          :metadata="metadata"
          :status="selectedInfo?.status"
          :loading="metadataLoading"
          :icon-gradient="dataset ? getGradient('ds-' + dataset) : undefined"
          show-preprocess-link
        />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useQuery } from '@tanstack/vue-query'
import { Coin, Refresh, Grid, Menu } from '@element-plus/icons-vue'
import type { EnvironmentInfo } from '@/api/environments'
import { getDatasetMetadata, type DatasetInfo, type DatasetMetadata } from '@/api/datasets'
import { getGpuStatus } from '@/api/gpu'
import DatasetMetadataPanel from './DatasetMetadataPanel.vue'
import EnvSelect from './EnvSelect.vue'
import { getGradient } from '@/composables/useGradient'
import { useSystemCapabilities } from '@/composables/useSystemCapabilities'

const props = defineProps<{
  envId: string
  customPythonPath: string
  modelName: string
  dataset: string
  gpu: number | null
  environments: EnvironmentInfo[]
  models: string[]
  datasets: DatasetInfo[]
  refreshing: boolean
}>()

const emit = defineEmits<{
  (e: 'update:envId', val: string): void
  (e: 'update:customPythonPath', val: string): void
  (e: 'update:modelName', val: string): void
  (e: 'update:dataset', val: string): void
  (e: 'update:gpu', val: number | null): void
  (e: 'confirm'): void
  (e: 'refresh'): void
}>()

const { hasGpu, gpuCount } = useSystemCapabilities()

const VIEW_MODE_KEY = 'kt-web:selection-view-mode'
const viewMode = ref<'grid' | 'list'>(
  localStorage.getItem(VIEW_MODE_KEY) === 'list' ? 'list' : 'grid',
)
watch(viewMode, (v) => localStorage.setItem(VIEW_MODE_KEY, v))

const gpuStatusQuery = useQuery({
  queryKey: ['gpu-status'],
  queryFn: getGpuStatus,
  refetchInterval: 5000,
})
const gpuOccupancy = computed<Record<number, number>>(() => {
  const map: Record<number, number> = {}
  for (const g of gpuStatusQuery.data.value?.gpus ?? []) {
    map[g.index] = g.processes.length
  }
  return map
})

// el-radio values must be string/number/boolean (null is treated as absent by
// Element Plus), so the radio group works in strings and translates to/from
// the number|null contract on the way in and out.
const gpuChoice = computed<string>({
  get: () =>
    props.gpu === null || props.gpu === undefined ? 'auto' : String(props.gpu),
  set: (v: string) => emit('update:gpu', v === 'auto' ? null : Number(v)),
})

const metadata = ref<DatasetMetadata | null>(null)
const metadataLoading = ref(false)
const selectedInfo = computed(() => props.datasets.find(d => d.name === props.dataset))

const metadataCache = ref<Record<string, DatasetMetadata>>({})

function onDatasetClick(name: string) {
  emit('update:dataset', name)
}

async function loadMetadata(name: string) {
  metadataLoading.value = true
  try {
    const data = await getDatasetMetadata(name)
    metadataCache.value[name] = data
    metadata.value = data
  } catch {
    metadata.value = null
  } finally {
    metadataLoading.value = false
  }
}

// Only ready datasets expose a metadata grid; downloaded/empty render a status
// prompt instead, so skip the network round-trip for them.
watch(selectedInfo, (info) => {
  if (!info || info.status !== 'ready') { metadata.value = null; return }
  if (metadataCache.value[info.name]) { metadata.value = metadataCache.value[info.name]; return }
  loadMetadata(info.name)
}, { immediate: true })

function clearCache() {
  metadataCache.value = {}
  const info = selectedInfo.value
  if (info && info.status === 'ready') {
    loadMetadata(info.name)
  } else {
    metadata.value = null
  }
}

defineExpose({ clearCache })
</script>

<style scoped>
.selection-step {
  display: flex;
  flex-direction: column;
  gap: 28px;
  counter-reset: step;
}

.section > .section-label::before,
.section-label-row .section-label::before {
  counter-increment: step;
  content: counter(step, decimal-leading-zero);
  display: inline-block;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 600;
  color: var(--accent-blue);
  background: var(--soft-blue);
  padding: 1px 5px;
  border-radius: 4px;
  margin-right: 8px;
  letter-spacing: 0.5px;
  vertical-align: 1px;
}

.section-label {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 12px;
}

.section-label-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
}

.section-label-row .section-label {
  margin-bottom: 0;
}

.refresh-btn {
  font-size: 13px;
}

.section-actions {
  display: flex;
  align-items: center;
  gap: 8px;
}

.view-toggle :deep(.el-radio-button__inner) {
  height: 32px;
  padding: 0 12px;
  line-height: 1;
  display: inline-flex;
  align-items: center;
  justify-content: center;
}

.gpu-radio-group {
  display: flex;
  flex-wrap: wrap;
  gap: 8px 20px;
}

.gpu-radio {
  margin-right: 0;
}

.gpu-occ {
  color: var(--text-tertiary);
  font-size: 12px;
  font-family: var(--font-mono);
}

.card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 12px;
}

.dataset-layout {
  display: flex;
  gap: 20px;
  min-height: 200px;
}

.dataset-cards {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 12px;
  flex: 1;
  min-width: 0;
  align-content: start;
}

.select-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  padding: 18px 12px 14px;
  background: var(--bg-surface);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  cursor: pointer;
  transition: all 0.15s ease;
  user-select: none;
}

.select-card:hover {
  border-color: var(--text-tertiary);
  background: var(--bg-elevated);
}

.select-card.active {
  border-color: var(--accent-blue);
  background: var(--soft-blue);
  box-shadow: 0 0 0 1px var(--accent-blue);
}

.card-icon {
  width: 40px;
  height: 40px;
  border-radius: var(--radius-sm);
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 16px;
  color: #fff;
  font-family: var(--font-mono);
  flex-shrink: 0;
}

.icon-dataset {
  font-size: 0;
}

.card-name {
  font-size: 13px;
  font-weight: 500;
  color: var(--text-primary);
  font-family: var(--font-mono);
  letter-spacing: 0.2px;
  text-align: center;
  max-width: 100%;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.select-list {
  display: flex;
  flex-direction: column;
  gap: 2px;
  max-height: 340px;
  overflow-y: auto;
  padding: 4px;
  border: 1px solid var(--border-muted);
  border-radius: var(--radius-md);
  background: var(--bg-surface);
}

.dataset-layout .select-list {
  flex: 1;
  min-width: 0;
}

.list-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 8px 10px;
  border-radius: var(--radius-sm);
  cursor: pointer;
  transition: background 0.15s ease;
  user-select: none;
}

.list-item:hover {
  background: var(--bg-elevated);
}

.list-item.active {
  background: var(--soft-blue);
  box-shadow: inset 0 0 0 1px var(--accent-blue);
}

.list-badge {
  width: 22px;
  height: 22px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  font-family: var(--font-mono);
  font-size: 12px;
  font-weight: 700;
  flex-shrink: 0;
}

.list-badge.icon-dataset {
  font-size: 0;
}

.list-name {
  font-size: 13px;
  color: var(--text-primary);
  font-family: var(--font-mono);
  letter-spacing: 0.2px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.list-item.active .list-name {
  color: var(--accent-blue);
}
</style>
