<template>
  <div class="preprocess-view">
    <!-- Config state -->
    <template v-if="phase === 'config'">
      <div class="preprocess-body">
        <div class="page-header">
          <h1 class="page-title">{{ t('route.title.preprocess') }}</h1>
          <p class="page-subtitle">{{ t('preprocess.subtitle') }}</p>
        </div>

        <div class="section">
          <div class="section-label">{{ t('preprocess.actionType') }}</div>
          <div class="radio-group">
            <label class="radio-item" :class="{ active: action === 'download' }">
              <input type="radio" v-model="action" value="download" />
              <el-icon :size="14"><Download /></el-icon>
              <span>{{ t('preprocess.download') }}</span>
            </label>
            <label class="radio-item" :class="{ active: action === 'process' }">
              <input type="radio" v-model="action" value="process" />
              <el-icon :size="14"><Upload /></el-icon>
              <span>{{ t('preprocess.process') }}</span>
            </label>
          </div>
        </div>

        <div class="section" v-if="environments.length">
          <div class="section-label">{{ t('selection.env') }}</div>
          <EnvSelect
            v-model="selectedEnvId"
            v-model:custom-path="customPythonPath"
            :environments="environments"
            :placeholder="t('preprocess.envDefaultPlaceholder')"
            clearable
          />
        </div>

        <div class="section">
          <div class="section-label">{{ t('selection.dataset') }}</div>
          <div class="dataset-layout">
            <el-skeleton :loading="loading" animated style="flex:1;min-width:0">
              <template #template>
                <div class="dataset-cards">
                  <div class="select-card" v-for="i in 4" :key="i">
                    <el-skeleton-item variant="rect" style="width:40px;height:40px;border-radius:var(--radius-sm)" />
                    <el-skeleton-item variant="text" style="width:60%" />
                  </div>
                </div>
              </template>
              <template #default>
                <div class="dataset-cards" v-if="visibleDatasets.length">
                  <div
                    v-for="ds in visibleDatasets"
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
              </template>
            </el-skeleton>

            <DatasetMetadataPanel
              :dataset="dataset"
              :metadata="datasetMetadata"
              :status="selectedInfo?.status"
              :loading="metadataLoading"
              :icon-gradient="dataset ? getGradient('ds-' + dataset) : undefined"
            />
          </div>
        </div>

        <PreprocessForm :schema="schema" v-model="opts" />
      </div>

      <CommandPreview :command="preview">
        <el-button
          type="primary"
          size="large"
          :disabled="!dataset || submitting"
          :loading="submitting"
          @click="onStart"
        >
          {{ submitting ? t('preprocess.starting') : action === 'download' ? t('preprocess.startDownload') : t('preprocess.startProcess') }}
        </el-button>
      </CommandPreview>
    </template>

    <!-- Running state -->
    <template v-if="phase === 'running' && taskInfo">
      <div class="running-header">
        <button v-if="taskInfo.status !== 'running'" class="back-btn" @click="onBack">
          <el-icon :size="14"><ArrowLeft /></el-icon>
          {{ t('common.back') }}
        </button>
        <span class="running-command mono">{{ taskInfo.command }}</span>
        <span class="status-badge" :class="`status-${taskInfo.status}`">{{ statusLabel }}</span>
        <button v-if="taskInfo.status === 'running'" class="stop-btn" :disabled="stopping" @click="onStop">
          {{ stopping ? t('common.status.stopping') : t('common.stop') }}
        </button>
      </div>

      <LogCard :ws-url="`/api/preprocess/${taskId}/logs/stream`" :task-status="taskInfo.status" :task-id="taskId" api-base="/preprocess" fill />
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useQuery, useMutation, useQueryClient } from '@tanstack/vue-query'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Coin, Download, Upload, ArrowLeft } from '@element-plus/icons-vue'
import { listDatasets, getDatasetMetadata, type DatasetInfo, type DatasetMetadata } from '@/api/datasets'
import { startPreprocess, getPreprocess, stopPreprocess, listPreprocess, getPreprocessSchema, previewPreprocess, type PreprocessTaskInfo } from '@/api/preprocess'
import { listEnvironments, type EnvironmentInfo } from '@/api/environments'
import type { ParamGroup } from '@/api/schemas'
import CommandPreview from '@/components/task/CommandPreview.vue'
import LogCard from '@/components/task/LogCard.vue'
import DatasetMetadataPanel from '@/components/task/DatasetMetadataPanel.vue'
import PreprocessForm from '@/components/task/PreprocessForm.vue'
import EnvSelect from '@/components/task/EnvSelect.vue'
import { getGradient } from '@/composables/useGradient'

type Phase = 'config' | 'running'
const route = useRoute()
const queryClient = useQueryClient()
const { t } = useI18n()
const phase = ref<Phase>('config')
const submitting = ref(false)

// Honor an `?action=download` deep link (e.g. from the empty-dataset prompt on
// the train page); otherwise default to process.
const action = ref<'download' | 'process'>(
  route.query.action === 'download' ? 'download' : 'process',
)
const dataset = ref('')

const selectedEnvId = ref<string | null>(null)
const customPythonPath = ref('')

const activeTasksQuery = useQuery({ queryKey: ['preprocess-active-tasks'], queryFn: listPreprocess })
const datasetsQuery = useQuery({ queryKey: ['datasets'], queryFn: listDatasets })
const envsQuery = useQuery({ queryKey: ['environments'], queryFn: listEnvironments })

const datasets = computed<DatasetInfo[]>(() => datasetsQuery.data.value ?? [])
const environments = computed<EnvironmentInfo[]>(() => envsQuery.data.value ?? [])
// Download offers every dataset; process only those with raw data or already processed.
const visibleDatasets = computed<DatasetInfo[]>(() =>
  action.value === 'download'
    ? datasets.value
    : datasets.value.filter(d => d.status !== 'empty')
)
const selectedInfo = computed(
  () => datasets.value.find(d => d.name === dataset.value) ?? null
)
const loading = computed(() =>
  activeTasksQuery.isPending.value ||
  datasetsQuery.isPending.value ||
  envsQuery.isPending.value
)

// Params/defaults/routes all come from backend reflection (same source as task launch).
const schemaQuery = useQuery({
  queryKey: computed(() => ['preprocess-schema', action.value]),
  queryFn: () => getPreprocessSchema(action.value),
})
const schema = computed<ParamGroup[]>(() => schemaQuery.data.value ?? [])

const defaults = computed<Record<string, any>>(() => {
  const d: Record<string, any> = {}
  for (const g of schema.value) {
    for (const [k, f] of Object.entries(g.params)) d[k] = f.default
  }
  return d
})
const opts = ref<Record<string, any>>({})
// Reset opts only on first load and action switch — NOT on schema refetch
// (e.g. window refocus), so user edits survive background revalidation.
const needReset = ref(true)
watch(action, () => {
  needReset.value = true
  // Switching to a stricter view (e.g. download→process) may hide the current
  // selection — clear it instead of letting it linger as an unprocessable target.
  if (dataset.value && !visibleDatasets.value.some(d => d.name === dataset.value)) {
    dataset.value = ''
  }
})
watch(schema, (sch) => {
  if (!sch.length || !needReset.value) return
  opts.value = { ...defaults.value }
  needReset.value = false
}, { immediate: true })

const taskId = ref(0)
const taskInfo = ref<PreprocessTaskInfo | null>(null)

const preprocessTaskQuery = useQuery({
  queryKey: computed(() => ['preprocess-task', taskId.value]),
  queryFn: () => getPreprocess(taskId.value),
  enabled: computed(() => phase.value === 'running' && taskId.value > 0),
  refetchInterval: (query) => {
    const data = query.state.data as PreprocessTaskInfo | undefined
    return data?.status === 'running' ? 3000 : false
  },
})

watch(() => preprocessTaskQuery.data.value, (data) => {
  if (!data) return
  taskInfo.value = data
  // A finished download/process changes on-disk state; refresh dataset statuses so cards update.
  if (data.status === 'completed') {
    queryClient.invalidateQueries({ queryKey: ['datasets'] })
  }
})

const datasetMetadataQuery = useQuery({
  queryKey: computed(() => ['dataset-metadata', dataset.value]),
  queryFn: () => getDatasetMetadata(dataset.value),
  enabled: computed(() => !!dataset.value && selectedInfo.value?.status === 'ready'),
})

const datasetMetadata = computed<DatasetMetadata | null>(() => datasetMetadataQuery.data.value ?? null)
const metadataLoading = computed(() => datasetMetadataQuery.isPending.value && !!dataset.value)

function onDatasetClick(name: string) {
  dataset.value = name
}

// Resume an in-flight task on mount: `immediate` handles the query cache being
// hit synchronously (data.value already set on remount, so a lazy watch wouldn't
// fire); stop manually once data is ready so `once` doesn't burn out on empty.
const stopRecover = watch(() => activeTasksQuery.data.value, (tasks) => {
  if (!tasks) return
  const activeTask = tasks.find(task => task.status === 'running')
  if (activeTask) {
    taskId.value = activeTask.id
    taskInfo.value = activeTask
    phase.value = 'running'
  }
  stopRecover()
}, { immediate: true })

watch(() => datasetsQuery.data.value, (data) => {
  if (!data) return
  const q = route.query.dataset as string
  // Only restore visible selections: an empty dataset in the URL can't be
  // processed, so don't preselect it in the process view.
  if (q && visibleDatasets.value.some(d => d.name === q)) {
    dataset.value = q
  }
}, { once: true })

const statusLabel = computed(() => {
  const map: Record<string, string> = {
    running: t('common.status.running'),
    completed: t('common.status.completed'),
    failed: t('common.status.failed'),
    stopped: t('common.status.stopped'),
  }
  return taskInfo.value ? map[taskInfo.value.status] || taskInfo.value.status : ''
})

// Command preview comes from the backend (single source of truth; no client
// mirror). No env prefix — mirrors the task-launch preview (train.py ...).
const baseCommand = () =>
  ['data_process.py', action.value, '-d', dataset.value || '<dataset>'].join(' ')

const preview = ref('')
let previewTimer: ReturnType<typeof setTimeout> | null = null
watch(
  [action, dataset, opts, selectedEnvId, customPythonPath],
  () => {
    if (previewTimer) clearTimeout(previewTimer)
    previewTimer = setTimeout(async () => {
      if (!dataset.value) { preview.value = baseCommand(); return }
      try {
        preview.value = await previewPreprocess({
          action: action.value,
          dataset: dataset.value,
          params: { ...opts.value },
          env_id: selectedEnvId.value,
          custom_python_path: selectedEnvId.value === 'custom:0' ? customPythonPath.value || null : null,
        })
      } catch {
        preview.value = baseCommand()
      }
    }, 200)
  },
  { deep: true, immediate: true }
)

const startMutation = useMutation({
  mutationFn: startPreprocess,
  onSuccess: (result) => {
    taskId.value = result.id
    taskInfo.value = result
    phase.value = 'running'
    queryClient.invalidateQueries({ queryKey: ['preprocess-active-tasks'] })
  },
})

const onStart = async () => {
  if (!dataset.value) return
  submitting.value = true
  try {
    // Send the whole form; backend _build_command skips defaults/None.
    startMutation.mutate({
      action: action.value,
      dataset: dataset.value,
      params: { ...opts.value },
      env_id: selectedEnvId.value,
      custom_python_path: selectedEnvId.value === 'custom:0' ? customPythonPath.value || null : null,
    })
  } finally {
    submitting.value = false
  }
}

const stopMutation = useMutation({
  mutationFn: (id: number) => stopPreprocess(id),
  onSuccess: () => {
    ElMessage.success(t('preprocess.stopSignalSent'))
    queryClient.invalidateQueries({ queryKey: ['preprocess-task', taskId.value] })
  },
})

const stopping = computed(() => stopMutation.isPending.value)

const onStop = () => {
  stopMutation.mutate(taskId.value)
}

const onBack = () => {
  phase.value = 'config'
  taskInfo.value = null
  queryClient.invalidateQueries({ queryKey: ['datasets'] })
}
</script>

<style scoped>
.preprocess-view {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.preprocess-body {
  flex: 1;
  overflow-y: auto;
  min-height: 0;
  padding: 20px;
}

.page-header {
  margin-bottom: 24px;
}

.page-title {
  font-size: 22px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0 0 4px 0;
  letter-spacing: -0.3px;
}

.page-subtitle {
  font-size: 14px;
  color: var(--text-tertiary);
  margin: 0;
}

.section {
  margin-bottom: 24px;
}

.section-label {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 12px;
}

.radio-group {
  display: flex;
  gap: 8px;
}

.radio-item {
  padding: 8px 20px;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-size: 13px;
  color: var(--text-secondary);
  transition: all 0.15s ease;
  display: flex;
  align-items: center;
  gap: 8px;
  background: var(--bg-surface);
}

.radio-item input {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  white-space: nowrap;
  border: 0;
}

.radio-item.active {
  border-color: var(--accent-blue);
  color: var(--accent-blue);
  background: var(--soft-blue);
}

.radio-item:focus-within {
  outline: 2px solid var(--accent-blue);
  outline-offset: 2px;
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

.running-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 20px;
  border-bottom: 1px solid var(--border-muted);
  flex-shrink: 0;
}

.back-btn {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 4px 10px;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-sm);
  background: var(--bg-surface);
  color: var(--text-secondary);
  cursor: pointer;
  font-size: 12px;
  font-family: var(--font-sans);
  transition: all 0.15s ease;
}

.back-btn:hover {
  color: var(--text-primary);
  border-color: var(--accent-blue);
}

.running-command {
  flex: 1;
  font-size: 12px;
  color: var(--text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.status-badge {
  font-size: 12px;
  font-weight: 500;
  padding: 3px 10px;
  border-radius: 12px;
  flex-shrink: 0;
}

.status-badge.status-running {
  background: var(--soft-blue);
  color: var(--accent-blue);
}

.status-badge.status-completed {
  background: var(--soft-green);
  color: var(--accent-green);
}

.status-badge.status-failed {
  background: var(--soft-red);
  color: var(--accent-red);
}

.status-badge.status-stopped {
  background: var(--soft-orange);
  color: var(--accent-orange);
}

.stop-btn {
  padding: 4px 12px;
  border: 1px solid var(--accent-orange);
  border-radius: var(--radius-sm);
  background: transparent;
  color: var(--accent-orange);
  cursor: pointer;
  font-size: 12px;
  font-family: var(--font-sans);
  transition: all 0.15s ease;
}

.stop-btn:hover {
  background: var(--soft-orange);
}

.stop-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
</style>
