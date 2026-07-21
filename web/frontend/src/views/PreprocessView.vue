<template>
  <div class="preprocess-view">
    <!-- Config state -->
    <template v-if="phase === 'config'">
      <div class="preprocess-body">
        <div class="page-header">
          <h1 class="page-title">数据预处理</h1>
          <p class="page-subtitle">配置并运行 data_process.py 进行数据集下载和处理</p>
        </div>

        <div class="section">
          <div class="section-label">操作类型</div>
          <div class="radio-group">
            <label class="radio-item" :class="{ active: action === 'download' }">
              <input type="radio" v-model="action" value="download" />
              <el-icon :size="14"><Download /></el-icon>
              <span>下载</span>
            </label>
            <label class="radio-item" :class="{ active: action === 'process' }">
              <input type="radio" v-model="action" value="process" />
              <el-icon :size="14"><Upload /></el-icon>
              <span>处理</span>
            </label>
          </div>
        </div>

        <div class="section" v-if="environments.length">
          <div class="section-label">运行环境</div>
          <EnvSelect
            v-model="selectedEnvId"
            v-model:custom-path="customPythonPath"
            :environments="environments"
            placeholder="使用默认环境"
            clearable
          />
        </div>

        <div class="section">
          <div class="section-label">数据集</div>
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

        <PreprocessForm
          ref="preprocessFormRef"
          :action="action"
          @update:download-opts="onDownloadOptsUpdate"
          @update:process-opts="onProcessOptsUpdate"
        />

      </div>

      <CommandPreview :command="previewCommand">
        <el-button
          type="primary"
          size="large"
          :disabled="!dataset || submitting"
          :loading="submitting"
          @click="onStart"
        >
          {{ submitting ? '启动中...' : action === 'download' ? '开始下载' : '开始处理' }}
        </el-button>
      </CommandPreview>
    </template>

    <!-- Running state -->
    <template v-if="phase === 'running' && taskInfo">
      <div class="running-header">
        <button v-if="taskInfo.status !== 'running'" class="back-btn" @click="onBack">
          <el-icon :size="14"><ArrowLeft /></el-icon>
          返回
        </button>
        <span class="running-command mono">{{ taskInfo.command }}</span>
        <span class="status-badge" :class="`status-${taskInfo.status}`">{{ statusLabel }}</span>
        <button v-if="taskInfo.status === 'running'" class="stop-btn" :disabled="stopping" @click="onStop">
          {{ stopping ? '停止中...' : '停止' }}
        </button>
      </div>

      <LogCard :ws-url="`/api/preprocess/${taskId}/logs/stream`" :task-status="taskInfo.status" :task-id="taskId" resize-prefix="/preprocess" fill />
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useQuery, useMutation, useQueryClient } from '@tanstack/vue-query'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { Coin, Download, Upload, ArrowLeft } from '@element-plus/icons-vue'
import { listDatasets, getDatasetMetadata, type DatasetInfo, type DatasetMetadata } from '@/api/datasets'
import { startPreprocess, getPreprocess, stopPreprocess, listPreprocess, type PreprocessTaskInfo } from '@/api/preprocess'
import { listEnvironments, type EnvironmentInfo } from '@/api/environments'
import CommandPreview from '@/components/task/CommandPreview.vue'
import LogCard from '@/components/task/LogCard.vue'
import DatasetMetadataPanel from '@/components/task/DatasetMetadataPanel.vue'
import PreprocessForm from '@/components/task/PreprocessForm.vue'
import EnvSelect from '@/components/task/EnvSelect.vue'
import { getGradient } from '@/composables/useGradient'

type Phase = 'config' | 'running'
const route = useRoute()
const queryClient = useQueryClient()
const phase = ref<Phase>('config')
const submitting = ref(false)

const action = ref<'download' | 'process'>('process')
const dataset = ref('')

const selectedEnvId = ref<string | null>(null)
const customPythonPath = ref('')

const preprocessFormRef = ref<InstanceType<typeof PreprocessForm> | null>(null)

const currentDownloadOpts = ref({
  force: false,
  max_retries: 3,
  num_threads: 4,
})

const currentProcessOpts = ref({
  min_seq_len: 10,
  max_seq_len: 200,
  kfold: 5,
  seed: 42,
  sample_size: null as number | null,
  sample_ratio: null as number | null,
  sample_strategy: '',
  sample_attempts_bins: '',
  sample_correct_bins: '',
  extra: '',
})

const onDownloadOptsUpdate = (opts: typeof currentDownloadOpts.value) => {
  currentDownloadOpts.value = opts
}

const onProcessOptsUpdate = (opts: typeof currentProcessOpts.value) => {
  currentProcessOpts.value = opts
}

const taskId = ref(0)
const taskInfo = ref<PreprocessTaskInfo | null>(null)

const activeTasksQuery = useQuery({
  queryKey: ['preprocess-active-tasks'],
  queryFn: listPreprocess,
})

const datasetsQuery = useQuery({ queryKey: ['datasets'], queryFn: listDatasets })
const envsQuery = useQuery({ queryKey: ['environments'], queryFn: listEnvironments })

const datasets = computed<DatasetInfo[]>(() => datasetsQuery.data.value ?? [])
const environments = computed<EnvironmentInfo[]>(() => envsQuery.data.value ?? [])
// Download offers every dataset; process only those with raw data or already
// processed — empty ones have nothing to process.
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
  // A finished download/process changes on-disk state — refresh dataset statuses
  // so cards transition empty→downloaded→ready without a manual reload.
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

// 挂载时恢复正在运行的任务：immediate 处理 query 缓存被同步命中的情况
// （重新挂载时 data.value 已有值，惰性 watch 不会触发）；
// 在数据就绪后才手动停止，避免 once 在 tasks 为空时提前失效导致后续真实数据不再触发
const stopRecover = watch(() => activeTasksQuery.data.value, (tasks) => {
  if (!tasks) return
  const activeTask = tasks.find(t => t.status === 'running')
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
  if (q && data.some(d => d.name === q)) {
    dataset.value = q
  }
}, { once: true })

const statusLabel = computed(() => {
  const map: Record<string, string> = {
    running: '运行中', completed: '已完成', failed: '已失败', stopped: '已停止',
  }
  return taskInfo.value ? map[taskInfo.value.status] || taskInfo.value.status : ''
})

const previewCommand = computed(() => {
  let envPrefix = 'python'
  if (selectedEnvId.value) {
    const env = environments.value.find(e => e.id === selectedEnvId.value)
    if (env) {
      if (env.type === 'pixi') envPrefix = `pixi run --environment ${env.name} python`
      else if (env.type === 'conda') envPrefix = `conda run -n ${env.name} --no-banner python`
      else if (customPythonPath.value) envPrefix = customPythonPath.value
    }
  }
  const parts = [envPrefix, 'data_process.py', action.value, '-d', dataset.value || '<dataset>']
  if (action.value === 'download') {
    const o = currentDownloadOpts.value
    if (o.force) parts.push('--force')
    if (o.max_retries !== 3) parts.push('--max_retries', String(o.max_retries))
    if (o.num_threads !== 4) parts.push('--num_threads', String(o.num_threads))
  } else {
    const o = currentProcessOpts.value
    if (o.min_seq_len !== 10) parts.push('--min_seq_len', String(o.min_seq_len))
    if (o.max_seq_len !== 200) parts.push('--max_seq_len', String(o.max_seq_len))
    if (o.kfold !== 5) parts.push('--kfold', String(o.kfold))
    if (o.seed !== 42) parts.push('--seed', String(o.seed))
    if (o.sample_size) parts.push('--sample_size', String(o.sample_size))
    if (o.sample_ratio) parts.push('--sample_ratio', String(o.sample_ratio))
    if (o.sample_strategy) parts.push('--sample_strategy', o.sample_strategy)
    if (o.sample_attempts_bins) parts.push('--sample_attempts_bins', o.sample_attempts_bins)
    if (o.sample_correct_bins) parts.push('--sample_correct_bins', o.sample_correct_bins)
    if (o.extra) parts.push('--extra', o.extra)
  }
  return parts.join(' ')
})

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
    const params: Record<string, any> = {}
    if (action.value === 'download') {
      const o = currentDownloadOpts.value
      if (o.force) params.force = true
      if (o.max_retries !== 3) params.max_retries = o.max_retries
      if (o.num_threads !== 4) params.num_threads = o.num_threads
    } else {
      const o = currentProcessOpts.value
      if (o.min_seq_len !== 10) params.min_seq_len = o.min_seq_len
      if (o.max_seq_len !== 200) params.max_seq_len = o.max_seq_len
      if (o.kfold !== 5) params.kfold = o.kfold
      if (o.seed !== 42) params.seed = o.seed
      if (o.sample_size) params.sample_size = o.sample_size
      if (o.sample_ratio) params.sample_ratio = o.sample_ratio
      if (o.sample_strategy) params.sample_strategy = o.sample_strategy
      if (o.sample_attempts_bins) params.sample_attempts_bins = o.sample_attempts_bins
      if (o.sample_correct_bins) params.sample_correct_bins = o.sample_correct_bins
      if (o.extra) params.extra = o.extra
    }
    startMutation.mutate({
      action: action.value,
      dataset: dataset.value,
      params,
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
    ElMessage.success('已发送停止信号')
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
  background: var(--bg-surface);
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
