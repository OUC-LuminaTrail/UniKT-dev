<template>
  <div class="preprocess-view">
    <!-- Config state -->
    <template v-if="phase === 'config'">
      <div class="preprocess-body">
        <div class="page-header">
          <h1 class="page-title">数据预处理</h1>
          <p class="page-subtitle">配置并运行 data_process.py 进行数据集下载和处理</p>
        </div>

        <div class="form-section">
          <div class="section-label">基本配置</div>
          <div class="form-row">
            <div class="form-field">
              <label class="field-label">操作类型</label>
              <div class="radio-group">
                <label class="radio-item" :class="{ active: action === 'download' }">
                  <input type="radio" v-model="action" value="download" />
                  <span>下载</span>
                </label>
                <label class="radio-item" :class="{ active: action === 'process' }">
                  <input type="radio" v-model="action" value="process" />
                  <span>处理</span>
                </label>
              </div>
            </div>
            <div class="form-field">
              <label class="field-label">数据集</label>
              <el-select v-model="dataset" placeholder="选择数据集" class="dataset-select">
                <el-option v-for="ds in datasets" :key="ds.name" :label="ds.name" :value="ds.name" />
              </el-select>
            </div>
          </div>
        </div>

        <div v-if="action === 'download'" class="form-section">
          <div class="section-label">下载选项</div>
          <div class="form-row">
            <div class="form-field">
              <label class="field-label">强制重新下载</label>
              <el-switch v-model="downloadOpts.force" />
            </div>
            <div class="form-field">
              <label class="field-label">最大重试次数</label>
              <el-input-number v-model="downloadOpts.max_retries" :min="1" :max="10" />
            </div>
            <div class="form-field">
              <label class="field-label">下载线程数</label>
              <el-input-number v-model="downloadOpts.num_threads" :min="1" :max="16" />
            </div>
          </div>
        </div>

        <div v-if="action === 'process'" class="form-section">
          <div class="section-label">序列参数</div>
          <div class="form-row">
            <div class="form-field">
              <label class="field-label">最小序列长度</label>
              <el-input-number v-model="processOpts.min_seq_len" :min="1" :max="1000" />
            </div>
            <div class="form-field">
              <label class="field-label">最大序列长度</label>
              <el-input-number v-model="processOpts.max_seq_len" :min="10" :max="2000" />
            </div>
            <div class="form-field">
              <label class="field-label">K 折交叉验证</label>
              <el-input-number v-model="processOpts.kfold" :min="1" :max="20" />
            </div>
            <div class="form-field">
              <label class="field-label">随机种子</label>
              <el-input-number v-model="processOpts.seed" :min="0" />
            </div>
          </div>
        </div>

        <div v-if="action === 'process'" class="form-section">
          <div class="section-label">采样参数</div>
          <div class="form-row">
            <div class="form-field">
              <label class="field-label">采样数量</label>
              <el-input-number v-model="processOpts.sample_size" :min="1" />
            </div>
            <div class="form-field">
              <label class="field-label">采样比例</label>
              <el-input-number v-model="processOpts.sample_ratio" :min="0.01" :max="1" :step="0.05" :precision="2" />
            </div>
            <div class="form-field">
              <label class="field-label">采样策略</label>
              <el-select v-model="processOpts.sample_strategy" clearable>
                <el-option value="random" label="随机" />
                <el-option value="stratified" label="分层" />
                <el-option value="time" label="按时间" />
              </el-select>
            </div>
          </div>
          <div class="form-row">
            <div class="form-field">
              <label class="field-label">尝试次数分箱</label>
              <el-input v-model="processOpts.sample_attempts_bins" placeholder="20 100" />
            </div>
            <div class="form-field">
              <label class="field-label">正确率分箱</label>
              <el-input v-model="processOpts.sample_correct_bins" placeholder="0.4 0.8" />
            </div>
          </div>
        </div>

        <div v-if="action === 'process'" class="form-section">
          <div class="section-label">额外选项</div>
          <div class="form-row">
            <div class="form-field">
              <label class="field-label">额外处理步骤</label>
              <el-input v-model="processOpts.extra" placeholder="windowslate" />
            </div>
          </div>
        </div>
      </div>

      <CommandPreview :command="previewCommand">
        <el-button
          type="primary"
          size="large"
          :disabled="!dataset || submitting"
          :loading="submitting"
          @click="onStart"
        >
          {{ submitting ? '启动中...' : '开始处理' }}
        </el-button>
      </CommandPreview>
    </template>

    <!-- Running state -->
    <template v-if="phase === 'running' && taskInfo">
      <div class="running-header">
        <button class="back-btn" @click="onBack">
          <svg width="14" height="14" viewBox="0 0 12 12" fill="none"><path d="M7.5 2L3.5 6L7.5 10" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/></svg>
          返回
        </button>
        <span class="running-command mono">{{ taskInfo.command }}</span>
        <span class="status-badge" :class="`status-${taskInfo.status}`">{{ statusLabel }}</span>
        <button v-if="taskInfo.status === 'running'" class="stop-btn" :disabled="stopping" @click="onStop">
          {{ stopping ? '停止中...' : '停止' }}
        </button>
      </div>
      <div class="terminal-wrapper">
        <LogTerminal :ws-url="`/api/preprocess/${taskId}/logs/stream`" :task-status="taskInfo.status" :task-id="taskId" />
      </div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { ElMessage } from 'element-plus'
import { listDatasets, type DatasetInfo } from '@/api/datasets'
import { startPreprocess, getPreprocess, stopPreprocess, type PreprocessTaskInfo } from '@/api/preprocess'
import CommandPreview from '@/components/task/CommandPreview.vue'
import LogTerminal from '@/components/task/LogTerminal.vue'

type Phase = 'config' | 'running'
const phase = ref<Phase>('config')
const submitting = ref(false)
const stopping = ref(false)

const action = ref<'download' | 'process'>('process')
const dataset = ref('')
const datasets = ref<DatasetInfo[]>([])

const downloadOpts = ref({
  force: false,
  max_retries: 3,
  num_threads: 4,
})

const processOpts = ref({
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

const taskInfo = ref<PreprocessTaskInfo | null>(null)
const taskId = ref(0)
let pollTimer: ReturnType<typeof setInterval> | null = null

const statusLabel = computed(() => {
  const map: Record<string, string> = {
    running: '运行中', completed: '已完成', failed: '已失败', stopped: '已停止',
  }
  return taskInfo.value ? map[taskInfo.value.status] || taskInfo.value.status : ''
})

const previewCommand = computed(() => {
  const parts = ['python', 'data_process.py', action.value, '-d', dataset.value || '<dataset>']
  if (action.value === 'download') {
    if (downloadOpts.value.force) parts.push('--force')
    if (downloadOpts.value.max_retries !== 3) parts.push('--max_retries', String(downloadOpts.value.max_retries))
    if (downloadOpts.value.num_threads !== 4) parts.push('--num_threads', String(downloadOpts.value.num_threads))
  } else {
    const o = processOpts.value
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

const onStart = async () => {
  if (!dataset.value) return
  submitting.value = true
  try {
    const params: Record<string, any> = {}
    if (action.value === 'download') {
      const o = downloadOpts.value
      if (o.force) params.force = true
      if (o.max_retries !== 3) params.max_retries = o.max_retries
      if (o.num_threads !== 4) params.num_threads = o.num_threads
    } else {
      const o = processOpts.value
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
    const result = await startPreprocess({ action: action.value, dataset: dataset.value, params })
    taskId.value = result.id
    taskInfo.value = result
    phase.value = 'running'
    startPolling()
  } catch (e: any) {
    ElMessage.error(e?.response?.data?.detail || '启动失败')
  } finally {
    submitting.value = false
  }
}

const onStop = async () => {
  stopping.value = true
  try {
    await stopPreprocess(taskId.value)
    ElMessage.success('已发送停止信号')
  } finally {
    stopping.value = false
  }
}

const onBack = () => {
  phase.value = 'config'
  taskInfo.value = null
  stopPolling()
}

const startPolling = () => {
  pollTimer = setInterval(async () => {
    if (!taskInfo.value || taskInfo.value.status !== 'running') return
    try { taskInfo.value = await getPreprocess(taskId.value) } catch {}
  }, 3000)
}

const stopPolling = () => {
  if (pollTimer) { clearInterval(pollTimer); pollTimer = null }
}

onMounted(async () => {
  try { datasets.value = await listDatasets() } catch {}
})

onUnmounted(() => { stopPolling() })
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
  margin-bottom: 20px;
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

.form-section {
  margin-bottom: 24px;
}

.section-label {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 12px;
}

.form-row {
  display: flex;
  gap: 20px;
  flex-wrap: wrap;
}

.form-field {
  display: flex;
  flex-direction: column;
  gap: 6px;
  min-width: 160px;
}

.field-label {
  font-size: 12px;
  color: var(--text-secondary);
  font-weight: 500;
}

.radio-group {
  display: flex;
  gap: 4px;
}

.radio-item {
  padding: 6px 16px;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-sm);
  cursor: pointer;
  font-size: 13px;
  color: var(--text-secondary);
  transition: all 0.15s ease;
  display: flex;
  align-items: center;
  gap: 6px;
}

.radio-item input {
  display: none;
}

.radio-item.active {
  border-color: var(--accent-blue);
  color: var(--accent-blue);
  background: rgba(9, 105, 218, 0.06);
}

.dataset-select {
  width: 240px;
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
  background: rgba(9, 105, 218, 0.1);
  color: var(--accent-blue);
}

.status-badge.status-completed {
  background: rgba(26, 127, 55, 0.1);
  color: var(--accent-green);
}

.status-badge.status-failed {
  background: rgba(207, 34, 46, 0.1);
  color: var(--accent-red);
}

.status-badge.status-stopped {
  background: rgba(154, 103, 0, 0.1);
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
  background: rgba(210, 153, 34, 0.08);
}

.stop-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.terminal-wrapper {
  flex: 1;
  min-height: 0;
  background: #1a1b26;
}
</style>
