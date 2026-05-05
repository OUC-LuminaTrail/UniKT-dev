<template>
  <div class="selection-step">
    <div class="section">
      <div class="section-label">运行环境</div>
      <el-select
        :model-value="envId"
        @update:model-value="emit('update:envId', $event)"
        placeholder="选择运行环境"
        class="env-select"
      >
        <el-option-group label="Pixi">
          <el-option
            v-for="env in pixiEnvs"
            :key="env.id"
            :label="env.display_name"
            :value="env.id"
          />
        </el-option-group>
        <el-option-group label="Conda" v-if="condaEnvs.length">
          <el-option
            v-for="env in condaEnvs"
            :key="env.id"
            :label="env.display_name"
            :value="env.id"
          />
        </el-option-group>
        <el-option-group label="Other" v-if="otherEnvs.length">
          <el-option
            v-for="env in otherEnvs"
            :key="env.id"
            :label="env.display_name"
            :value="env.id"
          />
        </el-option-group>
      </el-select>
      <div v-if="envId === 'custom:0'" class="custom-path-row">
        <el-input
          :model-value="customPythonPath"
          @update:model-value="emit('update:customPythonPath', $event)"
          placeholder="/path/to/python"
        />
      </div>
    </div>

    <div class="section">
      <div class="section-label">模型</div>
      <div class="card-grid">
        <div
          v-for="name in models"
          :key="name"
          class="select-card"
          :class="{ active: modelName === name }"
          @click="emit('update:modelName', name)"
        >
          <div class="card-icon" :style="{ background: getGradient(name) }">
            {{ name.charAt(0) }}
          </div>
          <div class="card-name" :title="name">{{ name }}</div>
        </div>
      </div>
    </div>

    <div class="section">
      <div class="section-label">数据集</div>
      <div class="dataset-layout">
        <div class="dataset-cards" v-if="datasets.length">
          <div
            v-for="name in datasets"
            :key="name"
            class="select-card"
            :class="{ active: dataset === name }"
            @click="onDatasetClick(name)"
          >
            <div class="card-icon icon-dataset" :style="{ background: getGradient('ds-' + name) }">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>
            </div>
            <div class="card-name" :title="name">{{ name }}</div>
          </div>
        </div>

        <div class="metadata-panel" v-if="dataset && metadata">
          <div class="metadata-header">
            <div class="metadata-icon" :style="{ background: getGradient('ds-' + dataset) }">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>
            </div>
            <span class="metadata-title">{{ dataset }}</span>
            <span v-if="metadata.sampled" class="sampled-badge">采样</span>
          </div>

          <div class="metadata-grid">
            <template v-for="key in displayKeys" :key="key">
              <div class="meta-item">
                <span class="meta-label">{{ formatKey(key) }}</span>
                <span class="meta-value">{{ formatValue(metadata[key]) }}</span>
              </div>
            </template>
          </div>

          <template v-for="key in nestedKeys" :key="key">
            <div class="nested-section" v-if="typeof metadata[key] === 'object' && metadata[key] !== null">
              <div class="nested-header">{{ formatKey(key) }}</div>
              <div class="nested-grid">
                <div class="meta-item" v-for="(val, subKey) in (metadata[key] as Record<string, any>)" :key="subKey">
                  <span class="meta-label">{{ formatKey(String(subKey)) }}</span>
                  <span class="meta-value">{{ formatValue(val) }}</span>
                </div>
              </div>
            </div>
          </template>
        </div>

        <div class="metadata-panel metadata-empty" v-else-if="dataset && !metadata">
          <div class="empty-text">加载中...</div>
        </div>

        <div class="metadata-panel metadata-placeholder" v-else>
          <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--text-tertiary)" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>
          <span>选择数据集以查看详细信息</span>
        </div>
      </div>
    </div>

    <div class="action-bar">
      <el-button
        type="primary"
        size="large"
        :disabled="!modelName || !dataset"
        @click="emit('confirm')"
      >
        确认选择
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import type { EnvironmentInfo } from '@/api/environments'
import { getDatasetMetadata, type DatasetMetadata } from '@/api/datasets'

const HIDDEN_KEYS = new Set([
  'question_data_md5', 'sequence_data_md5',
  'split_question_sequence_data_md5', 'split_skill_sequence_data_md5',
  'windowlate_data_md5', 'data_base_path', 'dataset',
])

const KEY_LABELS: Record<string, string> = {
  num_users: '用户数',
  num_questions: '题目数',
  num_skills: '技能数',
  num_assignments: '作业数',
  num_templates: '模板数',
  num_split_question_users: '划分用户(题目)',
  num_split_skill_users: '划分用户(技能)',
  kfold_n_splits: 'K 折',
  test_ratio: '测试比例',
  min_seq_len: '最小序列长度',
  max_seq_len: '最大序列长度',
  random_seed: '随机种子',
  sampled: '采样数据',
  sampling_config: '采样配置',
  sampling_stats: '采样统计',
  n_samples_requested: '请求样本数',
  n_samples_actual: '实际样本数',
  sampling_ratio: '采样比例',
  stratify: '分层采样',
  attempts_bins: '尝试次数分箱',
  correct_bins: '正确率分箱',
  original_users: '原始用户',
  sampled_users: '采样用户',
  original_records: '原始记录',
  sampled_records: '采样记录',
  strata_distribution: '分层分布',
}

const props = defineProps<{
  envId: string
  customPythonPath: string
  modelName: string
  dataset: string
  environments: EnvironmentInfo[]
  models: string[]
  datasets: string[]
}>()

const emit = defineEmits<{
  (e: 'update:envId', val: string): void
  (e: 'update:customPythonPath', val: string): void
  (e: 'update:modelName', val: string): void
  (e: 'update:dataset', val: string): void
  (e: 'confirm'): void
}>()

const metadata = ref<DatasetMetadata | null>(null)

const pixiEnvs = computed(() => props.environments.filter(e => e.type === 'pixi'))
const condaEnvs = computed(() => props.environments.filter(e => e.type === 'conda'))
const otherEnvs = computed(() => props.environments.filter(e => e.type !== 'pixi' && e.type !== 'conda'))

const displayKeys = computed(() => {
  if (!metadata.value) return []
  return Object.keys(metadata.value).filter(k => !HIDDEN_KEYS.has(k) && typeof metadata.value![k] !== 'object')
})

const nestedKeys = computed(() => {
  if (!metadata.value) return []
  return Object.keys(metadata.value).filter(k => !HIDDEN_KEYS.has(k) && typeof metadata.value![k] === 'object' && metadata.value![k] !== null)
})

const PALETTE = [
  ['#58a6ff', '#1f6feb'],
  ['#3fb950', '#238636'],
  ['#d29922', '#9e6a03'],
  ['#f85149', '#da3633'],
  ['#bc8cff', '#8b5cf6'],
  ['#39d2c0', '#0d9488'],
  ['#f778ba', '#db2777'],
  ['#79c0ff', '#388bfd'],
]

function hashStr(s: string): number {
  let h = 0
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h + s.charCodeAt(i)) | 0
  }
  return Math.abs(h)
}

function getGradient(name: string): string {
  const idx = hashStr(name) % PALETTE.length
  const [c1, c2] = PALETTE[idx]
  return `linear-gradient(135deg, ${c1}, ${c2})`
}

function formatKey(key: string): string {
  return KEY_LABELS[key] || key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

function formatValue(val: any): string {
  if (typeof val === 'boolean') return val ? '是' : '否'
  if (Array.isArray(val)) return val.map(String).join(', ')
  if (typeof val === 'number') return val.toLocaleString()
  return String(val)
}

const metadataCache = ref<Record<string, DatasetMetadata>>({})

function onDatasetClick(name: string) {
  emit('update:dataset', name)
  if (metadataCache.value[name]) {
    metadata.value = metadataCache.value[name]
    return
  }
  loadMetadata(name)
}

async function loadMetadata(name: string) {
  try {
    const data = await getDatasetMetadata(name)
    metadataCache.value[name] = data
    metadata.value = data
  } catch {
    metadata.value = null
  }
}

watch(() => props.dataset, (name) => {
  if (!name) { metadata.value = null; return }
  if (metadataCache.value[name]) { metadata.value = metadataCache.value[name]; return }
  loadMetadata(name)
}, { immediate: true })
</script>

<style scoped>
.selection-step {
  display: flex;
  flex-direction: column;
  gap: 28px;
}

.section-label {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
  margin-bottom: 12px;
}

.env-select {
  max-width: 400px;
}

.custom-path-row {
  margin-top: 8px;
  max-width: 400px;
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

.metadata-panel {
  width: 340px;
  flex-shrink: 0;
  background: var(--bg-surface);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  padding: 18px;
  display: flex;
  flex-direction: column;
  gap: 14px;
  overflow-y: auto;
  max-height: 420px;
}

.metadata-header {
  display: flex;
  align-items: center;
  gap: 10px;
}

.metadata-icon {
  width: 28px;
  height: 28px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  color: #fff;
  flex-shrink: 0;
}

.metadata-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
  font-family: var(--font-mono);
}

.sampled-badge {
  font-size: 11px;
  font-weight: 500;
  padding: 2px 8px;
  border-radius: 10px;
  background: color-mix(in srgb, var(--accent-orange) 12%, transparent);
  color: var(--accent-orange);
  border: 1px solid color-mix(in srgb, var(--accent-orange) 20%, transparent);
  margin-left: auto;
}

.metadata-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 10px 16px;
}

.meta-item {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.meta-label {
  font-size: 11px;
  color: var(--text-tertiary);
  font-weight: 500;
}

.meta-value {
  font-size: 13px;
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-weight: 600;
}

.nested-section {
  border-top: 1px solid var(--border-muted);
  padding-top: 12px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.nested-header {
  font-size: 12px;
  font-weight: 600;
  color: var(--accent-orange);
}

.nested-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px 16px;
}

.metadata-empty {
  align-items: center;
  justify-content: center;
}

.empty-text {
  color: var(--text-tertiary);
  font-size: 13px;
}

.metadata-placeholder {
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.metadata-placeholder span {
  color: var(--text-tertiary);
  font-size: 13px;
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
  border-width: 2px;
  background: rgba(88, 166, 255, 0.06);
  box-shadow: 0 0 0 1px rgba(88, 166, 255, 0.2);
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

.action-bar {
  padding-top: 8px;
  display: flex;
  justify-content: flex-end;
}

.action-bar .el-button {
  min-width: 160px;
  height: 44px;
  font-size: 15px;
  font-weight: 600;
  border-radius: var(--radius-md);
}
</style>