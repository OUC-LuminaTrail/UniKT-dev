<template>
  <div class="metadata-panel" v-if="metadata">
    <div class="metadata-header">
      <div class="metadata-icon" :style="{ background: iconGradient }">
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
          <div class="meta-item" v-for="(val, subKey) in (metadata[key] as Record<string, any>)" :key="String(subKey)">
            <span class="meta-label">{{ formatKey(String(subKey)) }}</span>
            <span class="meta-value">{{ formatValue(val) }}</span>
          </div>
        </div>
      </div>
    </template>
  </div>

  <div class="metadata-panel metadata-empty" v-else-if="dataset && loading">
    <div class="empty-text">加载中...</div>
  </div>

  <div class="metadata-panel metadata-no-data" v-else-if="dataset && !loading">
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--text-tertiary)" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>
    <span class="no-data-text">该数据集尚未预处理</span>
    <router-link v-if="showPreprocessLink" :to="`/preprocess?dataset=${dataset}`" class="preprocess-link">
      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
      去预处理
    </router-link>
  </div>

  <div class="metadata-panel metadata-placeholder" v-else>
    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--text-tertiary)" stroke-width="1.4" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>
    <span>选择数据集以查看详细信息</span>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { type DatasetMetadata } from '@/api/datasets'

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
  dataset: string
  metadata: DatasetMetadata | null
  loading?: boolean
  iconGradient?: string
  showPreprocessLink?: boolean
}>()

const displayKeys = computed(() => {
  if (!props.metadata) return []
  return Object.keys(props.metadata).filter(k => !HIDDEN_KEYS.has(k) && typeof props.metadata![k] !== 'object')
})

const nestedKeys = computed(() => {
  if (!props.metadata) return []
  return Object.keys(props.metadata).filter(k => !HIDDEN_KEYS.has(k) && typeof props.metadata![k] === 'object' && props.metadata![k] !== null)
})

function formatKey(key: string): string {
  return KEY_LABELS[key] || key.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

function formatValue(val: any): string {
  if (typeof val === 'boolean') return val ? '是' : '否'
  if (Array.isArray(val)) return val.map(String).join(', ')
  if (typeof val === 'number') return val.toLocaleString()
  return String(val)
}
</script>

<style scoped>
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

.metadata-no-data {
  align-items: center;
  justify-content: center;
  gap: 8px;
}

.no-data-text {
  color: var(--text-tertiary);
  font-size: 13px;
}

.preprocess-link {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border-radius: var(--radius-sm);
  font-size: 13px;
  font-weight: 500;
  text-decoration: none;
  color: var(--accent-blue);
  background: rgba(9, 105, 218, 0.08);
  border: 1px solid rgba(9, 105, 218, 0.2);
  transition: all 0.15s ease;
}

.preprocess-link:hover {
  background: rgba(9, 105, 218, 0.14);
  border-color: var(--accent-blue);
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
</style>
