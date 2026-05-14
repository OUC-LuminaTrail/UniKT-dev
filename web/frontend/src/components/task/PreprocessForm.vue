<template>
  <div class="preprocess-form">
    <div v-if="action === 'process'" class="param-group">
      <div class="group-header" @click="toggleGroup('sequence')">
        <div class="group-header-left">
          <span class="group-chevron" :class="{ expanded: expandedGroups.includes('sequence') }">
            <el-icon :size="12"><ArrowRight /></el-icon>
          </span>
          <span class="group-name">序列参数</span>
          <span class="group-count">4</span>
        </div>
      </div>
      <transition name="collapse">
        <div v-if="expandedGroups.includes('sequence')" class="group-body">
          <div class="fields-grid">
            <div class="field-item">
              <div class="field-top-row">
                <span class="field-key">min_seq_len</span>
                <span class="type-badge">int</span>
              </div>
              <span class="field-help">最小序列长度，短于此长度的序列将被过滤</span>
              <div class="field-input">
                <el-input-number v-model="processOpts.min_seq_len" :min="1" :max="1000" controls-position="right" style="width:100%" />
              </div>
              <div class="field-default">default: <span class="default-val">10</span></div>
            </div>
            <div class="field-item">
              <div class="field-top-row">
                <span class="field-key">max_seq_len</span>
                <span class="type-badge">int</span>
              </div>
              <span class="field-help">最大序列长度，超过此长度将被截断</span>
              <div class="field-input">
                <el-input-number v-model="processOpts.max_seq_len" :min="10" :max="2000" controls-position="right" style="width:100%" />
              </div>
              <div class="field-default">default: <span class="default-val">200</span></div>
            </div>
            <div class="field-item">
              <div class="field-top-row">
                <span class="field-key">kfold</span>
                <span class="type-badge">int</span>
              </div>
              <span class="field-help">K 折交叉验证折数，≥2 启用交叉验证</span>
              <div class="field-input">
                <el-input-number v-model="processOpts.kfold" :min="1" :max="20" controls-position="right" style="width:100%" />
              </div>
              <div class="field-default">default: <span class="default-val">5</span></div>
            </div>
            <div class="field-item">
              <div class="field-top-row">
                <span class="field-key">seed</span>
                <span class="type-badge">int</span>
              </div>
              <span class="field-help">随机种子，用于数据划分和采样</span>
              <div class="field-input">
                <el-input-number v-model="processOpts.seed" :min="0" controls-position="right" style="width:100%" />
              </div>
              <div class="field-default">default: <span class="default-val">42</span></div>
            </div>
          </div>
        </div>
      </transition>
    </div>

    <div v-if="action === 'process'" class="param-group">
      <div class="group-header" @click="toggleGroup('sampling')">
        <div class="group-header-left">
          <span class="group-chevron" :class="{ expanded: expandedGroups.includes('sampling') }">
            <el-icon :size="12"><ArrowRight /></el-icon>
          </span>
          <span class="group-name">采样参数</span>
          <span class="group-count">{{ samplingVisibleCount }}</span>
        </div>
      </div>
      <transition name="collapse">
        <div v-if="expandedGroups.includes('sampling')" class="group-body">
          <div class="fields-grid">
            <div class="field-item">
              <div class="field-top-row">
                <span class="field-key">sample_size</span>
                <span class="type-badge">int</span>
              </div>
              <span class="field-help">采样绝对数量（random/stratified 为用户数，time 为交互数）</span>
              <div class="field-input">
                <el-input-number v-model="processOpts.sample_size" :min="1" controls-position="right" placeholder="不启用" style="width:100%" />
              </div>
              <div class="field-default">default: <span class="default-val">null（不启用）</span></div>
            </div>
            <div class="field-item">
              <div class="field-top-row">
                <span class="field-key">sample_ratio</span>
                <span class="type-badge">float</span>
              </div>
              <span class="field-help">采样比例 0.0~1.0，设置后覆盖 sample_size</span>
              <div class="field-input">
                <el-input-number v-model="processOpts.sample_ratio" :min="0.01" :max="1" :step="0.05" :precision="2" controls-position="right" placeholder="不启用" style="width:100%" />
              </div>
              <div class="field-default">default: <span class="default-val">null（不启用）</span></div>
            </div>
            <div class="field-item">
              <div class="field-top-row">
                <span class="field-key">sample_strategy</span>
                <span class="type-badge">str</span>
              </div>
              <span class="field-help">采样策略：random / stratified（分层）/ time（按时间）</span>
              <div class="field-input">
                <el-select v-model="processOpts.sample_strategy" clearable style="width:100%">
                  <el-option value="random" label="random（随机）" />
                  <el-option value="stratified" label="stratified（分层）" />
                  <el-option value="time" label="time（按时间）" />
                </el-select>
              </div>
              <div class="field-default">default: <span class="default-val">random</span></div>
            </div>
            <template v-if="processOpts.sample_strategy === 'stratified'">
              <div class="field-item">
                <div class="field-top-row">
                  <span class="field-key">sample_attempts_bins</span>
                  <span class="type-badge">str</span>
                </div>
                <span class="field-help">分层采样的尝试次数分箱边界，空格分隔</span>
                <div class="field-input">
                  <el-input v-model="processOpts.sample_attempts_bins" placeholder="20 100" />
                </div>
                <div class="field-default">default: <span class="default-val">"20 100"</span></div>
              </div>
              <div class="field-item">
                <div class="field-top-row">
                  <span class="field-key">sample_correct_bins</span>
                  <span class="type-badge">str</span>
                </div>
                <span class="field-help">分层采样的正确率分箱边界，空格分隔</span>
                <div class="field-input">
                  <el-input v-model="processOpts.sample_correct_bins" placeholder="0.4 0.8" />
                </div>
                <div class="field-default">default: <span class="default-val">"0.4 0.8"</span></div>
              </div>
            </template>
          </div>
        </div>
      </transition>
    </div>

    <div v-if="action === 'process'" class="param-group">
      <div class="group-header" @click="toggleGroup('extra')">
        <div class="group-header-left">
          <span class="group-chevron" :class="{ expanded: expandedGroups.includes('extra') }">
            <el-icon :size="12"><ArrowRight /></el-icon>
          </span>
          <span class="group-name">额外选项</span>
          <span class="group-count">1</span>
        </div>
      </div>
      <transition name="collapse">
        <div v-if="expandedGroups.includes('extra')" class="group-body">
          <div class="fields-grid">
            <div class="field-item">
              <div class="field-top-row">
                <span class="field-key">extra</span>
                <span class="type-badge">str</span>
              </div>
              <span class="field-help">额外处理步骤，如 windowslate</span>
              <div class="field-input">
                <el-input v-model="processOpts.extra" placeholder="windowslate" />
              </div>
              <div class="field-default">default: <span class="default-val">null</span></div>
            </div>
          </div>
        </div>
      </transition>
    </div>

    <div v-if="action === 'download'" class="param-group">
      <div class="group-header" @click="toggleGroup('download')">
        <div class="group-header-left">
          <span class="group-chevron" :class="{ expanded: expandedGroups.includes('download') }">
            <el-icon :size="12"><ArrowRight /></el-icon>
          </span>
          <span class="group-name">下载选项</span>
          <span class="group-count">3</span>
        </div>
      </div>
      <transition name="collapse">
        <div v-if="expandedGroups.includes('download')" class="group-body">
          <div class="fields-grid">
            <div class="field-item">
              <div class="field-top-row">
                <span class="field-key">force</span>
                <span class="type-badge">bool</span>
              </div>
              <span class="field-help">强制重新下载，覆盖已有文件</span>
              <div class="field-input">
                <el-switch v-model="downloadOpts.force" />
              </div>
              <div class="field-default">default: <span class="default-val">false</span></div>
            </div>
            <div class="field-item">
              <div class="field-top-row">
                <span class="field-key">max_retries</span>
                <span class="type-badge">int</span>
              </div>
              <span class="field-help">下载失败时的最大重试次数</span>
              <div class="field-input">
                <el-input-number v-model="downloadOpts.max_retries" :min="1" :max="10" controls-position="right" style="width:100%" />
              </div>
              <div class="field-default">default: <span class="default-val">3</span></div>
            </div>
            <div class="field-item">
              <div class="field-top-row">
                <span class="field-key">num_threads</span>
                <span class="type-badge">int</span>
              </div>
              <span class="field-help">并发下载线程数</span>
              <div class="field-input">
                <el-input-number v-model="downloadOpts.num_threads" :min="1" :max="16" controls-position="right" style="width:100%" />
              </div>
              <div class="field-default">default: <span class="default-val">4</span></div>
            </div>
          </div>
        </div>
      </transition>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { ArrowRight } from '@element-plus/icons-vue'

const props = defineProps<{
  action: 'download' | 'process'
}>()

const emit = defineEmits<{
  (e: 'update:downloadOpts', opts: typeof downloadOpts.value): void
  (e: 'update:processOpts', opts: typeof processOpts.value): void
}>()

const expandedGroups = ref<string[]>([])

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

const samplingVisibleCount = computed(() => {
  let n = 3
  if (processOpts.value.sample_strategy === 'stratified') n += 2
  return n
})

const toggleGroup = (name: string) => {
  const idx = expandedGroups.value.indexOf(name)
  if (idx >= 0) expandedGroups.value.splice(idx, 1)
  else expandedGroups.value.push(name)
}

watch(
  () => props.action,
  (action) => {
    if (action === 'download') {
      expandedGroups.value = ['download']
    } else {
      expandedGroups.value = ['sequence', 'sampling', 'extra']
    }
  },
  { immediate: true }
)

watch(downloadOpts, (v) => emit('update:downloadOpts', v), { deep: true })
watch(processOpts, (v) => emit('update:processOpts', v), { deep: true })

defineExpose({ downloadOpts, processOpts })
</script>

<style scoped>
.preprocess-form {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.param-group {
  background: var(--bg-surface);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  overflow: hidden;
}

.group-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 20px;
  cursor: pointer;
  user-select: none;
  transition: background 0.15s ease;
}

.group-header:hover {
  background: var(--bg-elevated);
}

.group-header-left {
  display: flex;
  align-items: center;
  gap: 10px;
}

.group-chevron {
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-tertiary);
  transition: transform 0.2s ease;
}

.group-chevron.expanded {
  transform: rotate(90deg);
}

.group-name {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
}

.group-count {
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--text-tertiary);
  background: var(--bg-elevated);
  padding: 1px 7px;
  border-radius: 10px;
  line-height: 1.6;
}

.group-body {
  padding: 4px 20px 20px;
}

.fields-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px 28px;
}

.field-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.field-top-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.field-key {
  font-size: 13px;
  font-family: var(--font-mono);
  font-weight: 500;
  color: var(--text-primary);
  letter-spacing: 0.2px;
}

.type-badge {
  font-size: 10px;
  font-family: var(--font-mono);
  color: var(--text-tertiary);
  background: var(--bg-overlay);
  padding: 1px 6px;
  border-radius: var(--radius-sm);
  letter-spacing: 0.3px;
}

.field-help {
  font-size: 12px;
  color: var(--text-tertiary);
  line-height: 1.4;
}

.field-input {
  width: 100%;
}

.field-input :deep(.el-input__wrapper),
.field-input :deep(.el-select .el-input__wrapper) {
  height: 36px;
  border-radius: var(--radius-sm);
}

.field-input :deep(.el-input-number) {
  width: 100%;
}

.field-input :deep(.el-input-number .el-input__wrapper) {
  height: 36px;
  border-radius: var(--radius-sm);
}

.field-default {
  font-size: 11px;
  color: var(--text-tertiary);
  font-family: var(--font-mono);
}

.default-val {
  color: var(--accent-blue);
}

.collapse-enter-active,
.collapse-leave-active {
  transition: all 0.2s ease;
  overflow: hidden;
}

.collapse-enter-from,
.collapse-leave-to {
  opacity: 0;
  max-height: 0;
  padding-top: 0;
  padding-bottom: 0;
}

.collapse-enter-to,
.collapse-leave-from {
  opacity: 1;
  max-height: 2000px;
}
</style>
