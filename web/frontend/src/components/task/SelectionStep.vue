<template>
  <div class="selection-step">
    <div class="section">
      <div class="section-label">运行环境</div>
      <el-select
        :model-value="envId"
        @update:model-value="$emit('update:envId', $event)"
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
          @update:model-value="$emit('update:customPythonPath', $event)"
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
          @click="$emit('update:modelName', name)"
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
      <div class="card-grid" v-if="datasets.length">
        <div
          v-for="name in datasets"
          :key="name"
          class="select-card"
          :class="{ active: dataset === name }"
          @click="$emit('update:dataset', name)"
        >
          <div class="card-icon icon-dataset" :style="{ background: getGradient('ds-' + name) }">
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>
          </div>
          <div class="card-name" :title="name">{{ name }}</div>
        </div>
      </div>
    </div>

    <div class="action-bar">
      <el-button
        type="primary"
        size="large"
        :disabled="!modelName || !dataset"
        @click="$emit('confirm')"
      >
        确认选择
      </el-button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type { EnvironmentInfo } from '@/api/environments'

const props = defineProps<{
  envId: string
  customPythonPath: string
  modelName: string
  dataset: string
  environments: EnvironmentInfo[]
  models: string[]
  datasets: string[]
}>()

defineEmits<{
  (e: 'update:envId', val: string): void
  (e: 'update:customPythonPath', val: string): void
  (e: 'update:modelName', val: string): void
  (e: 'update:dataset', val: string): void
  (e: 'confirm'): void
}>()

const pixiEnvs = computed(() => props.environments.filter(e => e.type === 'pixi'))
const condaEnvs = computed(() => props.environments.filter(e => e.type === 'conda'))
const otherEnvs = computed(() => props.environments.filter(e => e.type !== 'pixi' && e.type !== 'conda'))

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