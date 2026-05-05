<template>
  <div class="command-preview">
    <div class="command-line">
      <span class="prompt">$</span>
      <span class="command-text">{{ command }}</span>
    </div>
    <div class="task-name">{{ taskName }}</div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  modelName: string
  dataset: string
  params: Record<string, any>
  schemaDefaultParams: Record<string, any>
}>()

const taskName = computed(() => {
  const m = props.modelName || ''
  const d = props.dataset || ''
  return d ? `${m}_${d}` : m || '...'
})

const command = computed(() => {
  const parts = ['python train.py']
  if (props.modelName) parts.push(`-m ${props.modelName}`)
  if (props.dataset) parts.push(`-d ${props.dataset}`)

  const overridden: string[] = []
  for (const [key, value] of Object.entries(props.params)) {
    if (value === null || value === undefined || value === '') continue
    const defaultVal = props.schemaDefaultParams[key]
    if (value === defaultVal) continue
    if (typeof value === 'boolean') {
      overridden.push(value ? `--${key}` : `--no_${key.replace(/^no_/, '')}`)
    } else {
      overridden.push(`--${key} ${value}`)
    }
  }
  if (overridden.length) parts.push(overridden.join(' '))

  return parts.join(' ')
})
</script>

<style scoped>
.command-preview {
  background: var(--bg-elevated);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  padding: 12px 18px;
  margin-bottom: 20px;
  position: sticky;
  top: 0;
  z-index: 5;
}

.command-line {
  display: flex;
  align-items: baseline;
  gap: 8px;
  overflow-x: auto;
  white-space: nowrap;
}

.prompt {
  color: var(--accent-green);
  font-family: var(--font-mono);
  font-size: 13px;
  font-weight: 700;
  flex-shrink: 0;
}

.command-text {
  color: var(--text-primary);
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 1.6;
}

.task-name {
  color: var(--text-tertiary);
  font-family: var(--font-mono);
  font-size: 11px;
  margin-top: 4px;
}
</style>
