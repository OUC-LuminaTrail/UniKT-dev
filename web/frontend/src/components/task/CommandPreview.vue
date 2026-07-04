<template>
  <div class="command-preview">
    <div class="command-preview-inner">
      <div class="command-info">
        <div class="command-line">
          <span class="prompt">$</span>
          <span class="command-text">{{ commandDisplay }}</span>
        </div>
        <div class="task-name">{{ taskName }}</div>
      </div>
      <div class="command-actions">
        <slot />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const sameAsDefault = (a: any, b: any): boolean => {
  if (Array.isArray(a) && Array.isArray(b)) {
    return a.length === b.length && a.every((v, i) => v === b[i])
  }
  return a === b
}

const props = defineProps<{
  command?: string
  modelName?: string
  dataset?: string
  params?: Record<string, any>
  schemaDefaultParams?: Record<string, any>
}>()

const taskName = computed(() => {
  if (props.command) return ''
  const m = props.modelName || ''
  const d = props.dataset || ''
  return d ? `${m}_${d}` : m || '...'
})

const commandDisplay = computed(() => {
  if (props.command) return props.command
  const parts = ['python train.py']
  if (props.modelName) parts.push(`-m ${props.modelName}`)
  if (props.dataset) parts.push(`-d ${props.dataset}`)

  const overridden: string[] = []
  for (const [key, value] of Object.entries(props.params || {})) {
    if (value === null || value === undefined || value === '') continue
    if (Array.isArray(value) && value.length === 0) continue
    const defaultVal = (props.schemaDefaultParams || {})[key]
    if (sameAsDefault(value, defaultVal)) continue
    if (typeof value === 'boolean') {
      overridden.push(value ? `--${key}` : `--no_${key.replace(/^no_/, '')}`)
    } else if (Array.isArray(value)) {
      overridden.push(`--${key} ${value.join(' ')}`)
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
  flex-shrink: 0;
  background: var(--bg-surface);
  border-top: 1px solid var(--border-muted);
  padding: 10px 20px;
}

.command-preview-inner {
  display: flex;
  align-items: center;
  gap: 16px;
}

.command-info {
  flex: 1;
  min-width: 0;
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
  margin-top: 2px;
}

.command-actions {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 8px;
}
</style>
