<template>
  <div class="command-preview">
    <div class="command-preview-inner">
      <div class="command-info">
        <div class="command-line">
          <span class="prompt">$</span>
          <span class="command-text">{{ command || taskLabel }}</span>
        </div>
      </div>
      <div class="command-actions">
        <slot />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{
  command?: string
  modelName?: string
}>()

const taskLabel = computed(() => props.modelName || '...')
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

.command-actions {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 8px;
}
</style>
