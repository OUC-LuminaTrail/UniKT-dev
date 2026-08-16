<template>
  <div class="tab-bar" role="tablist" :aria-label="ariaLabel">
    <button
      v-for="tab in tabs"
      :key="tab.value"
      role="tab"
      :aria-selected="modelValue === tab.value"
      :class="['tab-item', { active: modelValue === tab.value }]"
      @click="emit('update:modelValue', tab.value)"
    >
      {{ t(tab.label) }}
      <span v-if="tab.badge" class="tab-badge">{{ tab.badge }}</span>
    </button>
  </div>
</template>

<script setup lang="ts">
import { useI18n } from 'vue-i18n'

export interface ListTab {
  value: string
  label: string
  badge?: number
}

defineProps<{ tabs: ListTab[]; modelValue: string; ariaLabel?: string }>()
const emit = defineEmits<{ (e: 'update:modelValue', value: string): void }>()

const { t } = useI18n()
</script>

<style scoped>
.tab-bar {
  display: flex;
  gap: 4px;
  padding: 4px;
  background: var(--bg-surface);
  border-radius: var(--radius-md);
  border: 1px solid var(--border-muted);
  width: fit-content;
}

.tab-item {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  background: transparent;
  border: none;
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: color 0.15s, background 0.15s;
  font-family: var(--font-sans);
}

.tab-item:hover {
  color: var(--text-primary);
  background: var(--bg-elevated);
}

.tab-item.active {
  color: var(--accent-blue);
  background: var(--bg-elevated);
  box-shadow: inset 0 -2px 0 var(--accent-blue);
}

.tab-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  min-width: 18px;
  height: 18px;
  padding: 0 5px;
  border-radius: 9px;
  background: var(--accent-blue);
  color: #fff;
  font-size: 11px;
  font-weight: 600;
  line-height: 1;
}

/* Dark-mode accent is lighter, so darken badge text to keep contrast. */
html.dark .tab-badge {
  color: var(--bg-base, #0d1117);
}
</style>
