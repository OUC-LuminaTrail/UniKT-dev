<template>
  <div class="meta-grid">
    <div v-for="cell in visibleCells" :key="cell.label" class="meta-cell">
      <span class="meta-key">{{ cell.label }}</span>
      <span class="meta-val" :class="[cell.mono && 'mono', cell.valueClass]">{{ cell.value }}</span>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

export interface MetaCell {
  label: string
  value: string | number | null | undefined
  mono?: boolean
  valueClass?: string
}

const props = defineProps<{ cells: MetaCell[] }>()

// Cells may be conditionally omitted by passing null/undefined/'' (e.g. duration).
const visibleCells = computed(() => props.cells.filter((c) => c.value !== null && c.value !== undefined && c.value !== ''))
</script>

<style scoped>
.meta-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 14px 24px;
}

.meta-cell {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.meta-key {
  font-size: 11px;
  font-weight: 500;
  color: var(--text-tertiary);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}

.meta-val {
  font-size: 13px;
  color: var(--text-primary);
  font-family: var(--font-sans);
  line-height: 1.4;
  word-break: break-all;
}

.meta-val.mono {
  font-family: var(--font-mono);
  font-size: 12.5px;
}

@media (max-width: 900px) {
  .meta-grid {
    grid-template-columns: repeat(3, 1fr);
  }
}

@media (max-width: 600px) {
  .meta-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}
</style>
