<template>
  <div class="skeleton-table">
    <div class="skeleton-header">
      <div v-for="i in cols" :key="i" class="skeleton-th" :style="{ width: colWidths?.[i - 1] || 'auto' }">
        <div class="skeleton-bar short" />
      </div>
    </div>
    <div v-for="r in rows" :key="r" class="skeleton-row">
      <div v-for="c in cols" :key="c" class="skeleton-td" :style="{ width: colWidths?.[c - 1] || 'auto' }">
        <div class="skeleton-bar" :class="{ wide: (r + c) % 3 === 0 }" />
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  rows?: number
  cols?: number
  colWidths?: string[]
}>()
</script>

<style scoped>
.skeleton-table {
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  overflow: hidden;
}

.skeleton-header {
  display: flex;
  background: var(--bg-elevated);
  padding: 8px 12px;
  gap: 0;
}

.skeleton-th {
  padding: 4px 0;
}

.skeleton-row {
  display: flex;
  padding: 8px 12px;
  border-top: 1px solid var(--border-muted);
  gap: 0;
}

.skeleton-td {
  flex: 1;
  display: flex;
  align-items: center;
  min-width: 60px;
}

.skeleton-bar {
  height: 12px;
  border-radius: 6px;
  background: linear-gradient(
    90deg,
    var(--bg-overlay) 25%,
    color-mix(in srgb, var(--bg-overlay) 80%, var(--text-tertiary)) 50%,
    var(--bg-overlay) 75%
  );
  background-size: 200% 100%;
  animation: shimmer 1.5s ease-in-out infinite;
  width: 60%;
}

.skeleton-bar.short {
  height: 10px;
  width: 40%;
}

.skeleton-bar.wide {
  width: 80%;
}

@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
</style>
