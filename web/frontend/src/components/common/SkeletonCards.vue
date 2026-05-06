<template>
  <div class="skeleton-cards" :style="{ gridTemplateColumns: `repeat(auto-fill, minmax(${cardWidth}px, 1fr))` }">
    <div v-for="i in count" :key="i" class="skeleton-card">
      <div class="skeleton-icon shimmer-block" />
      <div class="skeleton-text shimmer-block" />
    </div>
  </div>
</template>

<script setup lang="ts">
defineProps<{
  count?: number
  cardWidth?: number
}>()
</script>

<style scoped>
.skeleton-cards {
  display: grid;
  gap: 12px;
}

.skeleton-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  padding: 18px 12px 14px;
  background: var(--bg-surface);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
}

.skeleton-icon {
  width: 40px;
  height: 40px;
  border-radius: var(--radius-sm);
}

.skeleton-text {
  height: 13px;
  width: 70%;
  border-radius: 6px;
}

.shimmer-block {
  background: linear-gradient(
    90deg,
    var(--bg-overlay) 25%,
    color-mix(in srgb, var(--bg-overlay) 80%, var(--text-tertiary)) 50%,
    var(--bg-overlay) 75%
  );
  background-size: 200% 100%;
  animation: shimmer 1.5s ease-in-out infinite;
}

@keyframes shimmer {
  0% { background-position: 200% 0; }
  100% { background-position: -200% 0; }
}
</style>
