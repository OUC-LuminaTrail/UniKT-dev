<template>
  <section class="detail-section">
    <div class="section-head">
      <h2 class="section-title">{{ title }}</h2>
      <span class="section-rule" />
      <slot name="actions" />
    </div>
    <slot />
  </section>
</template>

<script setup lang="ts">
defineProps<{ title: string }>()
</script>

<style scoped>
.detail-section {
  display: flex;
  flex-direction: column;
  gap: 10px;
}

.section-head {
  display: flex;
  align-items: center;
  gap: 12px;
}

.section-title {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-tertiary);
  letter-spacing: 0.05em;
  margin: 0;
  flex-shrink: 0;
  text-transform: uppercase;
}

.section-rule {
  flex: 1;
  height: 1px;
  background: var(--border-muted);
}
</style>
