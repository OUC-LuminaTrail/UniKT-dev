<template>
  <div class="empty-state">
    <el-icon :size="48"><component :is="icon" /></el-icon>
    <p>{{ title }}</p>
    <span v-if="hint">{{ hint }}</span>
  </div>
</template>

<script setup lang="ts">
import type { Component } from 'vue'
import { Document } from '@element-plus/icons-vue'

withDefaults(defineProps<{ icon?: Component; title: string; hint?: string }>(), {
  icon: () => Document,
})
</script>

<style scoped>
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 64px 24px;
  color: var(--text-tertiary);
  text-align: center;
}

.empty-state svg {
  margin-bottom: 16px;
  opacity: 0.4;
}

.empty-state p {
  margin: 0 0 4px;
  font-size: 15px;
  font-weight: 500;
  color: var(--text-secondary);
}

.empty-state span {
  font-size: 13px;
}
</style>
