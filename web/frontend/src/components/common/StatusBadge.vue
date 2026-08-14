<template>
  <!-- pill: detail-page badge with tinted background; inline: table cell dot + text -->
  <span v-if="variant === 'pill'" class="status-pill" :style="{ '--dot-color': entry?.color }">
    <span class="dot" :class="{ pulse: status === 'running' }" />
    <span class="label">{{ label }}</span>
  </span>
  <span v-else class="status-inline">
    <span class="dot" :style="{ '--dot-color': entry?.color }" :class="{ glow: status === 'running' }" />
    <span class="text">{{ label }}</span>
  </span>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { statusMap } from '@/composables/useStatusMap'

const props = withDefaults(
  defineProps<{ status: string; variant?: 'pill' | 'inline' }>(),
  { variant: 'inline' },
)

const { t } = useI18n()
const entry = computed(() => statusMap[props.status])
const label = computed(() => (entry.value ? t(entry.value.label) : props.status))
</script>

<style scoped>
.status-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 20px;
  background: color-mix(in srgb, var(--dot-color, var(--text-tertiary)) 12%, transparent);
  border: 1px solid color-mix(in srgb, var(--dot-color, var(--text-tertiary)) 20%, transparent);
  flex-shrink: 0;
  font-size: 12px;
}

.status-pill .label {
  font-size: 12px;
  font-weight: 500;
  color: var(--dot-color, var(--text-secondary));
  line-height: 1;
}

.status-inline {
  display: inline-flex;
  align-items: center;
  gap: 7px;
}

.dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
  background: var(--dot-color, var(--text-tertiary));
}

.dot.glow {
  box-shadow: 0 0 6px var(--dot-color, var(--accent-blue));
}

.dot.pulse {
  animation: pulse-glow 2s ease-in-out infinite;
}

@keyframes pulse-glow {
  0%,
  100% {
    box-shadow: 0 0 0 0 color-mix(in srgb, var(--accent-blue) 50%, transparent);
    opacity: 1;
  }
  50% {
    box-shadow: 0 0 0 6px color-mix(in srgb, var(--accent-blue) 0%, transparent);
    opacity: 0.7;
  }
}

.text {
  font-size: 12px;
  font-family: var(--font-mono);
  color: var(--text-secondary);
}

@media (prefers-reduced-motion: reduce) {
  .dot.pulse {
    animation: none;
  }
}
</style>
