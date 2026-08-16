<template>
  <header class="detail-header">
    <button class="back-btn" @click="goBack">
      <el-icon :size="18"><ArrowLeft /></el-icon>
    </button>
    <h1 class="name">{{ name }}</h1>
    <StatusBadge :status="status" variant="pill" />
    <div v-if="status === 'running' || status === 'pending'" class="header-actions">
      <button class="action-btn stop" :disabled="stopping" @click="emit('stop')">
        <el-icon :size="14"><SwitchButton /></el-icon>
        <span>{{ stopping ? t('common.stopping') : stopLabel }}</span>
      </button>
      <button v-if="status === 'running'" class="action-btn kill" :disabled="killing" @click="emit('kill')">
        <el-icon :size="14"><Bottom /></el-icon>
        <span>{{ killing ? t('common.killing') : t('common.forceKill') }}</span>
      </button>
    </div>
  </header>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { ArrowLeft, Bottom, SwitchButton } from '@element-plus/icons-vue'
import StatusBadge from './StatusBadge.vue'

const props = withDefaults(
  defineProps<{ name: string; status: string; stopLabel?: string; stopping?: boolean; killing?: boolean; fallbackRoute: string }>(),
  { stopLabel: '', stopping: false, killing: false },
)
const emit = defineEmits<{ (e: 'back'): void; (e: 'stop'): void; (e: 'kill'): void }>()

const { t } = useI18n()
const router = useRouter()

const stopLabel = computed(() =>
  props.status === 'pending' ? t('common.cancelQueue') : props.stopLabel || t('common.stop'),
)

const goBack = () => {
  if (window.history.length > 1) router.back()
  else router.replace({ name: props.fallbackRoute })
  emit('back')
}
</script>

<style scoped>
.detail-header {
  display: flex;
  align-items: center;
  gap: 12px;
  min-height: 36px;
}

.back-btn {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 32px;
  height: 32px;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-sm);
  background: var(--bg-surface);
  color: var(--text-secondary);
  cursor: pointer;
  transition: all 0.15s ease;
  flex-shrink: 0;
}

.back-btn:hover {
  background: var(--bg-elevated);
  color: var(--text-primary);
  border-color: var(--accent-blue);
}

.name {
  font-size: 18px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  letter-spacing: -0.01em;
}

.header-actions {
  display: flex;
  gap: 8px;
  margin-left: auto;
  flex-shrink: 0;
}

.action-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-default);
  background: var(--bg-surface);
  color: var(--text-secondary);
  font-size: 13px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.15s ease;
  font-family: var(--font-sans);
}

.action-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.action-btn.stop:hover:not(:disabled) {
  border-color: var(--accent-orange);
  color: var(--accent-orange);
  background: color-mix(in srgb, var(--accent-orange) 8%, var(--bg-surface));
}

.action-btn.kill:hover:not(:disabled) {
  border-color: var(--accent-red);
  color: var(--accent-red);
  background: color-mix(in srgb, var(--accent-red) 8%, var(--bg-surface));
}
</style>
