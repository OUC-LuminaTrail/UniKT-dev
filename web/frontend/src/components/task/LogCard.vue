<template>
  <section class="log-card" :style="cardStyle">
    <div class="log-card-header">
      <div class="log-card-title">{{ t('log.title') }}</div>
      <span class="header-rule"></span>
      <div class="header-right">
        <span class="conn-indicator" :class="`conn-${connState}`" :title="connTitle">
          <span class="conn-dot"></span>
          <span class="conn-label">{{ connLabel }}</span>
        </span>
        <button class="scroll-btn" @click="scrollToBottom">
          <el-icon :size="14"><Bottom /></el-icon>
          {{ t('log.scrollToBottom') }}
        </button>
      </div>
    </div>
    <div class="terminal-wrapper">
      <LogViewer
        ref="viewerRef"
        :ws-url="wsUrl"
        :task-status="taskStatus"
        :task-id="taskId"
        :api-base="apiBase"
        @state="onState"
      />
    </div>
  </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { Bottom } from '@element-plus/icons-vue'
import LogViewer from './LogViewer.vue'
import type { LogApiBase } from '@/api/logs'
import { CONN_MAP, type ConnState } from './log-conn'

const props = defineProps<{
  wsUrl: string
  taskStatus: string
  taskId: number
  apiBase?: LogApiBase
  fill?: boolean
}>()

const viewerRef = ref<InstanceType<typeof LogViewer>>()
const { t } = useI18n()

const cardStyle = computed(() =>
  props.fill ? { flex: '1 1 0', margin: '0 20px 20px' } : {}
)

const connState = ref<ConnState>('connecting')
const onState = (s: ConnState) => { connState.value = s }

const connLabel = computed(() => t(CONN_MAP[connState.value].label))
const connTitle = computed(() => t(CONN_MAP[connState.value].title))

const scrollToBottom = () => { viewerRef.value?.scrollToBottom() }
</script>

<style scoped>
.log-card {
  display: flex;
  flex-direction: column;
  flex: 1 1 0;
  min-height: 0;
}

.log-card-header {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 0 10px;
  flex-shrink: 0;
}

.log-card-title {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-tertiary);
  letter-spacing: 0.05em;
  flex-shrink: 0;
}

.header-rule {
  flex: 1;
  height: 1px;
  background: var(--border-muted);
}

.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.conn-indicator {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  font-size: 11px;
  color: var(--text-tertiary);
  font-family: var(--font-sans);
  letter-spacing: 0.05em;
}

.conn-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  background: var(--text-tertiary);
  flex-shrink: 0;
}

.conn-connecting .conn-dot {
  background: var(--accent-orange);
  animation: conn-pulse 1s infinite;
}

.conn-connected .conn-dot {
  background: var(--accent-green);
  box-shadow: 0 0 5px var(--accent-green);
}

.conn-connected .conn-label {
  color: var(--accent-green);
}

.conn-reconnecting .conn-dot {
  background: var(--accent-orange);
  animation: conn-pulse 0.8s infinite;
}

.conn-reconnecting .conn-label {
  color: var(--accent-orange);
}

.conn-ended .conn-dot {
  background: var(--text-tertiary);
}

@keyframes conn-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.3; }
}

.scroll-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: var(--text-tertiary);
  background: none;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-sm);
  padding: 2px 8px;
  cursor: pointer;
  font-family: var(--font-sans);
  transition: all 0.15s ease;
}

.scroll-btn:hover {
  color: var(--accent-blue);
  border-color: var(--accent-blue);
}

.terminal-wrapper {
  flex: 1 1 0;
  min-height: 0;
  background: var(--term-bg);
}

@media (prefers-reduced-motion: reduce) {
  .conn-connecting .conn-dot,
  .conn-reconnecting .conn-dot { animation: none; }
}
</style>
