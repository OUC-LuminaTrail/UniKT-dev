<template>
  <div class="app-layout">
    <Sidebar />
    <div class="right-panel">
      <header class="app-header">
        <span class="header-title">{{ currentTitle }}</span>
        <div class="header-right">
          <span class="header-time">{{ currentTime }}</span>
        </div>
      </header>
      <main class="main-content" :class="{ 'main-content--flush': flushContent }">
        <slot :flushContent="flushContent" />
      </main>
      <footer class="app-footer">
        <div class="status-bar">
          <div class="status-item">
            <span class="status-dot" :style="{ background: progressColor(sys.cpu_percent) }"></span>
            <span class="status-label">CPU</span>
            <div class="status-bar-track">
              <div class="status-bar-fill" :style="{ width: sys.cpu_percent + '%', background: progressColor(sys.cpu_percent) }"></div>
            </div>
            <span class="status-value">{{ sys.cpu_percent.toFixed(0) }}%</span>
          </div>
          <div class="status-sep"></div>
          <div class="status-item">
            <span class="status-dot" :style="{ background: progressColor(sys.memory_percent) }"></span>
            <span class="status-label">MEM</span>
            <div class="status-bar-track">
              <div class="status-bar-fill" :style="{ width: sys.memory_percent + '%', background: progressColor(sys.memory_percent) }"></div>
            </div>
            <span class="status-value">{{ sys.memory_used_gb.toFixed(1) }}/{{ sys.memory_total_gb.toFixed(0) }}G</span>
          </div>
          <div class="status-sep"></div>
          <div class="status-item">
            <span class="status-dot" :style="{ background: progressColor(sys.gpu_utilization) }"></span>
            <span class="status-label">GPU</span>
            <div class="status-bar-track">
              <div class="status-bar-fill" :style="{ width: sys.gpu_utilization + '%', background: progressColor(sys.gpu_utilization) }"></div>
            </div>
            <span class="status-value">{{ sys.gpu_utilization.toFixed(0) }}%</span>
          </div>
          <div class="status-sep"></div>
          <div class="status-item">
            <span class="status-dot" :style="{ background: progressColor(sys.gpu_memory_percent) }"></span>
            <span class="status-label">VRAM</span>
            <div class="status-bar-track">
              <div class="status-bar-fill" :style="{ width: sys.gpu_memory_percent + '%', background: progressColor(sys.gpu_memory_percent) }"></div>
            </div>
            <span class="status-value">{{ sys.gpu_memory_percent.toFixed(0) }}%</span>
          </div>
          <div class="status-sep"></div>
          <div class="status-item status-load">
            <span class="status-label">LOAD</span>
            <span class="status-value">{{ sys.load_1m.toFixed(2) }}</span>
            <span class="status-value-sub">{{ sys.load_5m.toFixed(2) }}</span>
            <span class="status-value-sub">{{ sys.load_15m.toFixed(2) }}</span>
          </div>
        </div>
      </footer>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import Sidebar from './Sidebar.vue'
import { getSystemStatus, type SystemStatus } from '@/api/gpu'

const route = useRoute()

const titleMap: Record<string, string> = {
  '/tasks': '训练任务',
  '/tasks/new': '新建训练任务',
  '/experiments': '实验日志',
  '/gpu': 'GPU 监控',
}

const flushContent = computed(() => route.path === '/tasks/new')

const currentTitle = computed(() => {
  if (route.path.match(/^\/tasks\/\d+$/)) return 'Task Detail'
  return titleMap[route.path] || 'KT Experiment Manager'
})

const currentTime = ref('')
let timeTimer: ReturnType<typeof setInterval> | null = null

const updateTime = () => {
  currentTime.value = new Date().toLocaleTimeString('en-US', { hour12: false })
}

const defaultSys: SystemStatus = {
  cpu_percent: 0, memory_used_gb: 0, memory_total_gb: 0, memory_percent: 0,
  gpu_utilization: 0, gpu_memory_percent: 0, load_1m: 0, load_5m: 0, load_15m: 0,
  updated_at: '',
}
const sys = ref<SystemStatus>(defaultSys)
let statusTimer: ReturnType<typeof setInterval> | null = null

const progressColor = (pct: number) => {
  if (pct > 90) return 'var(--accent-red)'
  if (pct > 70) return 'var(--accent-orange)'
  return 'var(--accent-green)'
}

const loadStatus = async () => {
  try { sys.value = await getSystemStatus() } catch {}
}

onMounted(() => {
  updateTime()
  timeTimer = setInterval(updateTime, 1000)
  loadStatus()
  statusTimer = setInterval(loadStatus, 5000)
})

onUnmounted(() => {
  if (timeTimer) clearInterval(timeTimer)
  if (statusTimer) clearInterval(statusTimer)
})
</script>

<style scoped>
.app-layout {
  display: flex;
  height: 100vh;
  overflow: hidden;
  background: var(--bg-base);
}

.right-panel {
  margin-left: 220px;
  flex: 1;
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}

.app-header {
  height: 48px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  background: var(--bg-surface);
  border-bottom: 1px solid var(--border-muted);
}

.header-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
  font-family: var(--font-sans);
  letter-spacing: -0.01em;
}

.header-right {
  display: flex;
  align-items: center;
  gap: 12px;
}

.header-time {
  font-size: 12px;
  color: var(--text-tertiary);
  font-family: var(--font-mono);
  font-weight: 500;
}

.main-content {
  flex: 1;
  padding: 24px;
  overflow-y: auto;
  background: var(--bg-base);
}

.main-content--flush {
  padding: 0;
  overflow: hidden;
}

.app-footer {
  flex-shrink: 0;
  height: 32px;
  background: var(--bg-surface);
  border-top: 1px solid var(--border-muted);
  display: flex;
  align-items: center;
  padding: 0 16px;
}

.status-bar {
  display: flex;
  align-items: center;
  gap: 0;
  width: 100%;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 5px;
  padding: 0 8px;
}

.status-sep {
  width: 1px;
  height: 14px;
  background: var(--border-default);
}

.status-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  flex-shrink: 0;
}

.status-label {
  font-size: 10px;
  color: var(--text-tertiary);
  font-weight: 600;
  font-family: var(--font-mono);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  white-space: nowrap;
}

.status-bar-track {
  width: 48px;
  height: 3px;
  background: var(--border-default);
  border-radius: 2px;
  overflow: hidden;
}

.status-bar-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.5s ease, background 0.3s ease;
}

.status-value {
  font-size: 10px;
  color: var(--text-secondary);
  font-family: var(--font-mono);
  font-weight: 500;
  white-space: nowrap;
  min-width: 36px;
}

.status-value-sub {
  font-size: 9px;
  color: var(--text-tertiary);
  font-family: var(--font-mono);
}

.status-load .status-value-sub {
  opacity: 0.6;
}
</style>
