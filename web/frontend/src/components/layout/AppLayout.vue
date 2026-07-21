<template>
  <el-container class="app-layout">
    <el-aside :width="isCollapsed ? '64px' : '220px'" class="app-aside">
      <Sidebar :collapsed="isCollapsed" @toggle-collapse="isCollapsed = !isCollapsed" />
    </el-aside>
    <el-container>
      <el-header class="app-header">
        <div class="header-left">
          <span class="header-title">{{ currentTitle }}</span>
        </div>
        <div class="header-right">
          <span class="header-time">{{ currentTime }}</span>
        </div>
      </el-header>
      <el-main class="main-content" :class="{ 'main-content--flush': flushContent }">
        <slot :flushContent="flushContent" />
      </el-main>
      <el-footer class="app-footer">
        <div class="status-bar">
          <el-tooltip :content="`CPU 使用率 ${sys.cpu_percent.toFixed(0)}%`" placement="top" :show-after="400">
            <div class="status-item" tabindex="0">
              <span class="status-dot" :style="{ background: progressColor(sys.cpu_percent) }"></span>
              <span class="status-label">CPU</span>
              <div class="status-bar-track">
                <div class="status-bar-fill" :style="{ width: sys.cpu_percent + '%', background: progressColor(sys.cpu_percent) }"></div>
              </div>
              <span class="status-value">{{ sys.cpu_percent.toFixed(0) }}%</span>
            </div>
          </el-tooltip>

          <div class="status-sep"></div>

          <el-tooltip :content="memTooltip" placement="top" :show-after="400">
            <div class="status-item" tabindex="0">
              <span class="status-dot" :style="{ background: progressColor(sys.memory_percent) }"></span>
              <span class="status-label">MEM</span>
              <div class="status-bar-track">
                <div class="status-bar-fill" :style="{ width: sys.memory_percent + '%', background: progressColor(sys.memory_percent) }"></div>
              </div>
              <span class="status-value">{{ sys.memory_used_gb.toFixed(1) }}/{{ sys.memory_total_gb.toFixed(0) }}G</span>
            </div>
          </el-tooltip>

          <template v-if="hasGpu">
            <div class="status-sep"></div>
            <el-tooltip content="GPU 平均计算利用率" placement="top" :show-after="400">
              <div class="status-item" tabindex="0">
                <span class="status-dot" :style="{ background: progressColor(sys.gpu_utilization) }"></span>
                <span class="status-label">GPU</span>
                <div class="status-bar-track">
                  <div class="status-bar-fill" :style="{ width: sys.gpu_utilization + '%', background: progressColor(sys.gpu_utilization) }"></div>
                </div>
                <span class="status-value">{{ sys.gpu_utilization.toFixed(0) }}%</span>
              </div>
            </el-tooltip>

            <div class="status-sep"></div>
            <el-tooltip content="GPU 显存占用率" placement="top" :show-after="400">
              <div class="status-item" tabindex="0">
                <span class="status-dot" :style="{ background: progressColor(sys.gpu_memory_percent) }"></span>
                <span class="status-label">VRAM</span>
                <div class="status-bar-track">
                  <div class="status-bar-fill" :style="{ width: sys.gpu_memory_percent + '%', background: progressColor(sys.gpu_memory_percent) }"></div>
                </div>
                <span class="status-value">{{ sys.gpu_memory_percent.toFixed(0) }}%</span>
              </div>
            </el-tooltip>
          </template>

          <div class="status-sep"></div>
          <el-tooltip :content="loadTooltip" placement="top" :show-after="400">
            <div class="status-item status-load" tabindex="0">
              <span class="status-label">LOAD</span>
              <span class="status-value">{{ sys.load_1m.toFixed(2) }}</span>
              <span class="status-value-sub">{{ sys.load_5m.toFixed(2) }}</span>
              <span class="status-value-sub">{{ sys.load_15m.toFixed(2) }}</span>
            </div>
          </el-tooltip>

          <div class="status-spacer"></div>

          <el-tooltip :content="live ? '遥测正常更新（每 5 秒）' : '无法连接服务端，数据已停止更新'" placement="top" :show-after="400">
            <div class="status-freshness" tabindex="0">
              <span class="freshness-dot" :class="{ offline: !live }"></span>
              <span class="freshness-text">{{ live ? '实时' : '离线' }}</span>
            </div>
          </el-tooltip>
        </div>
      </el-footer>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onUnmounted } from 'vue'
import { useRoute } from 'vue-router'
import Sidebar from './Sidebar.vue'
import { getSystemStatus, type SystemStatus } from '@/api/gpu'
import { useSystemCapabilities } from '@/composables/useSystemCapabilities'

const route = useRoute()

const isCollapsed = ref(false)
const { hasGpu } = useSystemCapabilities()

const flushContent = computed(() => !!route.meta?.flush)

const currentTitle = computed(() => {
  return (route.meta?.title as string) || 'KT Experiment Manager'
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
const live = ref(false)
let statusTimer: ReturnType<typeof setInterval> | null = null

const progressColor = (pct: number) => {
  if (pct > 90) return 'var(--accent-red)'
  if (pct > 70) return 'var(--accent-orange)'
  return 'var(--accent-green)'
}

const memTooltip = computed(() =>
  `内存 ${sys.value.memory_used_gb.toFixed(1)} / ${sys.value.memory_total_gb.toFixed(0)} GB（${sys.value.memory_percent.toFixed(0)}%）`
)

const loadTooltip = computed(() =>
  `系统平均负载（1 / 5 / 15 分钟）：${sys.value.load_1m.toFixed(2)} / ${sys.value.load_5m.toFixed(2)} / ${sys.value.load_15m.toFixed(2)}`
)

// CPU/MEM/LOAD are system-level and available with or without a GPU, so poll unconditionally.
const loadStatus = async () => {
  try {
    sys.value = await getSystemStatus()
    live.value = true
  } catch {
    live.value = false
  }
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
  height: 100vh;
  overflow: hidden;
  background: var(--bg-base);
}

.app-aside {
  transition: width 0.3s ease;
  overflow: hidden;
}

.app-header {
  --el-header-padding: 0 24px;
  --el-header-height: 48px;
  display: flex;
  align-items: center;
  justify-content: space-between;
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

.header-left {
  display: flex;
  align-items: center;
  gap: 10px;
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
  --el-main-padding: 24px;
  min-height: 0;
  background: var(--bg-base);
}

.main-content--flush {
  --el-main-padding: 0;
  overflow: hidden;
}

.app-footer {
  --el-footer-padding: 0 14px;
  --el-footer-height: 32px;
  display: flex;
  align-items: center;
  background: var(--bg-surface);
  border-top: 1px solid var(--border-muted);
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
  gap: 6px;
  padding: 0 10px;
  height: 32px;
  border-radius: var(--radius-sm);
  cursor: default;
  transition: background 0.15s ease;
}

.status-item:hover,
.status-item:focus-visible {
  background: var(--bg-overlay);
}

.status-sep {
  width: 1px;
  height: 14px;
  background: var(--border-default);
  flex-shrink: 0;
}

.status-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
}

.status-label {
  font-size: 11px;
  color: var(--text-tertiary);
  font-weight: 600;
  font-family: var(--font-mono);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  white-space: nowrap;
}

.status-bar-track {
  width: 54px;
  height: 4px;
  background: var(--border-default);
  border-radius: 2px;
  overflow: hidden;
  flex-shrink: 0;
}

.status-bar-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.5s ease, background 0.3s ease;
}

.status-value {
  font-size: 11px;
  color: var(--text-secondary);
  font-family: var(--font-mono);
  font-weight: 500;
  white-space: nowrap;
  min-width: 38px;
}

.status-value-sub {
  font-size: 10px;
  color: var(--text-tertiary);
  font-family: var(--font-mono);
}

.status-load .status-value-sub {
  opacity: 0.6;
}

.status-spacer {
  flex: 1 1 auto;
}

.status-freshness {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 0 10px;
  height: 32px;
  border-radius: var(--radius-sm);
  cursor: default;
}

.status-freshness:hover,
.status-freshness:focus-visible {
  background: var(--bg-overlay);
}

.freshness-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--accent-green);
  box-shadow: 0 0 0 0 var(--soft-green);
  animation: live-pulse 2.4s infinite;
}

.freshness-dot.offline {
  background: var(--accent-red);
  animation: none;
}

.freshness-text {
  font-size: 11px;
  font-weight: 600;
  font-family: var(--font-mono);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-tertiary);
}

@keyframes live-pulse {
  0% { box-shadow: 0 0 0 0 var(--soft-green); }
  70% { box-shadow: 0 0 0 5px transparent; }
  100% { box-shadow: 0 0 0 0 transparent; }
}

@media (prefers-reduced-motion: reduce) {
  .freshness-dot { animation: none; }
  .status-bar-fill { transition: none; }
}
</style>
