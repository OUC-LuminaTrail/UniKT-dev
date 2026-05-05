<template>
  <div class="app-layout">
    <aside class="sidebar">
      <div class="sidebar-brand">KT Experiment</div>
      <Sidebar />
    </aside>
    <div class="right-panel">
      <header class="app-header">
        <span class="header-title">{{ currentTitle }}</span>
        <span class="header-time">{{ currentTime }}</span>
      </header>
      <main class="main-content">
        <slot />
      </main>
      <footer class="app-footer">
        <div class="status-bar">
          <div class="status-item">
            <span class="status-label">CPU</span>
            <el-progress
              :percentage="sys.cpu_percent"
              :stroke-width="8"
              :color="progressColor(sys.cpu_percent)"
              style="width: 80px"
            />
          </div>
          <div class="status-item">
            <span class="status-label">内存</span>
            <el-progress
              :percentage="sys.memory_percent"
              :stroke-width="8"
              :color="progressColor(sys.memory_percent)"
              style="width: 80px"
            />
            <span class="status-value">{{ sys.memory_used_gb }}/{{ sys.memory_total_gb }}GB</span>
          </div>
          <div class="status-item">
            <span class="status-label">GPU</span>
            <el-progress
              :percentage="sys.gpu_utilization"
              :stroke-width="8"
              :color="progressColor(sys.gpu_utilization)"
              style="width: 80px"
            />
          </div>
          <div class="status-item">
            <span class="status-label">显存</span>
            <el-progress
              :percentage="sys.gpu_memory_percent"
              :stroke-width="8"
              :color="progressColor(sys.gpu_memory_percent)"
              style="width: 80px"
            />
          </div>
          <div class="status-item">
            <span class="status-label">Load</span>
            <span class="status-value">{{ sys.load_1m }} / {{ sys.load_5m }} / {{ sys.load_15m }}</span>
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
  '/tasks/new': '新建任务',
  '/experiments': '实验记录',
  '/gpu': 'GPU 监控',
}

const currentTitle = computed(() => {
  if (route.path.match(/^\/tasks\/\d+$/)) return '任务详情'
  return titleMap[route.path] || 'KT Experiment Manager'
})

const currentTime = ref('')
let timeTimer: ReturnType<typeof setInterval> | null = null

const updateTime = () => {
  currentTime.value = new Date().toLocaleString('zh-CN')
}

const defaultSys: SystemStatus = {
  cpu_percent: 0, memory_used_gb: 0, memory_total_gb: 0, memory_percent: 0,
  gpu_utilization: 0, gpu_memory_percent: 0, load_1m: 0, load_5m: 0, load_15m: 0,
  updated_at: '',
}
const sys = ref<SystemStatus>(defaultSys)
let statusTimer: ReturnType<typeof setInterval> | null = null

const progressColor = (pct: number) => {
  if (pct > 90) return '#f56c6c'
  if (pct > 70) return '#e6a23c'
  return '#67c23a'
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
}

.sidebar {
  width: 200px;
  flex-shrink: 0;
  background: #fff;
  border-right: 1px solid #e4e7ed;
  display: flex;
  flex-direction: column;
  position: fixed;
  top: 0;
  left: 0;
  bottom: 0;
  z-index: 10;
}

.sidebar-brand {
  height: 50px;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 16px;
  color: #303133;
  border-bottom: 1px solid #e4e7ed;
}

.right-panel {
  margin-left: 200px;
  flex: 1;
  display: flex;
  flex-direction: column;
  height: 100vh;
  overflow: hidden;
}

.app-header {
  height: 50px;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
  background: #fff;
  border-bottom: 1px solid #e4e7ed;
}

.header-title {
  font-size: 16px;
  font-weight: 600;
  color: #303133;
}

.header-time {
  font-size: 13px;
  color: #909399;
  font-family: monospace;
}

.main-content {
  flex: 1;
  padding: 20px 24px;
  overflow-y: auto;
  background: #f5f7fa;
}

.app-footer {
  flex-shrink: 0;
  height: 40px;
  background: #fff;
  border-top: 1px solid #e4e7ed;
  display: flex;
  align-items: center;
  padding: 0 24px;
}

.status-bar {
  display: flex;
  align-items: center;
  gap: 24px;
  width: 100%;
}

.status-item {
  display: flex;
  align-items: center;
  gap: 6px;
}

.status-label {
  font-size: 12px;
  color: #606266;
  font-weight: 500;
  white-space: nowrap;
}

.status-value {
  font-size: 11px;
  color: #909399;
  font-family: monospace;
  white-space: nowrap;
}
</style>
