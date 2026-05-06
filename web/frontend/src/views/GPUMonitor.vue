<template>
  <div class="gpu-monitor">
    <div class="page-header">
      <h2 class="page-title">GPU 监控</h2>
      <div class="live-badge">
        <span class="live-dot"></span>
        <span class="live-text">实时</span>
      </div>
    </div>

    <SkeletonCards v-if="loading" :count="2" :cardWidth="400" />
    <div class="gpu-grid" v-else-if="status && status.gpus.length > 0">
      <div class="gpu-card" v-for="gpu in status.gpus" :key="gpu.index">
        <div class="gpu-card-header">
          <span class="gpu-index">GPU {{ gpu.index }}</span>
          <span class="gpu-name">{{ gpu.name }}</span>
        </div>

        <div class="stats-grid">
          <div class="stat-block">
            <div class="stat-label">利用率</div>
            <div class="progress-track">
              <div
                class="progress-fill"
                :style="{
                  width: gpu.utilization_percent + '%',
                  background: progressColor(gpu.utilization_percent),
                }"
              ></div>
            </div>
            <div class="stat-value">{{ gpu.utilization_percent.toFixed(0) }}%</div>
          </div>

          <div class="stat-block">
            <div class="stat-label">显存</div>
            <div class="progress-track">
              <div
                class="progress-fill"
                :style="{
                  width: (gpu.memory_total_mb > 0 ? (gpu.memory_used_mb / gpu.memory_total_mb * 100) : 0) + '%',
                  background: progressColor(gpu.memory_used_mb / (gpu.memory_total_mb || 1) * 100),
                }"
              ></div>
            </div>
            <div class="stat-value">
              {{ gpu.memory_used_mb.toFixed(0) }} /
              {{ gpu.memory_total_mb.toFixed(0) }} MB
            </div>
          </div>

          <div class="stat-block">
            <div class="stat-label">温度</div>
            <div class="stat-row">
              <span
                class="temp-indicator"
                :style="{ background: tempColor(gpu.temperature_c) }"
              ></span>
              <span class="stat-value">{{ gpu.temperature_c }}°C</span>
            </div>
          </div>

          <div class="stat-block">
            <div class="stat-label">功耗</div>
            <div class="stat-value">{{ gpu.power_usage_w.toFixed(1) }}W</div>
          </div>
        </div>
      </div>
    </div>

    <div class="empty-state" v-else>
      <div class="empty-icon">⬡</div>
      <div class="empty-text">未检测到 GPU</div>
      <div class="empty-sub">请确保已安装 NVIDIA 驱动和 nvidia-smi</div>
    </div>

    <div class="updated-at" v-if="status">
      更新时间: {{ status.updated_at }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { getGpuStatus, type GpuStatus } from '@/api/gpu'
import SkeletonCards from '@/components/common/SkeletonCards.vue'

const status = ref<GpuStatus | null>(null)
const loading = ref(true)
let timer: ReturnType<typeof setInterval> | null = null

const progressColor = (pct: number) => {
  if (pct > 90) return 'var(--accent-red)'
  if (pct > 70) return 'var(--accent-orange)'
  return 'var(--accent-green)'
}

const tempColor = (temp: number) => {
  if (temp > 85) return 'var(--accent-red)'
  if (temp > 70) return 'var(--accent-orange)'
  return 'var(--accent-green)'
}

const loadStatus = async () => {
  try { status.value = await getGpuStatus() } catch {}
  loading.value = false
}

onMounted(() => { loadStatus(); timer = setInterval(loadStatus, 3000) })
onUnmounted(() => { if (timer) clearInterval(timer) })
</script>

<style scoped>
.gpu-monitor {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.page-title {
  font-size: 20px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.live-badge {
  display: flex;
  align-items: center;
  gap: 6px;
}

.live-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  background: var(--accent-green);
  animation: pulse 2s ease-in-out infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.4; }
}

.live-text {
  font-size: 12px;
  font-weight: 600;
  color: var(--accent-green);
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.gpu-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}

@media (max-width: 900px) {
  .gpu-grid {
    grid-template-columns: 1fr;
  }
}

.gpu-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  padding: 20px;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.gpu-card-header {
  display: flex;
  align-items: center;
  gap: 10px;
}

.gpu-index {
  background: var(--bg-elevated);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-sm);
  padding: 2px 8px;
  font-family: var(--font-mono);
  font-size: 11px;
  font-weight: 600;
  color: var(--accent-cyan);
  white-space: nowrap;
}

.gpu-name {
  font-size: 14px;
  font-weight: 500;
  color: var(--text-primary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

.stats-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 14px 20px;
}

.stat-block {
  display: flex;
  flex-direction: column;
  gap: 6px;
}

.stat-label {
  font-size: 11px;
  font-weight: 500;
  color: var(--text-tertiary);
  letter-spacing: 0.4px;
}

.stat-value {
  font-family: var(--font-mono);
  font-size: 13px;
  color: var(--text-primary);
}

.stat-row {
  display: flex;
  align-items: center;
  gap: 8px;
}

.temp-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}

.progress-track {
  width: 100%;
  height: 4px;
  background: var(--bg-elevated);
  border-radius: 2px;
  overflow: hidden;
}

.progress-fill {
  height: 100%;
  border-radius: 2px;
  transition: width 0.6s ease, background 0.4s ease;
}

.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 60px 20px;
  background: var(--bg-surface);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  gap: 8px;
}

.empty-icon {
  font-size: 36px;
  color: var(--text-tertiary);
  line-height: 1;
}

.empty-text {
  font-size: 15px;
  font-weight: 500;
  color: var(--text-secondary);
}

.empty-sub {
  font-size: 12px;
  color: var(--text-tertiary);
}

.updated-at {
  font-family: var(--font-mono);
  font-size: 11px;
  color: var(--text-tertiary);
  text-align: right;
}
</style>
