<template>
  <div class="gpu-monitor">
    <div class="page-header">
      <div class="page-title-group">
        <h2 class="page-title">GPU 监控</h2>
        <span class="page-sub">实时利用率与显存占用趋势 · 最近 3 分钟</span>
      </div>
      <div class="live-badge">
        <span class="live-dot"></span>
        <span class="live-text">实时 · 3s</span>
      </div>
    </div>

    <el-skeleton :loading="loading" animated>
      <template #template>
        <div class="gpu-grid">
          <div class="gpu-card" v-for="i in 2" :key="i">
            <div class="gpu-card-header">
              <el-skeleton-item variant="text" style="width:50px;height:18px" />
              <el-skeleton-item variant="text" style="width:120px;height:14px" />
            </div>
            <div class="stats-grid">
              <div class="stat-block" v-for="j in 4" :key="j">
                <el-skeleton-item variant="text" style="width:40px;height:10px" />
                <el-skeleton-item variant="text" style="width:70%;height:12px;margin-top:4px" />
              </div>
            </div>
          </div>
        </div>
      </template>
      <template #default>
        <div class="gpu-grid" v-if="status && status.gpus.length > 0">
          <div class="gpu-card" v-for="gpu in status.gpus" :key="gpu.index">
            <div class="gpu-card-header">
              <span class="gpu-index">GPU {{ gpu.index }}</span>
              <span class="gpu-name" :title="gpu.name">{{ gpu.name }}</span>
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

            <div class="gpu-trend">
              <VChart
                v-if="trendReady(gpu.index)"
                class="trend-chart"
                :option="buildOption(gpu.index)"
                :autoresize="true"
                :update-options="{ notMerge: true }"
              />
              <div v-else class="trend-placeholder">趋势采集中…</div>
            </div>

            <div class="occupancy">
              <div class="occupancy-label">占用任务</div>
              <div v-if="gpu.processes.length === 0" class="occupancy-empty">空闲</div>
              <div v-else class="occupancy-list">
                <div v-for="p in gpu.processes" :key="p.id" class="occ-item">
                  <span class="occ-dot" :class="`occ-${p.status}`"></span>
                  <span class="occ-name" :title="p.name">{{ p.name }}</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="empty-state" v-else>
          <el-icon :size="40" class="empty-icon"><Monitor /></el-icon>
          <div class="empty-text">未检测到 GPU</div>
          <div class="empty-sub">请确保已安装 NVIDIA 驱动与 nvidia-smi</div>
        </div>
      </template>
    </el-skeleton>

    <div class="updated-at" v-if="status">
      更新于 {{ formatDateTime(status.updated_at) }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, reactive, watch } from 'vue'
import { useDark } from '@vueuse/core'
import { useQuery } from '@tanstack/vue-query'
import { Monitor } from '@element-plus/icons-vue'
import { VChart } from '@/plugins/echarts'
import { getGpuStatus } from '@/api/gpu'
import { formatDateTime } from '@/utils/date'

const { data: status, isPending: loading } = useQuery({
  queryKey: ['gpu-status'],
  queryFn: getGpuStatus,
  refetchInterval: 3000,
})

const HISTORY = 60
interface GpuHistory { util: number[]; vram: number[] }
const history = reactive<Record<number, GpuHistory>>({})

watch(status, (s) => {
  if (!s) return
  for (const gpu of s.gpus) {
    if (!history[gpu.index]) history[gpu.index] = { util: [], vram: [] }
    const h = history[gpu.index]
    const vramPct = gpu.memory_total_mb > 0 ? (gpu.memory_used_mb / gpu.memory_total_mb) * 100 : 0
    h.util.push(gpu.utilization_percent)
    h.vram.push(vramPct)
    while (h.util.length > HISTORY) h.util.shift()
    while (h.vram.length > HISTORY) h.vram.shift()
  }
}, { immediate: true })

const isDark = useDark()
// Cache CSS tokens per theme; canvas can't resolve CSS variables, so re-read on theme switch.
const tokens = computed(() => {
  void isDark.value
  const v = (n: string) => getComputedStyle(document.documentElement).getPropertyValue(n).trim()
  return {
    blue: v('--accent-blue'),
    cyan: v('--accent-cyan'),
    textTertiary: v('--text-tertiary'),
    bgElevated: v('--bg-elevated'),
    borderDefault: v('--border-default'),
    textPrimary: v('--text-primary'),
  }
})

const trendReady = (idx: number) => (history[idx]?.util.length ?? 0) > 1

const buildOption = (gpuIndex: number) => {
  const h = history[gpuIndex] ?? { util: [], vram: [] }
  const t = tokens.value
  return {
    animation: false,
    grid: { left: 0, right: 0, top: 16, bottom: 0 },
    xAxis: { type: 'category', show: false, boundaryGap: false },
    yAxis: { type: 'value', show: false, min: 0, max: 100 },
    legend: {
      show: true, right: 0, top: -2,
      itemWidth: 10, itemHeight: 6, itemGap: 10, icon: 'roundRect',
      textStyle: { color: t.textTertiary, fontSize: 10 },
      data: ['利用率', '显存'],
    },
    tooltip: {
      trigger: 'axis', confine: true,
      backgroundColor: t.bgElevated,
      borderColor: t.borderDefault, borderWidth: 1,
      textStyle: { color: t.textPrimary, fontSize: 11 },
      formatter: (params: { marker: string; seriesName: string; value: number }[]) =>
        params.map((p) => `${p.marker}${p.seriesName} ${Number(p.value).toFixed(0)}%`).join('<br/>'),
    },
    series: [
      {
        name: '利用率', type: 'line', data: h.util, smooth: 0.3, symbol: 'none',
        lineStyle: { width: 1.5, color: t.blue },
        areaStyle: { opacity: 0.14, color: t.blue },
      },
      {
        name: '显存', type: 'line', data: h.vram, smooth: 0.3, symbol: 'none',
        lineStyle: { width: 1.5, color: t.cyan },
      },
    ],
  }
}

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
</script>

<style scoped>
.gpu-monitor {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.page-header {
  display: flex;
  align-items: flex-end;
  justify-content: space-between;
  gap: 12px;
}

.page-title-group {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.page-title {
  font-size: 20px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.page-sub {
  font-size: 12px;
  color: var(--text-tertiary);
}

.live-badge {
  display: flex;
  align-items: center;
  gap: 6px;
  flex-shrink: 0;
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
  font-size: 11px;
  font-weight: 600;
  color: var(--accent-green);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  font-family: var(--font-mono);
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

.gpu-trend {
  height: 64px;
}

.trend-chart {
  width: 100%;
  height: 64px;
}

.trend-placeholder {
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 11px;
  color: var(--text-tertiary);
  font-family: var(--font-mono);
  background: var(--bg-overlay);
  border-radius: var(--radius-sm);
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
  color: var(--text-tertiary);
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

.occupancy {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding-top: 12px;
  border-top: 1px solid var(--border-muted);
}

.occupancy-label {
  font-size: 11px;
  font-weight: 500;
  color: var(--text-tertiary);
  letter-spacing: 0.4px;
}

.occupancy-empty {
  font-size: 12px;
  color: var(--text-tertiary);
  font-family: var(--font-mono);
}

.occupancy-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.occ-item {
  display: flex;
  align-items: center;
  gap: 8px;
}

.occ-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
  background: var(--accent-blue);
}

.occ-dot.occ-running {
  background: var(--accent-blue);
  box-shadow: 0 0 5px var(--accent-blue);
}

.occ-dot.occ-stopping {
  background: var(--accent-orange);
}

.occ-dot.occ-interrupted {
  background: var(--text-tertiary);
}

.occ-name {
  font-size: 12px;
  font-family: var(--font-mono);
  color: var(--text-secondary);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

@media (prefers-reduced-motion: reduce) {
  .live-dot { animation: none; }
  .progress-fill { transition: none; }
}
</style>
