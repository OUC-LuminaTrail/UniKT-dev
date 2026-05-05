<template>
  <div class="gpu-monitor">
    <h2>GPU 监控</h2>

    <el-row :gutter="16" v-if="status && status.gpus.length > 0">
      <el-col :span="12" v-for="gpu in status.gpus" :key="gpu.index">
        <el-card style="margin-bottom: 16px">
          <template #header>
            <span>GPU {{ gpu.index }}: {{ gpu.name }}</span>
          </template>
          <el-descriptions :column="1">
            <el-descriptions-item label="使用率">
              <el-progress
                :percentage="gpu.utilization_percent"
                :color="progressColor(gpu.utilization_percent)"
              />
            </el-descriptions-item>
            <el-descriptions-item label="显存">
              <el-progress
                :percentage="gpu.memory_total_mb > 0 ? (gpu.memory_used_mb / gpu.memory_total_mb * 100) : 0"
                :color="progressColor(gpu.memory_used_mb / (gpu.memory_total_mb || 1) * 100)"
                :format="() => `${gpu.memory_used_mb.toFixed(0)} / ${gpu.memory_total_mb.toFixed(0)} MB`"
              />
            </el-descriptions-item>
            <el-descriptions-item label="温度">{{ gpu.temperature_c }}°C</el-descriptions-item>
            <el-descriptions-item label="功耗">{{ gpu.power_usage_w.toFixed(1) }}W</el-descriptions-item>
          </el-descriptions>
        </el-card>
      </el-col>
    </el-row>

    <el-empty v-else description="暂无 GPU 信息" />

    <div class="updated-at" v-if="status">
      最后更新: {{ status.updated_at }}
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted } from 'vue'
import { getGpuStatus, type GpuStatus } from '@/api/gpu'

const status = ref<GpuStatus | null>(null)
let timer: ReturnType<typeof setInterval> | null = null

const progressColor = (pct: number) => {
  if (pct > 90) return '#f56c6c'
  if (pct > 70) return '#e6a23c'
  return '#67c23a'
}

const loadStatus = async () => {
  try {
    status.value = await getGpuStatus()
  } catch {}
}

onMounted(() => {
  loadStatus()
  timer = setInterval(loadStatus, 3000)
})

onUnmounted(() => {
  if (timer) clearInterval(timer)
})
</script>

<style scoped>
.updated-at {
  font-size: 12px;
  color: #909399;
  margin-top: 8px;
}
</style>
