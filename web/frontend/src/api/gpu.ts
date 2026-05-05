import api from './index'

export interface GpuInfo {
  index: number
  name: string
  utilization_percent: number
  memory_used_mb: number
  memory_total_mb: number
  temperature_c: number
  power_usage_w: number
  processes: any[]
}

export interface GpuStatus {
  gpus: GpuInfo[]
  updated_at: string
}

export const getGpuStatus = () =>
  api.get<GpuStatus>('/gpu/status').then(r => r.data)
