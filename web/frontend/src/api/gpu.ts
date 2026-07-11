import api from './index'

export interface GpuProcess {
  id: number
  name: string
  status: string
  pid: number | null
}

export interface GpuInfo {
  index: number
  name: string
  utilization_percent: number
  memory_used_mb: number
  memory_total_mb: number
  temperature_c: number
  power_usage_w: number
  processes: GpuProcess[]
}

export interface GpuStatus {
  gpus: GpuInfo[]
  updated_at: string
}

export interface SystemStatus {
  cpu_percent: number
  memory_used_gb: number
  memory_total_gb: number
  memory_percent: number
  gpu_utilization: number
  gpu_memory_percent: number
  load_1m: number
  load_5m: number
  load_15m: number
  updated_at: string
}

export const getGpuStatus = () =>
  api.get<GpuStatus>('/gpu/status').then(r => r.data)

export const getSystemStatus = () =>
  api.get<SystemStatus>('/gpu/system').then(r => r.data)
