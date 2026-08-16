import api from './index'

export interface ResourceSnapshot {
  cpu_percent: number
  cpu_cores: number[]
  load_1m: number
  load_5m: number
  load_15m: number
  memory_used_gb: number
  memory_total_gb: number
  memory_percent: number
  swap_used_gb: number
  swap_total_gb: number
  swap_percent: number
}

export interface GpuHistorySeries {
  index: number
  name: string
  utilization_percent: (number | null)[]
  memory_percent: (number | null)[]
}

export interface ResourceHistory {
  timestamps: number[]
  cpu_percent: number[]
  memory_percent: number[]
  swap_percent: number[]
  net_up_bps: number[]
  net_down_bps: number[]
  disk_read_bps: number[]
  disk_write_bps: number[]
  gpus: GpuHistorySeries[]
  snapshot: ResourceSnapshot
  interval_seconds: number
}

export const getResourceHistory = (since?: number) =>
  api.get<ResourceHistory>('/resource/history', { params: since !== undefined ? { since } : {} })
    .then(r => r.data)
