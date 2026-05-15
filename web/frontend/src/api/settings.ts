import api from './index'

export interface AppSettings {
  max_concurrent: number
}

export const getSettings = () =>
  api.get<AppSettings>('/settings').then(r => r.data)

export const updateSettings = (data: Partial<AppSettings>) =>
  api.put<AppSettings>('/settings', data).then(r => r.data)

export interface QueueItem {
  id: number
  name: string
  model_name: string
  dataset_name: string
  env_name: string
  status: string
  created_at: string | null
}

export const getQueue = () =>
  api.get<QueueItem[]>('/tasks/queue/list').then(r => r.data)

export const reorderQueue = (task_ids: number[]) =>
  api.put('/tasks/queue/reorder', { task_ids }).then(r => r.data)

export interface InitStatus {
  initialized: boolean
}

export interface DefaultEnv {
  default_env_id: string | null
  custom_python_path: string | null
}

export interface DefaultEnvUpdate {
  env_id: string
  custom_python_path?: string | null
}

export const getInitStatus = () =>
  api.get<InitStatus>('/settings/initialized').then(r => r.data)

export const getDefaultEnv = () =>
  api.get<DefaultEnv>('/settings/default-env').then(r => r.data)

export const setDefaultEnv = (data: DefaultEnvUpdate) =>
  api.post<DefaultEnvUpdate & { ok: boolean }>('/settings/default-env', data).then(r => r.data)
