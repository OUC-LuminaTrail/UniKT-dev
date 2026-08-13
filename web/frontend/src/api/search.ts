import api from './index'
import type { TaskInfo, PaginatedTasks } from './tasks'

// Search tasks reuse the tasks table, so a search task IS a TaskInfo.
export type SearchTaskInfo = TaskInfo

export interface SearchCreateRequest {
  name?: string
  env_id: string
  custom_python_path?: string | null
  model_name: string
  dataset: string
  gpu?: number | null
  runconfig_params: Record<string, any>
  optuna_config: Record<string, any>
}

export const createSearch = (data: SearchCreateRequest) =>
  api.post<TaskInfo>('/search', data).then(r => r.data)

export const listSearches = (params?: {
  status?: string
  page?: number
  page_size?: number
}) => api.get<PaginatedTasks>('/search', { params }).then(r => r.data)

export const getSearch = (id: number) =>
  api.get<TaskInfo>(`/search/${id}`).then(r => r.data)

export const stopSearch = (id: number) =>
  api.post(`/search/${id}/stop`).then(r => r.data)

export const killSearch = (id: number) =>
  api.post(`/search/${id}/kill`).then(r => r.data)

export const deleteSearch = (id: number) =>
  api.delete(`/search/${id}`).then(r => r.data)

export interface SearchPreviewRequest {
  model_name: string
  dataset: string
  runconfig_params: Record<string, any>
  optuna_config: Record<string, any>
}

export const previewSearchCommand = (data: SearchPreviewRequest) =>
  api.post<{ command: string }>('/search/preview-command', data).then(r => r.data.command)

export interface SearchTrial {
  number: number
  state: string
  value: number | null
  params: Record<string, any>
  datetime_start: string | null
  datetime_complete: string | null
}

export interface SearchStudy {
  total: number
  completed: number
  running: number
  pruned: number
  failed: number
  direction: string
  best_trial: { number: number; value: number; params: Record<string, any> } | null
  trials: SearchTrial[]
}

export const getSearchTrials = (id: number) =>
  api.get<SearchStudy>(`/search/${id}/trials`).then(r => r.data)

export interface SearchStudyPath {
  study_db_path: string | null
  dashboard_command: string | null
  exists: boolean
}

export const getSearchStudyDb = (id: number) =>
  api.get<SearchStudyPath>(`/search/${id}/study-db`).then(r => r.data)
