import api from './index'

export interface TaskInfo {
  id: number
  name: string
  command: string
  model_name: string
  dataset_name: string
  env_type: string
  env_name: string
  python_path: string | null
  status: string
  pid: number | null
  exp_dir: string
  started_at: string | null
  finished_at: string | null
  exit_code: number | null
  created_at: string
  tags: string
  extra_params: string
}

export interface TaskCreateRequest {
  name: string
  env_id: string
  custom_python_path?: string | null
  model_name: string
  params: Record<string, any>
}

export const createTask = (data: TaskCreateRequest) =>
  api.post<TaskInfo>('/tasks', data).then(r => r.data)

export interface PaginatedTasks {
  items: TaskInfo[]
  total: number
  page: number
  page_size: number
}

export const listTasks = (params?: { status?: string; page?: number; page_size?: number }) =>
  api.get<PaginatedTasks>('/tasks', { params }).then(r => r.data)

export const getTask = (id: number) =>
  api.get<TaskInfo>(`/tasks/${id}`).then(r => r.data)

export const stopTask = (id: number) =>
  api.post(`/tasks/${id}/stop`).then(r => r.data)

export const killTask = (id: number) =>
  api.post(`/tasks/${id}/kill`).then(r => r.data)

export const deleteTask = (id: number) =>
  api.delete(`/tasks/${id}`).then(r => r.data)

export const getTaskLogs = (id: number, offset = 0, limit = 10000) =>
  api.get(`/tasks/${id}/logs`, { params: { offset, limit } }).then(r => r.data)

export const resizeTerminal = (id: number, cols: number, rows: number) =>
  api.post(`/tasks/${id}/resize`, { cols, rows }).then(r => r.data)
