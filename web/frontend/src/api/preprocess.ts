import api from './index'
import type { ParamGroup } from './schemas'

export interface PreprocessTaskInfo {
  id: number
  command: string
  status: string
  exit_code: number | null
  started_at: string | null
  finished_at: string | null
}

export interface PreprocessStartRequest {
  action: string
  dataset: string
  params: Record<string, any>
  env_id?: string | null
  custom_python_path?: string | null
}

export const listPreprocess = () =>
  api.get<PreprocessTaskInfo[]>('/preprocess').then(r => r.data)

export const startPreprocess = (data: PreprocessStartRequest) =>
  api.post<PreprocessTaskInfo>('/preprocess', data).then(r => r.data)

export const getPreprocess = (id: number) =>
  api.get<PreprocessTaskInfo>(`/preprocess/${id}`).then(r => r.data)

export const stopPreprocess = (id: number) =>
  api.post(`/preprocess/${id}/stop`).then(r => r.data)

export const getPreprocessSchema = (action: string) =>
  api.get<ParamGroup[]>(`/schemas/preprocess/${action}`).then(r => r.data)

export const previewPreprocess = (data: PreprocessStartRequest) =>
  api.post<{ command: string }>('/preprocess/preview', data).then(r => r.data.command)
