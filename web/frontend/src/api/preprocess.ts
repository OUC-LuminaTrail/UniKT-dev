import api from './index'

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
}

export const listPreprocess = () =>
  api.get<PreprocessTaskInfo[]>('/preprocess').then(r => r.data)

export const startPreprocess = (data: PreprocessStartRequest) =>
  api.post<PreprocessTaskInfo>('/preprocess', data).then(r => r.data)

export const getPreprocess = (id: number) =>
  api.get<PreprocessTaskInfo>(`/preprocess/${id}`).then(r => r.data)

export const stopPreprocess = (id: number) =>
  api.post(`/preprocess/${id}/stop`).then(r => r.data)
