import api from './index'

export interface ExperimentInfo {
  name: string
  path: string
  model_name: string | null
  dataset_name: string | null
  timestamp: string | null
  type: string
}

export interface ExperimentDetail {
  name: string
  path: string
  files: string[]
  hyperparams: Record<string, any> | null
}

export const listExperiments = (params?: { type?: string; model?: string; dataset?: string }) =>
  api.get<ExperimentInfo[]>('/experiments', { params }).then(r => r.data)

export const getExperiment = (path: string) =>
  api.get<ExperimentDetail>(`/experiments/${path}`).then(r => r.data)

export const readExperimentFile = (path: string, fileName: string) =>
  api.get(`/experiments/${path}/files/${fileName}`).then(r => r.data)
