import api from './index'

export interface DatasetInfo {
  name: string
  num_users: number | null
  num_questions: number | null
  num_skills: number | null
}

export interface DatasetMetadata {
  kfold?: number
  [key: string]: any
}

export const listDatasets = () =>
  api.get<DatasetInfo[]>('/datasets').then(r => r.data)

export const getDatasetMetadata = (name: string) =>
  api.get<DatasetMetadata>(`/datasets/${name}/metadata`).then(r => r.data)