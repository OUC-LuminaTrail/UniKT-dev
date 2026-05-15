import api from './index'

export interface ParamField {
  type: string
  default: any
  help: string
  required: boolean
  choices: string[] | null
  short: string | null
  nargs: string | null
}

export interface ParamGroup {
  group_name: string
  params: Record<string, ParamField>
}

export interface ModelSchema {
  model_name: string
  param_groups: ParamGroup[]
}

export const listModels = () =>
  api.get<string[]>('/schemas/models').then(r => r.data)

export const getModelParams = (model: string) =>
  api.get<ModelSchema>(`/schemas/models/${model}/params`).then(r => r.data)
