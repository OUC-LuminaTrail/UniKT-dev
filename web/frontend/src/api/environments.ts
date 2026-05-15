import api from './index'

export interface EnvironmentInfo {
  id: string
  type: string
  name: string
  display_name: string
  python_path: string | null
}

export interface EnvHealthResult {
  env_id: string
  python_available: boolean
  python_version: string | null
  torch_available: boolean
  torch_version: string | null
  error: string | null
}

export interface EnvHealthCheckRequest {
  env_id: string
  custom_python_path?: string | null
}

export const listEnvironments = () =>
  api.get<EnvironmentInfo[]>('/environments').then(r => r.data)

export const healthCheckEnv = (data: EnvHealthCheckRequest) =>
  api.post<EnvHealthResult>('/environments/health-check', data).then(r => r.data)
