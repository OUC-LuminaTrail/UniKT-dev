import api from './index'

export interface EnvironmentInfo {
  id: string
  type: string
  name: string
  display_name: string
  python_path: string | null
}

export const listEnvironments = () =>
  api.get<EnvironmentInfo[]>('/environments').then(r => r.data)
