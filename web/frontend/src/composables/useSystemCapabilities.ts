import { ref } from 'vue'
import api from '@/api/index'

interface Capabilities {
  has_gpu: boolean
  gpu_count: number
  gpu_names: string[]
}

const hasGpu = ref(true)
const gpuCount = ref(0)
let loaded = false

export function useSystemCapabilities() {
  if (!loaded) {
    loaded = true
    api.get<Capabilities>('/system/capabilities').then(r => {
      hasGpu.value = r.data.has_gpu
      gpuCount.value = r.data.gpu_count
    }).catch(() => {
      hasGpu.value = false
    })
  }
  return { hasGpu, gpuCount }
}
