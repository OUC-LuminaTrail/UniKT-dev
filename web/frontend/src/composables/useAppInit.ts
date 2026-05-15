import { ref } from 'vue'
import { getInitStatus } from '@/api/settings'

const isInitialized = ref<boolean | null>(null)

export function useAppInit() {
  const checkInit = async () => {
    try {
      const res = await getInitStatus()
      isInitialized.value = res.initialized
    } catch {
      isInitialized.value = null
    }
  }

  if (isInitialized.value === null) {
    checkInit()
  }

  return { isInitialized, checkInit }
}
