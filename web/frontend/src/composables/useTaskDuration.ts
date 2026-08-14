import { computed, onUnmounted, ref, watch, type Ref } from 'vue'

interface DurationTask {
  started_at?: string | null
  finished_at?: string | null
  status?: string | null
}

// Live run-duration clock shared by TaskDetail / SearchDetail.
// Ticks only while the task is active; stops at a terminal state so a fixed
// finished_at doesn't trigger a per-second recompute forever.
export function useTaskDuration(task: Ref<DurationTask | null | undefined>) {
  const now = ref(Date.now())
  let clockTimer: ReturnType<typeof setInterval> | null = null

  const isActive = (status?: string | null) =>
    status === 'running' || status === 'pending' || status === 'stopping'

  watch(
    () => task.value?.status,
    (status) => {
      if (isActive(status) && !clockTimer) {
        clockTimer = setInterval(() => (now.value = Date.now()), 1000)
      } else if (!isActive(status) && clockTimer) {
        clearInterval(clockTimer)
        clockTimer = null
      }
    },
    { immediate: true },
  )

  onUnmounted(() => {
    if (clockTimer) clearInterval(clockTimer)
  })

  const duration = computed(() => {
    const t = task.value
    if (!t?.started_at) return ''
    const start = new Date(t.started_at).getTime()
    if (isNaN(start)) return ''
    const end = t.finished_at ? new Date(t.finished_at).getTime() : now.value
    if (isNaN(end)) return ''
    const s = Math.max(0, Math.floor((end - start) / 1000))
    const h = Math.floor(s / 3600)
    const m = Math.floor((s % 3600) / 60)
    const sec = s % 60
    if (h > 0) return `${h}h ${m}m ${sec}s`
    if (m > 0) return `${m}m ${sec}s`
    return `${sec}s`
  })

  return { duration }
}
