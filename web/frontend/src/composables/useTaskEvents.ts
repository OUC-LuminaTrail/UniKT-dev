import { onUnmounted } from 'vue'
import { useQueryClient } from '@tanstack/vue-query'
import type { TaskInfo } from '@/api/tasks'

type TaskStatusEvent = { type: 'task_status'; id: number; status: string; pid?: number | null }
type PreprocessStatusEvent = { type: 'preprocess_status'; id: number; status: string }
type StatusEvent = TaskStatusEvent | PreprocessStatusEvent

export function useTaskEvents() {
  const queryClient = useQueryClient()

  const handle = (event: StatusEvent) => {
    if (event.type === 'task_status') {
      queryClient.setQueryData<TaskInfo>(['task', event.id], (old) =>
        old ? { ...old, status: event.status, pid: event.pid ?? old.pid } : old,
      )
      queryClient.invalidateQueries({ queryKey: ['tasks-list'] })
    } else {
      queryClient.invalidateQueries({ queryKey: ['preprocess-active-tasks'] })
      queryClient.invalidateQueries({ queryKey: ['preprocess-task', event.id] })
    }
  }

  const es = new EventSource('/api/events')
  es.onmessage = (e) => {
    try {
      handle(JSON.parse(e.data) as StatusEvent)
    } catch {
      /* ignore malformed frames */
    }
  }

  onUnmounted(() => es.close())
}
