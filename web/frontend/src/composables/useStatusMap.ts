// Shared task status → {color, label} map. The task and search list/detail views
// all render the same status badge, so the mapping lives here once.
export const statusMap: Record<string, { color: string; label: string }> = {
  running: { color: 'var(--accent-blue)', label: 'common.status.running' },
  completed: { color: 'var(--accent-green)', label: 'common.status.completed' },
  failed: { color: 'var(--accent-red)', label: 'common.status.failed' },
  stopped: { color: 'var(--accent-orange)', label: 'common.status.stopped' },
  stopping: { color: 'var(--accent-orange)', label: 'common.status.stopping' },
  pending: { color: 'var(--text-tertiary)', label: 'common.status.pending' },
}
