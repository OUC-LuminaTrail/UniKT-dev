// GPU display helper shared by task/search list and detail views.
// `unassignedText` differs per context (list shows "Auto", detail shows "—" for
// non-pending tasks), so callers pass their own fallback string.
export function formatGpu(val: number | null | undefined, unassignedText: string): string {
  if (val === null || val === undefined) return unassignedText
  return `GPU ${val}`
}
