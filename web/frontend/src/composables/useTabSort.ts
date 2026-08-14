import { computed, ref, watch, type Ref } from 'vue'

type SortOrder = 'ascending' | 'descending'
interface SortState {
  prop: string
  order: SortOrder
}

// Per-tab default sort policy shared by the task and search lists:
// terminal tabs default to newest-finished-first, the active tab keeps id order.
// el-table's default-sort only applies at mount, so views remount the table
// (:key="activeTab") and overrides are tracked per tab here.
export function useTabSort(defaults: Record<string, SortState>, activeTab: Ref<string>) {
  const tabSortOverrides = ref<Record<string, SortState>>({})

  const currentSort = computed<SortState>(
    () => tabSortOverrides.value[activeTab.value] ?? defaults[activeTab.value] ?? defaults.all,
  )

  const onSortChange = ({ prop, order }: { prop: string | null; order: SortOrder | null }) => {
    if (prop && order) {
      tabSortOverrides.value[activeTab.value] = { prop, order }
    } else {
      delete tabSortOverrides.value[activeTab.value]
    }
  }

  return { currentSort, onSortChange }
}
