<template>
  <el-skeleton :loading="loading && !task" animated>
    <template #template>
      <div class="search-detail">
        <header class="detail-header">
          <el-skeleton-item variant="rect" style="width:32px;height:32px;border-radius:var(--radius-sm);flex-shrink:0" />
          <el-skeleton-item variant="text" style="width:180px;height:20px" />
          <el-skeleton-item variant="rect" style="width:72px;height:24px;border-radius:20px" />
        </header>

        <div class="detail-body">
          <div class="info-col">
            <section class="detail-section">
              <div class="section-head">
                <el-skeleton-item variant="text" style="width:60px;height:11px" />
                <span class="section-rule"></span>
              </div>
              <div class="sk-meta-grid">
                <div v-for="i in 6" :key="i" class="sk-meta-cell">
                  <el-skeleton-item variant="text" style="width:40px;height:10px" />
                  <el-skeleton-item variant="text" style="width:70%;height:13px;margin-top:2px" />
                </div>
              </div>
            </section>
            <section class="detail-section">
              <div class="section-head">
                <el-skeleton-item variant="text" style="width:80px;height:11px" />
                <span class="section-rule"></span>
              </div>
              <el-skeleton-item variant="text" style="width:100%;height:8px;border-radius:99px" />
              <el-skeleton-item variant="rect" style="width:100%;height:64px;border-radius:var(--radius-lg);margin-top:4px" />
            </section>
          </div>
          <div class="log-col">
            <el-skeleton-item variant="rect" style="width:100%;height:100%" />
          </div>
        </div>
      </div>
    </template>
    <template #default>
      <div class="search-detail" v-if="task" :class="{ 'is-dragging': dragging }">
        <DetailHeader
          class="detail-header"
          :name="task.name"
          :status="task.status"
          :stop-label="t('search.stopTask')"
          :stopping="stopping"
          :killing="killing"
          fallback-route="searches"
          @stop="handleStop"
          @kill="handleKill"
        />

        <div ref="bodyEl" class="detail-body">
          <!-- Left: scrollable info column (user-resizable via the splitter) -->
          <div class="info-col" :style="{ width: splitPct + '%' }">
            <DetailSection :title="t('task.detail.sectionMeta')">
              <MetaGrid :cells="metaCells" />
            </DetailSection>

            <!-- Trial progress from study.db (live while running). -->
            <DetailSection v-if="study" :title="t('search.sectionProgress')">
              <!-- Segmented progress bar: completed / pruned / failed / running -->
              <div class="seg-bar" role="progressbar"
                :aria-label="t('search.sectionProgress')"
                :aria-valuenow="doneCount" :aria-valuemin="0" :aria-valuemax="study.total"
                :aria-valuetext="`${doneCount}/${study.total}`">
                <span class="seg s-ok" :style="{ flexGrow: study.completed || 0 }" />
                <span class="seg s-pruned" :style="{ flexGrow: study.pruned || 0 }" />
                <span class="seg s-fail" :style="{ flexGrow: study.failed || 0 }" />
                <span class="seg s-run" :style="{ flexGrow: study.running || 0 }" />
              </div>
              <div class="seg-legend">
                <span class="lg"><i class="dot d-ok" />{{ t('common.status.completed') }} <b>{{ study.completed }}</b></span>
                <span class="lg"><i class="dot d-pruned" />{{ t('search.pruned') }} <b>{{ study.pruned }}</b></span>
                <span class="lg"><i class="dot d-fail" />{{ t('search.failedTrials') }} <b>{{ study.failed }}</b></span>
                <span class="lg"><i class="dot d-run" />{{ t('common.status.running') }} <b>{{ study.running }}</b></span>
                <span class="lg lg-total">{{ t('search.trialTotal') }} <b>{{ study.total }}</b></span>
              </div>

              <div class="best-card" v-if="study.best_trial">
                <div class="best-head">
                  <span class="best-badge">{{ t('search.bestTrial') }}</span>
                  <span class="best-num">#{{ study.best_trial.number }}</span>
                  <span class="best-value">
                    <span class="best-metric" :title="dirTooltip">{{ metricLabel }} {{ dirArrow }}</span>
                    <b>{{ fmtValue(study.best_trial.value) }}</b>
                  </span>
                </div>
                <div class="best-params">
                  <span v-for="(v, k) in study.best_trial.params" :key="k" class="param-pill">
                    <span class="pk">{{ k }}</span>=<span class="pv">{{ fmtParam(v) }}</span>
                  </span>
                </div>
              </div>

              <el-table
                :data="sortedTrials"
                size="small"
                class="trial-table"
                :empty-text="t('search.emptyTrials')"
                max-height="420"
                :row-class-name="trialRowClass"
                @sort-change="onTrialSort"
              >
                <el-table-column prop="number" label="#" width="56" sortable="custom" />
                <el-table-column prop="state" :label="t('search.trialState')" width="104" sortable="custom">
                  <template #default="{ row }">
                    <span class="t-state" :class="stateClass(row.state)">
                      <span class="t-dot" />
                      {{ stateLabel(row.state) }}
                    </span>
                  </template>
                </el-table-column>
                <el-table-column prop="value" :label="`${metricLabel} ${dirArrow}`" width="110" align="right" sortable="custom">
                  <template #default="{ row }">
                    <span class="t-value" :class="{ 't-best': isBest(row) }">
                      {{ row.value == null ? '—' : fmtValue(row.value) }}
                    </span>
                  </template>
                </el-table-column>
                <el-table-column prop="duration" :label="t('search.colDuration')" width="92" align="right" sortable="custom">
                  <template #default="{ row }">
                    <span class="t-dur">{{ fmtDuration(row) }}</span>
                  </template>
                </el-table-column>
                <el-table-column :label="t('search.trialParams')" min-width="120">
                  <template #default="{ row }">
                    <div class="pill-wrap">
                      <span v-for="(v, k) in row.params" :key="k" class="param-pill">
                        <span class="pk">{{ k }}</span>=<span class="pv">{{ fmtParam(v) }}</span>
                      </span>
                    </div>
                  </template>
                </el-table-column>
              </el-table>
            </DetailSection>

            <!-- Optuna Dashboard launcher (study.db is a standard Optuna storage). -->
            <DetailSection v-if="studyPath && studyPath.study_db_path" :title="t('search.sectionDashboard')">
              <CommandBlock :command="studyPath.dashboard_command || ''" />
              <p class="dashboard-hint" v-if="!studyPath.exists">{{ t('search.dashboardNotReady') }}</p>
            </DetailSection>

            <CommandBlock :command="task.command" />
          </div>

          <!-- Drag handle: pointer + keyboard resizable, dbl-click or Enter resets -->
          <div
            class="splitter"
            role="separator"
            aria-orientation="vertical"
            :aria-label="t('search.resizePane')"
            :aria-valuenow="Math.round(splitPct)"
            :aria-valuemin="SPLIT_MIN"
            :aria-valuemax="SPLIT_MAX"
            tabindex="0"
            @pointerdown="onPointerDown"
            @pointermove="onPointerMove"
            @pointerup="endDrag"
            @pointercancel="endDrag"
            @dblclick="resetSplit"
            @keydown.left.prevent="nudgeSplit(-3)"
            @keydown.right.prevent="nudgeSplit(3)"
            @keydown.enter.prevent="resetSplit"
          >
            <span class="splitter-grip" />
          </div>

          <!-- Right: log pane filling the full column height -->
          <div class="log-col">
            <LogCard :ws-url="`/api/tasks/${taskId}/logs/stream`" :task-status="task?.status || 'pending'" :task-id="taskId" />
          </div>
        </div>
      </div>
    </template>
  </el-skeleton>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useQuery, useQueryClient } from '@tanstack/vue-query'
import { useRoute } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { formatDateTime } from '@/utils/date'
import { formatGpu } from '@/utils/format'
import { getSearch, stopSearch, killSearch, getSearchTrials, getSearchStudyDb, type SearchTaskInfo, type SearchTrial } from '@/api/search'
import LogCard from '@/components/task/LogCard.vue'
import DetailHeader from '@/components/common/DetailHeader.vue'
import DetailSection from '@/components/common/DetailSection.vue'
import MetaGrid, { type MetaCell } from '@/components/common/MetaGrid.vue'
import CommandBlock from '@/components/common/CommandBlock.vue'
import { useSystemCapabilities } from '@/composables/useSystemCapabilities'
import { useTaskDuration } from '@/composables/useTaskDuration'

const { hasGpu } = useSystemCapabilities()
const { t } = useI18n()
const route = useRoute()
const queryClient = useQueryClient()
const taskId = Number(route.params.id)

const { data: task, isPending: loading } = useQuery({
  queryKey: ['search', taskId],
  queryFn: () => getSearch(taskId),
  refetchInterval: (query) => {
    const status = (query.state.data as SearchTaskInfo | undefined)?.status
    return status && ['running', 'pending', 'stopping'].includes(status) ? 3000 : false
  },
})

// Trial progress: poll while the search is active, stop at a terminal state.
const { data: study } = useQuery({
  queryKey: ['search-trials', taskId],
  queryFn: () => getSearchTrials(taskId),
  refetchInterval: (query) => {
    const status = queryClient.getQueryData<SearchTaskInfo>(['search', taskId])?.status
    return status && ['running', 'pending', 'stopping'].includes(status) ? 5000 : false
  },
})

const { data: studyPath } = useQuery({
  queryKey: ['search-studydb', taskId],
  queryFn: () => getSearchStudyDb(taskId),
})

const stopping = ref(false)
const killing = ref(false)

const { duration } = useTaskDuration(task)

// --- Resizable info/log split (persisted; dbl-click or Enter resets) ---
const SPLIT_KEY = 'kt-web:search-detail-split'
const SPLIT_DEFAULT = 42
const SPLIT_MIN = 28
const SPLIT_MAX = 72
const splitPct = ref(Number(localStorage.getItem(SPLIT_KEY)) || SPLIT_DEFAULT)
const bodyEl = ref<HTMLElement | null>(null)
const dragging = ref(false)

const clampSplit = (pct: number) => Math.min(SPLIT_MAX, Math.max(SPLIT_MIN, pct))

const onPointerDown = (e: PointerEvent) => {
  dragging.value = true
  ;(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId)
}

const onPointerMove = (e: PointerEvent) => {
  if (!dragging.value || !bodyEl.value) return
  const rect = bodyEl.value.getBoundingClientRect()
  splitPct.value = clampSplit(((e.clientX - rect.left) / rect.width) * 100)
}

const endDrag = () => {
  if (!dragging.value) return
  dragging.value = false
  localStorage.setItem(SPLIT_KEY, String(Math.round(splitPct.value)))
}

const resetSplit = () => {
  splitPct.value = SPLIT_DEFAULT
  localStorage.removeItem(SPLIT_KEY)
}

const nudgeSplit = (delta: number) => {
  splitPct.value = clampSplit(splitPct.value + delta)
  localStorage.setItem(SPLIT_KEY, String(Math.round(splitPct.value)))
}

// --- Metric display: real metric name from task.extra_params (e.g. "auc");
// direction is conveyed by a compact arrow, words only in the tooltip. ---
const extraParams = computed<Record<string, any>>(() => {
  try {
    return JSON.parse(task.value?.extra_params || '{}')
  } catch {
    return {}
  }
})

const metricLabel = computed(() => {
  const m = extraParams.value.metric
  if (Array.isArray(m)) return m.join(' + ')
  return typeof m === 'string' && m ? m : t('search.objective')
})

const dirArrow = computed(() => (study.value?.direction === 'minimize' ? '↓' : '↑'))
const dirTooltip = computed(() =>
  t(study.value?.direction === 'minimize' ? 'search.objectiveMin' : 'search.objectiveMax'),
)

const exitCodeClass = computed(() => {
  if (task.value?.exit_code == null) return ''
  return task.value.exit_code === 0 ? 'exit-ok' : 'exit-err'
})

const gpuText = computed(() =>
  formatGpu(task.value?.gpu_assigned ?? task.value?.gpu_request, task.value?.status === 'pending' ? t('search.gpuAuto') : '—'),
)

const metaCells = computed<MetaCell[]>(() => {
  const taskVal = task.value
  if (!taskVal) return []
  return [
    { label: t('task.detail.metaModel'), value: taskVal.model_name },
    { label: t('task.detail.metaDataset'), value: taskVal.dataset_name },
    { label: t('task.detail.metaEnv'), value: `${taskVal.env_type}:${taskVal.env_name}` },
    ...(hasGpu.value ? [{ label: 'GPU', value: gpuText.value }] : []),
    { label: t('task.detail.metaPid'), value: taskVal.pid || '—', mono: true },
    { label: t('task.detail.metaStartedAt'), value: formatDateTime(taskVal.started_at), mono: true },
    { label: t('task.detail.metaDuration'), value: duration.value, mono: true },
    { label: t('task.detail.metaExitCode'), value: taskVal.exit_code ?? '—', mono: true, valueClass: exitCodeClass.value },
  ]
})

// --- Trial display helpers ---
const doneCount = computed(() => (study.value?.completed ?? 0) + (study.value?.pruned ?? 0) + (study.value?.failed ?? 0))

const fmtValue = (v: number) => (typeof v === 'number' ? v.toFixed(4) : String(v))
const fmtParam = (v: unknown) => (typeof v === 'number' ? +v.toFixed(5) : v)

// --- Trials table sorting: custom mode driven by @sort-change so we can also
// express the "running first" default that el-table's default-sort cannot. ---
type SortOrder = 'ascending' | 'descending' | null
const trialSort = ref<{ prop: string; order: SortOrder }>({ prop: '', order: null })

const onTrialSort = ({ prop, order }: { prop?: string; order?: SortOrder }) => {
  trialSort.value = { prop: prop ?? '', order: order ?? null }
}

const STATE_RANK: Record<string, number> = { COMPLETE: 0, PRUNED: 1, FAIL: 2, RUNNING: 3, WAITING: 4 }
const isUnfinished = (s: string) => s === 'RUNNING' || s === 'WAITING'

const durationSec = (row: SearchTrial): number | null => {
  if (!row.datetime_start || !row.datetime_complete) return null
  const ms = new Date(row.datetime_complete).getTime() - new Date(row.datetime_start).getTime()
  return isNaN(ms) || ms < 0 ? null : ms / 1000
}

const sortedTrials = computed<SearchTrial[]>(() => {
  const rows = [...(study.value?.trials ?? [])]
  const { prop, order } = trialSort.value

  // Default view: in-flight trials pinned to the top, newest first (# desc)
  // within each group so live results stay visible without scrolling.
  if (!prop || !order) {
    return rows.sort((a, b) => {
      const ua = isUnfinished(a.state) ? 0 : 1
      const ub = isUnfinished(b.state) ? 0 : 1
      return ua !== ub ? ua - ub : b.number - a.number
    })
  }

  const dir = order === 'ascending' ? 1 : -1
  return rows.sort((a, b) => {
    // Null metric/duration values always sink to the bottom, regardless of direction.
    if (prop === 'value') {
      if (a.value == null) return 1
      if (b.value == null) return -1
      return (a.value - b.value) * dir
    }
    if (prop === 'duration') {
      const da = durationSec(a)
      const db = durationSec(b)
      if (da == null) return 1
      if (db == null) return -1
      return (da - db) * dir
    }
    if (prop === 'state') return (STATE_RANK[a.state] - STATE_RANK[b.state]) * dir
    return (a.number - b.number) * dir
  })
})

const fmtDuration = (row: SearchTrial) => {
  const sec = durationSec(row)
  if (sec == null) return ''
  const s = Math.floor(sec)
  if (s < 60) return `${s}s`
  const m = Math.floor(s / 60)
  if (m < 60) return `${m}m ${s % 60}s`
  return `${Math.floor(m / 60)}h ${m % 60}m`
}

// Optuna trial states (study.db): RUNNING / COMPLETE / PRUNED / FAIL / WAITING.
const stateClass = (s: string) => ({
  COMPLETE: 'st-complete',
  PRUNED: 'st-pruned',
  FAIL: 'st-fail',
  RUNNING: 'st-running',
  WAITING: 'st-waiting',
} as Record<string, string>)[s] ?? 'st-waiting'

const stateLabel = (s: string) =>
  ({
    COMPLETE: t('common.status.completed'),
    PRUNED: t('search.pruned'),
    FAIL: t('search.failedTrials'),
    RUNNING: t('common.status.running'),
    WAITING: t('search.stateWaiting'),
  } as Record<string, string>)[s] ?? s

const isBest = (row: { number: number }) =>
  !!study.value?.best_trial && row.number === study.value.best_trial.number

const trialRowClass = ({ row }: { row: { number: number } }) => (isBest(row) ? 'best-row' : '')

const handleStop = async () => {
  try {
    await ElMessageBox.confirm(t('search.stopConfirmMsg'), t('search.stopConfirmTitle'), {
      confirmButtonText: t('common.stop'), cancelButtonText: t('common.cancel'), type: 'warning',
    })
  } catch {
    return
  }
  stopping.value = true
  try {
    await stopSearch(taskId)
    ElMessage.success(t('search.stopSent'))
  } finally {
    stopping.value = false
  }
}

const handleKill = async () => {
  try {
    await ElMessageBox.confirm(t('task.detail.killConfirm'), t('task.detail.killTitle'), {
      confirmButtonText: t('common.forceKill'), cancelButtonText: t('common.cancel'), type: 'error',
    })
  } catch {
    return
  }
  killing.value = true
  try {
    await killSearch(taskId)
    ElMessage.success(t('search.killed'))
  } finally {
    killing.value = false
  }
}
</script>

<style scoped>
/* Header row + body split into a resizable info column and a full-height log. */
.search-detail {
  display: flex;
  flex-direction: column;
  height: 100%;
  max-width: 1200px;
  margin: 0 auto;
  min-height: 0;
  color: var(--text-primary);
}

.detail-header {
  flex-shrink: 0;
  margin-bottom: 16px;
}

.detail-body {
  flex: 1;
  display: flex;
  min-height: 0;
}

.info-col {
  flex-shrink: 0;
  display: flex;
  flex-direction: column;
  gap: 16px;
  min-height: 0;
  overflow-y: auto;
  padding-right: 14px;
}

/* Drag handle between the info column and the log pane. */
.splitter {
  position: relative;
  flex: 0 0 9px;
  margin: 0 3px;
  cursor: col-resize;
  touch-action: none;
}

/* Hit area wider than the 3px visual grip for easier dragging. */
.splitter::before {
  content: '';
  position: absolute;
  inset: 0 -6px;
  border-radius: 99px;
  background: transparent;
  transition: background 0.15s;
}

.splitter::after {
  content: '';
  position: absolute;
  inset: 0 3px;
  border-radius: 99px;
  background: var(--border-muted);
  transition: background 0.15s;
}

.splitter:hover::after,
.splitter:focus-visible::after,
.search-detail.is-dragging .splitter::after {
  background: var(--accent-blue);
}

.splitter-grip {
  position: absolute;
  top: 50%;
  left: 50%;
  width: 3px;
  height: 26px;
  transform: translate(-50%, -50%);
  border-radius: 99px;
  background: var(--border-default);
  transition: background 0.15s;
  pointer-events: none;
}

.splitter:hover .splitter-grip,
.splitter:focus-visible .splitter-grip {
  background: var(--accent-blue);
  opacity: 0.4;
}

/* While dragging: consistent cursor + no text selection anywhere. */
.search-detail.is-dragging {
  cursor: col-resize;
  user-select: none;
}

.log-col {
  flex: 1;
  display: flex;
  min-height: 0;
  min-width: 0;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  background: var(--bg-surface);
  overflow: hidden;
}

/* LogCard fills the pane; its internal rule is replaced by a panel border. */
.log-col :deep(.log-card) {
  flex: 1;
  min-width: 0;
}

.log-col :deep(.log-card-header) {
  padding: 10px 14px;
  border-bottom: 1px solid var(--border-muted);
}

.log-col :deep(.header-rule) {
  display: none;
}

.log-col :deep(.terminal-wrapper) {
  border-radius: 0;
}

/* --- Segmented trial progress --- */
.seg-bar {
  display: flex;
  gap: 2px;
  height: 8px;
  border-radius: 99px;
  background: var(--bg-overlay);
  overflow: hidden;
}

.seg {
  flex-basis: 0;
  min-width: 0;
  border-radius: 2px;
}

.seg.s-ok { background: var(--accent-green); }
.seg.s-pruned { background: var(--accent-orange); }
.seg.s-fail { background: var(--accent-red); }
.seg.s-run { background: var(--accent-blue); animation: seg-pulse 1.6s ease-in-out infinite; }

@keyframes seg-pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.55; }
}

.seg-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 6px 16px;
  font-size: 12px;
  color: var(--text-secondary);
}

.seg-legend .lg {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.seg-legend b {
  font-family: var(--font-mono);
  font-weight: 600;
  color: var(--text-primary);
}

.seg-legend .lg-total {
  margin-left: auto;
  color: var(--text-tertiary);
}

.dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
}

.dot.d-ok { background: var(--accent-green); }
.dot.d-pruned { background: var(--accent-orange); }
.dot.d-fail { background: var(--accent-red); }
.dot.d-run { background: var(--accent-blue); }

/* --- Best trial card --- */
.best-card {
  background: color-mix(in srgb, var(--accent-green) 6%, var(--bg-surface));
  border: 1px solid color-mix(in srgb, var(--accent-green) 25%, var(--border-default));
  border-radius: var(--radius-lg);
  padding: 12px 16px;
}

.best-head {
  display: flex;
  align-items: baseline;
  gap: 10px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}

.best-badge {
  font-size: 11px;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: var(--accent-green);
}

.best-num {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-tertiary);
}

.best-value {
  margin-left: auto;
  font-size: 12px;
  color: var(--text-secondary);
}

.best-metric {
  font-family: var(--font-mono);
  cursor: help;
  margin-right: 6px;
}

.best-value b {
  font-family: var(--font-mono);
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
}

.best-params {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}

.param-pill {
  font-family: var(--font-mono);
  font-size: 12px;
  background: var(--bg-overlay);
  padding: 2px 8px;
  border-radius: var(--radius-sm);
  color: var(--text-secondary);
  white-space: nowrap;
}

.param-pill .pk { color: var(--text-tertiary); }
.param-pill .pv { color: var(--text-primary); }

/* --- Trials table --- */
.trial-table {
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  overflow: hidden;
}

.trial-table :deep(.best-row td.el-table__cell) {
  background: color-mix(in srgb, var(--accent-green) 7%, transparent) !important;
}

.t-state {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
  color: var(--text-secondary);
}

.t-dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  flex-shrink: 0;
}

.st-complete .t-dot { background: var(--accent-green); }
.st-complete { color: var(--accent-green); }
.st-pruned .t-dot { background: var(--accent-orange); }
.st-pruned { color: var(--accent-orange); }
.st-fail .t-dot { background: var(--accent-red); }
.st-fail { color: var(--accent-red); }
.st-running .t-dot { background: var(--accent-blue); box-shadow: 0 0 5px var(--accent-blue); }
.st-running { color: var(--accent-blue); }
.st-waiting .t-dot { background: var(--text-tertiary); }
.st-waiting { color: var(--text-tertiary); }

.t-value,
.t-dur {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-secondary);
}

.t-value.t-best {
  color: var(--accent-green);
  font-weight: 600;
}

.pill-wrap {
  display: flex;
  flex-wrap: wrap;
  gap: 4px 6px;
}

/* --- Dashboard --- */
.dashboard-hint { font-size: 12px; color: var(--text-tertiary); margin: 0; }

/* --- Skeleton --- */
.detail-section { display: flex; flex-direction: column; gap: 10px; }
.section-head { display: flex; align-items: center; gap: 12px; }
.section-rule { flex: 1; height: 1px; background: var(--border-muted); }
.sk-meta-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 14px 24px;
}
.sk-meta-cell { display: flex; flex-direction: column; gap: 2px; }

:deep(.exit-ok) { color: var(--accent-green); }
:deep(.exit-err) { color: var(--accent-red); }

/* --- Narrow screens: stack, fixed-height log, splitter hidden --- */
@media (max-width: 1100px) {
  .detail-body {
    flex-direction: column;
  }

  .info-col {
    width: 100% !important;
    overflow: visible;
    padding-right: 0;
  }

  .splitter {
    display: none;
  }

  .log-col {
    height: 480px;
    flex: none;
  }
}

@media (prefers-reduced-motion: reduce) {
  .seg.s-run { animation: none; }
}
</style>
