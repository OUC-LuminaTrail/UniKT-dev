<template>
  <el-skeleton :loading="loading && !task" animated>
    <template #default>
      <div class="search-detail" v-if="task">
        <header class="detail-header">
          <button class="back-btn" @click="goBack">
            <el-icon :size="18"><ArrowLeft /></el-icon>
          </button>
          <h1 class="task-name">{{ task.name }}</h1>
          <span class="status-badge" :style="{ '--dot-color': statusMap[task.status]?.color }">
            <span class="status-dot" :class="{ pulse: task.status === 'running' }"></span>
            <span>{{ statusMap[task.status] ? t(statusMap[task.status].label) : task.status }}</span>
          </span>
          <div class="header-actions" v-if="task.status === 'running'">
            <button class="action-btn stop" :disabled="stopping" @click="handleStop">
              <el-icon :size="14"><SwitchButton /></el-icon>
              <span>{{ stopping ? t('task.detail.stopping') : t('task.detail.stop') }}</span>
            </button>
            <button class="action-btn kill" :disabled="killing" @click="handleKill">
              <el-icon :size="14"><Bottom /></el-icon>
              <span>{{ killing ? t('task.detail.killing') : t('task.detail.forceKill') }}</span>
            </button>
          </div>
        </header>

        <section class="detail-section">
          <div class="section-head">
            <h2 class="section-title">{{ t('task.detail.sectionMeta') }}</h2>
            <span class="section-rule"></span>
          </div>
          <div class="meta-grid">
            <div class="meta-cell"><span class="meta-key">{{ t('task.detail.metaModel') }}</span><span class="meta-val">{{ task.model_name }}</span></div>
            <div class="meta-cell"><span class="meta-key">{{ t('task.detail.metaDataset') }}</span><span class="meta-val">{{ task.dataset_name }}</span></div>
            <div class="meta-cell"><span class="meta-key">{{ t('task.detail.metaEnv') }}</span><span class="meta-val">{{ task.env_type }}:{{ task.env_name }}</span></div>
            <div class="meta-cell" v-if="hasGpu"><span class="meta-key">GPU</span><span class="meta-val">{{ gpuDisplay }}</span></div>
            <div class="meta-cell"><span class="meta-key">{{ t('task.detail.metaPid') }}</span><span class="meta-val mono">{{ task.pid || '—' }}</span></div>
            <div class="meta-cell"><span class="meta-key">{{ t('task.detail.metaStartedAt') }}</span><span class="meta-val mono">{{ formatDateTime(task.started_at) }}</span></div>
            <div class="meta-cell" v-if="duration"><span class="meta-key">{{ t('task.detail.metaDuration') }}</span><span class="meta-val mono">{{ duration }}</span></div>
            <div class="meta-cell"><span class="meta-key">{{ t('task.detail.metaExitCode') }}</span><span class="meta-val mono" :class="exitCodeClass">{{ task.exit_code ?? '—' }}</span></div>
          </div>
        </section>

        <!-- Trial progress from study.db (live while running). -->
        <section class="detail-section" v-if="study">
          <div class="section-head">
            <h2 class="section-title">{{ t('search.sectionProgress') }}</h2>
            <span class="section-rule"></span>
          </div>
          <div class="progress-row">
            <el-progress
              :percentage="progressPct"
              :status="task.status === 'completed' ? 'success' : undefined"
            />
            <span class="progress-text">{{ study.completed }}/{{ study.total }} ({{ t('search.pruned') }} {{ study.pruned }} · {{ t('search.failedTrials') }} {{ study.failed }})</span>
          </div>
          <div class="best-card" v-if="study.best_trial">
            <div class="best-head">
              <span class="best-label">{{ t('search.bestTrial') }} #{{ study.best_trial.number }}</span>
              <span class="best-value">{{ metricLabel }} = {{ fmtValue(study.best_trial.value) }}</span>
            </div>
            <div class="best-params">
              <span v-for="(v, k) in study.best_trial.params" :key="k" class="param-pill">
                <span class="pk">{{ k }}</span>=<span class="pv">{{ v }}</span>
              </span>
            </div>
          </div>
          <el-table :data="study.trials" size="small" max-height="320" class="trial-table" empty-text="—">
            <el-table-column prop="number" label="#" width="60" />
            <el-table-column :label="t('search.trialState')" width="110">
              <template #default="{ row }">
                <span class="trial-state" :class="row.state.toLowerCase()">{{ row.state }}</span>
              </template>
            </el-table-column>
            <el-table-column :label="metricLabel" width="120">
              <template #default="{ row }">{{ row.value == null ? '—' : fmtValue(row.value) }}</template>
            </el-table-column>
            <el-table-column :label="t('search.trialParams')">
              <template #default="{ row }">
                <span class="trial-params">{{ formatParams(row.params) }}</span>
              </template>
            </el-table-column>
          </el-table>
        </section>

        <!-- Optuna Dashboard launcher (study.db is a standard Optuna storage). -->
        <section class="detail-section" v-if="studyPath && studyPath.study_db_path">
          <div class="section-head">
            <h2 class="section-title">{{ t('search.sectionDashboard') }}</h2>
            <span class="section-rule"></span>
          </div>
          <div class="dashboard-row">
            <code class="dashboard-cmd">{{ studyPath.dashboard_command }}</code>
            <button class="copy-btn" @click="copyDash">
              <el-icon :size="12"><CopyDocument /></el-icon>
              {{ t('task.detail.copy') }}
            </button>
          </div>
          <p class="dashboard-hint" v-if="!studyPath.exists">{{ t('search.dashboardNotReady') }}</p>
        </section>

        <section class="detail-section">
          <div class="section-head">
            <h2 class="section-title">{{ t('task.detail.sectionCommand') }}</h2>
            <span class="section-rule"></span>
            <button class="copy-btn" @click="copyCommand">{{ t('task.detail.copy') }}</button>
            <button class="copy-btn" @click="commandExpanded = !commandExpanded">
              <el-icon :size="12"><component :is="commandExpanded ? ArrowUp : ArrowDown" /></el-icon>
              <span>{{ commandExpanded ? t('task.detail.collapse') : t('task.detail.expand') }}</span>
            </button>
          </div>
          <pre class="command-text" :class="{ expanded: commandExpanded }"><code>{{ task.command }}</code></pre>
        </section>

        <LogCard :ws-url="`/api/tasks/${taskId}/logs/stream`" :task-status="task?.status || 'pending'" :task-id="taskId" />
      </div>
    </template>
  </el-skeleton>
</template>

<script setup lang="ts">
import { ref, computed, watch, onUnmounted } from 'vue'
import { useI18n } from 'vue-i18n'
import { useQuery, useQueryClient } from '@tanstack/vue-query'
import { useRoute, useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { ArrowLeft, SwitchButton, Bottom, ArrowDown, ArrowUp, CopyDocument } from '@element-plus/icons-vue'
import { formatDateTime } from '@/utils/date'
import { getSearch, stopSearch, killSearch, getSearchTrials, getSearchStudyDb, type SearchTaskInfo } from '@/api/search'
import LogCard from '@/components/task/LogCard.vue'
import { useSystemCapabilities } from '@/composables/useSystemCapabilities'
import { statusMap } from '@/composables/useStatusMap'

const { hasGpu } = useSystemCapabilities()
const { t } = useI18n()
const route = useRoute()
const router = useRouter()
const queryClient = useQueryClient()
const taskId = Number(route.params.id)

const goBack = () => {
  if (window.history.length > 1) router.back()
  else router.replace({ name: 'searches' })
}

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
const commandExpanded = ref(false)

const exitCodeClass = computed(() => {
  if (task.value?.exit_code == null) return ''
  return task.value.exit_code === 0 ? 'exit-ok' : 'exit-err'
})

const gpuDisplay = computed(() => {
  const taskVal = task.value
  if (!taskVal) return '—'
  const val = taskVal.gpu_assigned ?? taskVal.gpu_request
  if (val === null || val === undefined) return taskVal.status === 'pending' ? t('task.detail.gpuAuto') : '—'
  return `GPU ${val}`
})

const progressPct = computed(() => {
  const s = study.value
  if (!s || s.total === 0) return 0
  return Math.round(((s.completed + s.pruned + s.failed) / s.total) * 100)
})

const metricLabel = computed(() => {
  const dir = study.value?.direction
  return dir === 'minimize' ? t('search.objectiveMin') : t('search.objectiveMax')
})

const fmtValue = (v: number) => (typeof v === 'number' ? v.toFixed(4) : String(v))
const formatParams = (p: Record<string, any>) =>
  Object.entries(p || {})
    .map(([k, v]) => `${k}=${typeof v === 'number' ? +v.toFixed(5) : v}`)
    .join('  ')

const now = ref(Date.now())
let clockTimer: ReturnType<typeof setInterval> | null = null
watch(
  () => task.value?.status,
  (status) => {
    const active = status === 'running' || status === 'pending' || status === 'stopping'
    if (active && !clockTimer) clockTimer = setInterval(() => (now.value = Date.now()), 1000)
    else if (!active && clockTimer) {
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

const copyCommand = () => task.value && navigator.clipboard.writeText(task.value.command)
const copyDash = () => {
  if (studyPath.value?.dashboard_command) {
    navigator.clipboard.writeText(studyPath.value.dashboard_command)
    ElMessage.success(t('search.copied'))
  }
}

const handleStop = async () => {
  try {
    await ElMessageBox.confirm(t('task.detail.stopConfirm'), t('task.detail.stopTitle'), {
      confirmButtonText: t('task.detail.stop'), cancelButtonText: t('common.cancel'), type: 'warning',
    })
  } catch {
    return
  }
  stopping.value = true
  try {
    await stopSearch(taskId)
    ElMessage.success(t('task.detail.stopSignalSent'))
  } finally {
    stopping.value = false
  }
}

const handleKill = async () => {
  try {
    await ElMessageBox.confirm(t('task.detail.killConfirm'), t('task.detail.killTitle'), {
      confirmButtonText: t('task.detail.killButton'), cancelButtonText: t('common.cancel'), type: 'error',
    })
  } catch {
    return
  }
  killing.value = true
  try {
    await killSearch(taskId)
    ElMessage.success(t('task.detail.killed'))
  } finally {
    killing.value = false
  }
}
</script>

<style scoped>
.search-detail {
  display: flex;
  flex-direction: column;
  gap: 16px;
  max-width: 1200px;
  margin: 0 auto;
  height: 100%;
  min-height: 0;
  color: var(--text-primary);
}
.detail-header { display: flex; align-items: center; gap: 12px; min-height: 36px; }
.back-btn {
  display: flex; align-items: center; justify-content: center;
  width: 32px; height: 32px; border: 1px solid var(--border-default);
  border-radius: var(--radius-sm); background: var(--bg-surface);
  color: var(--text-secondary); cursor: pointer; transition: all 0.15s ease; flex-shrink: 0;
}
.back-btn:hover { background: var(--bg-elevated); color: var(--text-primary); border-color: var(--accent-blue); }
.task-name { font-size: 18px; font-weight: 600; color: var(--text-primary); margin: 0; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; letter-spacing: -0.01em; }
.status-badge { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px; border-radius: 20px; background: color-mix(in srgb, var(--dot-color, var(--text-tertiary)) 12%, transparent); border: 1px solid color-mix(in srgb, var(--dot-color, var(--text-tertiary)) 20%, transparent); flex-shrink: 0; font-size: 12px; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; background: var(--dot-color, var(--text-tertiary)); flex-shrink: 0; }
.status-dot.pulse { animation: pulse-glow 2s ease-in-out infinite; }
@keyframes pulse-glow { 0%, 100% { box-shadow: 0 0 0 0 color-mix(in srgb, var(--accent-blue) 50%, transparent); opacity: 1; } 50% { box-shadow: 0 0 0 6px color-mix(in srgb, var(--accent-blue) 0%, transparent); opacity: 0.7; } }
.header-actions { display: flex; gap: 8px; margin-left: auto; flex-shrink: 0; }
.action-btn { display: inline-flex; align-items: center; gap: 6px; padding: 6px 14px; border-radius: var(--radius-sm); border: 1px solid var(--border-default); background: var(--bg-surface); color: var(--text-secondary); font-size: 13px; font-weight: 500; cursor: pointer; transition: all 0.15s ease; }
.action-btn:disabled { opacity: 0.5; cursor: not-allowed; }
.action-btn.stop:hover:not(:disabled) { border-color: var(--accent-orange); color: var(--accent-orange); }
.action-btn.kill:hover:not(:disabled) { border-color: var(--accent-red); color: var(--accent-red); }
.meta-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 14px 24px; }
.meta-cell { display: flex; flex-direction: column; gap: 2px; }
.meta-key { font-size: 11px; font-weight: 500; color: var(--text-tertiary); text-transform: uppercase; letter-spacing: 0.05em; }
.meta-val { font-size: 13px; color: var(--text-primary); line-height: 1.4; word-break: break-all; }
.meta-val.mono { font-family: var(--font-mono); font-size: 12.5px; }
.meta-val.exit-ok { color: var(--accent-green); }
.meta-val.exit-err { color: var(--accent-red); }
.detail-section { display: flex; flex-direction: column; gap: 10px; }
.section-head { display: flex; align-items: center; gap: 12px; }
.section-title { font-size: 11px; font-weight: 600; color: var(--text-tertiary); letter-spacing: 0.05em; margin: 0; flex-shrink: 0; }
.section-rule { flex: 1; height: 1px; background: var(--border-muted); }
.progress-row { display: flex; align-items: center; gap: 14px; }
.progress-row .el-progress { flex: 1; }
.progress-text { font-size: 12px; color: var(--text-tertiary); font-family: var(--font-mono); white-space: nowrap; }
.best-card { background: color-mix(in srgb, var(--accent-green) 6%, var(--bg-surface)); border: 1px solid color-mix(in srgb, var(--accent-green) 25%, var(--border-default)); border-radius: var(--radius-lg); padding: 12px 16px; }
.best-head { display: flex; align-items: baseline; gap: 12px; margin-bottom: 8px; }
.best-label { font-size: 13px; font-weight: 600; color: var(--accent-green); }
.best-value { font-family: var(--font-mono); font-size: 13px; color: var(--text-primary); }
.best-params { display: flex; flex-wrap: wrap; gap: 6px; }
.param-pill { font-family: var(--font-mono); font-size: 12px; background: var(--bg-overlay); padding: 2px 8px; border-radius: var(--radius-sm); color: var(--text-secondary); }
.param-pill .pk { color: var(--text-tertiary); }
.param-pill .pv { color: var(--text-primary); }
.trial-table .trial-state { font-size: 11px; font-family: var(--font-mono); padding: 1px 6px; border-radius: var(--radius-sm); }
.trial-state.complete { color: var(--accent-green); background: color-mix(in srgb, var(--accent-green) 12%, transparent); }
.trial-state.pruned { color: var(--accent-orange); background: color-mix(in srgb, var(--accent-orange) 12%, transparent); }
.trial-state.fail { color: var(--accent-red); background: color-mix(in srgb, var(--accent-red) 12%, transparent); }
.trial-state.running { color: var(--accent-blue); background: color-mix(in srgb, var(--accent-blue) 12%, transparent); }
.trial-params { font-family: var(--font-mono); font-size: 12px; color: var(--text-tertiary); }
.dashboard-row { display: flex; align-items: center; gap: 8px; }
.dashboard-cmd { flex: 1; font-family: var(--font-mono); font-size: 12.5px; color: var(--accent-cyan); background: var(--term-bg); padding: 8px 12px; border-radius: var(--radius-sm); overflow-x: auto; white-space: nowrap; }
.copy-btn { display: inline-flex; align-items: center; gap: 4px; font-size: 11px; color: var(--text-tertiary); background: none; border: 1px solid var(--border-default); border-radius: var(--radius-sm); padding: 4px 8px; cursor: pointer; transition: all 0.15s ease; }
.copy-btn:hover { color: var(--accent-blue); border-color: var(--accent-blue); }
.dashboard-hint { font-size: 12px; color: var(--text-tertiary); margin: 0; }
.command-text { margin: 0; overflow-x: auto; }
.command-text code { font-family: var(--font-mono); font-size: 12.5px; color: var(--accent-cyan); line-height: 1.6; white-space: nowrap; }
.command-text.expanded code { white-space: pre-wrap; word-break: break-all; }
</style>
