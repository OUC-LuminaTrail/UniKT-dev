<template>
  <el-skeleton :loading="loading && !task" animated>
    <template #template>
      <div class="task-detail">
        <header class="detail-header">
          <el-skeleton-item variant="rect" style="width:32px;height:32px;border-radius:var(--radius-sm);flex-shrink:0" />
          <el-skeleton-item variant="text" style="width:180px;height:20px" />
          <el-skeleton-item variant="rect" style="width:72px;height:24px;border-radius:20px" />
        </header>

        <DetailSection :title="t('task.detail.sectionMeta')">
          <div class="sk-meta-grid">
            <div v-for="i in 6" :key="i" class="sk-meta-cell">
              <el-skeleton-item variant="text" style="width:40px;height:10px" />
              <el-skeleton-item variant="text" style="width:70%;height:13px;margin-top:2px" />
            </div>
          </div>
        </DetailSection>

        <DetailSection :title="t('task.detail.sectionCommand')">
          <div style="padding:12px 0">
            <el-skeleton-item variant="text" style="width:90%;height:14px" />
          </div>
        </DetailSection>

        <DetailSection :title="t('task.detail.sectionLog')">
          <div style="min-height:200px;background:var(--term-bg);border-radius:var(--radius-md)">
            <el-skeleton-item variant="text" style="margin:16px;width:60%;height:14px" />
          </div>
        </DetailSection>
      </div>
    </template>
    <template #default>
      <div class="task-detail" v-if="task">
        <DetailHeader
          :name="task.name"
          :status="task.status"
          :stopping="stopping"
          :killing="killing"
          fallback-route="tasks"
          @stop="handleStop"
          @kill="handleKill"
        />

        <DetailSection :title="t('task.detail.sectionMeta')">
          <MetaGrid :cells="metaCells" />
        </DetailSection>

        <CommandBlock :command="task.command" />

        <LogCard :ws-url="`/api/tasks/${taskId}/logs/stream`" :task-status="task?.status || 'pending'" :task-id="taskId" />
      </div>
    </template>
  </el-skeleton>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useQuery } from '@tanstack/vue-query'
import { useRoute } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { formatDateTime } from '@/utils/date'
import { formatGpu } from '@/utils/format'
import { getTask, stopTask, killTask, type TaskInfo } from '@/api/tasks'
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
const taskId = Number(route.params.id)

const { data: task, isPending: loading } = useQuery({
  queryKey: ['task', taskId],
  queryFn: () => getTask(taskId),
  refetchInterval: (query) => {
    const status = (query.state.data as TaskInfo | undefined)?.status
    return status && ['running', 'pending', 'stopping'].includes(status)
      ? 3000
      : false
  },
})

const stopping = ref(false)
const killing = ref(false)

const { duration } = useTaskDuration(task)

const exitCodeClass = computed(() => {
  if (task.value?.exit_code == null) return ''
  return task.value.exit_code === 0 ? 'exit-ok' : 'exit-err'
})

const gpuText = computed(() =>
  formatGpu(task.value?.gpu_assigned ?? task.value?.gpu_request, task.value?.status === 'pending' ? t('task.detail.gpuAuto') : '—'),
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

const handleStop = async () => {
  try {
    await ElMessageBox.confirm(t('task.detail.stopConfirm'), t('task.detail.stopTitle'), { confirmButtonText: t('task.detail.stop'), cancelButtonText: t('common.cancel'), type: 'warning' })
  } catch { return }
  stopping.value = true
  try {
    await stopTask(taskId)
    ElMessage.success(t('task.detail.stopSignalSent'))
  } finally { stopping.value = false }
}

const handleKill = async () => {
  try {
    await ElMessageBox.confirm(t('task.detail.killConfirm'), t('task.detail.killTitle'), { confirmButtonText: t('task.detail.killButton'), cancelButtonText: t('common.cancel'), type: 'error' })
  } catch { return }
  killing.value = true
  try {
    await killTask(taskId)
    ElMessage.success(t('task.detail.killed'))
  } finally { killing.value = false }
}
</script>

<style scoped>
.task-detail {
  display: flex;
  flex-direction: column;
  gap: 16px;
  max-width: 1200px;
  margin: 0 auto;
  height: 100%;
  min-height: 0;
  color: var(--text-primary);
}

/* Skeleton meta placeholders reuse MetaGrid's grid rhythm. */
.sk-meta-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 14px 24px;
}

.sk-meta-cell {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

:deep(.exit-ok) {
  color: var(--accent-green);
}

:deep(.exit-err) {
  color: var(--accent-red);
}
</style>
