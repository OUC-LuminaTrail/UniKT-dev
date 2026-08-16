<template>
  <div class="task-launch">
    <div ref="bodyRef" class="launch-body">
      <LaunchPageHeader
        :title="t('task.launch.title')"
        :subtitle="step === 'select' ? t('task.launch.subtitleSelect') : t('task.launch.subtitleParams')"
      />

      <el-skeleton :loading="loading" animated>
        <template #template>
          <SelectionSkeleton />
        </template>
        <template #default>
          <SelectionStep
            v-if="step === 'select'"
            ref="selectionStepRef"
            :envId="envId"
            :customPythonPath="customPythonPath"
            :modelName="modelName"
            :dataset="dataset"
            :gpu="gpu"
            :environments="environments"
            :models="models"
            :datasets="datasets"
            :refreshing="refreshing"
            @update:envId="envId = $event"
            @update:customPythonPath="customPythonPath = $event"
            @update:modelName="setModel"
            @update:dataset="dataset = $event"
            @update:gpu="gpu = $event"
            @confirm="onSelectConfirm"
            @refresh="refreshAll(['dataset-metadata'])"
          />

          <div v-if="step === 'params'" class="params-step">
            <div class="params-header">
              <el-button class="back-btn" @click="step = 'select'">
                <el-icon :size="14" style="margin-right:4px"><ArrowLeft /></el-icon>
                {{ t('task.launch.backToSelect') }}
              </el-button>
            </div>

            <ParamForm
              v-if="selectionSchema"
              :schema="selectionSchema"
              @update:params="params = $event"
            />
          </div>
        </template>
      </el-skeleton>

    </div>

    <CommandPreview
      :command="previewCommandText"
      :modelName="modelName"
    >
      <el-button
        v-if="step === 'select'"
        type="primary"
        size="large"
        :disabled="!canConfirm"
        @click="onSelectConfirm"
      >
        {{ t('task.launch.confirmSelection') }}
      </el-button>
      <el-select
        v-if="step === 'params' && showKfoldSelector"
        v-model="selectedFolds"
        multiple
        collapse-tags
        collapse-tags-tooltip
        :placeholder="t('task.launch.kfoldPlaceholder')"
        class="kfold-select"
      >
        <el-option
          v-for="fold in kfoldCount"
          :key="fold - 1"
          :label="`fold ${fold - 1}`"
          :value="fold - 1"
        />
      </el-select>
      <el-button
        v-if="step === 'params'"
        type="primary"
        size="large"
        :loading="submitting"
        @click="onStartTraining"
      >
        <span v-if="!submitting">{{ t('task.launch.startTraining') }}</span>
        <span v-else>{{ t('task.launch.creating') }}</span>
      </el-button>
    </CommandPreview>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, h, nextTick } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { useQuery } from '@tanstack/vue-query'
import { ElMessage, ElMessageBox } from 'element-plus'
import { ArrowLeft } from '@element-plus/icons-vue'
import { createTask, previewCommand } from '@/api/tasks'
import { getModelParams, type ModelSchema } from '@/api/schemas'
import CommandPreview from '@/components/task/CommandPreview.vue'
import SelectionStep from '@/components/task/SelectionStep.vue'
import ParamForm from '@/components/task/ParamForm.vue'
import LaunchPageHeader from '@/components/common/LaunchPageHeader.vue'
import SelectionSkeleton from '@/components/common/SelectionSkeleton.vue'
import { useLaunchInit } from '@/composables/useLaunchInit'

const router = useRouter()
const { t } = useI18n()
const step = ref<'select' | 'params'>('select')
const submitting = ref(false)
const params = ref<Record<string, any>>({})
const previewCommandText = ref('')

const {
  envId,
  customPythonPath,
  modelName,
  dataset,
  gpu,
  environments,
  models,
  datasets,
  selectedInfo,
  loading,
  refreshing,
  setModel,
  persistSelection,
  refreshAll,
  datasetMetaQuery,
} = useLaunchInit({
  initQueryKey: 'task-launch-init',
  registryFailedKey: 'task.launch.refreshRegistryFailed',
  refreshedKey: 'task.launch.refreshed',
})

const canConfirm = computed(
  () => !!envId.value && !!modelName.value && selectedInfo.value?.status === 'ready'
)

const kfoldCount = ref<number | null>(null)
const selectedFolds = ref<number[]>([])
const showKfoldSelector = computed(() => (kfoldCount.value ?? 0) >= 2)

const modelParamsQuery = useQuery({
  queryKey: computed(() => ['model-params', modelName.value]),
  queryFn: () => getModelParams(modelName.value),
  enabled: computed(() => !!modelName.value),
})

const selectionSchema = computed<ModelSchema | null>(() => modelParamsQuery.data.value ?? null)

const datasetMeta = datasetMetaQuery(dataset)

const bodyRef = ref<HTMLElement | null>(null)
const selectionStepRef = ref<InstanceType<typeof SelectionStep> | null>(null)

function onSelectConfirm() {
  if (!canConfirm.value) return
  persistSelection()
  step.value = 'params'
  nextTick(() => {
    bodyRef.value?.scrollTo({ top: 0, behavior: 'instant' as ScrollBehavior })
  })
}

watch(() => datasetMeta.data.value, (meta) => {
  if (!meta) {
    kfoldCount.value = null
    selectedFolds.value = []
    return
  }
  const kfold = typeof meta.kfold_n_splits === 'number' ? meta.kfold_n_splits : null
  kfoldCount.value = kfold
  selectedFolds.value = kfold && kfold >= 2 ? [0] : []
})

let _previewTimer: ReturnType<typeof setTimeout> | undefined

watch(
  () => [modelName.value, dataset.value, params.value] as const,
  ([model, ds, p]) => {
    clearTimeout(_previewTimer)
    if (!model) return
    _previewTimer = setTimeout(async () => {
      try {
        const { command } = await previewCommand({
          model_name: model,
          params: { ...p, dataset: ds },
        })
        previewCommandText.value = command
      } catch {
        previewCommandText.value = ''
      }
    }, 200)
  },
  { deep: true },
)

function buildTaskName(fold?: number) {
  return fold === undefined
    ? `${modelName.value}_${dataset.value}`
    : `${modelName.value}_${dataset.value}_fold${fold}`
}

async function confirmMultiFold(): Promise<void> {
  const folds = [...selectedFolds.value].sort((a, b) => a - b)
  const message = h('div', { style: 'line-height: 1.8' }, [
    h('div', t('task.launch.multiFoldModel', { name: modelName.value })),
    h('div', t('task.launch.multiFoldDataset', { name: dataset.value })),
    h('div', t('task.launch.multiFoldFolds', { folds: folds.join(', ') })),
    h('div', t('task.launch.multiFoldCount', { n: folds.length })),
  ])
  await ElMessageBox.confirm(message, t('task.launch.multiFoldTitle'), {
    confirmButtonText: t('task.launch.createButton'),
    cancelButtonText: t('common.cancel'),
    type: 'warning',
  })
}

async function onStartTraining() {
  submitting.value = true
  try {
    const taskParams = { ...params.value, dataset: dataset.value }

    if (showKfoldSelector.value && selectedFolds.value.length > 1) {
      await confirmMultiFold()
      let created = 0
      for (const fold of [...selectedFolds.value].sort((a, b) => a - b)) {
        await createTask({
          name: buildTaskName(fold),
          env_id: envId.value,
          custom_python_path: customPythonPath.value || null,
          model_name: modelName.value,
          params: { ...taskParams, fold },
          gpu: gpu.value,
        })
        created += 1
      }
      ElMessage.success(t('task.launch.createdN', { n: created }))
      router.replace({ name: 'tasks' })
      return
    }

    if (showKfoldSelector.value && selectedFolds.value.length === 1) {
      const fold = selectedFolds.value[0]
      const task = await createTask({
        name: buildTaskName(fold),
        env_id: envId.value,
        custom_python_path: customPythonPath.value || null,
        model_name: modelName.value,
        params: { ...taskParams, fold },
        gpu: gpu.value,
      })
      if (task.status === 'pending') {
        ElMessage.success(t('task.launch.queued'))
        router.replace({ name: 'tasks' })
      } else {
        ElMessage.success(t('task.launch.created'))
        router.replace({ name: 'task-detail', params: { id: task.id } })
      }
      return
    }

    const task = await createTask({
      name: buildTaskName(),
      env_id: envId.value,
      custom_python_path: customPythonPath.value || null,
      model_name: modelName.value,
      params: taskParams,
      gpu: gpu.value,
    })
    if (task.status === 'pending') {
      ElMessage.success(t('task.launch.queued'))
      router.replace({ name: 'tasks' })
    } else {
      ElMessage.success(t('task.launch.created'))
      router.replace({ name: 'task-detail', params: { id: task.id } })
    }
  } catch (err: any) {
    if (err?.message && err?.message !== 'cancel') {
      ElMessage.error(t('task.launch.createFailed', { msg: err.message }))
    }
  } finally {
    submitting.value = false
  }
}
</script>

<style scoped>
.task-launch {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.launch-body {
  flex: 1;
  overflow-y: auto;
  min-height: 0;
  padding: 20px;
}

.params-step {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.params-header {
  display: flex;
  justify-content: flex-end;
}

.back-btn {
  font-size: 13px;
}

.kfold-select {
  min-width: 180px;
}
</style>
