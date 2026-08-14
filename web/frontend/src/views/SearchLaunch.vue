<template>
  <div class="search-launch">
    <div ref="bodyRef" class="launch-body">
      <LaunchPageHeader
        :title="t('search.launchTitle')"
        :subtitle="step === 'select' ? t('search.launchSubtitleSelect') : t('search.launchSubtitleParams')"
      />

      <el-skeleton :loading="loading" animated>
        <template #template>
          <SelectionSkeleton :with-datasets="false" />
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
            @refresh="refreshAll()"
          />

          <div v-if="step === 'params'" class="params-step">
            <div class="params-header">
              <el-button class="back-btn" @click="step = 'select'">
                <el-icon :size="14" style="margin-right: 4px"><ArrowLeft /></el-icon>
                {{ t('search.backToSelect') }}
              </el-button>
            </div>

            <template v-if="selectionSchema">
              <SearchRunConfigForm :schema="selectionSchema" @update:params="runconfigParams = $event" />
              <div class="optuna-section">
                <div class="section-label">{{ t('search.optunaConfig') }}</div>
                <OptunaConfigForm v-model="optunaConfig" />
              </div>
            </template>
          </div>
        </template>
      </el-skeleton>
    </div>

    <CommandPreview :command="previewCommandText" :modelName="modelName">
      <el-button
        v-if="step === 'select'"
        type="primary"
        size="large"
        :disabled="!canConfirm"
        @click="onSelectConfirm"
      >
        {{ t('search.confirmSelection') }}
      </el-button>
      <el-button
        v-if="step === 'params'"
        type="primary"
        size="large"
        :loading="submitting"
        @click="onStartSearch"
      >
        <span v-if="!submitting">{{ t('search.startSearch') }}</span>
        <span v-else>{{ t('search.creating') }}</span>
      </el-button>
    </CommandPreview>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRouter } from 'vue-router'
import { useQuery } from '@tanstack/vue-query'
import { ElMessage } from 'element-plus'
import { ArrowLeft } from '@element-plus/icons-vue'
import { createSearch, previewSearchCommand } from '@/api/search'
import { getModelParams, type ModelSchema } from '@/api/schemas'
import CommandPreview from '@/components/task/CommandPreview.vue'
import SelectionStep from '@/components/task/SelectionStep.vue'
import SearchRunConfigForm from '@/components/task/SearchRunConfigForm.vue'
import OptunaConfigForm from '@/components/task/OptunaConfigForm.vue'
import LaunchPageHeader from '@/components/common/LaunchPageHeader.vue'
import SelectionSkeleton from '@/components/common/SelectionSkeleton.vue'
import { useLaunchInit } from '@/composables/useLaunchInit'

const router = useRouter()
const { t } = useI18n()
const step = ref<'select' | 'params'>('select')
const submitting = ref(false)
const runconfigParams = ref<Record<string, any>>({})
const optunaConfig = ref<Record<string, any>>({})
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
} = useLaunchInit({
  initQueryKey: 'search-launch-init',
  registryFailedKey: 'search.refreshRegistryFailed',
  refreshedKey: 'search.refreshed',
})

const canConfirm = computed(
  () => !!envId.value && !!modelName.value && selectedInfo.value?.status === 'ready',
)

const modelParamsQuery = useQuery({
  queryKey: computed(() => ['model-params', modelName.value]),
  queryFn: () => getModelParams(modelName.value),
  enabled: computed(() => !!modelName.value),
})

const selectionSchema = computed<ModelSchema | null>(() => modelParamsQuery.data.value ?? null)

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

// Debounced command preview: any of model/dataset/runconfig/optuna changing
// re-derives the CLI the backend would actually run.
let _previewTimer: ReturnType<typeof setTimeout> | undefined
watch(
  () => [modelName.value, dataset.value, runconfigParams.value, optunaConfig.value] as const,
  ([model, ds, rp, oc]) => {
    clearTimeout(_previewTimer)
    if (!model) return
    _previewTimer = setTimeout(async () => {
      try {
        previewCommandText.value = await previewSearchCommand({
          model_name: model,
          dataset: ds,
          runconfig_params: rp,
          optuna_config: oc,
        })
      } catch {
        previewCommandText.value = ''
      }
    }, 200)
  },
  { deep: true },
)

async function onStartSearch() {
  submitting.value = true
  try {
    const task = await createSearch({
      name: `${modelName.value}_${dataset.value}_search`,
      env_id: envId.value,
      custom_python_path: customPythonPath.value || null,
      model_name: modelName.value,
      dataset: dataset.value,
      gpu: gpu.value,
      runconfig_params: runconfigParams.value,
      optuna_config: optunaConfig.value,
    })
    if (task.status === 'pending') {
      ElMessage.success(t('search.queued'))
      router.replace({ name: 'searches' })
    } else {
      ElMessage.success(t('search.created'))
      router.replace({ name: 'search-detail', params: { id: task.id } })
    }
  } catch (err: any) {
    if (err?.message && err?.message !== 'cancel') {
      ElMessage.error(t('search.createFailed', { msg: err.message }))
    }
  } finally {
    submitting.value = false
  }
}
</script>

<style scoped>
.search-launch {
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

.optuna-section {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.section-label {
  font-size: 11px;
  font-weight: 600;
  color: var(--text-tertiary);
  letter-spacing: 0.05em;
}
</style>
