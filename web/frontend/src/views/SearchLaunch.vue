<template>
  <div class="search-launch">
    <div ref="bodyRef" class="search-launch-body">
      <div class="page-header">
        <h1 class="page-title">{{ t('search.launchTitle') }}</h1>
        <p class="page-subtitle">
          {{ step === 'select' ? t('search.launchSubtitleSelect') : t('search.launchSubtitleParams') }}
        </p>
      </div>

      <el-skeleton :loading="loading" animated>
        <template #template>
          <div class="selection-step-skeleton">
            <div class="sk-section">
              <el-skeleton-item variant="text" style="width: 80px; margin-bottom: 10px" />
              <el-skeleton-item variant="text" style="width: 240px; height: 32px" />
            </div>
            <div class="sk-section">
              <el-skeleton-item variant="text" style="width: 48px; margin-bottom: 10px" />
              <div class="sk-card-grid">
                <div class="sk-card" v-for="i in 6" :key="'m' + i">
                  <el-skeleton-item variant="rect" style="width: 40px; height: 40px" />
                  <el-skeleton-item variant="text" style="width: 50px" />
                </div>
              </div>
            </div>
          </div>
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
            :refreshing="loading"
            @update:envId="envId = $event"
            @update:customPythonPath="customPythonPath = $event"
            @update:modelName="onModelChange"
            @update:dataset="dataset = $event"
            @update:gpu="gpu = $event"
            @confirm="onSelectConfirm"
            @refresh="refreshAll"
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
import { useQuery, useQueryClient } from '@tanstack/vue-query'
import { ElMessage } from 'element-plus'
import { ArrowLeft } from '@element-plus/icons-vue'
import { createSearch, previewSearchCommand } from '@/api/search'
import { listDatasets, type DatasetInfo } from '@/api/datasets'
import { listEnvironments, type EnvironmentInfo } from '@/api/environments'
import { listModels, getModelParams, type ModelSchema } from '@/api/schemas'
import { getDefaultEnv } from '@/api/settings'
import { refreshRegistry } from '@/api/system'
import CommandPreview from '@/components/task/CommandPreview.vue'
import SelectionStep from '@/components/task/SelectionStep.vue'
import SearchRunConfigForm from '@/components/task/SearchRunConfigForm.vue'
import OptunaConfigForm from '@/components/task/OptunaConfigForm.vue'

const router = useRouter()
const { t } = useI18n()
const queryClient = useQueryClient()
const step = ref<'select' | 'params'>('select')
const submitting = ref(false)
const runconfigParams = ref<Record<string, any>>({})
const optunaConfig = ref<Record<string, any>>({})
const previewCommandText = ref('')

const envId = ref('')
const customPythonPath = ref('')
const modelName = ref('')
const dataset = ref('')
const gpu = ref<number | null>(null)

const STORAGE_KEY_ENV = 'kt-web:last-env-id'
const STORAGE_KEY_MODEL = 'kt-web:last-model-name'
const STORAGE_KEY_DATASET = 'kt-web:last-dataset'

const initDataQuery = useQuery({
  queryKey: ['search-launch-init'],
  queryFn: async () => {
    const [envs, modelList, defaultEnv] = await Promise.all([
      listEnvironments(),
      listModels(),
      getDefaultEnv().catch(() => ({ default_env_id: null, custom_python_path: null, remember_last_env: false })),
    ])
    return { envs, modelList, defaultEnv }
  },
})

const environments = computed<EnvironmentInfo[]>(() => initDataQuery.data.value?.envs ?? [])
const models = computed(() => initDataQuery.data.value?.modelList ?? [])
const refreshing = ref(false)
const loading = computed(() => initDataQuery.isPending.value || refreshing.value)

const datasetsQuery = useQuery({ queryKey: ['datasets'], queryFn: listDatasets })
const datasets = computed<DatasetInfo[]>(() =>
  (datasetsQuery.data.value ?? []).filter((d) => d.status !== 'empty'),
)
const selectedInfo = computed(
  () => datasets.value.find((d) => d.name === dataset.value) ?? null,
)
const canConfirm = computed(
  () => !!envId.value && !!modelName.value && selectedInfo.value?.status === 'ready',
)

const modelParamsQuery = useQuery({
  queryKey: computed(() => ['model-params', modelName.value]),
  queryFn: () => getModelParams(modelName.value),
  enabled: computed(() => !!modelName.value),
})

const selectionSchema = computed<ModelSchema | null>(() => modelParamsQuery.data.value ?? null)

async function onModelChange(val: string) {
  modelName.value = val
  if (!val) return
  localStorage.setItem(STORAGE_KEY_MODEL, val)
}

function onSelectConfirm() {
  if (!canConfirm.value) return
  localStorage.setItem(STORAGE_KEY_ENV, envId.value)
  localStorage.setItem(STORAGE_KEY_DATASET, dataset.value)
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

const initDone = ref(false)
const bodyRef = ref<HTMLElement | null>(null)
const selectionStepRef = ref<InstanceType<typeof SelectionStep> | null>(null)

async function refreshAll() {
  refreshing.value = true
  initDone.value = false
  selectionStepRef.value?.clearCache()
  try {
    await refreshRegistry()
  } catch (err: any) {
    if (!err?.response) {
      ElMessage.error(t('search.refreshRegistryFailed'))
    }
    refreshing.value = false
    return
  }
  await queryClient.refetchQueries({ queryKey: ['search-launch-init'] })
  await queryClient.refetchQueries({ queryKey: ['model-params'] })
  await queryClient.refetchQueries({ queryKey: ['datasets'] })
  refreshing.value = false
  ElMessage.success(t('search.refreshed'))
}

watch(
  () => initDataQuery.data.value,
  (data) => {
    if (!data || initDone.value) return
    initDone.value = true
    const { envs, modelList, defaultEnv } = data
    const savedEnv = localStorage.getItem(STORAGE_KEY_ENV)
    if (defaultEnv.remember_last_env && savedEnv && envs.some((e) => e.id === savedEnv)) {
      envId.value = savedEnv
      if (savedEnv === 'custom:0' && defaultEnv.custom_python_path) {
        customPythonPath.value = defaultEnv.custom_python_path
      }
    } else if (defaultEnv.default_env_id && envs.some((e) => e.id === defaultEnv.default_env_id)) {
      envId.value = defaultEnv.default_env_id
      if (defaultEnv.custom_python_path) customPythonPath.value = defaultEnv.custom_python_path
    } else if (savedEnv && envs.some((e) => e.id === savedEnv)) {
      envId.value = savedEnv
    }
    const savedModel = localStorage.getItem(STORAGE_KEY_MODEL)
    if (savedModel && modelList.includes(savedModel)) {
      modelName.value = savedModel
      onModelChange(savedModel)
    }
  },
  { immediate: true },
)

const datasetRestored = ref(false)
watch(datasets, (list) => {
  if (datasetRestored.value || !list.length) return
  datasetRestored.value = true
  const saved = localStorage.getItem(STORAGE_KEY_DATASET)
  if (saved && list.some((d) => d.name === saved)) dataset.value = saved
})
</script>

<style scoped>
.search-launch {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.search-launch-body {
  flex: 1;
  overflow-y: auto;
  min-height: 0;
  padding: 20px;
}

.page-header {
  margin-bottom: 20px;
}

.page-title {
  font-size: 22px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0 0 4px 0;
  letter-spacing: -0.3px;
}

.page-subtitle {
  font-size: 14px;
  color: var(--text-tertiary);
  margin: 0;
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

.selection-step-skeleton {
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.sk-section {
  display: flex;
  flex-direction: column;
}

.sk-card-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: 12px;
}

.sk-card {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  padding: 18px 12px 14px;
  background: var(--bg-surface);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
}
</style>
