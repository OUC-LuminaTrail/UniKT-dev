import { computed, ref, watch, type Ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { useQuery, useQueryClient } from '@tanstack/vue-query'
import { ElMessage } from 'element-plus'
import { getDatasetMetadata, listDatasets, type DatasetInfo } from '@/api/datasets'
import { listEnvironments, type EnvironmentInfo } from '@/api/environments'
import { listModels } from '@/api/schemas'
import { getDefaultEnv } from '@/api/settings'
import { refreshRegistry } from '@/api/system'

// Shared launch-page bootstrap: environment/model/dataset selection state,
// persisted last-choices restore, dataset filtering, and registry refresh.
// Used by both TaskLaunch and SearchLaunch so the two flows stay in lockstep.
export function useLaunchInit(opts: {
  initQueryKey: string
  registryFailedKey: string
  refreshedKey: string
}) {
  const { t } = useI18n()
  const queryClient = useQueryClient()

  const envId = ref('')
  const customPythonPath = ref('')
  const modelName = ref('')
  const dataset = ref('')
  const gpu = ref<number | null>(null)

  const STORAGE_KEY_ENV = 'kt-web:last-env-id'
  const STORAGE_KEY_MODEL = 'kt-web:last-model-name'
  const STORAGE_KEY_DATASET = 'kt-web:last-dataset'

  const initDataQuery = useQuery({
    queryKey: [opts.initQueryKey],
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
  // Launch views hide empty datasets; downloaded ones stay visible so the user
  // can be routed to preprocessing instead of training on unprocessed data.
  const datasets = computed<DatasetInfo[]>(() =>
    (datasetsQuery.data.value ?? []).filter((d) => d.status !== 'empty'),
  )
  const selectedInfo = computed(
    () => datasets.value.find((d) => d.name === dataset.value) ?? null,
  )

  function setModel(val: string) {
    modelName.value = val
    if (val) localStorage.setItem(STORAGE_KEY_MODEL, val)
  }

  function persistSelection() {
    localStorage.setItem(STORAGE_KEY_ENV, envId.value)
    localStorage.setItem(STORAGE_KEY_DATASET, dataset.value)
  }

  // Restore last-chosen env/model from the init payload (env) and
  // localStorage (model), honoring the server's default-env policy.
  const initDone = ref(false)
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
        setModel(savedModel)
      }
    },
    { immediate: true },
  )

  // Restore last-chosen dataset once the filtered list resolves; only ready
  // ones are eligible since empty datasets are hidden from launch views.
  const datasetRestored = ref(false)
  watch(datasets, (list) => {
    if (datasetRestored.value || !list.length) return
    datasetRestored.value = true
    const saved = localStorage.getItem(STORAGE_KEY_DATASET)
    if (saved && list.some((d) => d.name === saved)) dataset.value = saved
  })

  // Refresh the model registry and re-pull launch data. `extraRefetchKeys`
  // lets callers extend the refetch set (e.g. TaskLaunch's dataset-metadata).
  async function refreshAll(extraRefetchKeys: string[] = []) {
    refreshing.value = true
    initDone.value = false

    try {
      await refreshRegistry()
    } catch (err: any) {
      if (!err?.response) {
        ElMessage.error(t(opts.registryFailedKey))
      }
      refreshing.value = false
      return
    }

    await queryClient.refetchQueries({ queryKey: [opts.initQueryKey] })
    await queryClient.refetchQueries({ queryKey: ['model-params'] })
    await queryClient.refetchQueries({ queryKey: ['datasets'] })
    for (const key of extraRefetchKeys) {
      await queryClient.refetchQueries({ queryKey: [key] })
    }

    refreshing.value = false
    ElMessage.success(t(opts.refreshedKey))
  }

  // K-fold metadata (TaskLaunch only) is exposed via the same query pattern.
  const datasetMetaQuery = (name: Ref<string>) =>
    useQuery({
      queryKey: computed(() => ['dataset-metadata', name.value]),
      queryFn: () => getDatasetMetadata(name.value),
      enabled: computed(() => !!name.value),
    })

  return {
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
  }
}
