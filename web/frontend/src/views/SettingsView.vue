<template>
  <div class="settings-view">
    <div class="page-header">
      <h1 class="page-title">{{ t('route.title.settings') }}</h1>
      <p class="page-subtitle">{{ t('settings.pageSubtitle') }}</p>
    </div>

    <el-skeleton :loading="loading" animated>
      <template #template>
        <div class="settings-card">
          <div class="card-section">
            <div class="section-header">
              <div class="section-title-row">
                <el-skeleton-item variant="text" style="width:140px;height:16px" />
              </div>
              <el-skeleton-item variant="text" style="width:80%;height:12px;margin-top:6px" />
            </div>
            <div class="setting-row">
              <div class="setting-info">
                <el-skeleton-item variant="text" style="width:100px;height:13px" />
                <el-skeleton-item variant="text" style="width:160px;height:11px;margin-top:3px" />
              </div>
              <el-skeleton-item variant="rect" style="width:140px;height:32px;border-radius:var(--radius-sm)" />
            </div>
          </div>
        </div>
      </template>
      <template #default>
    <div class="settings-card">
      <div class="card-section">
        <div class="section-header">
          <div class="section-title-row">
            <span class="section-icon">
              <el-icon :size="16"><Monitor /></el-icon>
            </span>
            <span class="section-title">{{ t('settings.queueTitle') }}</span>
          </div>
          <span class="section-desc">{{ concurrencyDesc }}</span>
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <span class="setting-key">{{ concurrencyLabel }}</span>
            <span class="setting-help">{{ concurrencyHelp }}</span>
          </div>
          <div class="setting-control">
            <el-input-number
              v-model="gpuSlots"
              :min="1"
              :max="16"
              controls-position="right"
              style="width: 140px"
            />
          </div>
        </div>

        <div class="card-footer">
          <el-button type="primary" :loading="saving" @click="onSave">
            {{ saving ? t('settings.saving') : t('settings.saveSettings') }}
          </el-button>
          <span v-if="saved" class="saved-hint">{{ t('settings.saved') }}</span>
        </div>
      </div>
    </div>
      </template>
    </el-skeleton>

    <div class="settings-card" v-if="!loading">
      <div class="card-section">
        <div class="section-header">
          <div class="section-title-row">
            <span class="section-icon">
              <el-icon :size="16"><Cpu /></el-icon>
            </span>
            <span class="section-title">{{ t('settings.envTitle') }}</span>
          </div>
          <span class="section-desc">{{ t('settings.envDesc') }}</span>
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <span class="setting-key">{{ t('settings.defaultEnvLabel') }}</span>
            <span class="setting-help">{{ t('settings.defaultEnvHelp') }}</span>
          </div>
          <div class="setting-control">
            <el-select
              v-model="selectedEnvId"
              :placeholder="t('settings.envPlaceholder')"
              style="width: 220px"
              @change="onEnvChange"
            >
              <el-option
                v-for="env in envList"
                :key="env.id"
                :label="env.display_name"
                :value="env.id"
              />
            </el-select>
            <el-button
              size="small"
              :loading="healthChecking"
              :disabled="!selectedEnvId"
              @click="onHealthCheck"
              style="margin-left: 8px"
            >
              {{ healthChecking ? t('settings.checking') : t('settings.checkEnv') }}
            </el-button>
          </div>
        </div>
        <div v-if="healthResult" class="health-result">
          <div class="health-item">
            <span class="health-label">Python</span>
            <span v-if="healthResult.python_available" class="health-ok">
              <el-icon :size="12"><CircleCheck /></el-icon>
              {{ healthResult.python_version || t('settings.available') }}
            </span>
            <span v-else class="health-fail">
              <el-icon :size="12"><CircleClose /></el-icon>
              {{ healthResult.error || t('settings.unavailable') }}
            </span>
          </div>
          <div class="health-item">
            <span class="health-label">PyTorch</span>
            <span v-if="healthResult.torch_available" class="health-ok">
              <el-icon :size="12"><CircleCheck /></el-icon>
              {{ healthResult.torch_version }}
            </span>
            <span v-else class="health-fail">
              <el-icon :size="12"><CircleClose /></el-icon>
              {{ t('settings.unavailable') }}
            </span>
          </div>
        </div>
        <div v-if="selectedEnvId === 'custom:0'" class="custom-path-row">
          <div class="setting-info">
            <span class="setting-key">{{ t('settings.pythonPathLabel') }}</span>
            <span class="setting-help">{{ t('settings.pythonPathHelp') }}</span>
          </div>
          <el-input
            v-model="customPythonPath"
            placeholder="/path/to/python"
            style="max-width: 400px"
            @blur="onCustomPathBlur"
          />
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <span class="setting-key">{{ t('settings.rememberLastEnvLabel') }}</span>
            <span class="setting-help">{{ t('settings.rememberLastEnvHelp') }}</span>
          </div>
          <div class="setting-control">
            <el-switch
              v-model="rememberLastEnv"
              @change="onRememberLastEnvChange"
            />
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { useQuery, useMutation, useQueryClient } from '@tanstack/vue-query'
import { ElMessage } from 'element-plus'
import { Monitor, Cpu } from '@element-plus/icons-vue'
import { CircleCheck, CircleClose } from '@element-plus/icons-vue'
import Cookies from 'universal-cookie'
import { getSettings, updateSettings, getDefaultEnv, setDefaultEnv } from '@/api/settings'
import { listEnvironments, healthCheckEnv, type EnvironmentInfo, type EnvHealthResult } from '@/api/environments'
import { useSystemCapabilities } from '@/composables/useSystemCapabilities'

const { t } = useI18n()
const queryClient = useQueryClient()
const cookies = new Cookies()
const { hasGpu, gpuCount } = useSystemCapabilities()

const COOKIE_KEY = 'kt-settings'

const gpuSlots = ref(1)
const saved = ref(false)

const totalConcurrency = computed(
  () => (hasGpu.value ? gpuCount.value : 1) * gpuSlots.value,
)

const concurrencyLabel = computed(() =>
  hasGpu.value ? t('settings.concurrencyPerGpu') : t('settings.concurrencyTasks'),
)
const concurrencyHelp = computed(() =>
  hasGpu.value
    ? t('settings.concurrencyHelpGpu', { gpuCount: gpuCount.value, total: totalConcurrency.value })
    : t('settings.concurrencyHelpCpu'),
)
const concurrencyDesc = computed(() =>
  hasGpu.value
    ? t('settings.concurrencyDescGpu')
    : t('settings.concurrencyDescCpu'),
)

const selectedEnvId = ref<string | null>(null)
const customPythonPath = ref('')
const rememberLastEnv = ref(false)
const healthResult = ref<EnvHealthResult | null>(null)

const settingsQuery = useQuery({ queryKey: ['settings'], queryFn: getSettings })
const envsQuery = useQuery({ queryKey: ['environments'], queryFn: listEnvironments })
const defaultEnvQuery = useQuery({ queryKey: ['default-env'], queryFn: getDefaultEnv })

const loading = computed(() => settingsQuery.isPending.value || envsQuery.isPending.value || defaultEnvQuery.isPending.value)
const envList = computed(() => envsQuery.data.value ?? [])

const initDone = ref(false)
watch(
  () => ({ s: settingsQuery.data.value, e: defaultEnvQuery.data.value }),
  ({ s, e }) => {
    if (initDone.value) return
    if (s && e) {
      gpuSlots.value = s.gpu_slots
      selectedEnvId.value = e.default_env_id
      customPythonPath.value = e.custom_python_path || ''
      rememberLastEnv.value = e.remember_last_env
      initDone.value = true
    }
  },
  { immediate: true },
)

watch(() => settingsQuery.isError.value, (isError) => {
  if (isError && !initDone.value) {
    const cached = cookies.get<{ gpu_slots: number }>(COOKIE_KEY)
    if (cached?.gpu_slots) {
      gpuSlots.value = cached.gpu_slots
    }
    initDone.value = true
  }
})

const updateSettingsMutation = useMutation({
  mutationFn: (data: Parameters<typeof updateSettings>[0]) => updateSettings(data),
  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: ['settings'] })
    cookies.set(COOKIE_KEY, { gpu_slots: gpuSlots.value }, { maxAge: 365 * 86400, path: '/' })
    saved.value = true
    setTimeout(() => { saved.value = false }, 2000)
    ElMessage.success(t('settings.saveSuccessToast'))
  },
})

const saving = computed(() => updateSettingsMutation.isPending.value)

const onSave = () => {
  saved.value = false
  updateSettingsMutation.mutate({ gpu_slots: gpuSlots.value })
}

const setDefaultEnvMutation = useMutation({
  mutationFn: (data: Parameters<typeof setDefaultEnv>[0]) => setDefaultEnv(data),
  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: ['default-env'] })
  },
})

const onEnvChange = (envId: string) => {
  setDefaultEnvMutation.mutate({
    env_id: envId,
    custom_python_path: envId === 'custom:0' ? customPythonPath.value || null : null,
  })
  ElMessage.success(t('settings.envUpdatedToast'))
}

const onCustomPathBlur = () => {
  if (selectedEnvId.value !== 'custom:0') return
  setDefaultEnvMutation.mutate({
    env_id: selectedEnvId.value,
    custom_python_path: customPythonPath.value || null,
  })
}

const healthCheckMutation = useMutation({
  mutationFn: (data: Parameters<typeof healthCheckEnv>[0]) => healthCheckEnv(data),
  onSuccess: (result) => {
    healthResult.value = result
  },
})

const healthChecking = computed(() => healthCheckMutation.isPending.value)

const onHealthCheck = () => {
  if (!selectedEnvId.value) return
  healthResult.value = null
  healthCheckMutation.mutate({
    env_id: selectedEnvId.value,
    custom_python_path: selectedEnvId.value === 'custom:0' ? customPythonPath.value || null : null,
  })
}

const onRememberLastEnvChange = (val: boolean) => {
  setDefaultEnvMutation.mutate({
    env_id: selectedEnvId.value || '',
    remember_last_env: val,
  })
}
</script>

<style scoped>
.settings-view {
  display: flex;
  flex-direction: column;
  gap: 24px;
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

.settings-card {
  background: var(--bg-surface);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  overflow: hidden;
}

.card-section {
  padding: 20px 24px;
}

.section-header {
  margin-bottom: 20px;
}

.section-title-row {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 6px;
}

.section-icon {
  display: flex;
  align-items: center;
  color: var(--text-secondary);
}

.section-title {
  font-size: 15px;
  font-weight: 600;
  color: var(--text-primary);
}

.section-desc {
  font-size: 13px;
  color: var(--text-tertiary);
  line-height: 1.5;
}

.setting-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 16px;
  background: var(--bg-elevated);
  border-radius: var(--radius-md);
}

.setting-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.setting-key {
  font-size: 13px;
  font-weight: 500;
  color: var(--text-primary);
}

.setting-help {
  font-size: 12px;
  color: var(--text-tertiary);
}

.card-footer {
  display: flex;
  align-items: center;
  gap: 12px;
  margin-top: 16px;
}

.saved-hint {
  font-size: 12px;
  color: var(--accent-green);
}

.custom-path-row {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 12px 16px;
  background: var(--bg-elevated);
  border-radius: var(--radius-md);
}

.setting-control {
  flex-shrink: 0;
  display: flex;
  align-items: center;
}

.health-result {
  display: flex;
  gap: 16px;
  padding: 10px 16px;
  background: var(--bg-elevated);
  border-radius: var(--radius-md);
  margin-top: 8px;
}

.health-item {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 12px;
}

.health-label {
  color: var(--text-tertiary);
  font-weight: 500;
}

.health-ok {
  color: var(--accent-green);
  display: flex;
  align-items: center;
  gap: 3px;
}

.health-fail {
  color: var(--accent-red);
  display: flex;
  align-items: center;
  gap: 3px;
}
</style>
