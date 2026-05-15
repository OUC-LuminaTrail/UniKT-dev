<template>
  <div class="settings-view">
    <div class="page-header">
      <h1 class="page-title">设置</h1>
      <p class="page-subtitle">配置训练任务的全局参数</p>
    </div>

    <div class="settings-card" v-if="loading">
      <div class="card-section">
        <div class="section-header">
          <div class="section-title-row">
            <div class="sk-bar" style="width:140px;height:16px;border-radius:4px" />
          </div>
          <div class="sk-bar" style="width:80%;height:12px;border-radius:4px;margin-top:6px" />
        </div>
        <div class="setting-row">
          <div class="setting-info">
            <div class="sk-bar" style="width:100px;height:13px;border-radius:4px" />
            <div class="sk-bar" style="width:160px;height:11px;border-radius:4px;margin-top:3px" />
          </div>
          <div class="sk-bar" style="width:140px;height:32px;border-radius:var(--radius-sm)" />
        </div>
      </div>
    </div>

    <div class="settings-card" v-else>
      <div class="card-section">
        <div class="section-header">
          <div class="section-title-row">
            <span class="section-icon">
              <el-icon :size="16"><Monitor /></el-icon>
            </span>
            <span class="section-title">任务队列</span>
          </div>
          <span class="section-desc">控制训练任务的并发执行数量。修改后立即生效，如果增大并发数，队列中的等待任务会自动启动。</span>
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <span class="setting-key">最大并发任务数</span>
            <span class="setting-help">同时运行的训练任务上限</span>
          </div>
          <div class="setting-control">
            <el-input-number
              v-model="maxConcurrent"
              :min="1"
              :max="8"
              controls-position="right"
              style="width: 140px"
            />
          </div>
        </div>

        <div class="card-footer">
          <el-button type="primary" :loading="saving" @click="onSave">
            {{ saving ? '保存中...' : '保存设置' }}
          </el-button>
          <span v-if="saved" class="saved-hint">已保存</span>
        </div>
      </div>
    </div>

    <div class="settings-card" v-if="!loading">
      <div class="card-section">
        <div class="section-header">
          <div class="section-title-row">
            <span class="section-icon">
              <el-icon :size="16"><Cpu /></el-icon>
            </span>
            <span class="section-title">训练环境</span>
          </div>
          <span class="section-desc">选择提交训练任务时使用的默认 Python 环境。环境列表由系统自动检测。</span>
        </div>

        <div class="setting-row">
          <div class="setting-info">
            <span class="setting-key">默认训练环境</span>
            <span class="setting-help">训练任务将通过此环境执行</span>
          </div>
          <div class="setting-control">
            <el-select
              v-model="selectedEnvId"
              placeholder="选择环境"
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
              {{ healthChecking ? '检测中...' : '检测环境' }}
            </el-button>
          </div>
        </div>
        <div v-if="healthResult" class="health-result">
          <div class="health-item">
            <span class="health-label">Python</span>
            <span v-if="healthResult.python_available" class="health-ok">
              <el-icon :size="12"><CircleCheck /></el-icon>
              {{ healthResult.python_version || '可用' }}
            </span>
            <span v-else class="health-fail">
              <el-icon :size="12"><CircleClose /></el-icon>
              {{ healthResult.error || '不可用' }}
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
              不可用
            </span>
          </div>
        </div>
        <div v-if="selectedEnvId === 'custom:0'" class="custom-path-row">
          <div class="setting-info">
            <span class="setting-key">Python 路径</span>
            <span class="setting-help">自定义 Python 解释器的完整路径</span>
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
            <span class="setting-key">记住上次使用的环境</span>
            <span class="setting-help">开启后，新建训练任务时默认使用上次选择的环境，而非此处的默认环境</span>
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
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { Monitor, Cpu } from '@element-plus/icons-vue'
import { CircleCheck, CircleClose } from '@element-plus/icons-vue'
import Cookies from 'universal-cookie'
import { getSettings, updateSettings, getDefaultEnv, setDefaultEnv } from '@/api/settings'
import { listEnvironments, healthCheckEnv, type EnvironmentInfo, type EnvHealthResult } from '@/api/environments'

const cookies = new Cookies()

const COOKIE_KEY = 'kt-settings'

const maxConcurrent = ref(1)
const saving = ref(false)
const saved = ref(false)
const loading = ref(true)

const selectedEnvId = ref<string | null>(null)
const customPythonPath = ref('')
const rememberLastEnv = ref(false)
const envList = ref<EnvironmentInfo[]>([])
const healthChecking = ref(false)
const healthResult = ref<EnvHealthResult | null>(null)

onMounted(async () => {
  try {
    const [settings, envs, envRes] = await Promise.all([
      getSettings(),
      listEnvironments(),
      getDefaultEnv(),
    ])
    maxConcurrent.value = settings.max_concurrent
    envList.value = envs
    selectedEnvId.value = envRes.default_env_id
    customPythonPath.value = envRes.custom_python_path || ''
    rememberLastEnv.value = envRes.remember_last_env
  } catch {
    const cached = cookies.get<{ max_concurrent: number }>(COOKIE_KEY)
    if (cached?.max_concurrent) {
      maxConcurrent.value = cached.max_concurrent
    }
  } finally {
    loading.value = false
  }
})

const onSave = async () => {
  saving.value = true
  saved.value = false
  try {
    await updateSettings({ max_concurrent: maxConcurrent.value })
    cookies.set(COOKIE_KEY, { max_concurrent: maxConcurrent.value }, { maxAge: 365 * 86400, path: '/' })
    saved.value = true
    setTimeout(() => { saved.value = false }, 2000)
    ElMessage.success('设置已保存')
  } catch {
  } finally {
    saving.value = false
  }
}

const onEnvChange = async (envId: string) => {
  try {
    await setDefaultEnv({
      env_id: envId,
      custom_python_path: envId === 'custom:0' ? customPythonPath.value || null : null,
    })
    ElMessage.success('默认环境已更新')
  } catch {}
}

const onCustomPathBlur = async () => {
  if (selectedEnvId.value !== 'custom:0') return
  try {
    await setDefaultEnv({
      env_id: selectedEnvId.value,
      custom_python_path: customPythonPath.value || null,
    })
  } catch {}
}

const onHealthCheck = async () => {
  if (!selectedEnvId.value) return
  healthChecking.value = true
  healthResult.value = null
  try {
    healthResult.value = await healthCheckEnv({
      env_id: selectedEnvId.value,
      custom_python_path: selectedEnvId.value === 'custom:0' ? customPythonPath.value || null : null,
    })
  } catch {
    healthResult.value = null
  } finally {
    healthChecking.value = false
  }
}

const onRememberLastEnvChange = async (val: boolean) => {
  try {
    await setDefaultEnv({
      env_id: selectedEnvId.value || '',
      remember_last_env: val,
    })
  } catch {}
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
