<template>
  <div class="settings-view">
    <div class="page-header">
      <h1 class="page-title">设置</h1>
      <p class="page-subtitle">配置训练任务的全局参数</p>
    </div>

    <div class="settings-card">
      <div class="card-section">
        <div class="section-header">
          <div class="section-title-row">
            <span class="section-icon">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="3" width="20" height="14" rx="2" ry="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/></svg>
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
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import Cookies from 'universal-cookie'
import { getSettings, updateSettings } from '@/api/settings'

const cookies = new Cookies()

const COOKIE_KEY = 'kt-settings'

const maxConcurrent = ref(1)
const saving = ref(false)
const saved = ref(false)

onMounted(async () => {
  try {
    const s = await getSettings()
    maxConcurrent.value = s.max_concurrent
  } catch {
    const cached = cookies.get<{ max_concurrent: number }>(COOKIE_KEY)
    if (cached?.max_concurrent) {
      maxConcurrent.value = cached.max_concurrent
    }
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
  } catch (e: any) {
    ElMessage.error(e?.response?.data?.detail || '保存失败')
  } finally {
    saving.value = false
  }
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

.setting-control {
  flex-shrink: 0;
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
</style>
