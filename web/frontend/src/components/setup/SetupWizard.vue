<template>
  <div class="wizard-overlay">
    <div class="wizard-card">
      <div class="wizard-header">
        <div class="wizard-icon">KT</div>
        <h2 class="wizard-title">欢迎使用 KT 实验管理平台</h2>
        <p class="wizard-desc">请选择默认的训练环境，后续可在设置中修改</p>
      </div>

      <div class="wizard-loading" v-if="loading">
        <el-icon class="is-loading" :size="24"><Loading /></el-icon>
        <span>正在扫描可用环境...</span>
      </div>

      <div class="wizard-envs" v-else-if="envs.length > 0">
        <div
          v-for="env in envs"
          :key="env.id"
          class="env-card"
          :class="{ selected: selected === env.id }"
          @click="selected = env.id"
        >
          <div class="env-type-badge">{{ env.type }}</div>
          <div class="env-name">{{ env.display_name }}</div>
        </div>
        <div v-if="selected === 'custom:0'" class="custom-path-row">
          <el-input
            v-model="customPath"
            placeholder="/path/to/python"
          />
        </div>
      </div>

      <div class="wizard-empty" v-else>
        <p>未检测到可用环境</p>
        <p class="wizard-empty-sub">请确保已安装 pixi 或 conda</p>
      </div>

      <div class="wizard-actions">
        <el-button @click="emit('skip')">跳过</el-button>
        <el-button
          type="primary"
          :disabled="!selected"
          :loading="submitting"
          @click="onComplete"
        >
          完成设置
        </el-button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { Loading } from '@element-plus/icons-vue'
import { listEnvironments, type EnvironmentInfo } from '@/api/environments'
import { setDefaultEnv } from '@/api/settings'

const emit = defineEmits<{
  skip: []
  done: []
}>()

const envs = ref<EnvironmentInfo[]>([])
const selected = ref<string | null>(null)
const customPath = ref('')
const loading = ref(true)
const submitting = ref(false)

onMounted(async () => {
  try {
    envs.value = await listEnvironments()
  } catch {} finally {
    loading.value = false
  }
})

const onComplete = async () => {
  if (!selected.value) return
  submitting.value = true
  try {
    await setDefaultEnv({
      env_id: selected.value,
      custom_python_path: selected.value === 'custom:0' ? customPath.value || null : null,
    })
    emit('done')
  } catch {} finally {
    submitting.value = false
  }
}
</script>

<style scoped>
.wizard-overlay {
  position: fixed;
  inset: 0;
  z-index: 9999;
  background: rgba(0, 0, 0, 0.5);
  display: flex;
  align-items: center;
  justify-content: center;
}

.wizard-card {
  background: var(--bg-surface);
  border-radius: var(--radius-lg);
  border: 1px solid var(--border-default);
  padding: 32px;
  width: 520px;
  max-width: 90vw;
  max-height: 80vh;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 24px;
}

.wizard-header {
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.wizard-icon {
  width: 48px;
  height: 48px;
  border-radius: var(--radius-md);
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 16px;
  color: #fff;
  font-family: var(--font-mono);
  margin-bottom: 4px;
}

.wizard-title {
  font-size: 20px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
}

.wizard-desc {
  font-size: 13px;
  color: var(--text-tertiary);
  margin: 0;
}

.wizard-loading {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  padding: 32px;
  color: var(--text-secondary);
  font-size: 13px;
}

.wizard-envs {
  display: flex;
  flex-direction: column;
  gap: 8px;
  max-height: 300px;
  overflow-y: auto;
}

.env-card {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 16px;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  cursor: pointer;
  transition: all 0.15s ease;
}

.env-card:hover {
  border-color: var(--accent-blue);
  background: var(--bg-overlay);
}

.env-card.selected {
  border-color: var(--accent-blue);
  background: rgba(88, 166, 255, 0.08);
}

.env-type-badge {
  font-size: 10px;
  font-weight: 600;
  font-family: var(--font-mono);
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--accent-cyan);
  background: var(--bg-elevated);
  padding: 2px 8px;
  border-radius: var(--radius-sm);
  border: 1px solid var(--border-default);
  flex-shrink: 0;
}

.env-name {
  font-size: 13px;
  color: var(--text-primary);
  font-weight: 500;
}

.wizard-empty {
  text-align: center;
  padding: 24px;
  color: var(--text-secondary);
  font-size: 14px;
}

.wizard-empty-sub {
  font-size: 12px;
  color: var(--text-tertiary);
  margin-top: 4px;
}

.wizard-actions {
  display: flex;
  justify-content: flex-end;
  gap: 8px;
}

.custom-path-row {
  margin-top: 4px;
}
</style>
