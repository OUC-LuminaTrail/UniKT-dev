<template>
  <div class="task-launch">
    <div class="task-launch-body">
      <div class="page-header">
        <h1 class="page-title">新建训练任务</h1>
        <p class="page-subtitle">{{ step === 'select' ? '选择运行环境、模型和数据集' : '调整模型参数并开始训练' }}</p>
      </div>

      <div v-if="loading" class="selection-step-skeleton">
        <div class="sk-section">
          <div class="sk-bar" style="width:80px;height:14px;border-radius:4px;margin-bottom:10px" />
          <div class="sk-bar" style="width:240px;height:32px;border-radius:var(--radius-sm)" />
        </div>
        <div class="sk-section">
          <div class="sk-bar" style="width:48px;height:14px;border-radius:4px;margin-bottom:10px" />
          <div class="sk-card-grid">
            <div class="sk-card" v-for="i in 6" :key="'m'+i">
              <div class="sk-bar" style="width:40px;height:40px;border-radius:var(--radius-sm)" />
              <div class="sk-bar" style="width:50px;height:12px;border-radius:4px" />
            </div>
          </div>
        </div>
        <div class="sk-section">
          <div class="sk-bar" style="width:60px;height:14px;border-radius:4px;margin-bottom:10px" />
          <div class="sk-card-grid">
            <div class="sk-card" v-for="i in 4" :key="'d'+i">
              <div class="sk-bar" style="width:40px;height:40px;border-radius:var(--radius-sm)" />
              <div class="sk-bar" style="width:70px;height:12px;border-radius:4px" />
            </div>
          </div>
        </div>
      </div>
      <template v-else>
        <SelectionStep
          v-if="step === 'select'"
          :envId="envId"
          :customPythonPath="customPythonPath"
          :modelName="modelName"
          :dataset="dataset"
          :environments="environments"
          :models="models"
          :datasets="datasets"
          @update:envId="envId = $event"
          @update:customPythonPath="customPythonPath = $event"
          @update:modelName="onModelChange"
          @update:dataset="dataset = $event"
          @confirm="onSelectConfirm"
        />

        <div v-if="step === 'params'" class="params-step">
          <div class="params-header">
            <el-button class="back-btn" @click="step = 'select'">
              <el-icon :size="14" style="margin-right:4px"><ArrowLeft /></el-icon>
              返回选择
            </el-button>
          </div>

          <ParamForm
            v-if="selectionSchema"
            :schema="selectionSchema"
            @update:params="params = $event"
          />
        </div>
      </template>

    </div>

    <CommandPreview
      :modelName="modelName"
      :dataset="dataset"
      :params="params"
      :schemaDefaultParams="schemaDefaultParams"
    >
      <el-button
        v-if="step === 'select'"
        type="primary"
        size="large"
        :disabled="!envId || !modelName || !dataset"
        @click="onSelectConfirm"
      >
        确认选择
      </el-button>
      <el-select
        v-if="step === 'params' && showKfoldSelector"
        v-model="selectedFolds"
        multiple
        collapse-tags
        collapse-tags-tooltip
        placeholder="多折训练"
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
        <span v-if="!submitting">开始训练</span>
        <span v-else>创建任务...</span>
      </el-button>
    </CommandPreview>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, watch, h } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage, ElMessageBox } from 'element-plus'
import { ArrowLeft } from '@element-plus/icons-vue'
import { createTask } from '@/api/tasks'
import { getDatasetMetadata } from '@/api/datasets'
import { listEnvironments, type EnvironmentInfo } from '@/api/environments'
import { listModels, getModelParams, type ModelSchema } from '@/api/schemas'
import { getDefaultEnv } from '@/api/settings'
import CommandPreview from '@/components/task/CommandPreview.vue'
import SelectionStep from '@/components/task/SelectionStep.vue'
import ParamForm from '@/components/task/ParamForm.vue'

const router = useRouter()
const step = ref<'select' | 'params'>('select')
const submitting = ref(false)
const loading = ref(true)
const selectionSchema = ref<ModelSchema | null>(null)
const params = ref<Record<string, any>>({})

const environments = ref<EnvironmentInfo[]>([])
const models = ref<string[]>([])
const datasets = ref<string[]>([])

const envId = ref('')
const customPythonPath = ref('')
const modelName = ref('')
const dataset = ref('')

const kfoldCount = ref<number | null>(null)
const selectedFolds = ref<number[]>([])
const showKfoldSelector = computed(() => (kfoldCount.value ?? 0) >= 2)

const STORAGE_KEY_ENV = 'kt-web:last-env-id'
const STORAGE_KEY_MODEL = 'kt-web:last-model-name'
const STORAGE_KEY_DATASET = 'kt-web:last-dataset'

const schemaDefaultParams = computed(() => {
  if (!selectionSchema.value) return {}
  const defaults: Record<string, any> = {}
  for (const g of selectionSchema.value.param_groups) {
    for (const [k, f] of Object.entries(g.params)) {
      defaults[k] = f.default
    }
  }
  return defaults
})

async function onModelChange(val: string) {
  modelName.value = val
  if (!val) return
  localStorage.setItem(STORAGE_KEY_MODEL, val)
  selectionSchema.value = await getModelParams(val)
}

function onSelectConfirm() {
  if (!envId.value || !modelName.value || !dataset.value) return
  localStorage.setItem(STORAGE_KEY_ENV, envId.value)
  localStorage.setItem(STORAGE_KEY_DATASET, dataset.value)
  step.value = 'params'
}

watch(dataset, async (name) => {
  if (!name) {
    kfoldCount.value = null
    selectedFolds.value = []
    return
  }
  try {
    const meta = await getDatasetMetadata(name)
    const kfold = typeof meta.kfold_n_splits === 'number' ? meta.kfold_n_splits : null
    kfoldCount.value = kfold
    selectedFolds.value = kfold && kfold >= 2 ? [0] : []
  } catch {
    kfoldCount.value = null
    selectedFolds.value = []
  }
})

function buildTaskName(fold?: number) {
  return fold === undefined
    ? `${modelName.value}_${dataset.value}`
    : `${modelName.value}_${dataset.value}_fold${fold}`
}

async function confirmMultiFold(): Promise<void> {
  const folds = [...selectedFolds.value].sort((a, b) => a - b)
  const message = h('div', { style: 'line-height: 1.8' }, [
    h('div', `模型：${modelName.value}`),
    h('div', `数据集：${dataset.value}`),
    h('div', `折号：${folds.join(', ')}`),
    h('div', `任务数量：${folds.length}`),
  ])
  await ElMessageBox.confirm(message, '确认多折训练任务', {
    confirmButtonText: '创建任务',
    cancelButtonText: '取消',
    type: 'warning',
  })
}

async function onStartTraining() {
  submitting.value = true
  try {
    const taskParams = { ...params.value, dataset: dataset.value }

    if (showKfoldSelector.value && selectedFolds.value.length > 0) {
      await confirmMultiFold()
      let created = 0
      for (const fold of selectedFolds.value) {
        await createTask({
          name: buildTaskName(fold),
          env_id: envId.value,
          custom_python_path: customPythonPath.value || null,
          model_name: modelName.value,
          params: { ...taskParams, fold },
        })
        created += 1
      }
      ElMessage.success(`已创建 ${created} 个任务`)
      router.replace({ name: 'tasks' })
      return
    }

    const task = await createTask({
      name: buildTaskName(),
      env_id: envId.value,
      custom_python_path: customPythonPath.value || null,
      model_name: modelName.value,
      params: taskParams,
    })
    if (task.status === 'pending') {
      ElMessage.success('任务已加入队列')
      router.replace({ name: 'tasks' })
    } else {
      ElMessage.success('任务已创建')
      router.replace({ name: 'task-detail', params: { id: task.id } })
    }
  } catch (err: any) {
    if (err?.message && err?.message !== 'cancel') {
      ElMessage.error(`创建失败：${err.message}`)
    }
  } finally {
    submitting.value = false
  }
}

onMounted(async () => {
  try {
    const [envs, modelList, defaultEnv] = await Promise.all([
      listEnvironments(),
      listModels(),
      getDefaultEnv().catch(() => ({ default_env_id: null, custom_python_path: null, remember_last_env: false })),
    ])
    environments.value = envs
    models.value = modelList

    const savedEnv = localStorage.getItem(STORAGE_KEY_ENV)
    if (defaultEnv.remember_last_env && savedEnv && envs.some(e => e.id === savedEnv)) {
      envId.value = savedEnv
      if (savedEnv === 'custom:0' && defaultEnv.custom_python_path) {
        customPythonPath.value = defaultEnv.custom_python_path
      }
    } else if (defaultEnv.default_env_id && envs.some(e => e.id === defaultEnv.default_env_id)) {
      envId.value = defaultEnv.default_env_id
      if (defaultEnv.custom_python_path) {
        customPythonPath.value = defaultEnv.custom_python_path
      }
    } else if (savedEnv && envs.some(e => e.id === savedEnv)) {
      envId.value = savedEnv
    }

    const savedModel = localStorage.getItem(STORAGE_KEY_MODEL)
    if (savedModel && modelList.includes(savedModel)) {
      modelName.value = savedModel
      onModelChange(savedModel)
    }

    const anyModel = modelList[0]
    if (anyModel) {
      const s = await getModelParams(anyModel)
      for (const g of s.param_groups) {
        if (g.params['dataset']?.choices?.length) {
          datasets.value = g.params['dataset'].choices as string[]
          break
        }
      }
    }

    const savedDataset = localStorage.getItem(STORAGE_KEY_DATASET)
    if (savedDataset && datasets.value.includes(savedDataset)) {
      dataset.value = savedDataset
    }
  } finally {
    loading.value = false
  }
})
</script>

<style scoped>
.task-launch {
  display: flex;
  flex-direction: column;
  height: 100%;
}

.task-launch-body {
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

.kfold-select {
  min-width: 180px;
}
</style>