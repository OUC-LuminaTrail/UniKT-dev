<template>
  <div class="task-launch">
    <div class="task-launch-body">
      <div class="page-header">
        <h1 class="page-title">新建训练任务</h1>
        <p class="page-subtitle">{{ step === 'select' ? '选择运行环境、模型和数据集' : '调整模型参数并开始训练' }}</p>
      </div>

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
            <svg width="14" height="14" viewBox="0 0 12 12" fill="none" style="margin-right:4px">
              <path d="M7.5 2L3.5 6L7.5 10" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            返回选择
          </el-button>
        </div>

        <ParamForm
          v-if="selectionSchema"
          :schema="selectionSchema"
          @update:params="params = $event"
        />
      </div>
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
        :disabled="!modelName || !dataset"
        @click="onSelectConfirm"
      >
        确认选择
      </el-button>
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
import { ref, computed, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { createTask } from '@/api/tasks'
import { listEnvironments, type EnvironmentInfo } from '@/api/environments'
import { listModels, getModelParams, type ModelSchema } from '@/api/schemas'
import CommandPreview from '@/components/task/CommandPreview.vue'
import SelectionStep from '@/components/task/SelectionStep.vue'
import ParamForm from '@/components/task/ParamForm.vue'

const router = useRouter()
const step = ref<'select' | 'params'>('select')
const submitting = ref(false)
const selectionSchema = ref<ModelSchema | null>(null)
const params = ref<Record<string, any>>({})

const environments = ref<EnvironmentInfo[]>([])
const models = ref<string[]>([])
const datasets = ref<string[]>([])

const envId = ref('')
const customPythonPath = ref('')
const modelName = ref('')
const dataset = ref('')

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
  if (!modelName.value || !dataset.value) return
  localStorage.setItem(STORAGE_KEY_ENV, envId.value)
  localStorage.setItem(STORAGE_KEY_DATASET, dataset.value)
  step.value = 'params'
}

async function onStartTraining() {
  submitting.value = true
  try {
    const taskName = `${modelName.value}_${dataset.value}`
    const taskParams = { ...params.value, dataset: dataset.value }
    const task = await createTask({
      name: taskName,
      env_id: envId.value,
      custom_python_path: customPythonPath.value || null,
      model_name: modelName.value,
      params: taskParams,
    })
    ElMessage.success('任务已创建')
    router.push(`/tasks/${task.id}`)
  } catch (e: any) {
    ElMessage.error(e?.response?.data?.detail || '创建失败')
  } finally {
    submitting.value = false
  }
}

onMounted(async () => {
  const [envs, modelList] = await Promise.all([listEnvironments(), listModels()])
  environments.value = envs
  models.value = modelList

  const savedEnv = localStorage.getItem(STORAGE_KEY_ENV)
  if (savedEnv && envs.some(e => e.id === savedEnv)) envId.value = savedEnv

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
</style>