<template>
  <div class="task-launch">
    <h2>新建训练任务</h2>

    <el-form :model="form" label-width="120px" style="max-width: 900px">
      <el-form-item label="任务名称">
        <el-input v-model="form.name" placeholder="例如: GIKT_assist09_fold0" />
      </el-form-item>

      <el-form-item label="运行环境">
        <el-select v-model="form.env_id" placeholder="选择环境" style="width: 100%">
          <el-option
            v-for="env in environments"
            :key="env.id"
            :label="env.display_name"
            :value="env.id"
          />
        </el-select>
        <el-input
          v-if="form.env_id === 'custom:0'"
          v-model="form.custom_python_path"
          placeholder="输入 Python 可执行文件路径"
          style="margin-top: 8px"
        />
      </el-form-item>

      <el-form-item label="选择模型">
        <el-select
          v-model="form.model_name"
          placeholder="选择模型"
          style="width: 100%"
          @change="onModelChange"
        >
          <el-option v-for="m in models" :key="m" :label="m" :value="m" />
        </el-select>
      </el-form-item>

      <el-divider v-if="currentSchema" />

      <ParamForm
        v-if="currentSchema"
        :schema="currentSchema"
        @update:params="form.params = $event"
      />

      <el-form-item style="margin-top: 20px">
        <el-button type="primary" :loading="submitting" @click="onSubmit">
          启动训练
        </el-button>
      </el-form-item>
    </el-form>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { ElMessage } from 'element-plus'
import { listEnvironments, type EnvironmentInfo } from '@/api/environments'
import { listModels, getModelParams, type ModelSchema } from '@/api/schemas'
import { createTask } from '@/api/tasks'
import ParamForm from '@/components/task/ParamForm.vue'

const router = useRouter()
const environments = ref<EnvironmentInfo[]>([])
const models = ref<string[]>([])
const currentSchema = ref<ModelSchema | null>(null)
const submitting = ref(false)

const form = ref({
  name: '',
  env_id: '',
  custom_python_path: '',
  model_name: '',
  params: {} as Record<string, any>,
})

const STORAGE_KEY_ENV = 'kt-web:last-env-id'
const STORAGE_KEY_MODEL = 'kt-web:last-model-name'

onMounted(async () => {
  const [envs, modelList] = await Promise.all([listEnvironments(), listModels()])
  environments.value = envs
  models.value = modelList

  const savedEnv = localStorage.getItem(STORAGE_KEY_ENV)
  if (savedEnv && envs.some(e => e.id === savedEnv)) {
    form.value.env_id = savedEnv
  }

  const savedModel = localStorage.getItem(STORAGE_KEY_MODEL)
  if (savedModel && modelList.includes(savedModel)) {
    form.value.model_name = savedModel
    onModelChange(savedModel)
  }
})

const onModelChange = async (model: string) => {
  currentSchema.value = await getModelParams(model)
  const dataset = form.value.params.dataset || ''
  form.value.name = `${model}_${dataset}`.replace(/_$/, '')
}

const onSubmit = async () => {
  if (!form.value.model_name) {
    ElMessage.warning('请选择模型')
    return
  }
  localStorage.setItem(STORAGE_KEY_ENV, form.value.env_id)
  localStorage.setItem(STORAGE_KEY_MODEL, form.value.model_name)
  submitting.value = true
  try {
    const task = await createTask({
      name: form.value.name || `${form.value.model_name}_task`,
      env_id: form.value.env_id,
      custom_python_path: form.value.custom_python_path || null,
      model_name: form.value.model_name,
      params: form.value.params,
    })
    ElMessage.success('任务已创建')
    router.push(`/tasks/${task.id}`)
  } catch (e: any) {
    ElMessage.error(e?.response?.data?.detail || '创建失败')
  } finally {
    submitting.value = false
  }
}
</script>

<style scoped>
.task-launch {
  max-width: 960px;
}
</style>
