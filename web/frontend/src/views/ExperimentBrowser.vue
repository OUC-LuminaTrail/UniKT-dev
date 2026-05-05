<template>
  <div class="experiment-browser">
    <h2>实验记录</h2>

    <el-form inline style="margin-bottom: 16px">
      <el-form-item label="类型">
        <el-select v-model="filters.type" style="width: 150px">
          <el-option label="普通训练" value="normal" />
          <el-option label="超参搜索" value="hyperparam_search" />
          <el-option label="消融实验" value="ablation" />
        </el-select>
      </el-form-item>
      <el-form-item label="模型">
        <el-input v-model="filters.model" placeholder="过滤模型" clearable />
      </el-form-item>
      <el-form-item>
        <el-button type="primary" @click="loadExperiments">查询</el-button>
      </el-form-item>
    </el-form>

    <el-table :data="experiments" stripe>
      <el-table-column prop="name" label="实验名称" min-width="300" />
      <el-table-column prop="model_name" label="模型" width="120" />
      <el-table-column prop="dataset_name" label="数据集" width="140" />
      <el-table-column prop="timestamp" label="时间" width="180" />
      <el-table-column label="操作" width="100">
        <template #default="{ row }">
          <el-button size="small" @click="showDetail(row.path)">详情</el-button>
        </template>
      </el-table-column>
    </el-table>

    <el-drawer v-model="drawerVisible" title="实验详情" size="50%">
      <template v-if="detail">
        <el-descriptions :column="1" border>
          <el-descriptions-item label="名称">{{ detail.name }}</el-descriptions-item>
        </el-descriptions>

        <h4 style="margin-top: 16px">文件列表</h4>
        <el-tree
          :data="fileTree"
          :props="{ label: 'label', children: 'children' }"
          @node-click="onFileClick"
        />

        <el-dialog v-model="fileDialogVisible" :title="currentFile.name" width="70%">
          <pre class="file-content">{{ currentFile.content }}</pre>
        </el-dialog>
      </template>
    </el-drawer>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import {
  listExperiments,
  getExperiment,
  readExperimentFile,
  type ExperimentInfo,
  type ExperimentDetail,
} from '@/api/experiments'

const filters = ref({ type: 'normal', model: '' })
const experiments = ref<ExperimentInfo[]>([])
const drawerVisible = ref(false)
const detail = ref<ExperimentDetail | null>(null)
const fileDialogVisible = ref(false)
const currentFile = ref({ name: '', content: '' })

const fileTree = ref<any[]>([])

const loadExperiments = async () => {
  experiments.value = await listExperiments({
    type: filters.value.type,
    model: filters.value.model || undefined,
  })
}

const showDetail = async (path: string) => {
  detail.value = await getExperiment(path)
  fileTree.value = detail.value.files.map((f) => ({ label: f }))
  drawerVisible.value = true
}

const onFileClick = async (node: any) => {
  if (!detail.value) return
  const data = await readExperimentFile(detail.value.path, node.label)
  currentFile.value = data
  fileDialogVisible.value = true
}

onMounted(loadExperiments)
</script>

<style scoped>
.file-content {
  font-size: 12px;
  max-height: 500px;
  overflow: auto;
  background: #1e1e1e;
  color: #d4d4d4;
  padding: 12px;
  border-radius: 4px;
}
</style>
