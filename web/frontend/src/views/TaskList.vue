<template>
  <div class="task-list">
    <div class="header">
      <h2>训练任务</h2>
      <el-button type="primary" @click="$router.push('/tasks/new')">新建任务</el-button>
    </div>

    <el-tabs v-model="activeTab" @tab-change="loadTasks">
      <el-tab-pane label="运行中" name="running" />
      <el-tab-pane label="已完成" name="completed" />
      <el-tab-pane label="失败" name="failed" />
      <el-tab-pane label="全部" name="" />
    </el-tabs>

    <el-table :data="tasks" stripe style="width: 100%">
      <el-table-column prop="id" label="ID" width="60" />
      <el-table-column prop="name" label="名称" min-width="200" />
      <el-table-column prop="model_name" label="模型" width="120" />
      <el-table-column prop="dataset_name" label="数据集" width="140" />
      <el-table-column prop="env_name" label="环境" width="100" />
      <el-table-column prop="status" label="状态" width="100">
        <template #default="{ row }">
          <el-tag :type="statusTagType(row.status)">{{ row.status }}</el-tag>
        </template>
      </el-table-column>
      <el-table-column prop="created_at" label="创建时间" width="180">
        <template #default="{ row }">
          {{ formatTime(row.created_at) }}
        </template>
      </el-table-column>
      <el-table-column label="操作" width="200" fixed="right">
        <template #default="{ row }">
          <el-button size="small" @click="$router.push(`/tasks/${row.id}`)">详情</el-button>
          <el-button
            v-if="row.status === 'running'"
            size="small"
            type="warning"
            @click="handleStop(row.id)"
          >停止</el-button>
          <el-button
            v-if="row.status !== 'running'"
            size="small"
            type="danger"
            @click="handleDelete(row.id)"
          >删除</el-button>
        </template>
      </el-table-column>
    </el-table>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ElMessage } from 'element-plus'
import { listTasks, stopTask, deleteTask, type TaskInfo } from '@/api/tasks'

const tasks = ref<TaskInfo[]>([])
const activeTab = ref('running')

const statusTagType = (status: string) => {
  const map: Record<string, string> = {
    running: 'primary', completed: 'success', failed: 'danger', stopped: 'warning', pending: 'info',
  }
  return map[status] || 'info'
}

const formatTime = (t: string | null) => {
  if (!t) return '-'
  return new Date(t).toLocaleString('zh-CN')
}

const loadTasks = async () => {
  tasks.value = await listTasks({ status: activeTab.value || undefined })
}

const handleStop = async (id: number) => {
  await stopTask(id)
  ElMessage.success('已发送停止信号')
  loadTasks()
}

const handleDelete = async (id: number) => {
  await deleteTask(id)
  ElMessage.success('已删除')
  loadTasks()
}

onMounted(loadTasks)
</script>

<style scoped>
.header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
</style>
