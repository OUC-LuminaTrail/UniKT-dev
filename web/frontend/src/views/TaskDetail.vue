<template>
  <div class="task-detail" v-if="task">
    <div class="header">
      <el-button @click="$router.push('/tasks')" text>&larr; 返回列表</el-button>
      <h2>{{ task.name }}</h2>
      <div class="actions">
        <el-tag :type="statusTagType(task.status)" size="large">{{ task.status }}</el-tag>
        <el-button v-if="task.status === 'running'" type="warning" @click="handleStop">停止</el-button>
        <el-button v-if="task.status === 'running'" type="danger" @click="handleKill">强制终止</el-button>
      </div>
    </div>

    <el-descriptions :column="3" border style="margin-bottom: 16px">
      <el-descriptions-item label="模型">{{ task.model_name }}</el-descriptions-item>
      <el-descriptions-item label="数据集">{{ task.dataset_name }}</el-descriptions-item>
      <el-descriptions-item label="环境">{{ task.env_type }}:{{ task.env_name }}</el-descriptions-item>
      <el-descriptions-item label="PID">{{ task.pid || '-' }}</el-descriptions-item>
      <el-descriptions-item label="启动时间">{{ formatTime(task.started_at) }}</el-descriptions-item>
      <el-descriptions-item label="退出码">{{ task.exit_code ?? '-' }}</el-descriptions-item>
      <el-descriptions-item label="命令" :span="3">
        <code class="command">{{ task.command }}</code>
      </el-descriptions-item>
    </el-descriptions>

    <el-card header="训练日志">
      <div class="terminal-wrapper">
        <LogTerminal :messages="logMessages" />
      </div>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute } from 'vue-router'
import { ElMessage } from 'element-plus'
import { getTask, stopTask, killTask, type TaskInfo } from '@/api/tasks'
import { useWebSocket } from '@/composables/useWebSocket'
import LogTerminal from '@/components/task/LogTerminal.vue'

const route = useRoute()
const taskId = Number(route.params.id)
const task = ref<TaskInfo | null>(null)

const statusTagType = (status: string) => {
  const map: Record<string, string> = {
    running: 'primary', completed: 'success', failed: 'danger', stopped: 'warning', pending: 'info',
  }
  return map[status] || 'info'
}

const formatTime = (t: string | null) => t ? new Date(t).toLocaleString('zh-CN') : '-'

const { messages: logMessages } = useWebSocket(`/api/tasks/${taskId}/logs/stream`)

const loadTask = async () => {
  task.value = await getTask(taskId)
}

const handleStop = async () => {
  await stopTask(taskId)
  ElMessage.success('已发送停止信号')
  setTimeout(loadTask, 1000)
}

const handleKill = async () => {
  await killTask(taskId)
  ElMessage.success('已强制终止')
  setTimeout(loadTask, 1000)
}

onMounted(loadTask)
</script>

<style scoped>
.header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 16px;
}
.header h2 { margin: 0; }
.actions { display: flex; align-items: center; gap: 8px; margin-left: auto; }
.command {
  font-size: 12px;
  word-break: break-all;
  background: #f5f7fa;
  padding: 4px 8px;
  border-radius: 4px;
}
.terminal-wrapper {
  height: 500px;
}
</style>
