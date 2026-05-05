<template>
  <div class="experiment-browser">
    <div class="page-header">
      <h2 class="page-title">Experiment Records</h2>
      <div class="filter-bar">
        <select v-model="filters.type" class="filter-select">
          <option value="normal">Normal Training</option>
          <option value="hyperparam_search">Hyperparameter Search</option>
          <option value="ablation">Ablation Study</option>
        </select>
        <input
          v-model="filters.model"
          placeholder="Filter by model"
          class="filter-input"
        />
        <button class="filter-btn" @click="loadExperiments">Query</button>
      </div>
    </div>

    <div class="table-wrapper">
      <table class="data-table">
        <thead>
          <tr>
            <th class="col-name">Experiment Name</th>
            <th class="col-model">Model</th>
            <th class="col-dataset">Dataset</th>
            <th class="col-time">Timestamp</th>
            <th class="col-action">Action</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="row in experiments" :key="row.path">
            <td class="col-name">{{ row.name }}</td>
            <td class="mono">{{ row.model_name }}</td>
            <td class="mono">{{ row.dataset_name }}</td>
            <td class="mono">{{ row.timestamp }}</td>
            <td>
              <button class="action-btn" @click="showDetail(row.path)">Detail</button>
            </td>
          </tr>
          <tr v-if="experiments.length === 0">
            <td colspan="5" class="empty-cell">No experiments found</td>
          </tr>
        </tbody>
      </table>
    </div>

    <el-drawer v-model="drawerVisible" title="Experiment Detail" size="50%">
      <template v-if="detail">
        <div class="detail-section">
          <div class="detail-row">
            <span class="detail-label">Name</span>
            <span class="detail-value">{{ detail.name }}</span>
          </div>
        </div>

        <h4 class="section-heading">Files</h4>
        <div class="file-list">
          <div
            v-for="f in fileTree"
            :key="f.label"
            class="file-item"
            @click="onFileClick(f)"
          >
            {{ f.label }}
          </div>
        </div>

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
.experiment-browser {
  display: flex;
  flex-direction: column;
  gap: 20px;
}

.page-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  flex-wrap: wrap;
}

.page-title {
  font-size: 20px;
  font-weight: 600;
  color: var(--text-primary);
  margin: 0;
  white-space: nowrap;
}

.filter-bar {
  display: flex;
  align-items: center;
  gap: 10px;
}

.filter-select,
.filter-input {
  background: var(--bg-surface);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-sm);
  color: var(--text-primary);
  font-family: var(--font-sans);
  font-size: 13px;
  padding: 6px 12px;
  outline: none;
  transition: border-color 0.2s;
}

.filter-select:focus,
.filter-input:focus {
  border-color: var(--accent-blue);
}

.filter-select {
  min-width: 160px;
  cursor: pointer;
}

.filter-input {
  width: 180px;
}

.filter-input::placeholder {
  color: var(--text-tertiary);
}

.filter-btn {
  background: var(--accent-blue);
  border: none;
  border-radius: var(--radius-sm);
  color: #0d1117;
  font-family: var(--font-sans);
  font-size: 13px;
  font-weight: 600;
  padding: 6px 18px;
  cursor: pointer;
  transition: opacity 0.2s;
}

.filter-btn:hover {
  opacity: 0.85;
}

.table-wrapper {
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  overflow: hidden;
}

.data-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}

.data-table thead {
  background: var(--bg-elevated);
}

.data-table th {
  color: var(--text-secondary);
  font-weight: 500;
  text-align: left;
  padding: 10px 16px;
  border-bottom: 1px solid var(--border-default);
  white-space: nowrap;
}

.data-table td {
  color: var(--text-primary);
  padding: 10px 16px;
  border-bottom: 1px solid var(--border-muted);
}

.data-table tbody tr:last-child td {
  border-bottom: none;
}

.data-table tbody tr:hover {
  background: var(--bg-elevated);
}

.col-name {
  min-width: 280px;
}

.col-model,
.col-dataset {
  width: 130px;
}

.col-time {
  width: 180px;
}

.col-action {
  width: 90px;
}

.mono {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-secondary);
}

.empty-cell {
  text-align: center;
  color: var(--text-tertiary);
  padding: 40px 16px !important;
}

.action-btn {
  background: transparent;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-sm);
  color: var(--accent-blue);
  font-family: var(--font-sans);
  font-size: 12px;
  padding: 3px 12px;
  cursor: pointer;
  transition: background 0.2s, border-color 0.2s;
}

.action-btn:hover {
  background: rgba(88, 166, 255, 0.1);
  border-color: var(--accent-blue);
}

.detail-section {
  background: var(--bg-surface);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-md);
  padding: 12px 16px;
}

.detail-row {
  display: flex;
  align-items: baseline;
  gap: 12px;
}

.detail-label {
  color: var(--text-secondary);
  font-size: 13px;
  min-width: 60px;
  flex-shrink: 0;
}

.detail-value {
  color: var(--text-primary);
  font-size: 13px;
  word-break: break-all;
}

.section-heading {
  color: var(--text-primary);
  font-size: 14px;
  font-weight: 600;
  margin: 20px 0 10px;
}

.file-list {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.file-item {
  padding: 6px 12px;
  border-radius: var(--radius-sm);
  color: var(--accent-blue);
  font-family: var(--font-mono);
  font-size: 12px;
  cursor: pointer;
  transition: background 0.15s;
}

.file-item:hover {
  background: var(--bg-elevated);
}

.file-content {
  font-family: var(--font-mono);
  font-size: 12px;
  line-height: 1.6;
  max-height: 500px;
  overflow: auto;
  background: var(--bg-base);
  color: var(--text-primary);
  padding: 16px;
  border-radius: var(--radius-md);
  border: 1px solid var(--border-default);
  margin: 0;
  white-space: pre-wrap;
  word-break: break-all;
}
</style>
