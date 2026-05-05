<template>
  <div class="param-form">
    <el-card
      v-for="group in schema.param_groups"
      :key="group.group_name"
      shadow="never"
      class="group-card"
    >
      <template #header>
        <div class="group-header">
          <span class="group-name">{{ group.group_name }}</span>
          <span class="group-count">{{ Object.keys(group.params).length }} 个参数</span>
        </div>
      </template>
      <el-form label-position="left" label-width="180px" size="default">
        <el-row :gutter="20">
          <el-col
            v-for="(field, key) in group.params"
            :key="key"
            :span="colSpan(field)"
          >
            <el-form-item>
              <template #label>
                <span class="param-label">{{ formatLabel(key) }}</span>
                <span class="param-key">{{ key }}</span>
              </template>

              <template v-if="field.type === 'bool'">
                <el-switch v-model="formData[key]" />
              </template>

              <template v-else-if="field.choices && field.choices.length > 0">
                <el-select v-model="formData[key]" :placeholder="field.help" clearable style="width: 100%">
                  <el-option v-for="choice in field.choices" :key="choice" :label="String(choice)" :value="choice" />
                </el-select>
              </template>

              <template v-else-if="field.type === 'int'">
                <el-input-number
                  v-model="formData[key]"
                  controls-position="right"
                  style="width: 100%"
                />
              </template>

              <template v-else-if="field.type === 'float'">
                <el-input-number
                  v-model="formData[key]"
                  :precision="6"
                  :step="0.001"
                  controls-position="right"
                  style="width: 100%"
                />
              </template>

              <template v-else>
                <el-input v-model="formData[key]" :placeholder="field.help || key" />
              </template>

              <div class="field-meta">
                <span class="field-type">{{ field.type }}</span>
                <span v-if="field.default !== null && field.default !== undefined" class="field-default">
                  默认: {{ field.default }}
                </span>
                <span v-if="field.required" class="field-required">必填</span>
              </div>
            </el-form-item>
          </el-col>
        </el-row>
      </el-form>
    </el-card>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import type { ModelSchema, ParamField } from '@/api/schemas'

const props = defineProps<{ schema: ModelSchema }>()
const emit = defineEmits<{ (e: 'update:params', params: Record<string, any>): void }>()

const expandedGroups = ref<string[]>([])
const formData = ref<Record<string, any>>({})

const colSpan = (field: ParamField) => {
  if (field.type === 'bool') return 8
  if (field.choices && field.choices.length > 0) return 8
  return 12
}

const formatLabel = (key: string) => {
  return key
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase())
}

const emitParams = (data: Record<string, any>) => {
  const cleaned: Record<string, any> = {}
  for (const [k, v] of Object.entries(data)) {
    if (v !== null && v !== undefined && v !== '') {
      cleaned[k] = v
    }
  }
  emit('update:params', cleaned)
}

watch(
  () => props.schema,
  (schema) => {
    const data: Record<string, any> = {}
    const groups: string[] = []
    for (const group of schema.param_groups) {
      groups.push(group.group_name)
      for (const [key, field] of Object.entries(group.params)) {
        data[key] = field.default
      }
    }
    formData.value = data
    expandedGroups.value = groups
    emitParams(data)
  },
  { immediate: true }
)

watch(
  formData,
  (data) => emitParams(data),
  { deep: true }
)
</script>

<style scoped>
.group-card {
  margin-bottom: 16px;
}

.group-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.group-name {
  font-weight: 600;
  font-size: 15px;
}

.group-count {
  font-size: 12px;
  color: #909399;
}

.param-label {
  display: block;
  font-weight: 500;
  font-size: 13px;
}

.param-key {
  display: block;
  font-size: 11px;
  color: #a0a0a0;
  font-family: monospace;
}

.field-meta {
  display: flex;
  gap: 8px;
  margin-top: 4px;
  font-size: 11px;
  line-height: 1;
}

.field-type {
  color: #909399;
  font-family: monospace;
  background: #f4f4f5;
  padding: 1px 4px;
  border-radius: 3px;
}

.field-default {
  color: #909399;
}

.field-required {
  color: #e6a23c;
  font-weight: 500;
}
</style>
