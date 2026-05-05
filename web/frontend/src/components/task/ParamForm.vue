<template>
  <div class="param-form">
    <el-collapse v-model="expandedGroups">
      <el-collapse-item
        v-for="group in schema.param_groups"
        :key="group.group_name"
        :title="group.group_name"
        :name="group.group_name"
      >
        <el-form label-position="top" size="default">
          <el-row :gutter="16">
            <el-col :span="12" v-for="(field, key) in group.params" :key="key">
              <el-form-item :label="key" :required="field.required">
                <template v-if="field.type === 'bool'">
                  <el-switch v-model="formData[key]" :active-text="field.help" />
                </template>
                <template v-else-if="field.choices && field.choices.length > 0">
                  <el-select v-model="formData[key]" :placeholder="field.help" clearable>
                    <el-option v-for="choice in field.choices" :key="choice" :label="choice" :value="choice" />
                  </el-select>
                </template>
                <template v-else-if="field.type === 'int'">
                  <el-input-number v-model="formData[key]" :placeholder="field.help" controls-position="right" style="width: 100%" />
                </template>
                <template v-else-if="field.type === 'float'">
                  <el-input-number v-model="formData[key]" :placeholder="field.help" :precision="6" :step="0.001" controls-position="right" style="width: 100%" />
                </template>
                <template v-else>
                  <el-input v-model="formData[key]" :placeholder="field.help" />
                </template>
                <div class="field-help" v-if="field.help && field.type !== 'bool'">{{ field.help }}</div>
              </el-form-item>
            </el-col>
          </el-row>
        </el-form>
      </el-collapse-item>
    </el-collapse>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import type { ModelSchema } from '@/api/schemas'

const props = defineProps<{ schema: ModelSchema }>()
const emit = defineEmits<{ (e: 'update:params', params: Record<string, any>): void }>()

const expandedGroups = ref<string[]>([])
const formData = ref<Record<string, any>>({})

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
  },
  { immediate: true }
)

watch(
  formData,
  (data) => {
    const cleaned: Record<string, any> = {}
    for (const [k, v] of Object.entries(data)) {
      if (v !== null && v !== undefined && v !== '') {
        cleaned[k] = v
      }
    }
    emit('update:params', cleaned)
  },
  { deep: true }
)
</script>

<style scoped>
.field-help {
  font-size: 12px;
  color: #909399;
  margin-top: 2px;
  line-height: 1.4;
}
</style>
