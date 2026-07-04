<template>
  <div class="param-form">
    <div
      v-for="group in schema.param_groups"
      :key="group.group_name"
      class="param-group"
    >
      <div class="group-header" @click="toggleGroup(group.group_name)">
        <div class="group-header-left">
          <span class="group-chevron" :class="{ expanded: isExpanded(group.group_name) }">
            <el-icon :size="12"><ArrowRight /></el-icon>
          </span>
          <span class="group-name">{{ group.group_name }}</span>
          <span class="group-count">{{ visibleParams(group).length }}</span>
        </div>
      </div>

      <transition name="collapse">
        <div v-if="isExpanded(group.group_name)" class="group-body">
          <div class="fields-grid">
            <div
              v-for="field in visibleParams(group)"
              :key="field.key"
              class="field-item"
            >
              <div class="field-top-row">
                <span class="field-key">{{ field.key }}</span>
                <span class="type-badge">{{ field.def.type }}</span>
              </div>
              <span v-if="field.def.help" class="field-help">{{ field.def.help }}</span>

              <div class="field-input">
                <template v-if="field.def.type === 'bool'">
                  <el-switch v-model="formData[field.key]" />
                </template>

                <template v-else-if="field.def.choices && field.def.choices.length > 0">
                  <el-select v-model="formData[field.key]" :placeholder="field.def.help || 'Select'" clearable style="width: 100%">
                    <el-option v-for="choice in field.def.choices" :key="choice" :label="String(choice)" :value="choice" />
                  </el-select>
                </template>

                <template v-else-if="field.def.type === 'int'">
                  <el-input-number
                    v-model="formData[field.key]"
                    controls-position="right"
                    style="width: 100%"
                  />
                </template>

                <template v-else-if="field.def.type === 'float'">
                  <el-input-number
                    v-model="formData[field.key]"
                    :precision="6"
                    :step="0.001"
                    controls-position="right"
                    style="width: 100%"
                  />
                </template>

                <template v-else-if="field.def.type === 'list'">
                  <el-input
                    :model-value="formatList(formData[field.key])"
                    @update:model-value="(v: string) => (formData[field.key] = parseList(v))"
                    :placeholder="field.def.help || field.key"
                  />
                </template>

                <template v-else>
                  <el-input v-model="formData[field.key]" :placeholder="field.def.help || field.key" />
                </template>
              </div>

              <div v-if="field.def.default !== null && field.def.default !== undefined" class="field-default">
                default: <span class="default-val">{{ field.def.default }}</span>
              </div>
            </div>
          </div>
        </div>
      </transition>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, watch } from 'vue'
import { ArrowRight } from '@element-plus/icons-vue'
import type { ModelSchema, ParamField } from '@/api/schemas'

interface FieldEntry {
  key: string
  def: ParamField
}

const TOP_BAR_PARAMS = new Set(['dataset', 'fold'])

const props = defineProps<{ schema: ModelSchema }>()
const emit = defineEmits<{ (e: 'update:params', params: Record<string, any>): void }>()

const expandedGroups = ref<string[]>([])
const formData = ref<Record<string, any>>({})

// List params round-trip as JS arrays (so the backend expands them into
// separate CLI tokens); the input field shows them comma/space-separated.
const formatList = (v: unknown): string => {
  if (Array.isArray(v)) return v.join(', ')
  return v == null ? '' : String(v)
}

const parseList = (v: string): any[] =>
  v
    .split(/[\s,]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map((s) => (Number.isFinite(Number(s)) ? Number(s) : s))

const visibleParams = (group: { params: Record<string, ParamField> }): FieldEntry[] => {
  return Object.entries(group.params)
    .filter(([key]) => !TOP_BAR_PARAMS.has(key))
    .map(([key, def]) => ({ key, def }))
}

const isExpanded = (name: string) => expandedGroups.value.includes(name)

const toggleGroup = (name: string) => {
  const idx = expandedGroups.value.indexOf(name)
  if (idx >= 0) {
    expandedGroups.value.splice(idx, 1)
  } else {
    expandedGroups.value.push(name)
  }
}

const emitParams = (data: Record<string, any>) => {
  const cleaned: Record<string, any> = {}
  for (const [k, v] of Object.entries(data)) {
    if (TOP_BAR_PARAMS.has(k)) continue
    if (v === null || v === undefined || v === '') continue
    if (Array.isArray(v) && v.length === 0) continue
    cleaned[k] = v
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
.param-form {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.param-group {
  background: var(--bg-surface);
  border: 1px solid var(--border-default);
  border-radius: var(--radius-lg);
  overflow: hidden;
}

.group-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 20px;
  cursor: pointer;
  user-select: none;
  transition: background 0.15s ease;
}

.group-header:hover {
  background: var(--bg-elevated);
}

.group-header-left {
  display: flex;
  align-items: center;
  gap: 10px;
}

.group-chevron {
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--text-tertiary);
  transition: transform 0.2s ease;
}

.group-chevron.expanded {
  transform: rotate(90deg);
}

.group-name {
  font-size: 14px;
  font-weight: 600;
  color: var(--text-primary);
}

.group-count {
  font-size: 11px;
  font-family: var(--font-mono);
  color: var(--text-tertiary);
  background: var(--bg-elevated);
  padding: 1px 7px;
  border-radius: 10px;
  line-height: 1.6;
}

.group-body {
  padding: 4px 20px 20px;
}

.fields-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px 28px;
}

.field-item {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.field-top-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
}

.field-key {
  font-size: 13px;
  font-family: var(--font-mono);
  font-weight: 500;
  color: var(--text-primary);
  letter-spacing: 0.2px;
}

.type-badge {
  font-size: 10px;
  font-family: var(--font-mono);
  color: var(--text-tertiary);
  background: var(--bg-overlay);
  padding: 1px 6px;
  border-radius: var(--radius-sm);
  letter-spacing: 0.3px;
}

.field-help {
  font-size: 12px;
  color: var(--text-tertiary);
  line-height: 1.4;
}

.field-input {
  width: 100%;
}

.field-input :deep(.el-input__wrapper),
.field-input :deep(.el-select .el-input__wrapper) {
  height: 36px;
  border-radius: var(--radius-sm);
}

.field-input :deep(.el-input-number) {
  width: 100%;
}

.field-input :deep(.el-input-number .el-input__wrapper) {
  height: 36px;
  border-radius: var(--radius-sm);
}

.field-default {
  font-size: 11px;
  color: var(--text-tertiary);
  font-family: var(--font-mono);
}

.default-val {
  color: var(--accent-blue);
}

.collapse-enter-active,
.collapse-leave-active {
  transition: all 0.2s ease;
  overflow: hidden;
}

.collapse-enter-from,
.collapse-leave-to {
  opacity: 0;
  max-height: 0;
  padding-top: 0;
  padding-bottom: 0;
}

.collapse-enter-to,
.collapse-leave-from {
  opacity: 1;
  max-height: 2000px;
}
</style>
