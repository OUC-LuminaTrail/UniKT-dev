<template>
  <div class="param-groups">
    <div
      v-for="group in props.groups"
      :key="group.name"
      class="param-group"
    >
      <div class="group-header" @click="toggleGroup(group.name)">
        <div class="group-header-left">
          <span class="group-chevron" :class="{ expanded: isExpanded(group.name) }">
            <el-icon :size="12"><ArrowRight /></el-icon>
          </span>
          <span class="group-name">{{ group.name }}</span>
          <span class="group-count">{{ group.fields.length }}</span>
        </div>
      </div>

      <transition name="collapse">
        <div v-if="isExpanded(group.name)" class="group-body">
          <div class="fields-grid">
            <div
              v-for="field in group.fields"
              :key="field.key"
              class="field-item"
            >
              <div class="field-top-row">
                <span class="field-key">{{ field.key }}</span>
                <span class="type-badge">{{ field.type }}</span>
              </div>
              <span v-if="field.help" class="field-help">{{ field.help }}</span>

              <div class="field-input">
                <el-switch
                  v-if="field.type === 'bool'"
                  v-model="form[field.key]"
                />

                <el-select
                  v-else-if="field.choices && field.choices.length > 0"
                  v-model="form[field.key]"
                  :placeholder="field.help || 'Select'"
                  clearable
                  style="width: 100%"
                >
                  <el-option
                    v-for="choice in field.choices"
                    :key="String(choice)"
                    :label="String(choice)"
                    :value="choice"
                  />
                </el-select>

                <el-input-number
                  v-else-if="field.type === 'int'"
                  v-model="form[field.key]"
                  controls-position="right"
                  style="width: 100%"
                />

                <el-input-number
                  v-else-if="field.type === 'float'"
                  v-model="form[field.key]"
                  :step="0.001"
                  :precision="6"
                  controls-position="right"
                  style="width: 100%"
                />

                <el-input
                  v-else-if="field.type === 'list'"
                  :model-value="formatList(form[field.key])"
                  @update:model-value="(v: string) => (form[field.key] = parseList(v))"
                  :placeholder="field.help || field.key"
                />

                <el-input
                  v-else
                  v-model="form[field.key]"
                  :placeholder="field.help || field.key"
                />
              </div>

              <div v-if="field.default !== null && field.default !== undefined" class="field-default">
                default: <span class="default-val">{{ field.default }}</span>
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
import type { ParamGroupDef } from './param-groups'

const props = defineProps<{
  groups: ParamGroupDef[]
  modelValue: Record<string, any>
}>()
const emit = defineEmits<{ 'update:modelValue': [val: Record<string, any>] }>()

const expandedGroups = ref<string[]>([])
const form = ref<Record<string, any>>({ ...props.modelValue })

const sameValues = (a: Record<string, any>, b: Record<string, any>): boolean => {
  const ak = Object.keys(a)
  return ak.length === Object.keys(b).length && ak.every((k) => a[k] === b[k])
}

// Sync local form when the parent resets modelValue (e.g. action switch), but
// skip echoes of our own emit (same values) to avoid a watch loop.
watch(
  () => props.modelValue,
  (v) => {
    if (!sameValues(v, form.value)) form.value = { ...v }
  }
)
watch(form, (v) => emit('update:modelValue', { ...v }), { deep: true })

const isExpanded = (name: string) => expandedGroups.value.includes(name)

const toggleGroup = (name: string) => {
  const idx = expandedGroups.value.indexOf(name)
  if (idx >= 0) expandedGroups.value.splice(idx, 1)
  else expandedGroups.value.push(name)
}

// Expand every group by default; re-expand when the group set changes.
watch(
  () => props.groups.map((g) => g.name).join('\n'),
  (names) => {
    expandedGroups.value = names ? names.split('\n') : []
  },
  { immediate: true }
)

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
</script>

<style scoped>
.param-groups {
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
