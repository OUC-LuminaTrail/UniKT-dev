<template>
  <div class="search-runconfig">
    <!-- Search space: derived from the model config's optuna metadata, read-only. -->
    <div v-if="searchSpace.length" class="search-space">
      <div class="space-head">
        <span class="space-title">{{ t('search.searchSpace') }}</span>
        <el-tooltip :content="t('search.searchSpaceHint')" placement="top">
          <el-icon :size="13" class="space-hint"><InfoFilled /></el-icon>
        </el-tooltip>
      </div>
      <div class="space-grid">
        <div v-for="s in searchSpace" :key="s.key" class="space-item">
          <span class="space-key">{{ s.key }}</span>
          <span class="space-range">{{ s.range }}</span>
        </div>
      </div>
    </div>

    <!-- Editable RunConfig knobs (dataset is chosen in the selection step;
         optuna-searched fields are excluded since the sampler overrides them). -->
    <ParamGroups :groups="editableGroups" v-model="formData" />
  </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import { InfoFilled } from '@element-plus/icons-vue'
import ParamGroups from './ParamGroups.vue'
import type { ParamGroupDef } from './param-groups'
import type { ModelSchema } from '@/api/schemas'

const { t } = useI18n()
const props = defineProps<{ schema: ModelSchema }>()
const emit = defineEmits<{ (e: 'update:params', params: Record<string, any>): void }>()

const rangeText = (spec: Record<string, any>): string => {
  if (spec.type === 'categorical') return `[${(spec.choices || []).join(', ')}]`
  let txt = `[${spec.low}, ${spec.high}]`
  if (spec.log) txt += ' · log'
  if (spec.step) txt += ` · step ${spec.step}`
  return txt
}

const searchSpace = computed(() => {
  const out: { key: string; range: string }[] = []
  for (const group of props.schema.param_groups) {
    for (const [key, def] of Object.entries(group.params)) {
      if (def.optuna) out.push({ key, range: rangeText(def.optuna) })
    }
  }
  return out
})

const editableGroups = computed<ParamGroupDef[]>(() =>
  props.schema.param_groups
    .map((group) => ({
      name: group.group_name,
      fields: Object.entries(group.params)
        .filter(([key, def]) => key !== 'dataset' && !def.optuna)
        .map(([key, def]) => ({
          key,
          type: def.type,
          help: def.help,
          default: def.default,
          choices: def.choices ?? undefined,
        })),
    }))
    .filter((g) => g.fields.length > 0)
)

const formData = ref<Record<string, any>>({})

const emitParams = (data: Record<string, any>) => {
  const cleaned: Record<string, any> = {}
  for (const [k, v] of Object.entries(data)) {
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
    for (const group of schema.param_groups) {
      for (const [key, def] of Object.entries(group.params)) {
        if (key === 'dataset' || def.optuna) continue
        data[key] = def.default
      }
    }
    formData.value = data
    emitParams(data)
  },
  { immediate: true }
)

watch(formData, (data) => emitParams(data), { deep: true })
</script>

<style scoped>
.search-runconfig {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.search-space {
  background: color-mix(in srgb, var(--accent-blue) 6%, var(--bg-surface));
  border: 1px solid color-mix(in srgb, var(--accent-blue) 25%, var(--border-default));
  border-radius: var(--radius-lg);
  padding: 14px 20px;
}

.space-head {
  display: flex;
  align-items: center;
  gap: 6px;
  margin-bottom: 10px;
}

.space-title {
  font-size: 13px;
  font-weight: 600;
  color: var(--accent-blue);
}

.space-hint {
  color: var(--text-tertiary);
  cursor: help;
}

.space-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
  gap: 8px 20px;
}

.space-item {
  display: flex;
  align-items: baseline;
  gap: 8px;
}

.space-key {
  font-family: var(--font-mono);
  font-size: 12.5px;
  color: var(--text-primary);
  font-weight: 500;
}

.space-range {
  font-family: var(--font-mono);
  font-size: 12px;
  color: var(--text-tertiary);
}
</style>
