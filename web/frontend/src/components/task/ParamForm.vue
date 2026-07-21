<template>
  <ParamGroups :groups="groups" v-model="formData" />
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import ParamGroups from './ParamGroups.vue'
import type { ParamGroupDef } from './param-groups'
import type { ModelSchema } from '@/api/schemas'

// These are handled in the top bar (dataset selector / k-fold), not here.
const TOP_BAR_PARAMS = new Set(['dataset', 'fold'])

const props = defineProps<{ schema: ModelSchema }>()
const emit = defineEmits<{ (e: 'update:params', params: Record<string, any>): void }>()

const formData = ref<Record<string, any>>({})

const groups = computed<ParamGroupDef[]>(() =>
  props.schema.param_groups.map((group) => ({
    name: group.group_name,
    fields: Object.entries(group.params)
      .filter(([key]) => !TOP_BAR_PARAMS.has(key))
      .map(([key, def]) => ({
        key,
        type: def.type,
        help: def.help,
        default: def.default,
        choices: def.choices ?? undefined,
      })),
  }))
)

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
    for (const group of schema.param_groups) {
      for (const [key, field] of Object.entries(group.params)) {
        if (!TOP_BAR_PARAMS.has(key)) data[key] = field.default
      }
    }
    formData.value = data
    emitParams(data)
  },
  { immediate: true }
)

watch(formData, (data) => emitParams(data), { deep: true })
</script>
