<template>
  <ParamGroups
    :groups="groups"
    :model-value="modelValue"
    @update:model-value="$emit('update:modelValue', $event)"
  />
</template>

<script setup lang="ts">
import { computed } from 'vue'
import ParamGroups from './ParamGroups.vue'
import type { ParamGroupDef } from './param-groups'
import type { ParamGroup } from '@/api/schemas'

const props = defineProps<{
  schema: ParamGroup[]
  modelValue: Record<string, any>
}>()

defineEmits<{ 'update:modelValue': [val: Record<string, any>] }>()

// Backend ParamGroup -> frontend ParamGroupDef. UI hints (min/max/placeholder/…)
// are intentionally absent — this mirrors ParamForm (schema-driven, no hints).
const groups = computed<ParamGroupDef[]>(() =>
  props.schema.map((group) => ({
    name: group.group_name,
    fields: Object.entries(group.params).map(([key, def]) => ({
      key,
      type: def.type,
      help: def.help,
      default: def.default,
      choices: def.choices ?? undefined,
    })),
  }))
)
</script>
