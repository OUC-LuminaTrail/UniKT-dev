<template>
  <el-select
    :model-value="modelValue"
    @update:model-value="$emit('update:modelValue', $event)"
    :placeholder="placeholder || t('env.selectPlaceholder')"
    :clearable="clearable"
    class="env-select"
  >
    <el-option-group label="Pixi">
      <el-option
        v-for="env in pixiEnvs"
        :key="env.id"
        :label="envLabel(env)"
        :value="env.id"
      />
    </el-option-group>
    <el-option-group label="Conda" v-if="condaEnvs.length">
      <el-option
        v-for="env in condaEnvs"
        :key="env.id"
        :label="envLabel(env)"
        :value="env.id"
      />
    </el-option-group>
    <el-option-group label="Other" v-if="otherEnvs.length">
      <el-option
        v-for="env in otherEnvs"
        :key="env.id"
        :label="envLabel(env)"
        :value="env.id"
      />
    </el-option-group>
  </el-select>
  <div v-if="modelValue === 'custom:0'" class="custom-path-row">
    <el-input
      :model-value="customPath"
      @update:model-value="$emit('update:customPath', $event)"
      placeholder="/path/to/python"
    />
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import type { EnvironmentInfo } from '@/api/environments'

const { t } = useI18n()

const props = withDefaults(defineProps<{
  modelValue: string | null
  customPath: string
  environments: EnvironmentInfo[]
  placeholder?: string
  clearable?: boolean
}>(), {
  clearable: false,
})

defineEmits<{
  (e: 'update:modelValue', val: string | null): void
  (e: 'update:customPath', val: string): void
}>()

const envLabel = (env: EnvironmentInfo) =>
  env.type === 'custom' ? t('env.customPythonPath') : env.display_name

const pixiEnvs = computed(() => props.environments.filter(e => e.type === 'pixi'))
const condaEnvs = computed(() => props.environments.filter(e => e.type === 'conda'))
const otherEnvs = computed(() => props.environments.filter(e => e.type !== 'pixi' && e.type !== 'conda'))
</script>

<style scoped>
.env-select {
  max-width: 400px;
}

.custom-path-row {
  margin-top: 8px;
  max-width: 400px;
}
</style>
