<template>
  <ParamGroups :groups="groups" v-model="formData" />
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import ParamGroups from './ParamGroups.vue'
import type { ParamGroupDef } from './param-groups'

// Flat field list rendered by the generic ParamGroups engine; the nested
// sampler_kwargs / pruner_kwargs that OptunaConfig expects are (re)assembled
// on every change via build() / flatten().
const FIELDS: { key: string; type: string; choices?: (string | number)[]; default: unknown }[] = [
  { key: 'metric', type: 'str', choices: ['auc', 'acc', 'rmse', 'loss'], default: 'auc' },
  { key: 'n_trials', type: 'int', default: 100 },
  { key: 'sampler', type: 'str', choices: ['tpe', 'random', 'grid', 'cmaes'], default: 'tpe' },
  { key: 'seed', type: 'int', default: 42 },
  { key: 'sampler_n_startup_trials', type: 'int', default: 10 },
  { key: 'pruner', type: 'str', choices: ['median', 'percentile', 'successive_halving', 'hyperband', 'none'], default: 'median' },
  { key: 'pruner_n_startup_trials', type: 'int', default: 5 },
  { key: 'pruner_n_warmup_steps', type: 'int', default: 10 },
  { key: 'n_jobs', type: 'int', default: 1 },
  { key: 'timeout', type: 'int', default: 0 },
  { key: 'study_name', type: 'str', default: 'hyperparameter_search' },
  { key: 'verbose', type: 'int', choices: [0, 1, 2], default: 1 },
]

// Single source of default values; flatten/build fall back to this instead of
// re-hardcoding literals, so changing FIELDS[].default propagates everywhere.
const DEFAULTS: Record<string, any> = Object.fromEntries(
  FIELDS.map((f) => [f.key, f.default]),
)

const { t, te } = useI18n()
const props = defineProps<{ modelValue?: Record<string, any> }>()
const emit = defineEmits<{ 'update:modelValue': [val: Record<string, any>] }>()

const helpText = (key: string) => {
  const k = `search.fieldHelp.${key}`
  return te(k) ? t(k) : ''
}

const groups = computed<ParamGroupDef[]>(() => [
  {
    name: 'optuna_config',
    fields: FIELDS.map((f) => ({
      key: f.key,
      type: f.type,
      choices: f.choices,
      default: f.default,
      help: helpText(f.key),
    })),
  },
])

const defaults = (): Record<string, any> =>
  Object.fromEntries(FIELDS.map((f) => [f.key, f.default]))

function flatten(cfg: Record<string, any>): Record<string, any> {
  const sk = cfg.sampler_kwargs || {}
  const pk = cfg.pruner_kwargs || {}
  return {
    metric: cfg.metric ?? DEFAULTS.metric,
    n_trials: cfg.n_trials ?? DEFAULTS.n_trials,
    sampler: cfg.sampler ?? DEFAULTS.sampler,
    seed: cfg.seed ?? sk.seed ?? DEFAULTS.seed,
    sampler_n_startup_trials: sk.n_startup_trials ?? DEFAULTS.sampler_n_startup_trials,
    pruner: cfg.pruner == null ? 'none' : cfg.pruner ?? DEFAULTS.pruner,
    pruner_n_startup_trials: pk.n_startup_trials ?? DEFAULTS.pruner_n_startup_trials,
    pruner_n_warmup_steps: pk.n_warmup_steps ?? DEFAULTS.pruner_n_warmup_steps,
    n_jobs: cfg.n_jobs ?? DEFAULTS.n_jobs,
    timeout: cfg.timeout ?? DEFAULTS.timeout,
    study_name: cfg.study_name ?? DEFAULTS.study_name,
    verbose: cfg.verbose ?? DEFAULTS.verbose,
  }
}

function build(form: Record<string, any>): Record<string, any> {
  const num = (v: any) => (v === '' || v == null ? undefined : Number(v))
  return {
    metric: form.metric,
    n_trials: num(form.n_trials) ?? DEFAULTS.n_trials,
    sampler: form.sampler,
    seed: num(form.seed) ?? DEFAULTS.seed,
    sampler_kwargs: { seed: num(form.seed) ?? DEFAULTS.seed, n_startup_trials: num(form.sampler_n_startup_trials) ?? DEFAULTS.sampler_n_startup_trials },
    pruner: form.pruner === 'none' ? null : form.pruner,
    pruner_kwargs: {
      n_startup_trials: num(form.pruner_n_startup_trials) ?? DEFAULTS.pruner_n_startup_trials,
      n_warmup_steps: num(form.pruner_n_warmup_steps) ?? DEFAULTS.pruner_n_warmup_steps,
    },
    n_jobs: num(form.n_jobs) ?? DEFAULTS.n_jobs,
    timeout: Number(form.timeout) > 0 ? num(form.timeout) : null,
    study_name: form.study_name,
    verbose: num(form.verbose) ?? DEFAULTS.verbose,
  }
}

const formData = ref<Record<string, any>>(
  props.modelValue && Object.keys(props.modelValue).length
    ? flatten(props.modelValue)
    : defaults()
)

watch(formData, (v) => emit('update:modelValue', build(v)), { deep: true, immediate: true })
</script>
