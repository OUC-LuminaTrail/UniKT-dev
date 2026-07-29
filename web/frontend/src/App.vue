<template>
  <el-config-provider :locale="epLocale">
    <SetupWizard
      v-if="showWizard"
      @skip="onSkipWizard"
      @done="onWizardDone"
    />
    <AppLayout>
      <router-view />
    </AppLayout>
  </el-config-provider>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import AppLayout from '@/components/layout/AppLayout.vue'
import SetupWizard from '@/components/setup/SetupWizard.vue'
import { getInitStatus } from '@/api/settings'
import { useLocale } from '@/composables/useLocale'
import { useTaskEvents } from '@/composables/useTaskEvents'

useTaskEvents()
const { epLocale } = useLocale()

const showWizard = ref(false)

onMounted(async () => {
  try {
    const res = await getInitStatus()
    showWizard.value = !res.initialized
  } catch {}
})

const onSkipWizard = () => {
  showWizard.value = false
}

const onWizardDone = () => {
  showWizard.value = false
}
</script>
