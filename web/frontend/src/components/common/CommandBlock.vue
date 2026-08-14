<template>
  <DetailSection :title="title || t('task.detail.sectionCommand')">
    <template #actions>
      <button class="copy-btn" @click="copy">
        <el-icon :size="12"><CopyDocument /></el-icon>
        <span>{{ t('task.detail.copy') }}</span>
      </button>
      <button class="copy-btn" @click="expanded = !expanded">
        <el-icon :size="12"><component :is="expanded ? ArrowUp : ArrowDown" /></el-icon>
        <span>{{ expanded ? t('task.detail.collapse') : t('task.detail.expand') }}</span>
      </button>
    </template>
    <pre class="command-text" :class="{ expanded }"><code>{{ command }}</code></pre>
  </DetailSection>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { ArrowDown, ArrowUp, CopyDocument } from '@element-plus/icons-vue'
import DetailSection from './DetailSection.vue'

const props = defineProps<{ command: string; title?: string }>()

const { t } = useI18n()
const expanded = ref(false)

const copy = () => {
  if (!props.command) return
  navigator.clipboard.writeText(props.command)
  ElMessage.success(t('common.copied'))
}
</script>

<style scoped>
.copy-btn {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  font-size: 11px;
  color: var(--text-tertiary);
  background: none;
  border: 1px solid var(--border-default);
  border-radius: var(--radius-sm);
  padding: 4px 8px;
  cursor: pointer;
  font-family: var(--font-sans);
  transition: all 0.15s ease;
}

.copy-btn:hover {
  color: var(--accent-blue);
  border-color: var(--accent-blue);
}

.command-text {
  margin: 0;
  overflow-x: auto;
}

.command-text code {
  font-family: var(--font-mono);
  font-size: 12.5px;
  color: var(--accent-cyan);
  line-height: 1.6;
  white-space: nowrap;
}

.command-text.expanded code {
  white-space: pre-wrap;
  word-break: break-all;
}
</style>
