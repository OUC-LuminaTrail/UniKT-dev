<template>
  <div class="sidebar" :class="{ collapsed }">
    <div class="sidebar-brand">
      <div class="brand-icon">KT</div>
      <Transition name="fade">
        <div v-show="!collapsed" class="brand-text">
          <span class="brand-title">KT Exp</span>
          <span class="brand-sub">training manager</span>
        </div>
      </Transition>
    </div>

    <el-menu
      :default-active="activeIndex"
      :collapse="collapsed"
      :router="true"
      class="sidebar-menu"
    >
      <el-menu-item v-for="item in navItems" :key="item.path" :index="item.path">
        <el-icon><component :is="item.icon" /></el-icon>
        <template #title>{{ item.label }}</template>
      </el-menu-item>
    </el-menu>

    <div class="sidebar-bottom">
      <el-menu
        :default-active="activeIndex"
        :collapse="collapsed"
        :router="true"
        class="sidebar-menu-bottom"
      >
        <el-menu-item index="/settings">
          <el-icon><Setting /></el-icon>
          <template #title>{{ t('nav.settings') }}</template>
        </el-menu-item>
        <el-menu-item @click="toggleDark()">
          <el-icon><component :is="isDark ? Sunny : Moon" /></el-icon>
          <template #title>{{ isDark ? t('nav.lightMode') : t('nav.darkMode') }}</template>
        </el-menu-item>
      </el-menu>

      <div class="sidebar-actions">
        <button
          class="action-btn"
          @click="$emit('toggle-collapse')"
          :title="collapsed ? t('nav.expandSidebar') : t('nav.collapseSidebar')"
        >
          <el-icon :size="15"><Fold v-if="!collapsed" /><Expand v-else /></el-icon>
          <Transition name="fade">
            <span v-show="!collapsed" class="action-label">{{ t('nav.collapse') }}</span>
          </Transition>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useI18n } from 'vue-i18n'
import { useRoute } from 'vue-router'
import { useDark, useToggle } from '@vueuse/core'
import { Upload, Monitor, Grid, Setting, Sunny, Moon, Expand, Fold } from '@element-plus/icons-vue'
import { useSystemCapabilities } from '@/composables/useSystemCapabilities'

defineProps<{
  collapsed: boolean
}>()

defineEmits<{
  'toggle-collapse': []
}>()

const route = useRoute()

const isDark = useDark()
const toggleDark = useToggle(isDark)

const { t } = useI18n()

const { hasGpu } = useSystemCapabilities()

const navItems = computed(() => {
  const items = [
    { path: '/preprocess', label: t('nav.preprocess'), icon: Upload },
    { path: '/tasks', label: t('nav.tasks'), icon: Monitor },
  ]
  if (hasGpu.value) {
    items.push({ path: '/gpu', label: t('nav.gpu'), icon: Grid })
  }
  return items
})

const activeIndex = computed(() => {
  const path = route.path
  if (path === '/settings') return '/settings'
  if (path.startsWith('/tasks')) return '/tasks'
  return path || '/tasks'
})
</script>

<style scoped>
.sidebar {
  height: 100%;
  background: var(--bg-surface);
  border-right: 1px solid var(--border-muted);
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.sidebar-brand {
  height: 48px;
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 0 16px;
  border-bottom: 1px solid var(--border-muted);
  flex-shrink: 0;
}

.collapsed .sidebar-brand {
  padding: 0;
  justify-content: center;
}

.brand-icon {
  width: 28px;
  height: 28px;
  border-radius: var(--radius-sm);
  background: linear-gradient(135deg, var(--accent-blue), var(--accent-cyan));
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: 700;
  font-size: 10px;
  color: #fff;
  font-family: var(--font-mono);
  flex-shrink: 0;
}

.brand-text {
  display: flex;
  flex-direction: column;
  line-height: 1.2;
  white-space: nowrap;
}

.brand-title {
  font-weight: 600;
  font-size: 13px;
  color: var(--text-primary);
  letter-spacing: -0.01em;
}

.brand-sub {
  font-size: 10px;
  color: var(--text-tertiary);
  font-family: var(--font-mono);
  letter-spacing: 0.3px;
}

.sidebar-menu,
.sidebar-menu-bottom {
  border-right: none !important;
  --el-menu-bg-color: transparent;
  --el-menu-text-color: var(--text-secondary);
  --el-menu-active-color: var(--accent-blue);
  --el-menu-hover-bg-color: var(--bg-overlay);
  --el-menu-item-height: 40px;
  --el-menu-item-font-size: 13px;
}

.sidebar-menu {
  flex: 1;
  overflow-y: auto;
  padding: 4px 0;
}

.sidebar-menu:not(.el-menu--collapse),
.sidebar-menu-bottom:not(.el-menu--collapse) {
  width: 100%;
}

.sidebar-menu :deep(.el-menu-item),
.sidebar-menu-bottom :deep(.el-menu-item) {
  margin: 1px 8px;
  border-radius: var(--radius-sm);
}

.sidebar-menu.el-menu--collapse :deep(.el-menu-item),
.sidebar-menu-bottom.el-menu--collapse :deep(.el-menu-item) {
  margin: 1px 0;
}

.sidebar-menu :deep(.el-menu-item.is-active),
.sidebar-menu-bottom :deep(.el-menu-item.is-active) {
  background: rgba(88, 166, 255, 0.12);
}

.sidebar-bottom {
  border-top: 1px solid var(--border-muted);
  flex-shrink: 0;
}

.sidebar-menu-bottom {
  padding: 4px 0;
}

.sidebar-actions {
  display: flex;
  flex-direction: column;
  gap: 1px;
  padding: 4px 8px 8px;
}

.collapsed .sidebar-actions {
  padding: 4px 0 8px;
}

.action-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  height: 32px;
  padding: 0 12px;
  border: none;
  background: transparent;
  color: var(--text-tertiary);
  cursor: pointer;
  border-radius: var(--radius-sm);
  font-size: 12px;
  font-family: var(--font-sans);
  transition: all 0.15s ease;
  white-space: nowrap;
}

.collapsed .action-btn {
  justify-content: center;
  padding: 0;
}

.action-btn:hover {
  background: var(--bg-overlay);
  color: var(--text-primary);
}

.action-label {
  font-weight: 450;
  letter-spacing: -0.01em;
}

.fade-enter-active,
.fade-leave-active {
  transition: opacity 0.15s ease;
}

.fade-enter-from,
.fade-leave-to {
  opacity: 0;
}
</style>
