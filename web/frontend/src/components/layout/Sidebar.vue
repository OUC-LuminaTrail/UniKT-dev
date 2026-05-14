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
          <el-icon><SettingsIcon /></el-icon>
          <template #title>设置</template>
        </el-menu-item>
      </el-menu>

      <div class="sidebar-actions">
        <button class="action-btn" @click="toggleDark()" :title="isDark ? '切换到亮色模式' : '切换到暗色模式'">
          <svg v-if="isDark" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/>
          </svg>
          <svg v-else width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
          </svg>
          <Transition name="fade">
            <span v-show="!collapsed" class="action-label">{{ isDark ? '亮色模式' : '暗色模式' }}</span>
          </Transition>
        </button>
        <button class="action-btn" @click="$emit('toggle-collapse')" :title="collapsed ? '展开侧栏' : '折叠侧栏'">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polyline v-if="collapsed" points="9,18 15,12 9,6"/>
            <polyline v-else points="15,18 9,12 15,6"/>
          </svg>
          <Transition name="fade">
            <span v-show="!collapsed" class="action-label">折叠</span>
          </Transition>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, h } from 'vue'
import { useRoute } from 'vue-router'
import { useDark, useToggle } from '@vueuse/core'

defineProps<{
  collapsed: boolean
}>()

defineEmits<{
  'toggle-collapse': []
}>()

const route = useRoute()

const isDark = useDark()
const toggleDark = useToggle(isDark)

const PreprocessIcon = () => h('svg', { width: 18, height: 18, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, [
  h('polyline', { points: '16 3 21 3 21 8' }),
  h('line', { x1: 4, y1: 20, x2: 21, y2: 3 }),
  h('polyline', { points: '21 16 21 21 16 21' }),
  h('line', { x1: 15, y1: 15, x2: 21, y2: 21 }),
  h('line', { x1: 4, y1: 4, x2: 9, y2: 9 }),
])

const TasksIcon = () => h('svg', { width: 18, height: 18, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, [
  h('rect', { x: 2, y: 3, width: 20, height: 14, rx: 2, ry: 2 }),
  h('line', { x1: 8, y1: 21, x2: 16, y2: 21 }),
  h('line', { x1: 12, y1: 17, x2: 12, y2: 21 }),
])

const GpuIcon = () => h('svg', { width: 18, height: 18, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, [
  h('rect', { x: 4, y: 4, width: 16, height: 16, rx: 2, ry: 2 }),
  h('rect', { x: 9, y: 9, width: 6, height: 6 }),
  h('line', { x1: 1, y1: 9, x2: 4, y2: 9 }),
  h('line', { x1: 1, y1: 14, x2: 4, y2: 14 }),
])

const SettingsIcon = () => h('svg', { width: 18, height: 18, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, [
  h('circle', { cx: 12, cy: 12, r: 3 }),
  h('path', { d: 'M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z' }),
])

const navItems = [
  { path: '/preprocess', label: '数据预处理', icon: PreprocessIcon },
  { path: '/tasks', label: '训练任务', icon: TasksIcon },
  { path: '/gpu', label: 'GPU 监控', icon: GpuIcon },
]

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
