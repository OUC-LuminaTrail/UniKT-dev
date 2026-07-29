import { createRouter, createWebHistory } from 'vue-router'
import i18n from '@/plugins/i18n'
import { useSystemCapabilities } from '@/composables/useSystemCapabilities'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', redirect: '/tasks' },
    {
      path: '/tasks',
      name: 'tasks',
      component: () => import('@/views/TaskList.vue'),
      meta: { title: 'route.title.tasks' },
    },
    {
      path: '/tasks/new',
      name: 'task-new',
      component: () => import('@/views/TaskLaunch.vue'),
      meta: { title: 'route.title.task-new', parent: 'tasks', flush: true },
    },
    {
      path: '/tasks/:id',
      name: 'task-detail',
      component: () => import('@/views/TaskDetail.vue'),
      meta: { title: 'route.title.task-detail', parent: 'tasks' },
      beforeEnter: (to) => {
        const id = Number(to.params.id)
        if (isNaN(id) || id <= 0) return { name: 'tasks' }
      },
    },
    {
      path: '/gpu',
      name: 'gpu',
      component: () => import('@/views/GPUMonitor.vue'),
      meta: { title: 'route.title.gpu' },
      beforeEnter: () => {
        const { hasGpu } = useSystemCapabilities()
        if (!hasGpu.value) return { name: 'tasks' }
      },
    },
    {
      path: '/preprocess',
      name: 'preprocess',
      component: () => import('@/views/PreprocessView.vue'),
      meta: { title: 'route.title.preprocess', flush: true },
    },
    {
      path: '/settings',
      name: 'settings',
      component: () => import('@/views/SettingsView.vue'),
      meta: { title: 'route.title.settings' },
    },
    { path: '/:pathMatch(.*)*', redirect: '/tasks' },
  ],
})

router.afterEach((to) => {
  const key = to.meta?.title as string | undefined
  if (key) document.title = i18n.global.t(key) || 'KT Experiment Manager'
})

export default router
