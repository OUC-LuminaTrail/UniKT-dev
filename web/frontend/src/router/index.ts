import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', redirect: '/tasks' },
    {
      path: '/tasks',
      name: 'tasks',
      component: () => import('@/views/TaskList.vue'),
      meta: { title: '训练任务' },
    },
    {
      path: '/tasks/new',
      name: 'task-new',
      component: () => import('@/views/TaskLaunch.vue'),
      meta: { title: '新建训练任务', parent: 'tasks', flush: true },
    },
    {
      path: '/tasks/:id',
      name: 'task-detail',
      component: () => import('@/views/TaskDetail.vue'),
      meta: { title: '任务详情', parent: 'tasks' },
      beforeEnter: (to) => {
        const id = Number(to.params.id)
        if (isNaN(id) || id <= 0) return { name: 'tasks' }
      },
    },
    {
      path: '/gpu',
      name: 'gpu',
      component: () => import('@/views/GPUMonitor.vue'),
      meta: { title: 'GPU 监控' },
    },
    {
      path: '/preprocess',
      name: 'preprocess',
      component: () => import('@/views/PreprocessView.vue'),
      meta: { title: '数据预处理', flush: true },
    },
    {
      path: '/settings',
      name: 'settings',
      component: () => import('@/views/SettingsView.vue'),
      meta: { title: '设置' },
    },
    { path: '/:pathMatch(.*)*', redirect: '/tasks' },
  ],
})

export default router
