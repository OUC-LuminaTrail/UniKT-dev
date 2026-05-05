import { createRouter, createWebHistory } from 'vue-router'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', redirect: '/tasks' },
    { path: '/tasks', component: () => import('@/views/TaskList.vue') },
    { path: '/tasks/new', component: () => import('@/views/TaskLaunch.vue') },
    { path: '/tasks/:id', component: () => import('@/views/TaskDetail.vue') },
    { path: '/experiments', component: () => import('@/views/ExperimentBrowser.vue') },
    { path: '/gpu', component: () => import('@/views/GPUMonitor.vue') },
  ],
})

export default router
