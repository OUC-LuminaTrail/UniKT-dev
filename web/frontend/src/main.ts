import { createApp } from 'vue'
import { QueryClient, VueQueryPlugin } from '@tanstack/vue-query'
import 'element-plus/dist/index.css'
import 'element-plus/theme-chalk/dark/css-vars.css'
import './styles/global.css'
import App from './App.vue'
import i18n from './plugins/i18n'
import router from './router'

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 5000, retry: 2 } },
})

const app = createApp(App)
app.use(i18n)
app.use(router)
app.use(VueQueryPlugin, { queryClient })
app.mount('#app')
