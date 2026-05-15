import { createApp } from 'vue'
import { VueQueryPlugin } from '@tanstack/vue-query'
import 'element-plus/dist/index.css'
import 'element-plus/theme-chalk/dark/css-vars.css'
import './styles/global.css'
import App from './App.vue'
import router from './router'

const app = createApp(App)
app.use(router)
app.use(VueQueryPlugin)
app.mount('#app')
