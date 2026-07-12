import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'

const FRONTEND_PORT = parseInt(process.env.KT_WEB_PORT || '5173', 10)

export default defineConfig({
  plugins: [
    vue(),
    AutoImport({
      resolvers: [ElementPlusResolver()],
    }),
    Components({
      resolvers: [ElementPlusResolver()],
    }),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('@xterm/xterm') || id.includes('@xterm/addon-fit') || id.includes('@xterm/addon-web-links') || id.includes('@xterm/addon-search')) {
            return 'xterm'
          }
          if (id.includes('node_modules/vue/') || id.includes('node_modules/vue-router/')) {
            return 'vue-vendor'
          }
        }
      }
    }
  },
  server: {
    port: FRONTEND_PORT,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8765',
        ws: true,
      },
    },
  },
  preview: {
    port: FRONTEND_PORT,
    proxy: {
      '/api': {
        target: API_TARGET,
        ws: true,
      },
    },
  },
})
