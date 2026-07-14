import { fileURLToPath, URL } from 'node:url'
import { defineConfig, loadEnv } from 'vite'
import vue from '@vitejs/plugin-vue'
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'

const projectRoot = fileURLToPath(new URL('../../', import.meta.url))

export default defineConfig(({ mode }) => {
  // Load repo-root .env (no prefix filter) so KT_WEB_PORT reaches the dev/preview servers
  const env = loadEnv(mode, projectRoot, '')
  const FRONTEND_PORT = parseInt(env.KT_WEB_PORT || '5173', 10)

  return {
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
      allowedHosts: true,
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:8765',
          ws: true,
        },
      },
    },
    preview: {
      port: FRONTEND_PORT,
      allowedHosts: true,
      proxy: {
        '/api': {
          target: 'http://127.0.0.1:8765',
          ws: true,
        },
      },
    },
  }
})
