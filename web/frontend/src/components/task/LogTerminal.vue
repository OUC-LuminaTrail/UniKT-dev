<template>
  <div ref="terminalContainer" class="log-terminal"></div>
</template>

<script setup lang="ts">
import { ref, onMounted, onUnmounted, watch } from 'vue'
import { Terminal } from '@xterm/xterm'
import { FitAddon } from '@xterm/addon-fit'
import '@xterm/xterm/css/xterm.css'

const props = defineProps<{ messages: string[] }>()

const terminalContainer = ref<HTMLElement>()
let terminal: Terminal | null = null
let fitAddon: FitAddon | null = null
let lastRenderedIndex = 0

onMounted(() => {
  if (!terminalContainer.value) return
  terminal = new Terminal({
    theme: { background: '#1e1e1e', foreground: '#d4d4d4' },
    fontSize: 13,
    fontFamily: 'Menlo, Monaco, Consolas, monospace',
    convertEol: true,
    scrollback: 10000,
  })
  fitAddon = new FitAddon()
  terminal.loadAddon(fitAddon)
  terminal.open(terminalContainer.value)
  fitAddon.fit()
  const resizeObserver = new ResizeObserver(() => fitAddon?.fit())
  resizeObserver.observe(terminalContainer.value)
})

watch(
  () => props.messages.length,
  () => {
    if (!terminal) return
    while (lastRenderedIndex < props.messages.length) {
      terminal.write(props.messages[lastRenderedIndex])
      lastRenderedIndex++
    }
  }
)

onUnmounted(() => { terminal?.dispose() })
</script>

<style scoped>
.log-terminal {
  width: 100%;
  height: 100%;
  min-height: 400px;
}
</style>
