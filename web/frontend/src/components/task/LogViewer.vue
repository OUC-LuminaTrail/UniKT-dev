<template>
  <div ref="scrollEl" class="log-viewer" @scroll="onScroll">
    <div v-if="loading" class="load-hint">加载更早日志…</div>
    <div class="log-spacer" :style="{ height: totalSize + 'px' }">
      <div
        v-for="item in virtualItems"
        :key="String(item.key)"
        :data-index="item.index"
        :ref="(el) => measureItem(el)"
        class="log-row"
        :style="{ transform: `translateY(${item.start}px)` }"
      >
        <template v-if="lines[item.index] && lines[item.index].length">
          <span
            v-for="(seg, i) in lines[item.index]"
            :key="i"
            :style="segmentStyle(seg)"
            class="log-seg"
          >
            <template v-for="(tok, j) in urlTokens(seg.t)" :key="j">
              <a v-if="tok.url" :href="tok.url" target="_blank" rel="noopener" class="log-url">{{
                tok.text
              }}</a>
              <template v-else>{{ tok.text }}</template>
            </template>
          </span>
        </template>
        <span v-else class="log-blank">&nbsp;</span>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed, nextTick, onMounted, onUnmounted, ref } from 'vue'
import { useVirtualizer } from '@tanstack/vue-virtual'
import { getLogLines, type LogApiBase, type RenderedLine } from '@/api/logs'
import { segmentStyle } from './log-palette'
import type { ConnState } from './log-conn'

const props = withDefaults(
  defineProps<{
    wsUrl: string
    taskStatus: string
    taskId: number
    apiBase?: LogApiBase
  }>(),
  { apiBase: '/tasks' },
)
const emit = defineEmits<{
  (e: 'state', state: ConnState): void
}>()

// Lines per page (initial tail + each upward load).
const PAGE = 500
// Punctuation that is prose, not part of a URL — stripped from link hrefs.
const URL_RE = /(https?:\/\/[^\s]+|file:\/\/[^\s]+)/g
const URL_TRAIL_RE = /[.,;:!?)\]"']+$/

const scrollEl = ref<HTMLElement>()
const lines = ref<RenderedLine[]>([])
// Index of the oldest currently loaded line; loaded range is [oldest, total).
const oldest = ref(0)
const total = ref(0)
const loading = ref(false)
// Whether the viewport is pinned to the bottom (auto-follow new output).
const atBottom = ref(true)
let streamEnded = false
let ws: WebSocket | null = null
let reconnectTimer: ReturnType<typeof setTimeout> | null = null

const isRunning = () => props.taskStatus === 'running' || props.taskStatus === 'stopping'
// A queued task has no output yet but its stream is still expected, so a drop
// while pending must reconnect.
const mayStillStream = () => isRunning() || props.taskStatus === 'pending'

// Stable key = global line index, so prepended rows keep their measured height
// and the virtualizer doesn't re-measure the whole list on upward pagination.
const virtualizer = useVirtualizer(
  computed(() => ({
    count: lines.value.length,
    getScrollElement: () => scrollEl.value ?? null,
    estimateSize: () => 20,
    overscan: 12,
    getItemKey: (i: number) => oldest.value + i,
  })),
)
const virtualItems = computed(() => virtualizer.value.getVirtualItems())
const totalSize = computed(() => virtualizer.value.getTotalSize())

function measureItem(el: unknown) {
  if (el instanceof HTMLElement) virtualizer.value.measureElement(el)
}

// Split a segment's text into plain/url tokens so URLs stay clickable while
// inheriting the segment's color style. Trailing prose punctuation is kept out
// of the href (it stays as plain text after the link).
function urlTokens(text: string): { text: string; url?: string }[] {
  if (!text) return []
  const out: { text: string; url?: string }[] = []
  let last = 0
  for (const m of text.matchAll(URL_RE)) {
    const idx = m.index ?? 0
    if (idx > last) out.push({ text: text.slice(last, idx) })
    const raw = m[0]
    const trail = raw.match(URL_TRAIL_RE)?.[0] ?? ''
    const url = trail ? raw.slice(0, -trail.length) : raw
    out.push({ text: url, url })
    if (trail) out.push({ text: trail })
    last = idx + raw.length
  }
  if (last < text.length) out.push({ text: text.slice(last) })
  return out.length ? out : [{ text }]
}

function scrollToBottom() {
  nextTick(() => {
    if (lines.value.length === 0) return
    // scrollToIndex reconciles after dynamic-height measurement settles,
    // unlike a raw scrollTop assignment which races the ResizeObserver.
    virtualizer.value.scrollToIndex(lines.value.length - 1, { align: 'end' })
  })
}

function onScroll() {
  const el = scrollEl.value
  if (!el) return
  // Threshold scales with the last visible row so a single tall wrapped row
  // doesn't keep the viewer falsely pinned to the bottom.
  const items = virtualItems.value
  const last = items[items.length - 1]
  const threshold = last ? last.size + 8 : 24
  atBottom.value = el.scrollTop + el.clientHeight >= el.scrollHeight - threshold
  if (el.scrollTop < el.clientHeight * 0.5 && oldest.value > 0 && !loading.value) {
    loadOlder()
  }
}

async function loadInitial() {
  // Read total with a 1-line probe, then fetch the most recent page.
  const head = await getLogLines(props.apiBase, props.taskId, 0, 1)
  total.value = head.total
  const offset = Math.max(0, total.value - PAGE)
  const res = await getLogLines(props.apiBase, props.taskId, offset, PAGE)
  oldest.value = offset
  lines.value = res.lines
  total.value = res.total
  await nextTick()
  scrollToBottom()
}

async function loadOlder() {
  loading.value = true
  // Capture the first visible LOCAL index before mutating; after prepending N
  // rows that same row sits at local index + N. Scrolling to it by index (not
  // scrollHeight delta) keeps the viewport steady even if a WS patch appends
  // tail rows during the await.
  const startIdx = virtualItems.value[0]?.index ?? 0
  const newOffset = Math.max(0, oldest.value - PAGE)
  const limit = oldest.value - newOffset
  try {
    const res = await getLogLines(props.apiBase, props.taskId, newOffset, limit)
    total.value = res.total
    oldest.value = newOffset
    lines.value = [...res.lines, ...lines.value]
    await nextTick()
    virtualizer.value.scrollToIndex(startIdx + res.lines.length, { align: 'start' })
  } finally {
    loading.value = false
  }
}

function applyPatch(p: { from_line: number; total: number; lines: RenderedLine[] }) {
  if (p.from_line < oldest.value) {
    // Server state rewound below our loaded window (log rotation/truncation) —
    // drop the stale local history and adopt the server's view fresh.
    oldest.value = Math.max(0, p.from_line)
    lines.value = p.lines
    total.value = p.total
    if (atBottom.value) scrollToBottom()
    return
  }
  total.value = p.total
  const localStart = p.from_line - oldest.value
  const clamped = Math.min(localStart, lines.value.length)
  lines.value = lines.value.slice(0, clamped).concat(p.lines)
  if (atBottom.value) scrollToBottom()
}

function connect() {
  if (streamEnded) return
  const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:'
  let url = props.wsUrl.startsWith('/')
    ? `${protocol}//${location.host}${props.wsUrl}`
    : props.wsUrl
  const sep = url.includes('?') ? '&' : '?'
  // Resume from the highest loaded line, not total: if output grew between the
  // probe and the page fetch, the gap is closed by the initial WS alignment.
  url += `${sep}from_line=${oldest.value + lines.value.length}`
  ws = new WebSocket(url)
  ws.onopen = () => emit('state', 'connected')
  ws.onmessage = (event) => {
    let data
    try {
      data = JSON.parse(event.data)
    } catch {
      return
    }
    if (data.type === 'patch') {
      applyPatch(data)
    } else if (data.type === 'done') {
      streamEnded = true
      lines.value = [...lines.value, [{ t: '——— Process exited ———', fg: 'brightblack' }]]
      nextTick(scrollToBottom)
      emit('state', 'ended')
    } else if (data.type === 'error') {
      // A backend error is terminal (the socket closes right after) — stop
      // retrying so we don't flood handshakes until the status poll catches up.
      streamEnded = true
      lines.value = [...lines.value, [{ t: data.content ?? '', fg: 'red' }]]
      nextTick(scrollToBottom)
      emit('state', 'ended')
    }
  }
  ws.onclose = () => {
    if (streamEnded || !mayStillStream()) {
      emit('state', 'ended')
      return
    }
    emit('state', 'reconnecting')
    reconnectTimer = setTimeout(connect, 2000)
  }
  ws.onerror = () => ws?.close()
}

onMounted(async () => {
  emit('state', 'connecting')
  try {
    await loadInitial()
  } catch {
    // Initial fetch failed (e.g. task not yet visible) — WS reconnect recovers.
  }
  connect()
})

onUnmounted(() => {
  streamEnded = true
  if (reconnectTimer) clearTimeout(reconnectTimer)
  ws?.close()
})

defineExpose({ scrollToBottom })
</script>

<style scoped>
.log-viewer {
  width: 100%;
  height: 100%;
  overflow: auto;
  background: var(--term-bg);
  font-family: var(--font-mono);
  font-size: 13px;
  line-height: 1.5;
  color: #a9b1d6;
  padding: 8px 0;
}

.log-spacer {
  position: relative;
  width: 100%;
}

.log-row {
  position: absolute;
  top: 0;
  left: 0;
  width: max-content;
  min-width: 100%;
  padding: 0 8px;
  box-sizing: border-box;
  white-space: pre;
  min-height: 1.5em;
}

.log-seg {
  white-space: pre;
}

.log-url {
  color: inherit;
  text-decoration: underline;
  opacity: 0.9;
}

.load-hint {
  position: sticky;
  top: 0;
  z-index: 2;
  padding: 2px 8px;
  font-size: 11px;
  color: var(--text-tertiary);
  background: var(--term-bg);
  font-family: var(--font-sans);
}
</style>
