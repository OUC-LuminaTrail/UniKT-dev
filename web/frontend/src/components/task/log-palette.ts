import type { CSSProperties } from 'vue'
import type { Segment } from '@/api/logs'

// Tokyo Night palette — matches the prior xterm terminal theme exactly so the
// rendered output looks identical after migrating off xterm.
const FG_DEFAULT = '#a9b1d6'
const BG_DEFAULT = '#1a1b26'

const NAMED: Record<string, string> = {
  default: FG_DEFAULT,
  black: '#15161e',
  red: '#f7768e',
  green: '#9ece6a',
  yellow: '#e0af68',
  blue: '#7aa2f7',
  magenta: '#bb9af7',
  cyan: '#7dcfff',
  white: '#a9b1d6',
  brightblack: '#414868',
  brightred: '#f7768e',
  brightgreen: '#9ece6a',
  brightyellow: '#e0af68',
  brightblue: '#7aa2f7',
  brightmagenta: '#bb9af7',
  brightcyan: '#7dcfff',
  brightwhite: '#c0caf5',
}

// pyte emits colors as lowercase names ("default"|"red"|"brightblue"|...) or a
// 6-digit hex string. Resolve either to a CSS color.
function resolveColor(c: string | undefined, fallback: string): string {
  if (!c || c === 'default') return fallback
  return c in NAMED ? NAMED[c] : `#${c}`
}

export function segmentStyle(seg: Segment): CSSProperties {
  let fg = resolveColor(seg.fg, FG_DEFAULT)
  let bg = resolveColor(seg.bg, BG_DEFAULT)
  if (seg.reverse) [fg, bg] = [bg, fg]

  const style: CSSProperties = { color: fg }
  if (seg.bg || seg.reverse) style.backgroundColor = bg
  if (seg.bold) style.fontWeight = 'bold'
  if (seg.italic) style.fontStyle = 'italic'
  if (seg.underline || seg.strike) {
    const dec: string[] = []
    if (seg.underline) dec.push('underline')
    if (seg.strike) dec.push('line-through')
    style.textDecoration = dec.join(' ')
  }
  return style
}
