const PALETTE = [
  ['#58a6ff', '#1f6feb'],
  ['#3fb950', '#238636'],
  ['#d29922', '#9e6a03'],
  ['#f85149', '#da3633'],
  ['#bc8cff', '#8b5cf6'],
  ['#39d2c0', '#0d9488'],
  ['#f778ba', '#db2777'],
  ['#79c0ff', '#388bfd'],
]

function hashStr(s: string): number {
  let h = 0
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h + s.charCodeAt(i)) | 0
  }
  return Math.abs(h)
}

export function getGradient(name: string): string {
  const idx = hashStr(name) % PALETTE.length
  const [c1, c2] = PALETTE[idx]
  return `linear-gradient(135deg, ${c1}, ${c2})`
}
