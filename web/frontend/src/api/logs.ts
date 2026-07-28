import api from './index'

export type LogApiBase = '/tasks' | '/preprocess'

export interface Segment {
  t: string
  fg?: string
  bg?: string
  bold?: boolean
  italic?: boolean
  underline?: boolean
  strike?: boolean
  reverse?: boolean
}

export type RenderedLine = Segment[]

export interface LogLinesResult {
  lines: RenderedLine[]
  total: number
}

// Paginate rendered log lines by line index. The backend collapses PTY/ANSI
// bytes (rich colors, CR/erase-line/progress) into final display lines; the
// frontend only renders them.
export const getLogLines = (
  apiBase: LogApiBase,
  id: number,
  offset = 0,
  limit = 500,
): Promise<LogLinesResult> =>
  api.get(`${apiBase}/${id}/logs`, { params: { offset, limit } }).then((r) => r.data)
