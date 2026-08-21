/**
 * Display helpers. Times, durations, sizes, and a deliberately small markdown
 * renderer.
 *
 * The markdown one matters: the answer text quotes the user's own mail. It is
 * escaped first and *then* six patterns are allowed back in — bold, italics,
 * ordered and unordered lists, and links whose scheme is http, https or mailto.
 * Everything else stays as the characters the person typed.
 */

import type { Degraded, ResolvedWindow, ServiceName } from './types'

/* ------------------------------------------------------------------- markdown */

const CONTROL_CHARS = /[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g

export function escapeHtml(value: string): string {
  return value
    .replace(CONTROL_CHARS, '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;')
}

/** Only three schemes get to be a link. Everything else renders as plain text. */
function safeUrl(escapedHref: string): string | null {
  const probe = escapedHref
    .replace(/&amp;/g, '&')
    .replace(/[\s\u0000-\u001F]/g, '')
    .toLowerCase()
  if (/^(https?:\/\/|mailto:)/.test(probe)) return escapedHref
  if (/^[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}$/.test(probe)) return `mailto:${escapedHref}`
  return null
}

function emphasis(text: string): string {
  return text
    .replace(/\*\*([^*\n]+)\*\*/g, '<strong>$1</strong>')
    .replace(/__([^_\n]+)__/g, '<strong>$1</strong>')
    .replace(/\*([^*\n]+)\*/g, '<em>$1</em>')
    .replace(/(^|[\s(])_([^_\n]+)_(?=$|[\s.,!?):;])/g, '$1<em>$2</em>')
}

/**
 * Links and code are stashed before emphasis runs, so underscores in a URL or a
 * backticked value survive to the other side instead of turning into italics.
 */
function inline(text: string): string {
  const stash: string[] = []
  const park = (html: string) => {
    stash.push(html)
    return `\u0000${stash.length - 1}\u0000`
  }

  const coded = text.replace(/`([^`\n]+)`/g, (_match, code: string) => park(`<code>${code}</code>`))

  const staged = coded.replace(
    /\[([^\]\n]+)\]\(([^)\s]+)\)/g,
    (_match, label: string, href: string) => {
      const safe = safeUrl(href)
      const inner = emphasis(label)
      return park(
        safe
          ? `<a href="${safe}" target="_blank" rel="noopener noreferrer nofollow">${inner}</a>`
          : inner,
      )
    },
  )
  return emphasis(staged).replace(
    /\u0000(\d+)\u0000/g,
    (_match, index: string) => stash[Number(index)] ?? '',
  )
}

const BULLET = /^\s{0,3}[-*+]\s+(.*)$/
const NUMBERED = /^\s{0,3}\d{1,3}[.)]\s+(.*)$/
const HEADING = /^\s{0,3}(#{1,6})\s+(.*)$/
const RULE = /^\s{0,3}([-*_])\1{2,}\s*$/

/**
 * Markdown → HTML for the answer text. Safe to drop into
 * `dangerouslySetInnerHTML`: the input is escaped before any tag is added.
 */
export function renderMarkdown(source: string | null | undefined): string {
  if (!source) return ''
  const lines = escapeHtml(source.replace(/\r\n?/g, '\n')).split('\n')

  const out: string[] = []
  let paragraph: string[] = []
  let items: string[] = []
  let ordered = false

  const flushParagraph = () => {
    if (!paragraph.length) return
    out.push(`<p>${inline(paragraph.join('\n')).replace(/\n/g, '<br>')}</p>`)
    paragraph = []
  }
  const flushList = () => {
    if (!items.length) return
    const tag = ordered ? 'ol' : 'ul'
    out.push(`<${tag}>${items.map((item) => `<li>${inline(item)}</li>`).join('')}</${tag}>`)
    items = []
  }

  for (const line of lines) {
    const bullet = BULLET.exec(line)
    const numbered = NUMBERED.exec(line)
    const heading = HEADING.exec(line)

    // A rule has to be checked before a bullet: `---` matches both, and a
    // horizontal line read as a one-item list is a stray empty bullet.
    if (RULE.test(line)) {
      flushParagraph()
      flushList()
      out.push('<hr>')
      continue
    }
    if (heading) {
      flushParagraph()
      flushList()
      // Everything renders at the same weight. These are answers, not
      // documents, and a giant H1 mid-conversation just shouts.
      out.push(`<p class="md-heading">${inline(heading[2])}</p>`)
      continue
    }

    if (bullet) {
      flushParagraph()
      if (items.length && ordered) flushList()
      ordered = false
      items.push(bullet[1])
      continue
    }
    if (numbered) {
      flushParagraph()
      if (items.length && !ordered) flushList()
      ordered = true
      items.push(numbered[1])
      continue
    }
    if (!line.trim()) {
      flushParagraph()
      flushList()
      continue
    }
    flushList()
    paragraph.push(line)
  }

  flushParagraph()
  flushList()
  return out.join('')
}

/** One line of plain text from markdown — for titles and previews. */
export function stripMarkdown(source: string | null | undefined): string {
  if (!source) return ''
  return source
    .replace(/\[([^\]\n]+)\]\([^)\s]+\)/g, '$1')
    .replace(/[*_`#>]/g, '')
    .replace(/\s+/g, ' ')
    .trim()
}

/* ---------------------------------------------------------------------- time */

function toDate(value: string | number | Date | null | undefined): Date | null {
  if (value == null) return null
  const date = value instanceof Date ? value : new Date(value)
  return Number.isNaN(date.getTime()) ? null : date
}

const MINUTE = 60_000
const HOUR = 60 * MINUTE
const DAY = 24 * HOUR

/** "just now" · "4m ago" · "in 2h" · "12 Aug" */
export function relativeTime(
  value: string | number | Date | null | undefined,
  now: number = Date.now(),
): string {
  const date = toDate(value)
  if (!date) return '—'
  const delta = now - date.getTime()
  const ahead = delta < 0
  const abs = Math.abs(delta)

  if (abs < 45_000) return 'just now'
  const wrap = (body: string) => (ahead ? `in ${body}` : `${body} ago`)
  if (abs < HOUR) return wrap(`${Math.round(abs / MINUTE)}m`)
  if (abs < DAY) return wrap(`${Math.round(abs / HOUR)}h`)
  if (abs < 7 * DAY) return wrap(`${Math.round(abs / DAY)}d`)
  return formatDay(date)
}

/** The name the components use for the same thing. */
export const formatRelative = relativeTime

/** The compact form for the sync bar: "4m", "1h", "now". */
export function formatAgo(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return '—'
  if (seconds < 45) return 'now'
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`
  if (seconds < 86_400) return `${Math.round(seconds / 3600)}h`
  return `${Math.round(seconds / 86_400)}d`
}

/** Sync lag, in seconds, as the bar shows it. */
export const formatLag = formatAgo

export function formatTime(
  value: string | number | Date | null | undefined,
  tz?: string,
): string {
  const date = toDate(value)
  if (!date) return '—'
  return new Intl.DateTimeFormat(undefined, {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    timeZone: tz,
  }).format(date)
}

/** "Thu 21 Aug" — with the year when it is not this one. */
export function formatDay(
  value: string | number | Date | null | undefined,
  tz?: string,
): string {
  const date = toDate(value)
  if (!date) return '—'
  const thisYear = new Date().getFullYear() === date.getFullYear()
  return new Intl.DateTimeFormat(undefined, {
    weekday: 'short',
    day: 'numeric',
    month: 'short',
    year: thisYear ? undefined : 'numeric',
    timeZone: tz,
  }).format(date)
}

/** "Thu 21 Aug, 14:00" */
export function formatDateTime(
  value: string | number | Date | null | undefined,
  tz?: string,
): string {
  const date = toDate(value)
  if (!date) return '—'
  return `${formatDay(date, tz)}, ${formatTime(date, tz)}`
}

/**
 * The two ends of a resolved window, in the zone it was resolved in.
 * Half-open [start, end), so the last day named is `end` minus a millisecond —
 * "Mon 24 Aug – Sun 30 Aug", never "– Mon 31 Aug".
 */
export function formatDateRange(
  start: string | number | Date | null | undefined,
  end: string | number | Date | null | undefined,
  tz?: string,
): string {
  const from = toDate(start)
  const to = toDate(end)
  if (!from && !to) return '—'
  if (!from) return formatDay(to, tz)
  if (!to) return formatDay(from, tz)

  const lastMoment = new Date(to.getTime() - 1)
  const startDay = formatDay(from, tz)
  const endDay = formatDay(lastMoment, tz)
  if (startDay === endDay) {
    // One day. If it is not a whole day, show the clock instead.
    const wholeDay = to.getTime() - from.getTime() >= DAY - 1
    return wholeDay ? startDay : `${startDay}, ${formatTime(from, tz)}–${formatTime(to, tz)}`
  }
  return `${startDay} – ${endDay}`
}

/**
 * A resolved window, as the planner understood it. Windows are half-open, so
 * the last day shown is `end` minus a millisecond.
 *
 * `interpretation` is the planner's own note to itself — "next week: 2026-08-24
 * to 2026-08-31, week_start=1; half-open". It names the open end of the range,
 * which is the one day the window does not cover, so it is the fallback for a
 * window with no dates, never the label.
 */
export function windowLabel(window: ResolvedWindow | null | undefined): string {
  if (!window) return ''
  if (window.start || window.end) return formatDateRange(window.start, window.end, window.tz)
  return window.interpretation ?? ''
}

/* ------------------------------------------------------------------ numbers */

/** "208ms" · "1.5s" · "12s" · "2m 04s" */
export function formatDuration(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms)) return '—'
  if (ms < 1000) return `${Math.round(ms)}ms`
  const seconds = ms / 1000
  if (seconds < 10) return `${seconds.toFixed(1)}s`
  if (seconds < 60) return `${Math.round(seconds)}s`
  const minutes = Math.floor(seconds / 60)
  const rest = Math.round(seconds % 60)
  return `${minutes}m ${String(rest).padStart(2, '0')}s`
}

const UNITS = ['B', 'KB', 'MB', 'GB', 'TB']

export function formatBytes(bytes: number | null | undefined): string {
  if (bytes == null || !Number.isFinite(bytes) || bytes < 0) return '—'
  let value = bytes
  let unit = 0
  while (value >= 1024 && unit < UNITS.length - 1) {
    value /= 1024
    unit += 1
  }
  const decimals = unit === 0 ? 0 : value < 10 ? 1 : 0
  return `${value.toFixed(decimals)} ${UNITS[unit]}`
}

export function formatCount(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '—'
  return value.toLocaleString()
}

export function formatUsd(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '—'
  if (value === 0) return '$0'
  if (value < 0.01) return `$${value.toFixed(4)}`
  return `$${value.toFixed(2)}`
}

export function formatPercent(fraction: number | null | undefined): string {
  if (fraction == null || !Number.isFinite(fraction)) return '—'
  return `${Math.round(fraction * 100)}%`
}

export function plural(count: number, one: string, many?: string): string {
  return `${formatCount(count)} ${count === 1 ? one : (many ?? `${one}s`)}`
}

export function truncate(value: string, max: number): string {
  if (value.length <= max) return value
  return `${value.slice(0, Math.max(0, max - 1)).trimEnd()}…`
}

export function initials(name: string | null | undefined): string {
  if (!name) return '?'
  const parts = name.trim().split(/\s+/).slice(0, 2)
  return parts.map((part) => part.charAt(0).toUpperCase()).join('') || '?'
}

/* ------------------------------------------------------------------- domain */

const SERVICE_LABELS: Record<string, string> = {
  gmail: 'Gmail',
  gcal: 'Calendar',
  gdrive: 'Drive',
}

export function serviceLabel(service: ServiceName | string): string {
  return SERVICE_LABELS[service] ?? service
}

/**
 * What to call the thing that went wrong in a `degraded` entry.
 *
 * A null `service` means the fault was ours, not Google's, so there is no
 * service to name — say whose it was instead of leaving a dangling dash.
 */
export function degradedSubject(entry: Degraded): string {
  if (entry.service) return serviceLabel(entry.service)
  if (entry.fault === 'plan') return 'The plan'
  if (entry.fault === 'dependency') return 'A step it was waiting on'
  return 'Something on our side'
}

const OP_LABELS: Record<string, string> = {
  'ask.user': 'Ask the person',
  'meta.summarise': 'Summarise',
  'meta.compare': 'Compare',
}

export function parseOp(op: string): { service: string; verb: string } {
  const dot = op.indexOf('.')
  if (dot === -1) return { service: '', verb: op.replace(/_/g, ' ') }
  return {
    service: serviceLabel(op.slice(0, dot)),
    verb: op.slice(dot + 1).replace(/_/g, ' '),
  }
}

/** "gmail.search_emails" → "Gmail · search emails" */
export function opLabel(op: string | null | undefined): string {
  if (!op) return ''
  if (OP_LABELS[op]) return OP_LABELS[op]
  const { service, verb } = parseOp(op)
  return service ? `${service} · ${verb}` : verb
}

/** Why a step ended the way it did, in words a person can read. */
export function reasonLabel(reason: string | null | undefined): string {
  if (!reason) return ''
  const known: Record<string, string> = {
    circuit_open: 'service was down',
    gate_false: 'condition not met',
    dependency_failed: 'a step it needed failed',
    timeout: 'took too long',
    rate_limited: 'rate limited',
    quota_exhausted: 'quota exhausted',
    not_found: 'nothing to act on',
    cancelled: 'cancelled',
    target_changed: 'the target changed',
    superseded: 'superseded',
  }
  return known[reason] ?? reason.replace(/_/g, ' ')
}
