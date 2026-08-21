/**
 * How up to date your information is.
 *
 * The strip at the top of the app says almost nothing on purpose — a row of
 * counters is noise you learn to ignore, and then miss on the day it matters.
 * This is where somebody who *wants* the detail comes to look for it, so it
 * can be specific. Still in plain words: "checked 4 minutes ago", never
 * "lag_seconds: 240".
 */

import { useEffect, useState } from 'react'

import { errorMessage, getSyncStatus, triggerSync } from '../lib/api'
import { formatCount, relativeTime } from '../lib/format'
import type { ServiceName, SyncServiceState, SyncStatus as Status } from '../lib/types'

interface SyncStatusProps {
  onClose: () => void
}

const SERVICES: Array<{ key: ServiceName; label: string; what: string }> = [
  { key: 'gmail', label: 'Mail', what: 'messages' },
  { key: 'gcal', label: 'Calendar', what: 'events' },
  { key: 'gdrive', label: 'Files', what: 'files' },
]

type Condition = 'fine' | 'importing' | 'stale' | 'broken' | 'waiting'

function conditionOf(state: SyncServiceState | undefined, target: number): Condition {
  if (!state || !state.last_success_at) return 'waiting'
  if (state.circuit_open_until && Date.parse(state.circuit_open_until) > Date.now()) return 'broken'
  if (state.consecutive_failures > 0 || state.last_error) return 'broken'
  if (!state.backfill_complete) return 'importing'
  if (state.lag_seconds != null && state.lag_seconds > target * 6) return 'stale'
  return 'fine'
}

const WORDS: Record<Condition, { text: string; dot: string; tone: string }> = {
  fine: { text: 'Up to date', dot: 'bg-ok', tone: 'text-ok' },
  importing: { text: 'First import running', dot: 'bg-warn', tone: 'text-warn' },
  stale: { text: 'Behind', dot: 'bg-warn', tone: 'text-warn' },
  broken: { text: 'Not responding', dot: 'bg-bad', tone: 'text-bad' },
  waiting: { text: 'Not checked yet', dot: 'bg-idle', tone: 'text-muted' },
}

export default function SyncStatus({ onClose }: SyncStatusProps) {
  const [status, setStatus] = useState<Status | null>(null)
  const [loading, setLoading] = useState(true)
  const [checking, setChecking] = useState(false)
  const [note, setNote] = useState<string | null>(null)

  async function load(signal?: AbortSignal) {
    try {
      setStatus(await getSyncStatus(signal))
      setNote(null)
    } catch (err) {
      if (!signal?.aborted) setNote(errorMessage(err))
    } finally {
      if (!signal?.aborted) setLoading(false)
    }
  }

  useEffect(() => {
    const ctrl = new AbortController()
    void load(ctrl.signal)
    // Kept current while it is open. This is the one screen somebody opens
    // *because* they want to watch it change.
    const timer = window.setInterval(() => void load(), 5000)
    return () => {
      ctrl.abort()
      window.clearInterval(timer)
    }
  }, [])

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [onClose])

  async function checkNow() {
    setChecking(true)
    try {
      await triggerSync()
      await load()
    } catch (err) {
      setNote(errorMessage(err))
    } finally {
      setChecking(false)
    }
  }

  const target = status?.freshness?.target_seconds ?? 900
  const total = SERVICES.reduce(
    (sum, s) => sum + (status?.services?.[s.key]?.items_indexed ?? 0),
    0,
  )

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/50 px-0 sm:items-center sm:px-4">
      <button type="button" aria-label="Close" onClick={onClose} className="absolute inset-0" />

      <div className="relative flex max-h-[90vh] w-full max-w-md flex-col overflow-hidden rounded-t-2xl border border-line bg-surface shadow-xl sm:rounded-2xl">
        <div className="flex items-start justify-between gap-3 border-b border-line px-5 py-4">
          <div>
            <h2 className="text-sm font-semibold text-ink">Your information</h2>
            <p className="mt-0.5 text-xs text-muted">
              {loading && !status
                ? 'Checking…'
                : total > 0
                  ? `${formatCount(total)} items Ryuk can search`
                  : 'Nothing imported yet'}
            </p>
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="-mr-1 rounded-lg p-1.5 text-muted transition hover:bg-elevated hover:text-ink"
          >
            <svg viewBox="0 0 16 16" aria-hidden className="h-4 w-4">
              <path
                d="M4 4l8 8M12 4l-8 8"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.75"
                strokeLinecap="round"
              />
            </svg>
          </button>
        </div>

        <div className="flex-1 overflow-y-auto px-5 py-4">
          <ul className="space-y-2">
            {SERVICES.map(({ key, label, what }) => {
              const state = status?.services?.[key]
              const condition = conditionOf(state, target)
              const words = WORDS[condition]
              return (
                <li key={key} className="rounded-xl border border-line px-4 py-3">
                  <div className="flex items-center gap-2">
                    <span className={`h-2 w-2 shrink-0 rounded-full ${words.dot}`} aria-hidden />
                    <p className="text-sm font-medium text-ink">{label}</p>
                    <p className={`ml-auto text-xs ${words.tone}`}>{words.text}</p>
                  </div>

                  <dl className="mt-2 space-y-1 text-xs">
                    <div className="flex justify-between gap-3">
                      <dt className="text-muted">Imported</dt>
                      <dd className="tabular-nums text-ink">
                        {formatCount(state?.items_indexed ?? 0)} {what}
                      </dd>
                    </div>
                    <div className="flex justify-between gap-3">
                      <dt className="text-muted">Last checked</dt>
                      <dd className="text-ink">
                        {state?.last_success_at ? relativeTime(state.last_success_at) : 'never'}
                      </dd>
                    </div>
                  </dl>

                  {condition === 'importing' && (
                    <p className="mt-2 text-2xs leading-relaxed text-muted">
                      The first import works through your history in the background. Answers get
                      more complete as it goes.
                    </p>
                  )}
                  {condition === 'broken' && (
                    <p className="mt-2 text-2xs leading-relaxed text-bad">
                      {state?.last_error?.class === 'AUTH_EXPIRED' ||
                      state?.last_error?.class === 'AUTH_REVOKED'
                        ? 'Google needs reconnecting before this can update.'
                        : 'Google turned the last request away. It will try again on its own.'}
                    </p>
                  )}
                </li>
              )
            })}
          </ul>

          {status?.next_scheduled_at && (
            <p className="mt-3 px-1 text-2xs text-muted">
              Next check {relativeTime(status.next_scheduled_at)}. Ryuk checks about every{' '}
              {Math.round(target / 60)} minutes on its own.
            </p>
          )}

          {note && <p className="mt-3 px-1 text-2xs text-bad">{note}</p>}
        </div>

        <div className="border-t border-line px-5 py-3">
          <button
            type="button"
            onClick={() => void checkNow()}
            disabled={checking}
            className="w-full rounded-lg border border-line px-3 py-2 text-sm font-medium text-ink transition hover:bg-elevated disabled:opacity-50"
          >
            {checking ? 'Checking…' : 'Check for new items now'}
          </button>
        </div>
      </div>
    </div>
  )
}
