/**
 * What the assistant is doing, shown while it does it.
 *
 * The point is reassurance, not instrumentation. A person waiting eight seconds
 * needs to know something is happening and roughly what — not the op name, the
 * node id, or how many milliseconds Postgres took. So this shows one line at a
 * time in plain words while the work runs, then collapses to a single summary
 * the moment there is an answer to read.
 *
 * The detail is still one click away, because when an answer looks wrong the
 * first question is always "where did you get that".
 */

import { useEffect, useState } from 'react'

import type { TraceStep } from '../lib/types'

interface Props {
  steps: TraceStep[]
  /** True while the run is still going. Drives the live view vs the summary. */
  running: boolean
  /** "Searched Sun 23 – Sat 29 Aug", when a date range was worked out. */
  window?: string | null
}

const DOT: Record<string, string> = {
  running: 'bg-accent animate-pulse',
  succeeded: 'bg-ok',
  failed: 'bg-bad',
  skipped: 'bg-idle',
  timeout: 'bg-bad',
  cancelled: 'bg-idle',
  pending: 'bg-idle',
}

/** The progress label, with anything engineer-shaped stripped out. */
function phrase(step: TraceStep): string {
  const label = (step.label || '').trim()
  // A label with a dot in it is an op name that escaped — "gcal.search_events"
  // is the sort of thing this whole function exists to keep off the screen.
  if (label && !label.includes('.')) return label

  const op = step.op || ''
  // The exact op first — "Writing a draft" and "Looking through your mail"
  // are different activities, and labelling a compose as a search taught a
  // real user that composing was searching.
  const exact: Record<string, string> = {
    'gmail.draft_email': 'Writing a draft',
    'gmail.send_email': 'Preparing an email',
    'gmail.update_labels': 'Updating labels',
    'gmail.get_email': 'Opening the email',
    'gcal.create_event': 'Adding to your calendar',
    'gcal.update_event': 'Updating the event',
    'gcal.delete_event': 'Removing the event',
    'gcal.get_events': 'Opening the event',
    'gcal.find_free_slots': 'Finding free time',
    'gcal.find_conflicts': 'Checking for clashes',
    'gdrive.get_file': 'Opening the file',
    'gdrive.share_file': 'Preparing to share',
    'gdrive.move_file': 'Preparing to move',
    'gdrive.create_folder': 'Making a folder',
    'chat.set_title': 'Naming this chat',
  }
  if (exact[op]) return exact[op]
  if (op.startsWith('gmail')) return 'Looking through your mail'
  if (op.startsWith('gcal') || op.startsWith('calendar')) return 'Checking your calendar'
  if (op.startsWith('gdrive') || op.startsWith('drive')) return 'Looking through your files'
  if (op.startsWith('search')) return 'Searching'
  if (op.startsWith('ask')) return 'Waiting for you'
  if (op.startsWith('llm') || op.startsWith('meta')) return 'Reading what it found'
  if (op.startsWith('resolve')) return 'Working out who you mean'
  if (op.startsWith('data')) return 'Narrowing it down'
  if (op.startsWith('page')) return 'Getting more'
  if (op.startsWith('action')) return 'Updating the draft'
  return 'Working'
}

export default function Activity({ steps, running, window: windowLabel }: Props) {
  const [open, setOpen] = useState(false)
  const done = steps.filter((s) => s.status !== 'pending' && s.status !== 'running')
  const active = steps.find((s) => s.status === 'running')

  // While it runs, the newest thing is the interesting thing.
  const [shown, setShown] = useState<string | null>(null)
  useEffect(() => {
    if (active) setShown(phrase(active))
    else if (!running) setShown(null)
  }, [active, running])

  if (!steps.length && !running) return null

  if (running) {
    // Show the work as it happens, not a single word standing in for it.
    //
    // The plan arrives one step at a time over a couple of seconds and each
    // step then runs, so there is real movement to show. Collapsing all of
    // that to "Thinking…" and only revealing the list once the run has
    // finished is why the steps looked like they arrived *after* the answer:
    // they were there the whole time, just not drawn.
    return (
      <div className="mt-2 space-y-1.5">
        {/* The header names the phase; the list below names the steps. Showing
            the active step in both puts the same sentence on screen twice. */}
        <div className="flex items-center gap-2 text-sm text-muted">
          <span className="flex gap-1" aria-hidden>
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted [animation-delay:-0.3s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted [animation-delay:-0.15s]" />
            <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted" />
          </span>
          <span>{steps.length ? 'Working' : (shown ?? 'Thinking')}…</span>
        </div>

        {steps.length > 0 && (
          <ul className="ml-1 space-y-1 border-l border-line pl-3">
            {steps.map((step, i) => {
              const state = step.status
              const settled = state !== 'pending' && state !== 'running'
              return (
                <li
                  key={`${step.node_id}-${i}`}
                  className={`flex items-center gap-2 text-xs transition-opacity ${
                    state === 'pending' ? 'opacity-45' : 'opacity-100'
                  }`}
                >
                  <span
                    aria-hidden
                    className={`h-1.5 w-1.5 shrink-0 rounded-full ${
                      state === 'running'
                        ? 'animate-pulse bg-accent'
                        : settled
                          ? DOT[state] ?? 'bg-ok'
                          : 'bg-line'
                    }`}
                  />
                  <span className={state === 'running' ? 'text-ink' : 'text-muted'}>
                    {phrase(step)}
                  </span>
                  {state === 'running' && (
                    <span className="text-2xs text-faint">running</span>
                  )}
                </li>
              )
            })}
          </ul>
        )}
      </div>
    )
  }

  const failed = done.filter((s) => s.status === 'failed' || s.status === 'timeout').length
  const summary = failed
    ? `${done.length} ${done.length === 1 ? 'step' : 'steps'} · ${failed} did not finish`
    : `${done.length} ${done.length === 1 ? 'step' : 'steps'}`

  return (
    <div className="mt-2">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex items-center gap-1.5 rounded-md py-0.5 text-xs text-muted transition hover:text-ink"
      >
        <svg
          viewBox="0 0 12 12"
          aria-hidden
          className={`h-3 w-3 transition-transform ${open ? 'rotate-90' : ''}`}
        >
          <path d="M4 2l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
        </svg>
        <span>{summary}</span>
        {windowLabel && <span className="hidden sm:inline">· {windowLabel}</span>}
      </button>

      {open && (
        <ol className="mt-2 space-y-1.5 border-l border-line pl-3">
          {done.map((step, i) => (
            <li key={`${step.node_id}-${i}`} className="flex items-start gap-2 text-xs">
              <span
                aria-hidden
                className={`mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full ${DOT[step.status] ?? 'bg-idle'}`}
              />
              <span className="min-w-0 flex-1">
                <span className="text-ink">{phrase(step)}</span>
                {step.status === 'skipped' && step.skipReason && (
                  <span className="text-muted"> — {step.skipReason}</span>
                )}
                {(step.status === 'failed' || step.status === 'timeout') && (
                  <span className="text-bad"> — did not finish</span>
                )}
              </span>
              {typeof step.duration_ms === 'number' && step.duration_ms > 0 && (
                <span className="shrink-0 tabular-nums text-muted">
                  {step.duration_ms < 1000
                    ? `${step.duration_ms}ms`
                    : `${(step.duration_ms / 1000).toFixed(1)}s`}
                </span>
              )}
            </li>
          ))}
        </ol>
      )}
    </div>
  )
}
