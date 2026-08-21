import { useState } from 'react'

import { formatCount, formatDateTime, opLabel, relativeTime } from '../lib/format'
import type { ActionCardProps, ActionRecord, ActionStatus } from '../lib/types'

/**
 * A prepared write, shown before it happens.
 *
 * `draft` means prepared and *not* done — the Gmail draft already exists in the
 * user's own account, and nothing has been sent. The body preview grows as
 * `content.delta` events stream the drafted text in.
 */

const STATUS: Record<ActionStatus, { label: string; className: string }> = {
  draft: { label: 'Draft — nothing sent yet', className: 'bg-elevated text-muted' },
  approved: { label: 'Approved', className: 'bg-accent-soft text-accent' },
  queued: { label: 'Queued', className: 'bg-accent-soft text-accent' },
  running: { label: 'Sending…', className: 'bg-warn/15 text-warn' },
  done: { label: 'Sent ✓', className: 'bg-ok/15 text-ok' },
  failed: { label: 'Failed', className: 'bg-bad/15 text-bad' },
  cancelled: { label: 'Cancelled', className: 'bg-elevated text-faint' },
}

/** Fields the card lays out itself. Everything else gets the generic list. */
const OWN_FIELDS = new Set(['to', 'cc', 'bcc', 'subject', 'body', 'body_excerpt', 'attachments'])

const ISO_LIKE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/

function previewValue(value: unknown): string {
  if (value == null) return '—'
  if (Array.isArray(value)) return value.map(previewValue).join(', ')
  if (typeof value === 'object') return JSON.stringify(value)
  const text = String(value)
  if (ISO_LIKE.test(text) && !Number.isNaN(Date.parse(text))) return formatDateTime(text)
  return text
}

/** The link on a finished action, when the result carries one we can trust. */
function resultLink(action: ActionRecord): { href: string; label: string } | null {
  const result = (action.result ?? {}) as Record<string, unknown>
  for (const key of ['html_link', 'htmlLink', 'url', 'link', 'web_link']) {
    const candidate = result[key]
    if (typeof candidate === 'string' && /^https:\/\//i.test(candidate)) {
      return { href: candidate, label: 'Open it' }
    }
  }
  const messageId = result.message_id
  if (typeof messageId === 'string' && action.op.startsWith('gmail.')) {
    return {
      href: `https://mail.google.com/mail/u/0/#all/${encodeURIComponent(messageId)}`,
      label: 'Open in Gmail',
    }
  }
  return null
}

/** The server keeps the previous payloads; this view shows them if they came. */
function revisionsOf(action: ActionRecord): unknown[] {
  const raw = (action as ActionRecord & { revisions?: unknown }).revisions
  return Array.isArray(raw) ? raw : []
}

export interface ActionCardExtraProps {
  /** True while this run is still writing the body into the card. */
  streaming?: boolean
  /** Show approve/decline here. Off when the gating prompt has its own card. */
  showControls?: boolean
  /** Position when one prompt gates several actions: "1 of 2". */
  index?: number
  total?: number
  /** Per-item toggle, for the same several-actions case. */
  included?: boolean
  onToggleIncluded?: (next: boolean) => void
  /** Failed action: ask the question again rather than pretending we can resend. */
  onRetry?: () => void
}

export default function ActionCard({
  action,
  prompt,
  streamingDraft,
  busy = false,
  error = null,
  onRespond,
  onCancel,
  streaming = false,
  showControls = false,
  index,
  total,
  included = true,
  onToggleIncluded,
  onRetry,
}: ActionCardProps & ActionCardExtraProps) {
  const [showOriginal, setShowOriginal] = useState(false)

  const preview = action.preview ?? {}
  const status = STATUS[action.status] ?? STATUS.draft
  const link = action.status === 'done' ? resultLink(action) : null
  const revisions = revisionsOf(action)
  const edited = revisions.length > 0 || (action.revision ?? 0) > 0

  // The drafted body only ever grows: prefer whichever text is longer, so a
  // late `action.prepared` excerpt never shortens what already streamed in.
  const streamed = streamingDraft ?? ''
  const settled = (preview.body as string | undefined) ?? preview.body_excerpt ?? ''
  const body = streamed.length >= settled.length ? streamed : settled
  const stillWriting = streaming && streamed.length >= settled.length && action.status === 'draft'

  const grouped = typeof total === 'number' && total > 1
  const extras = Object.entries(preview).filter(
    ([key, value]) => !OWN_FIELDS.has(key) && value != null && value !== '',
  )
  const attachments = Array.isArray(preview.attachments) ? preview.attachments : []
  const controls = showControls && prompt?.status === 'pending'

  return (
    <article
      className={`animate-rise-in overflow-hidden rounded-xl border border-line bg-surface shadow-card ${
        grouped && !included ? 'opacity-55' : ''
      }`}
    >
      <header className="flex flex-wrap items-center gap-2 border-b border-line bg-elevated/60 px-3 py-2">
        {grouped && (
          <span className="flex items-center gap-1.5">
            <input
              id={`include-${action.id}`}
              type="checkbox"
              checked={included}
              disabled={!onToggleIncluded || prompt?.status !== 'pending' || busy}
              onChange={(event) => onToggleIncluded?.(event.target.checked)}
              className="h-3.5 w-3.5 rounded border-line accent-accent"
            />
            <label
              htmlFor={`include-${action.id}`}
              className="text-2xs font-medium tabular-nums text-muted"
            >
              {index != null ? `${index + 1} of ${total}` : 'include'}
            </label>
          </span>
        )}
        <h4 className="text-xs font-semibold text-ink">{opLabel(action.op)}</h4>
        <span className={`rounded-full px-2 py-0.5 text-2xs font-medium ${status.className}`}>
          {status.label}
        </span>
        {link && (
          <a
            href={link.href}
            target="_blank"
            rel="noopener noreferrer"
            className="text-2xs font-medium text-accent underline underline-offset-2"
          >
            {link.label}
          </a>
        )}
        {action.expires_at && action.status === 'draft' && (
          <span className="ml-auto text-2xs text-faint">
            expires {relativeTime(action.expires_at)}
          </span>
        )}
      </header>

      <div className="space-y-2 px-3 py-2.5">
        {Array.isArray(preview.to) && preview.to.length > 0 && (
          <p className="flex gap-2 text-xs">
            <span className="w-14 shrink-0 text-faint">To</span>
            <span className="min-w-0 flex-1 break-words text-ink">{preview.to.join(', ')}</span>
          </p>
        )}
        {Array.isArray(preview.cc) && preview.cc.length > 0 && (
          <p className="flex gap-2 text-xs">
            <span className="w-14 shrink-0 text-faint">Cc</span>
            <span className="min-w-0 flex-1 break-words text-muted">{preview.cc.join(', ')}</span>
          </p>
        )}
        {preview.subject && (
          <p className="flex gap-2 text-xs">
            <span className="w-14 shrink-0 text-faint">Subject</span>
            <span className="min-w-0 flex-1 break-words font-medium text-ink">
              {preview.subject}
            </span>
          </p>
        )}

        {extras.length > 0 && (
          <dl className="space-y-1">
            {extras.map(([key, value]) => (
              <div key={key} className="flex gap-2 text-xs">
                <dt className="w-14 shrink-0 truncate text-faint" title={key}>
                  {key.replace(/_/g, ' ')}
                </dt>
                <dd className="min-w-0 flex-1 break-words text-ink">{previewValue(value)}</dd>
              </div>
            ))}
          </dl>
        )}

        {(body || stillWriting) && (
          <div className="max-h-56 overflow-y-auto whitespace-pre-wrap rounded-lg bg-canvas px-3 py-2 text-xs leading-relaxed text-ink ring-1 ring-inset ring-line">
            {body}
            {stillWriting && (
              <span className="ml-px inline-block animate-caret text-accent" aria-hidden="true">
                ▍
              </span>
            )}
          </div>
        )}

        {attachments.length > 0 && (
          <p className="text-2xs text-muted">
            {formatCount(attachments.length)} attachment{attachments.length === 1 ? '' : 's'}
          </p>
        )}

        {edited && (
          <div>
            <button
              type="button"
              onClick={() => setShowOriginal((value) => !value)}
              disabled={revisions.length === 0}
              className="text-2xs text-muted underline underline-offset-2 disabled:no-underline disabled:opacity-70"
            >
              edited
              {(action.revision ?? revisions.length) > 1
                ? ` ${formatCount(action.revision ?? revisions.length)} times`
                : ''}
              {revisions.length > 0 ? (showOriginal ? ' — hide the original' : ' — see the original') : ''}
            </button>
            {showOriginal && revisions.length > 0 && (
              <pre className="mt-1 max-h-48 overflow-auto rounded-md bg-canvas px-3 py-2 font-mono text-2xs leading-relaxed text-muted ring-1 ring-inset ring-line">
                {JSON.stringify(revisions[0], null, 2)}
              </pre>
            )}
          </div>
        )}

        {action.status === 'done' && action.executed_at && (
          <p className="text-2xs text-ok">Sent {relativeTime(action.executed_at)}.</p>
        )}

        {action.status === 'failed' && (
          <div className="rounded-lg bg-bad/10 px-3 py-2">
            <p className="text-2xs text-bad">
              {action.error?.message ?? 'It did not go through.'}
              {action.attempts ? ` (${formatCount(action.attempts)} attempts)` : ''}
            </p>
            {onRetry && (
              <button
                type="button"
                onClick={onRetry}
                className="mt-1.5 rounded-full border border-bad/40 px-2.5 py-1 text-2xs font-medium text-bad transition-colors hover:bg-bad/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-bad/40"
              >
                Try again
              </button>
            )}
          </div>
        )}

        {error && <p className="text-2xs text-bad">{error}</p>}

        {controls && (
          <div className="flex flex-wrap gap-2 pt-0.5">
            <button
              type="button"
              disabled={busy}
              onClick={() => onRespond({ approve: true })}
              className="rounded-lg bg-accent px-3 py-1.5 text-xs font-semibold text-accent-ink transition-opacity hover:opacity-90 disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
            >
              {busy ? 'Sending…' : 'Send it'}
            </button>
            <button
              type="button"
              disabled={busy}
              onClick={onCancel}
              className="rounded-lg border border-line px-3 py-1.5 text-xs font-medium text-muted transition-colors hover:bg-elevated hover:text-ink disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
            >
              Not now
            </button>
          </div>
        )}
      </div>
    </article>
  )
}
