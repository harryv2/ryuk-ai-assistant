import { useMemo, type ReactNode } from 'react'

import { api } from '../lib/api'
import { degradedSubject, stripMarkdown, truncate } from '../lib/format'
import type { ChatProps, PendingInput } from '../lib/types'
import Composer from './Composer'
import Empty from './Empty'
import MessageList from './MessageList'
import { LANE } from '../lib/layout'

/**
 * The left column: what was said, the cards that need an answer, and the box.
 *
 * The shell owns the run and the handlers; this is the arrangement of them. The
 * composer stays live while a card is open on purpose — "yes" and "the second
 * one" are answers.
 */

function Banner({
  tone,
  children,
}: {
  tone: 'warn' | 'bad' | 'accent'
  children: ReactNode
}) {
  const className =
    tone === 'bad'
      ? 'bg-bad/10 text-ink'
      : tone === 'warn'
        ? 'bg-warn/10 text-ink'
        : 'bg-accent-soft text-ink'
  return (
    <div className={`animate-fade-in border-b border-line px-4 py-2 text-xs ${className}`}>
      <div className={`${LANE} flex flex-wrap items-center gap-x-3 gap-y-1`}>
        {children}
      </div>
    </div>
  )
}

export default function Chat({
  messages,
  run,
  prompts,
  actions,
  draft,
  onDraftChange,
  composerEnabled,
  busy,
  onSend,
  onRespond,
  onCancelPrompt,
  onSearchDifferently,
  onRedo,
  notice,
  steps = [],
  remaining,
  limit,
  emptyHint,
}: ChatProps) {
  // The card the composer is standing in front of, if any.
  const openPrompt = useMemo<PendingInput | null>(() => {
    const ordered = run.promptOrder
      .map((id) => prompts[id])
      .filter((prompt): prompt is PendingInput => Boolean(prompt))
    const pool = ordered.length ? ordered : Object.values(prompts)
    return pool.find((prompt) => prompt.status === 'pending') ?? null
  }, [run.promptOrder, prompts])

  // Echoing the card's own text above the box is the same sentence twice, and
  // when the card is a reconnect notice the "answer" framing is simply wrong.
  const answeringLabel = openPrompt
    ? openPrompt.prompt.variant === 'reconnect'
      ? 'Reconnect Google to carry on'
      : `Answering: ${truncate(stripMarkdown(openPrompt.prompt.question), 44)}`
    : null

  const needsReauth = run.error?.code === 'GOOGLE_REAUTH_REQUIRED'

  return (
    <section className="flex h-full min-h-0 flex-col bg-canvas" aria-label="Conversation">
      {notice && (
        <Banner tone="accent">
          <span>{notice}</span>
        </Banner>
      )}

      {run.error && (
        <Banner tone="bad">
          <span>
            <span className="font-semibold text-bad">{run.error.message}</span>
            {run.error.partial && ' Some of the answer still landed.'}
          </span>
          {needsReauth ? (
            <a
              href={api.googleAuthUrl('/')}
              className="ml-auto rounded-lg bg-accent px-2.5 py-1 text-2xs font-semibold text-accent-ink transition-opacity hover:opacity-90"
            >
              Reconnect Google
            </a>
          ) : (
            run.query &&
            !busy && (
              <button
                type="button"
                onClick={() => run.query && onSend(run.query)}
                className="ml-auto rounded-lg bg-accent px-2.5 py-1 text-2xs font-semibold text-accent-ink transition-opacity hover:opacity-90"
              >
                Try again
              </button>
            )
          )}
        </Banner>
      )}

      {run.degraded.length > 0 && (
        <Banner tone="warn">
          <span>
            {run.degraded
              .map((entry) => `${degradedSubject(entry)} — ${entry.detail ?? entry.reason}`)
              .join(' · ')}
            . That part is missing from this answer.
          </span>
          {run.query && !busy && (
            <button
              type="button"
              onClick={() => run.query && onSend(run.query)}
              className="ml-auto rounded-lg border border-line bg-surface px-2.5 py-1 text-2xs font-semibold text-ink transition-colors hover:bg-canvas"
            >
              Try again
            </button>
          )}
        </Banner>
      )}

      {run.status === 'awaiting_input' && (
        <Banner tone="accent">
          <span>
            Waiting on your answer above. You can also just type below.
          </span>
        </Banner>
      )}

      {messages.length === 0 ? (
        <Empty onPick={onSend} hints={emptyHint} disabled={busy} />
      ) : (
        <MessageList
          messages={messages}
          run={run}
          steps={steps}
          prompts={prompts}
          actions={actions}
          onRespond={onRespond}
          onCancelPrompt={onCancelPrompt}
          onSearchDifferently={onSearchDifferently}
          onRedo={onRedo}
          onAsk={(question) => {
            // A question that ends mid-sentence — "Move X to " — is the widget
            // asking the person to finish it, so it goes in the box rather
            // than straight down the wire.
            if (/[\s:]$/.test(question)) onDraftChange(question)
            else onSend(question)
          }}
        />
      )}

      <Composer
        value={draft}
        onChange={onDraftChange}
        onSend={onSend}
        enabled={composerEnabled}
        busy={busy}
        answeringLabel={answeringLabel}
        remaining={remaining}
        limit={limit}
      />
    </section>
  )
}
