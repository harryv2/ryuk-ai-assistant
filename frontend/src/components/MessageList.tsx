import { useEffect, useMemo, useRef, useState } from 'react'

import { opLabel, relativeTime, renderMarkdown, windowLabel } from '../lib/format'
import { LANE } from '../lib/layout'
import type { ActionRecord, ChatMessage, Id, MessageListProps } from '../lib/types'
import ActionCard from './ActionCard'
import Activity from './Activity'
import Widget, { canRender } from './Widget'
import PromptCard from './PromptCard'

/**
 * The chat column's messages.
 *
 * Every message is an ordered list of content blocks. A `text` block is
 * markdown; an `action` or `input` block is a card, resolved out of the hydrated
 * maps the shell passes down. A ref with no matching object was dropped on read
 * — it is skipped, never drawn as an empty box.
 */

const PROSE =
  [
    'text-sm leading-relaxed text-ink',
    '[&_p]:mb-2 [&_p:last-child]:mb-0',
    '[&_ul]:mb-2 [&_ul]:list-disc [&_ul]:pl-5',
    '[&_ol]:mb-2 [&_ol]:list-decimal [&_ol]:pl-5',
    '[&_li]:mb-0.5 [&_li]:pl-0.5',
    '[&_strong]:font-semibold [&_em]:italic',
    '[&_a]:text-accent [&_a]:underline [&_a]:underline-offset-2',
    // A backticked value is usually a booking reference or a flight number, so
    // it gets a quiet tint that keeps it readable rather than terminal green.
    '[&_code]:rounded [&_code]:bg-elevated [&_code]:px-1 [&_code]:py-px [&_code]:text-[0.85em] [&_code]:font-medium',
    '[&_.md-heading]:mt-3 [&_.md-heading]:font-semibold [&_.md-heading:first-child]:mt-0',
    '[&_hr]:my-3 [&_hr]:border-line',
  ].join(' ')

function describeAction(action: ActionRecord | undefined): string {
  if (!action) return ''
  const subject = action.preview?.subject
  return subject ? `${opLabel(action.op)} — ${subject}` : opLabel(action.op)
}

/**
 * Copy one message's text. The tick is the entire feedback — a toast for a
 * clipboard write is more interruption than information.
 */
function CopyButton({ text, className = '' }: { text: string; className?: string }) {
  const [copied, setCopied] = useState(false)
  if (!text.trim()) return null
  return (
    <button
      type="button"
      aria-label={copied ? 'Copied' : 'Copy message'}
      title={copied ? 'Copied' : 'Copy'}
      onClick={() => {
        void navigator.clipboard?.writeText(text).then(() => {
          setCopied(true)
          window.setTimeout(() => setCopied(false), 1600)
        })
      }}
      className={`inline-flex h-6 w-6 items-center justify-center rounded-md text-faint transition-colors hover:bg-elevated hover:text-ink ${className}`}
    >
      {copied ? (
        <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="none" aria-hidden="true">
          <path d="M3 8.5 6.5 12 13 4.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      ) : (
        <svg viewBox="0 0 16 16" className="h-3.5 w-3.5" fill="none" aria-hidden="true">
          <rect x="5.5" y="5.5" width="8" height="8" rx="1.5" stroke="currentColor" strokeWidth="1.3" />
          <path d="M10.5 3.5v-1a1 1 0 0 0-1-1h-6a1 1 0 0 0-1 1v6a1 1 0 0 0 1 1h1" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
        </svg>
      )}
    </button>
  )
}

export default function MessageList({
  messages,
  run,
  steps = [],
  prompts,
  actions,
  onRespond,
  onCancelPrompt,
  onSearchDifferently,
  onRedo,
  onAsk,
}: MessageListProps) {
  const bottomRef = useRef<HTMLDivElement | null>(null)

  /** Which items the person unticked, per gating prompt. */
  const [excluded, setExcluded] = useState<Record<Id, Id[]>>({})

  // Follow the conversation as it grows and as text streams in.
  const signature = useMemo(() => {
    const last = messages[messages.length - 1]
    return [
      messages.length,
      last?.id ?? '',
      last?.content.length ?? 0,
      run.answerText.length,
      run.promptOrder.length,
      run.actionOrder.length,
      // Steps arrive before any answer text does. Without them the view sits
      // still through the whole working phase and only jumps at the end,
      // which is exactly when it looks like nothing was happening.
      steps.length,
    ].join(':')
  }, [messages, run.answerText, run.promptOrder, run.actionOrder, steps.length])

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: 'end', behavior: 'smooth' })
  }, [signature])

  const toggleExcluded = (promptId: Id, actionId: Id, included: boolean) => {
    setExcluded((current) => {
      const list = current[promptId] ?? []
      const next = included ? list.filter((id) => id !== actionId) : [...new Set([...list, actionId])]
      return { ...current, [promptId]: next }
    })
  }

  /** Unticked items ride along as a note; the approval itself is whole-card. */
  const respondFor = (promptId: Id) => (value: unknown) => {
    const skipped = excluded[promptId] ?? []
    const approving =
      Boolean(value) &&
      typeof value === 'object' &&
      !Array.isArray(value) &&
      (value as { approve?: unknown }).approve === true

    if (skipped.length && approving) {
      const names = skipped.map((id) => describeAction(actions[id])).filter(Boolean)
      onRespond(promptId, {
        ...(value as Record<string, unknown>),
        note: `Skip: ${names.join('; ')}`,
      })
      return
    }
    onRespond(promptId, value)
  }

  // The activity belongs to the turn in progress, which is the END of the
  // thread — not to "the last assistant message", which during a fresh run is
  // still the *previous* answer further up the page. Anchoring there is why it
  // looked like the steps arrived late: they were rendering, just not here.
  const last = messages[messages.length - 1]
  const anchorId = last?.role === 'assistant' ? last.id : null
  // Before this turn's answer exists there is nothing to hang it on, so it
  // trails the list instead.
  const trailingActivity = !anchorId && (steps.length > 0 || run.streaming)

  const renderMessage = (message: ChatMessage) => {
    if (message.role === 'user') {
      const text = message.content
        .map((block) => (block.type === 'text' ? block.data.markdown : ''))
        .join('\n')
        .trim()
      return (
        <li key={message.id} className="group flex flex-col items-end gap-1">
          <div
            className={`max-w-[80%] whitespace-pre-wrap rounded-[20px] bg-accent-soft px-4 py-2.5 text-[15px] leading-6 text-ink ${
              message.optimistic ? 'opacity-70' : ''
            }`}
          >
            {text}
          </div>
          <CopyButton
            text={text}
            className="opacity-0 transition-opacity group-hover:opacity-100 group-focus-within:opacity-100 max-sm:opacity-60"
          />
        </li>
      )
    }

    // One prompt can gate several actions — the calendar change and the email
    // that announces it. Number them and let each be unticked.
    const groups = new Map<Id, Id[]>()
    for (const block of message.content) {
      if (block.type !== 'action') continue
      const action = actions[block.ref]
      const gate = action?.requires_input_id
      if (!action || !gate) continue
      groups.set(gate, [...(groups.get(gate) ?? []), action.id])
    }

    // A widget carries the markdown it replaced, so that the answer survives a
    // client that cannot draw it. When this one can, showing both is the same
    // answer twice.
    const replaced = new Set(
      message.content
        .filter(
          (block) =>
            block.type === 'widget' &&
            canRender(block) &&
            // Only a widget that *is* the answer stands in for the words. One
            // the model offered sits under prose that says more than it does.
            (block as { replaces_text?: boolean }).replaces_text === true,
        )
        .map((block) => (block as { text: string }).text.trim()),
    )

    const textBlocks = message.content.filter((block) => block.type === 'text')
    const lastTextIndex = message.content.lastIndexOf(textBlocks[textBlocks.length - 1])
    const hasVisibleText = textBlocks.some(
      (block) => block.type === 'text' && block.data.markdown.trim(),
    )

    const rendered = message.content.flatMap((block, index) => {
      if (block.type === 'text') {
        const markdown = block.data.markdown
        if (!markdown.trim()) return []
        if (replaced.has(markdown.trim())) return []
        return (
          <div key={`${message.id}-text-${index}`} className="relative">
            <div className={PROSE} dangerouslySetInnerHTML={{ __html: renderMarkdown(markdown) }} />
            {message.streaming && index === lastTextIndex && (
              <span className="ml-px inline-block animate-caret text-accent" aria-hidden="true">
                ▍
              </span>
            )}
          </div>
        )
      }

      if (block.type === 'widget') {
        // The text block above already carries this answer in words. A widget
        // this client cannot draw is therefore a non-event, not a hole.
        if (!canRender(block)) return []
        return (
          <div key={`${message.id}-widget-${index}`} className="pt-0.5">
            <Widget block={block} onAsk={onAsk} />
          </div>
        )
      }

      if (block.type === 'action') {
        const action = actions[block.ref]
        if (!action) return []
        const gate = action.requires_input_id ?? null
        const prompt = gate ? (prompts[gate] ?? null) : null
        const group = gate ? (groups.get(gate) ?? []) : []
        const grouped = group.length > 1
        const promptHasOwnBlock = message.content.some(
          (other) => other.type === 'input' && other.ref === gate,
        )
        return (
          <ActionCard
            key={`${message.id}-action-${block.ref}`}
            action={action}
            prompt={prompt}
            streamingDraft={message.drafts?.[action.id]}
            streaming={Boolean(message.streaming)}
            busy={gate ? run.promptBusy[gate] : false}
            error={gate ? (run.promptErrors[gate] ?? null) : null}
            showControls={!promptHasOwnBlock}
            index={grouped ? group.indexOf(action.id) : undefined}
            total={grouped ? group.length : undefined}
            included={!gate || !(excluded[gate] ?? []).includes(action.id)}
            onToggleIncluded={
              grouped && gate ? (next) => toggleExcluded(gate, action.id, next) : undefined
            }
            onRespond={gate ? respondFor(gate) : () => undefined}
            onCancel={gate ? () => onCancelPrompt(gate) : () => undefined}
            onRetry={prompt ? () => onRedo(prompt) : undefined}
          />
        )
      }

      const prompt = prompts[block.ref]
      if (!prompt) return []
      const gated = Object.values(actions).find(
        (action) => action.requires_input_id === prompt.id,
      )
      const skipped = excluded[prompt.id] ?? []
      return (
        <div key={`${message.id}-input-${block.ref}`}>
          {skipped.length > 0 && prompt.status === 'pending' && (
            <p className="mb-1.5 text-2xs text-muted">
              Unticked items go through as a note on the approval — the card is approved as a
              whole.
            </p>
          )}
          <PromptCard
            key={prompt.id}
            prompt={prompt}
            action={gated ?? null}
            streamingDraft={gated ? message.drafts?.[gated.id] : undefined}
            busy={run.promptBusy[prompt.id] ?? false}
            error={run.promptErrors[prompt.id] ?? null}
            onRespond={respondFor(prompt.id)}
            onCancel={() => onCancelPrompt(prompt.id)}
            onSearchDifferently={() => onSearchDifferently(prompt)}
            onRedo={() => onRedo(prompt)}
          />
        </div>
      )
    })

    // The activity belongs to the newest answer, above it — you watch it work,
    // then the answer lands underneath. It also stands in for the "working…"
    // line, since it says the same thing with more in it.
    const showActivity =
      message.id === anchorId && (steps.length > 0 || Boolean(message.streaming))

    const working =
      message.streaming && !showActivity && !hasVisibleText && rendered.length === 0 ? (
        <p className="flex items-center gap-2 text-sm text-muted">
          <span className="relative flex h-2 w-2">
            <span className="absolute h-2 w-2 animate-halo rounded-full bg-accent" />
            <span className="relative h-2 w-2 animate-breathe rounded-full bg-accent" />
          </span>
          {run.progress?.label ?? 'Working on it…'}
        </p>
      ) : null

    return (
      <li key={message.id} className="group animate-fade-in space-y-3">
        {showActivity && (
          <Activity
            steps={steps}
            running={Boolean(message.streaming)}
            window={windowLabel(run.resolvedWindow)}
          />
        )}
        {working}
        {rendered}
        {!message.streaming && (
          <div className="flex items-center gap-1.5">
            <p className="text-2xs text-faint">{relativeTime(message.created_at)}</p>
            <CopyButton
              text={message.content
                .map((block) => (block.type === 'text' ? block.data.markdown : ''))
                .filter(Boolean)
                .join('\n\n')
                .trim()}
              className="opacity-0 transition-opacity group-hover:opacity-100 max-sm:opacity-60"
            />
          </div>
        )}
      </li>
    )
  }

  return (
    <div className="min-h-0 flex-1 overflow-y-auto py-6">
      <ul className={`${LANE} flex flex-col gap-7`}>
        {messages.map(renderMessage)}
        {trailingActivity && (
          <li className="animate-fade-in">
            <Activity
              steps={steps}
              running={run.streaming}
              window={windowLabel(run.resolvedWindow)}
            />
          </li>
        )}
      </ul>
      <div ref={bottomRef} className="h-px" />
    </div>
  )
}
