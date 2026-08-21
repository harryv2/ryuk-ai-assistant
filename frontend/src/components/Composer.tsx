/**
 * The box you type in.
 *
 * One rounded field with the controls tucked inside it, in the same lane as
 * the conversation above — a composer that spans the window while the thread
 * sits in a narrow column reads as two different pages.
 *
 * The allowance is a ring, not a counter. "96 of 100" all day is noise that
 * trains you to ignore the one time it says 4.
 */

import { useEffect, useRef, useState } from 'react'

import { LANE } from '../lib/layout'
import type { ComposerProps } from '../lib/types'
import UsageRing from './UsageRing'

export default function Composer({
  value,
  onChange,
  onSend,
  enabled,
  busy,
  answeringLabel,
  placeholder,
  remaining,
  limit,
}: ComposerProps) {
  const box = useRef<HTMLTextAreaElement>(null)
  const [focused, setFocused] = useState(false)

  // Grow with the text, up to a point, then scroll. A box that grows forever
  // pushes the conversation off the screen you are writing about.
  //
  // An empty box gets no inline height at all, so the CSS minimum governs. The
  // first measurement can land before the surrounding layout has settled, and
  // a wrong number written into `style.height` would otherwise stay there for
  // as long as the box stays empty — which is exactly when you are looking at
  // it. Re-measuring on the next frame covers that first pass.
  useEffect(() => {
    const el = box.current
    if (!el) return

    const fit = () => {
      el.style.height = 'auto'
      el.style.height = value ? `${Math.min(el.scrollHeight, 220)}px` : ''
    }

    fit()
    const frame = requestAnimationFrame(fit)
    return () => cancelAnimationFrame(frame)
  }, [value])

  const canSend = enabled && !busy && value.trim().length > 0
  const showRing = typeof remaining === 'number' && typeof limit === 'number' && limit > 0

  function submit() {
    if (!canSend) return
    onSend(value.trim())
  }

  return (
    <div className="bg-surface pb-3 pt-2 sm:pb-4">
      <div className={LANE}>
        {answeringLabel && (
          <p className="mb-2 flex items-center gap-1.5 px-1 text-xs text-muted">
            <span className="h-1.5 w-1.5 rounded-full bg-accent" aria-hidden />
            {answeringLabel}
          </p>
        )}

        <div
          data-focus-shell
          className={`flex items-end gap-2 rounded-[26px] border bg-base px-3 py-2 transition ${
            enabled
              ? focused
                ? // One quiet step, not a highlight. You already know where the
                  // cursor is — the border only has to confirm it.
                  'border-line-strong shadow-md'
                : 'border-line shadow-sm hover:border-line-strong'
              : 'border-line opacity-60'
          }`}
        >
          <textarea
            ref={box}
            rows={1}
            value={value}
            disabled={!enabled}
            onChange={(e) => onChange(e.target.value)}
            onFocus={() => setFocused(true)}
            onBlur={() => setFocused(false)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                submit()
              }
            }}
            aria-label="Message"
            placeholder={placeholder ?? 'Message Ryuk'}
            className="max-h-[220px] min-h-[28px] flex-1 resize-none bg-transparent px-1.5 py-1 text-[15px] leading-6 text-ink outline-none placeholder:text-muted disabled:cursor-not-allowed"
          />

          <div className="flex shrink-0 items-center gap-1 pb-0.5">
            {showRing && <UsageRing remaining={remaining} limit={limit} />}

            <button
              type="button"
              onClick={submit}
              disabled={!canSend}
              aria-label="Send"
              className="flex h-8 w-8 items-center justify-center rounded-full bg-accent text-accent-ink transition enabled:hover:opacity-90 disabled:bg-line disabled:text-muted"
            >
              {busy ? (
                <span className="h-3 w-3 animate-spin rounded-full border-2 border-current border-t-transparent" />
              ) : (
                <svg viewBox="0 0 16 16" aria-hidden className="h-4 w-4">
                  <path
                    d="M8 13V3.5M8 3.5L4.25 7.25M8 3.5l3.75 3.75"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.75"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
              )}
            </button>
          </div>
        </div>

        <p className="mt-2 hidden text-center text-2xs text-faint sm:block">
          Ryuk can make mistakes. Check anything that matters.
        </p>
      </div>
    </div>
  )
}
