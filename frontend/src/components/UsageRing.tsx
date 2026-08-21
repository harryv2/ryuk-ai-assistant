/**
 * How much of this hour's allowance is left, as a ring.
 *
 * A ring rather than a number because most of the time the answer is "plenty"
 * and a glance should be enough to know that. The number is one tap away for
 * the times it isn't.
 */

import { useEffect, useRef, useState } from 'react'

const RADIUS = 7
const CIRCUMFERENCE = 2 * Math.PI * RADIUS

interface UsageRingProps {
  remaining: number
  limit: number
}

export default function UsageRing({ remaining, limit }: UsageRingProps) {
  const [open, setOpen] = useState(false)
  const box = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const away = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false)
    }
    const esc = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false)
    window.addEventListener('mousedown', away)
    window.addEventListener('keydown', esc)
    return () => {
      window.removeEventListener('mousedown', away)
      window.removeEventListener('keydown', esc)
    }
  }, [open])

  const left = Math.max(0, Math.min(1, limit > 0 ? remaining / limit : 1))
  const used = 1 - left
  // Green while there is room, amber when it starts to matter, red at the end.
  const tone = left <= 0 ? 'text-bad' : left <= 0.1 ? 'text-warn' : 'text-accent'

  return (
    <div ref={box} className="relative">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-label={`${remaining} of ${limit} questions left this hour`}
        title={`${remaining} of ${limit} left this hour`}
        className="flex h-8 w-8 items-center justify-center rounded-lg text-muted transition hover:bg-elevated"
      >
        <svg viewBox="0 0 18 18" aria-hidden className="h-[18px] w-[18px] -rotate-90">
          <circle
            cx="9"
            cy="9"
            r={RADIUS}
            fill="none"
            strokeWidth="2.5"
            className="stroke-line"
          />
          <circle
            cx="9"
            cy="9"
            r={RADIUS}
            fill="none"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeDasharray={`${used * CIRCUMFERENCE} ${CIRCUMFERENCE}`}
            className={`stroke-current ${tone}`}
          />
        </svg>
      </button>

      {open && (
        <div className="absolute bottom-full right-0 z-50 mb-2 w-64 rounded-xl border border-line bg-surface p-3.5 shadow-lg">
          <div className="flex items-baseline justify-between gap-2">
            <span className="text-xs font-medium text-ink">Questions this hour</span>
            <span className="text-xs tabular-nums text-muted">
              {limit - remaining} / {limit}
            </span>
          </div>
          <div className="mt-2 h-1.5 w-full overflow-hidden rounded-full bg-line">
            <div
              className={`h-full rounded-full transition-all ${
                left <= 0 ? 'bg-bad' : left <= 0.1 ? 'bg-warn' : 'bg-accent'
              }`}
              style={{ width: `${used * 100}%` }}
            />
          </div>
          <p className="mt-2.5 text-2xs leading-relaxed text-muted">
            {left <= 0
              ? 'You have used this hour’s questions. It refills gradually.'
              : `${remaining} left. Refills gradually.`}
          </p>
        </div>
      )}
    </div>
  )
}
