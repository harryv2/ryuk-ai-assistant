import type { ReactNode } from 'react'
import { LANE } from '../lib/layout'

/**
 * The first-run state.
 *
 * These four are not decoration — they are the queries that work against the
 * seeded corpus, and two of them are ambiguous on purpose, so the ask-instead-of-
 * guess behaviour is one click away rather than something to take on faith.
 */

interface Suggestion {
  query: string
  note?: string
}

const SUGGESTIONS: Suggestion[] = [
  { query: "What's on my calendar next week?" },
  { query: 'Cancel my Turkish Airlines flight' },
  { query: 'Cancel my flight', note: 'two match — it will ask which' },
  { query: 'Move the meeting with John', note: 'two Johns — it will ask which' },
]

export interface EmptyProps {
  /** Clicking a suggestion sends it. */
  onPick: (query: string) => void
  /** Override the built-in list. */
  hints?: string[]
  disabled?: boolean
  children?: ReactNode
}

export default function Empty({ onPick, hints, disabled = false, children }: EmptyProps) {
  const suggestions: Suggestion[] = hints?.length
    ? hints.map((query) => ({ query }))
    : SUGGESTIONS

  return (
    <div className="flex min-h-0 flex-1 items-center overflow-y-auto py-10">
      <div className={LANE}>
        {/* No explainer. The suggestions below say what this can do far better
            than a sentence about it does, and the promise about approvals
            belongs on the card that is actually asking for one. */}
        <h2 className="text-center text-2xl font-semibold tracking-tight text-ink">
          What are you looking for?
        </h2>

        <ul className="mt-5 space-y-2">
          {suggestions.map((suggestion) => (
            <li key={suggestion.query}>
              <button
                type="button"
                disabled={disabled}
                onClick={() => onPick(suggestion.query)}
                className="group flex w-full items-center gap-3 rounded-xl border border-line bg-surface px-3.5 py-2.5 text-left transition-colors hover:border-accent/50 hover:bg-elevated disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50"
              >
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm text-ink">{suggestion.query}</span>
                  {suggestion.note && (
                    <span className="mt-0.5 block text-2xs text-faint">{suggestion.note}</span>
                  )}
                </span>
                <span
                  className="shrink-0 text-muted transition-transform group-hover:translate-x-0.5"
                  aria-hidden="true"
                >
                  →
                </span>
              </button>
            </li>
          ))}
        </ul>

        {children}
      </div>
    </div>
  )
}
