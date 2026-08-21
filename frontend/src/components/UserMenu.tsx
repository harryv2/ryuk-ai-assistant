/**
 * The account button in the corner.
 *
 * Everything that is about *you* rather than about the conversation lives here:
 * who you are, light or dark, and signing out. The header stays a title and this
 * one button, because a header full of status is a header nobody reads.
 */

import { useEffect, useRef, useState } from 'react'

interface Props {
  email: string
  displayName?: string | null
  theme: 'light' | 'dark'
  onToggleTheme: () => void
  onSignOut: () => void
  onOpenIntegrations?: () => void
  onOpenSyncStatus?: () => void
  /** Something in the sync needs looking at; shown as a dot, not a sentence. */
  needsAttention?: boolean
  /**
   * `avatar` is the circle in a header. `row` is the full-width strip at the
   * foot of the sidebar, where there is room for the name and the menu opens
   * upward because there is nothing below it.
   */
  variant?: 'avatar' | 'row'
}

function initials(name: string, email: string): string {
  const source = (name || email).trim()
  const parts = source.split(/[\s.@_-]+/).filter(Boolean)
  if (parts.length >= 2) return (parts[0][0] + parts[1][0]).toUpperCase()
  return source.slice(0, 2).toUpperCase()
}

export default function UserMenu({
  email,
  displayName,
  theme,
  onToggleTheme,
  onSignOut,
  onOpenIntegrations,
  onOpenSyncStatus,
  needsAttention = false,
  variant = 'avatar',
}: Props) {
  const [open, setOpen] = useState(false)
  const box = useRef<HTMLDivElement>(null)

  // Clicking anywhere else, or pressing Escape, closes it. Both, because a menu
  // that can only be closed by hitting the same small button again is annoying
  // on a desktop and nearly unusable on a phone.
  useEffect(() => {
    if (!open) return
    const away = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false)
    }
    const esc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', away)
    document.addEventListener('keydown', esc)
    return () => {
      document.removeEventListener('mousedown', away)
      document.removeEventListener('keydown', esc)
    }
  }, [open])

  const name = displayName || email

  return (
    <div ref={box} className="relative">
      {variant === 'row' ? (
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-haspopup="menu"
          aria-expanded={open}
          className="flex w-full items-center gap-2.5 rounded-lg px-2 py-2 text-left transition hover:bg-elevated"
        >
          <span
            aria-hidden
            className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-accent text-2xs font-semibold text-accent-ink"
          >
            {initials(displayName || '', email)}
          </span>
          <span className="min-w-0 flex-1 truncate text-sm text-ink">{name}</span>
          {needsAttention && (
            <span
              title="Something needs looking at"
              className="h-1.5 w-1.5 shrink-0 rounded-full bg-warn"
            />
          )}
          <svg viewBox="0 0 16 16" aria-hidden className="h-3.5 w-3.5 shrink-0 text-muted">
            <path
              d="M4 10l4-4 4 4"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.6"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
        </button>
      ) : (
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          aria-haspopup="menu"
          aria-expanded={open}
          aria-label="Account"
          className="flex h-9 w-9 items-center justify-center rounded-full bg-accent text-xs font-semibold text-accent-ink transition hover:opacity-90"
        >
          {initials(displayName || '', email)}
        </button>
      )}

      {open && (
        <div
          role="menu"
          className={`absolute z-50 overflow-hidden rounded-xl border border-line bg-surface shadow-lg ${
            variant === 'row'
              ? 'bottom-full left-0 mb-1 w-[calc(100%-0.5rem)] min-w-[13rem]'
              : 'right-0 mt-2 w-60'
          }`}
        >
          <div className="border-b border-line px-4 py-3">
            <p className="truncate text-sm font-medium text-ink">{name}</p>
            {displayName && <p className="truncate text-xs text-muted">{email}</p>}
          </div>

          {onOpenIntegrations && (
            <button
              type="button"
              role="menuitem"
              onClick={() => {
                setOpen(false)
                onOpenIntegrations()
              }}
              className="flex w-full items-center gap-2.5 px-4 py-2.5 text-sm text-ink hover:bg-elevated"
            >
              <svg viewBox="0 0 16 16" aria-hidden className="h-4 w-4 text-muted">
                <path
                  d="M6 2.5v3.2a1 1 0 0 1-.3.7L4.2 7.9a1 1 0 0 0-.2 1.1l1.5 3a1 1 0 0 0 .9.5h3.2a1 1 0 0 0 .9-.5l1.5-3a1 1 0 0 0-.2-1.1l-1.5-1.5a1 1 0 0 1-.3-.7V2.5"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.4"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              <span>Integrations</span>
            </button>
          )}


          {onOpenSyncStatus && (
            <button
              type="button"
              role="menuitem"
              onClick={() => {
                setOpen(false)
                onOpenSyncStatus()
              }}
              className="flex w-full items-center gap-2.5 px-4 py-2.5 text-sm text-ink hover:bg-elevated"
            >
              <svg viewBox="0 0 16 16" aria-hidden className="h-4 w-4 text-muted">
                <path
                  d="M14 8a6 6 0 1 1-1.8-4.3M14 2v3.5h-3.5"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.4"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </svg>
              <span>Your information</span>
              {needsAttention && (
                <span className="ml-auto h-1.5 w-1.5 rounded-full bg-warn" aria-hidden />
              )}
            </button>
          )}

          <button
            type="button"
            role="menuitem"
            onClick={onToggleTheme}
            className="flex w-full items-center justify-between border-t border-line px-4 py-2.5 text-sm text-ink hover:bg-elevated"
          >
            <span>Dark mode</span>
            {/* A switch, not a button labelled with the other state — "Dark" on a
                light page is ambiguous about whether it describes now or next. */}
            <span
              aria-hidden
              className={`relative h-5 w-9 rounded-full transition ${
                theme === 'dark' ? 'bg-accent' : 'bg-line'
              }`}
            >
              <span
                className={`absolute top-0.5 h-4 w-4 rounded-full bg-white transition-all ${
                  theme === 'dark' ? 'left-[1.15rem]' : 'left-0.5'
                }`}
              />
            </span>
          </button>

          <button
            type="button"
            role="menuitem"
            onClick={onSignOut}
            className="w-full border-t border-line px-4 py-2.5 text-left text-sm text-ink hover:bg-elevated"
          >
            Sign out
          </button>
        </div>
      )}
    </div>
  )
}
