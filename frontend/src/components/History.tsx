/**
 * Previous conversations.
 *
 * Grouped by how long ago you touched them, because that is how people look
 * for a thread — "the one from yesterday", not "the one from the 14th".
 */

import { useEffect, useRef, useState, type ReactNode } from 'react'

import { relativeTime } from '../lib/format'
import type { ConversationSummary, Id } from '../lib/types'

interface HistoryProps {
  /** The brand lockup, which lives at the top of the column. */
  header?: ReactNode
  /** The account row, which sits at the bottom of it. */
  footer?: ReactNode
  items: ConversationSummary[]
  activeId: Id | null
  open: boolean
  loading: boolean
  onSelect: (id: Id) => void
  onNew: () => void
  onRename: (id: Id, title: string) => void
  onDelete: (id: Id) => void
  onClose: () => void
}

/** Buckets, newest first. A thread lands in the first one it fits. */
const BUCKETS: Array<{ label: string; within: number }> = [
  { label: 'Today', within: 1 },
  { label: 'Yesterday', within: 2 },
  { label: 'Previous 7 days', within: 8 },
  { label: 'Previous 30 days', within: 31 },
  { label: 'Older', within: Infinity },
]

function daysAgo(iso: string): number {
  const then = new Date(iso)
  if (Number.isNaN(then.getTime())) return Infinity
  const startOfToday = new Date()
  startOfToday.setHours(0, 0, 0, 0)
  return Math.floor((startOfToday.getTime() - then.getTime()) / 86_400_000) + 1
}

function group(items: ConversationSummary[]) {
  const out: Array<{ label: string; items: ConversationSummary[] }> = []
  for (const bucket of BUCKETS) {
    const inBucket = items.filter((c) => {
      const age = daysAgo(c.last_message_at || c.created_at)
      const earlier = BUCKETS.slice(0, BUCKETS.indexOf(bucket))
      return age <= bucket.within && !earlier.some((b) => age <= b.within)
    })
    if (inBucket.length) out.push({ label: bucket.label, items: inBucket })
  }
  return out
}

export default function History({
  header,
  footer,
  items,
  activeId,
  open,
  loading,
  onSelect,
  onNew,
  onRename,
  onDelete,
  onClose,
}: HistoryProps) {
  const [menuFor, setMenuFor] = useState<Id | null>(null)
  const [editing, setEditing] = useState<Id | null>(null)
  const [draft, setDraft] = useState('')
  const [confirming, setConfirming] = useState<Id | null>(null)
  const [query, setQuery] = useState('')
  const input = useRef<HTMLInputElement>(null)

  useEffect(() => {
    if (editing) input.current?.focus()
  }, [editing])

  useEffect(() => {
    if (!menuFor) return
    const shut = () => setMenuFor(null)
    window.addEventListener('click', shut)
    return () => window.removeEventListener('click', shut)
  }, [menuFor])

  // On a phone the sidebar is a sheet over the chat, so Escape has to close it.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return
      if (editing) setEditing(null)
      else if (confirming) setConfirming(null)
      else onClose()
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [editing, confirming, onClose])

  function commitRename(id: Id) {
    const next = draft.trim()
    setEditing(null)
    if (next) onRename(id, next)
  }

  const needle = query.trim().toLowerCase()
  const shown = needle
    ? items.filter((c) => c.title.toLowerCase().includes(needle))
    : items
  const groups = group(shown)

  return (
    <>
      {/* Phone: dim the chat behind the sheet, and let a tap outside close it. */}
      {open && (
        <button
          type="button"
          aria-label="Close history"
          onClick={onClose}
          className="fixed inset-0 z-30 bg-black/40 md:hidden"
        />
      )}

      <aside
        className={`fixed inset-y-0 left-0 z-40 flex w-[272px] flex-col border-r border-line bg-canvas transition-transform duration-200 md:static md:z-auto md:translate-x-0 ${
          open ? 'translate-x-0' : '-translate-x-full md:w-0 md:overflow-hidden md:border-r-0'
        }`}
      >
        {header}

        <div className="flex items-center gap-1 px-2 py-2">
          {/* A row, not a boxed button. The sidebar is a list of places you can
              go and this is the first of them; drawing a border around one item
              in a list makes it look like it belongs to something else. */}
          <button
            type="button"
            onClick={onNew}
            className="flex flex-1 items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm font-medium text-ink transition hover:bg-elevated"
          >
            <svg viewBox="0 0 18 18" aria-hidden className="h-[18px] w-[18px] text-muted">
              <path
                d="M12.2 2.6a1.5 1.5 0 0 1 2.1 2.1l-7 7-2.8.7.7-2.8 7-7Z"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.4"
                strokeLinejoin="round"
              />
              <path
                d="M15.5 10.5v3.6a1.4 1.4 0 0 1-1.4 1.4H3.9a1.4 1.4 0 0 1-1.4-1.4V3.9a1.4 1.4 0 0 1 1.4-1.4h3.6"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.4"
                strokeLinecap="round"
              />
            </svg>
            New chat
          </button>
          <button
            type="button"
            onClick={onClose}
            aria-label="Hide history"
            className="rounded-lg p-2 text-muted transition hover:bg-base hover:text-ink md:hidden"
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

        {items.length > 8 && (
          <div className="px-2 pb-1">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search chats"
              aria-label="Search chats"
              className="w-full rounded-lg bg-elevated px-2.5 py-1.5 text-sm text-ink outline-none placeholder:text-muted focus:ring-1 focus:ring-accent/40"
            />
          </div>
        )}

        <nav className="flex-1 overflow-y-auto px-2 pb-4">
          {loading && items.length === 0 && (
            <div className="space-y-1.5 px-1 py-2">
              {[0, 1, 2, 3].map((i) => (
                <div key={i} className="h-8 animate-pulse rounded-lg bg-line/60" />
              ))}
            </div>
          )}

          {!loading && items.length === 0 && (
            <p className="px-2.5 py-6 text-xs leading-relaxed text-muted">
              Your conversations will appear here.
            </p>
          )}

          {!loading && items.length > 0 && shown.length === 0 && (
            <p className="px-2.5 py-6 text-xs text-muted">No chats match “{query}”.</p>
          )}

          {groups.map((bucket) => (
            <div key={bucket.label} className="mb-3">
              <p className="px-2.5 pb-1 pt-3 text-xs font-medium text-faint">{bucket.label}</p>
              <ul className="space-y-0.5">
                {bucket.items.map((c) => {
                  const active = c.id === activeId
                  return (
                    <li key={c.id} className="group relative">
                      {editing === c.id ? (
                        <input
                          ref={input}
                          value={draft}
                          onChange={(e) => setDraft(e.target.value)}
                          onBlur={() => commitRename(c.id)}
                          onKeyDown={(e) => {
                            if (e.key === 'Enter') commitRename(c.id)
                          }}
                          className="w-full rounded-lg border border-accent bg-base px-2.5 py-2 text-sm text-ink outline-none"
                        />
                      ) : (
                        <button
                          type="button"
                          onClick={() => onSelect(c.id)}
                          title={c.title}
                          className={`flex w-full items-center gap-2 rounded-lg py-2 pl-2.5 pr-7 text-left text-sm transition ${
                            active
                              ? 'bg-elevated font-medium text-ink'
                              : 'text-ink/80 hover:bg-elevated hover:text-ink'
                          }`}
                        >
                          <span className="flex-1 truncate">{c.title}</span>
                          {c.pending_input_count > 0 && (
                            <span
                              title="Waiting on you"
                              className="h-1.5 w-1.5 shrink-0 rounded-full bg-accent"
                            />
                          )}
                        </button>
                      )}

                      {editing !== c.id && (
                        <button
                          type="button"
                          aria-label="Conversation options"
                          onClick={(e) => {
                            e.stopPropagation()
                            setMenuFor(menuFor === c.id ? null : c.id)
                          }}
                          className={`absolute right-1 top-1.5 rounded p-1 text-muted transition hover:bg-line hover:text-ink ${
                            menuFor === c.id ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
                          } focus:opacity-100`}
                        >
                          <svg viewBox="0 0 16 16" aria-hidden className="h-3.5 w-3.5">
                            <circle cx="8" cy="3.5" r="1.2" fill="currentColor" />
                            <circle cx="8" cy="8" r="1.2" fill="currentColor" />
                            <circle cx="8" cy="12.5" r="1.2" fill="currentColor" />
                          </svg>
                        </button>
                      )}

                      {menuFor === c.id && (
                        <div
                          onClick={(e) => e.stopPropagation()}
                          className="absolute right-1 top-8 z-50 w-40 overflow-hidden rounded-lg border border-line bg-surface py-1 shadow-lg"
                        >
                          <p className="px-3 pb-1 text-2xs text-muted">
                            {relativeTime(c.last_message_at || c.created_at)}
                          </p>
                          <button
                            type="button"
                            onClick={() => {
                              setDraft(c.title)
                              setEditing(c.id)
                              setMenuFor(null)
                            }}
                            className="block w-full px-3 py-1.5 text-left text-sm text-ink transition hover:bg-base"
                          >
                            Rename
                          </button>
                          <button
                            type="button"
                            onClick={() => {
                              setConfirming(c.id)
                              setMenuFor(null)
                            }}
                            className="block w-full px-3 py-1.5 text-left text-sm text-bad transition hover:bg-base"
                          >
                            Delete
                          </button>
                        </div>
                      )}
                    </li>
                  )
                })}
              </ul>
            </div>
          ))}
        </nav>

        {footer}
      </aside>

      {confirming && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4">
          <div className="w-full max-w-xs rounded-xl border border-line bg-surface p-4 shadow-xl">
            <p className="text-sm font-medium text-ink">Delete this conversation?</p>
            <p className="mt-1 text-xs leading-relaxed text-muted">
              It will be removed from your list. Anything already sent stays sent.
            </p>
            <div className="mt-4 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setConfirming(null)}
                className="rounded-lg px-3 py-1.5 text-sm text-muted transition hover:text-ink"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => {
                  onDelete(confirming)
                  setConfirming(null)
                }}
                className="rounded-lg bg-bad px-3 py-1.5 text-sm text-white transition"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  )
}
