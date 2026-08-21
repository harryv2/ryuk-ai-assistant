/**
 * What Ryuk is connected to.
 *
 * Signing in and connecting a workspace are two different things here. You can
 * sign in with a password and connect any Google account afterwards — they do
 * not have to be the same address, and this is the page that says which one is
 * actually being read.
 */

import { useState } from 'react'

import { googleAuthUrl } from '../lib/api'
import GoogleMark from './GoogleMark'

interface IntegrationsProps {
  connected: boolean
  needsReauth: boolean
  /** The workspace being read, which need not be the sign-in address. */
  accountEmail?: string | null
  onClose: () => void
  onDisconnect: () => void
}

interface Upcoming {
  name: string
  blurb: string
  mark: string
  tint: string
}

const UPCOMING: Upcoming[] = [
  { name: 'Outlook', blurb: 'Mail and calendar', mark: 'O', tint: 'bg-[#0F6CBD] text-white' },
  { name: 'Slack', blurb: 'Messages and channels', mark: 'S', tint: 'bg-[#4A154B] text-white' },
  { name: 'Jira', blurb: 'Issues and projects', mark: 'J', tint: 'bg-[#1868DB] text-white' },
]

export default function Integrations({
  connected,
  needsReauth,
  accountEmail,
  onClose,
  onDisconnect,
}: IntegrationsProps) {
  const [confirming, setConfirming] = useState(false)

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/50 px-0 sm:items-center sm:px-4">
      <button type="button" aria-label="Close" onClick={onClose} className="absolute inset-0" />

      <div className="relative flex max-h-[90vh] w-full max-w-md flex-col overflow-hidden rounded-t-2xl border border-line bg-surface shadow-xl sm:rounded-2xl">
        <div className="flex items-start justify-between gap-3 border-b border-line px-5 py-4">
          <div>
            <h2 className="text-sm font-semibold text-ink">Integrations</h2>
            <p className="mt-0.5 text-xs text-muted">Apps Ryuk can read.</p>
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
          <div className="rounded-xl border border-line p-4">
            <div className="flex items-start gap-3">
              <span
                aria-hidden
                className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-white ring-1 ring-line"
              >
                <GoogleMark className="h-5 w-5" />
              </span>

              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <p className="text-sm font-medium text-ink">Google Workspace</p>
                  {connected && !needsReauth && (
                    <span className="rounded-full bg-ok/15 px-2 py-0.5 text-2xs font-medium text-ok">
                      Connected
                    </span>
                  )}
                  {connected && needsReauth && (
                    <span className="rounded-full bg-warn/15 px-2 py-0.5 text-2xs font-medium text-warn">
                      Needs attention
                    </span>
                  )}
                </div>
                <p className="mt-0.5 truncate text-xs text-muted">
                  {connected && accountEmail ? accountEmail : 'Gmail, Calendar and Drive'}
                </p>
              </div>
            </div>

            <div className="mt-3.5 flex flex-wrap gap-2">
              {!connected && (
                <a
                  href={googleAuthUrl('/')}
                  className="rounded-lg bg-accent px-3 py-1.5 text-xs font-semibold text-accent-ink transition hover:opacity-90"
                >
                  Connect
                </a>
              )}
              {connected && needsReauth && (
                <a
                  href={googleAuthUrl('/')}
                  className="rounded-lg bg-accent px-3 py-1.5 text-xs font-semibold text-accent-ink transition hover:opacity-90"
                >
                  Reconnect
                </a>
              )}
              {connected && !needsReauth && (
                <a
                  href={googleAuthUrl('/')}
                  className="rounded-lg border border-line px-3 py-1.5 text-xs font-medium text-ink transition hover:bg-elevated"
                >
                  Switch account
                </a>
              )}
              {connected && (
                <button
                  type="button"
                  onClick={() => setConfirming(true)}
                  className="rounded-lg px-3 py-1.5 text-xs font-medium text-muted transition hover:text-bad"
                >
                  Disconnect
                </button>
              )}
            </div>

            {!connected && (
              <p className="mt-3 text-2xs leading-relaxed text-muted">
                Any Google account works. It does not have to match your sign-in address.
              </p>
            )}
          </div>

          <p className="mt-5 px-1 text-2xs font-medium uppercase tracking-wide text-muted">
            Coming soon
          </p>
          <ul className="mt-2 space-y-2">
            {UPCOMING.map((item) => (
              <li
                key={item.name}
                className="flex items-center gap-3 rounded-xl border border-dashed border-line px-4 py-3 opacity-70"
              >
                <span
                  aria-hidden
                  className={`flex h-9 w-9 shrink-0 items-center justify-center rounded-lg text-sm font-bold ${item.tint}`}
                >
                  {item.mark}
                </span>
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium text-ink">{item.name}</p>
                  <p className="truncate text-xs text-muted">{item.blurb}</p>
                </div>
                <span className="shrink-0 rounded-full border border-line px-2 py-0.5 text-2xs text-muted">
                  Soon
                </span>
              </li>
            ))}
          </ul>
        </div>
      </div>

      {confirming && (
        <div className="absolute inset-0 z-10 flex items-center justify-center bg-black/50 px-4">
          <div className="w-full max-w-xs rounded-xl border border-line bg-surface p-4 shadow-xl">
            <p className="text-sm font-medium text-ink">Disconnect Google Workspace?</p>
            <p className="mt-1 text-xs leading-relaxed text-muted">
              Ryuk stops reading your mail, calendar and files, and the copy it keeps for
              searching is deleted. Your Google account is not affected.
            </p>
            <div className="mt-4 flex justify-end gap-2">
              <button
                type="button"
                onClick={() => setConfirming(false)}
                className="rounded-lg px-3 py-1.5 text-sm text-muted transition hover:text-ink"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => {
                  setConfirming(false)
                  onDisconnect()
                }}
                className="rounded-lg bg-bad px-3 py-1.5 text-sm text-white transition"
              >
                Disconnect
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
