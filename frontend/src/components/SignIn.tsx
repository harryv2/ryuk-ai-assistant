/**
 * Signing in: an address and a password, then a code sent to that address.
 *
 * The two steps live in one component because they are one decision to the
 * person doing them. Which step is showing is derived from whether the server
 * has told us a code is pending — there is no separate "screen" to get out of
 * sync with the server's idea of where we are.
 */

import { useEffect, useRef, useState } from 'react'

import { ApiError } from '../lib/api'
import { attemptsRemaining, login, register, resendOtp, verifyOtp, type OtpPending } from '../lib/auth'
import { googleAuthUrl } from '../lib/api'
import GoogleMark from './GoogleMark'
import type { MeResponse } from '../lib/types'

const CODE_LENGTH = 6
const RESEND_COOLDOWN_S = 30

interface Props {
  onSignedIn: (me: MeResponse) => void
  /** Prefilled after signing out, so switching back is one password away. */
  initialEmail?: string
}

export default function SignIn({ onSignedIn, initialEmail = '' }: Props) {
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [email, setEmail] = useState(initialEmail)
  const [password, setPassword] = useState('')
  const [pending, setPending] = useState<OtpPending | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  async function submitCredentials(event: React.FormEvent) {
    event.preventDefault()
    setError(null)
    setBusy(true)
    try {
      const fn = mode === 'register' ? register : login
      setPending(await fn(email.trim(), password))
      setPassword('')          // never keep it around once it has been spent
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'Something went wrong. Try again.')
    } finally {
      setBusy(false)
    }
  }

  if (pending) {
    return (
      <OtpStep
        pending={pending}
        onSignedIn={onSignedIn}
        onBack={() => {
          setPending(null)
          setError(null)
        }}
      />
    )
  }

  return (
    <div className="flex flex-1 items-center justify-center px-6 py-10">
      <div className="w-full max-w-sm rounded-xl border border-line bg-surface p-6 shadow-card">
        <div className="flex items-center gap-2.5">
          <span
            aria-hidden
            className="flex h-8 w-8 items-center justify-center rounded-lg bg-accent text-sm font-bold text-accent-ink"
          >
            R
          </span>
          <div className="leading-tight">
            <p className="text-sm font-semibold text-ink">Ryuk</p>
            <p className="text-2xs text-muted">AI Assistant</p>
          </div>
        </div>

        {/* No product pitch here. Whoever is on this screen either knows what
            this is or was sent a link — either way they came to get in, not to
            read. The one line that earns its place is the one that sets an
            expectation about what happens next. */}
        <h1 className="mt-6 text-lg font-semibold text-ink">
          {mode === 'login' ? 'Sign in' : 'Create your account'}
        </h1>
        {mode === 'register' && (
          <p className="mt-1 text-sm text-muted">You’ll connect Google Workspace next.</p>
        )}

        <form onSubmit={submitCredentials} className="mt-5 space-y-3">
          <label className="block">
            <span className="text-xs font-medium text-muted">Email</span>
            <input
              type="email"
              required
              autoComplete="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-1 w-full rounded-lg border border-line bg-base px-3 py-2 text-sm text-ink outline-none focus:border-accent"
              placeholder="you@example.com"
            />
          </label>

          <label className="block">
            <span className="text-xs font-medium text-muted">Password</span>
            <input
              type="password"
              required
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 w-full rounded-lg border border-line bg-base px-3 py-2 text-sm text-ink outline-none focus:border-accent"
              placeholder={mode === 'register' ? 'At least 10 characters' : ''}
            />
          </label>

          {error && (
            <p role="alert" className="rounded-lg border border-bad/30 bg-bad/10 px-3 py-2 text-sm text-bad">
              {error}
            </p>
          )}

          <button
            type="submit"
            disabled={busy || !email.trim() || !password}
            className="w-full rounded-lg bg-accent px-4 py-2 text-sm font-medium text-accent-ink disabled:opacity-50"
          >
            {busy ? 'One moment…' : mode === 'login' ? 'Continue' : 'Create account'}
          </button>
        </form>

        <button
          type="button"
          onClick={() => {
            setMode(mode === 'login' ? 'register' : 'login')
            setError(null)
          }}
          className="mt-3 w-full text-center text-xs text-muted underline-offset-2 hover:underline"
        >
          {mode === 'login' ? 'Create an account instead' : 'I already have an account'}
        </button>

        <div className="my-5 flex items-center gap-3 text-2xs text-muted">
          <span className="h-px flex-1 bg-line" />
          or
          <span className="h-px flex-1 bg-line" />
        </div>

        {/* A link, never a fetch: the response is a 302 to Google's consent screen. */}
        <a
          href={googleAuthUrl('/')}
          className="flex w-full items-center justify-center gap-2.5 rounded-lg border border-line px-4 py-2 text-sm font-medium text-ink hover:bg-base"
        >
          <GoogleMark className="h-4 w-4" />
          Continue with Google
        </a>
        {/* One line, and only because it changes what the button does: this
            path also connects the workspace, which the email path does not. */}
        <p className="mt-3 text-center text-2xs text-muted">
          Signs you in and connects your workspace.
        </p>
      </div>
    </div>
  )
}

function OtpStep({
  pending,
  onSignedIn,
  onBack,
}: {
  pending: OtpPending
  onSignedIn: (me: MeResponse) => void
  onBack: () => void
}) {
  const [digits, setDigits] = useState<string[]>(Array(CODE_LENGTH).fill(''))
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [remaining, setRemaining] = useState<number | null>(null)
  const [cooldown, setCooldown] = useState(0)
  const boxes = useRef<(HTMLInputElement | null)[]>([])

  useEffect(() => {
    boxes.current[0]?.focus()
  }, [])

  useEffect(() => {
    if (cooldown <= 0) return
    const t = setTimeout(() => setCooldown((n) => n - 1), 1000)
    return () => clearTimeout(t)
  }, [cooldown])

  const code = digits.join('')

  async function submit(value: string) {
    setBusy(true)
    setError(null)
    try {
      onSignedIn(await verifyOtp(pending.email, value))
    } catch (err) {
      setError(err instanceof ApiError ? err.message : 'That did not work. Try again.')
      setRemaining(attemptsRemaining(err))
      setDigits(Array(CODE_LENGTH).fill(''))
      boxes.current[0]?.focus()
    } finally {
      setBusy(false)
    }
  }

  function setAt(index: number, value: string) {
    // Pasting the whole code into any box should just work.
    const clean = value.replace(/\D/g, '')
    if (clean.length > 1) {
      const next = clean.slice(0, CODE_LENGTH).split('')
      const filled = Array.from({ length: CODE_LENGTH }, (_, i) => next[i] ?? '')
      setDigits(filled)
      if (filled.every(Boolean)) void submit(filled.join(''))
      else boxes.current[Math.min(next.length, CODE_LENGTH - 1)]?.focus()
      return
    }
    const next = [...digits]
    next[index] = clean
    setDigits(next)
    if (clean && index < CODE_LENGTH - 1) boxes.current[index + 1]?.focus()
    if (next.every(Boolean)) void submit(next.join(''))
  }

  return (
    <div className="flex flex-1 items-center justify-center px-6 py-10">
      <div className="w-full max-w-sm rounded-xl border border-line bg-surface p-6 shadow-card">
        <h1 className="text-base font-semibold text-ink">Check your email</h1>
        <p className="mt-2 text-sm leading-relaxed text-muted">
          We sent a {CODE_LENGTH}-digit code to <span className="text-ink">{pending.email}</span>.
        </p>

        {pending.dev_mode && (
          <p className="mt-3 rounded-lg border border-warn/30 bg-warn/10 px-3 py-2 text-sm text-warn">
            Development mode — the code is <span className="font-mono font-semibold">123456</span>.
            No email was actually sent.
          </p>
        )}

        <div className="mt-5 flex justify-between gap-2">
          {digits.map((digit, i) => (
            <input
              key={i}
              ref={(el) => {
                boxes.current[i] = el
              }}
              value={digit}
              inputMode="numeric"
              autoComplete="one-time-code"
              maxLength={CODE_LENGTH}
              disabled={busy}
              onChange={(e) => setAt(i, e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Backspace' && !digits[i] && i > 0) boxes.current[i - 1]?.focus()
              }}
              className="h-12 w-11 rounded-lg border border-line bg-base text-center font-mono text-lg text-ink outline-none focus:border-accent disabled:opacity-50"
            />
          ))}
        </div>

        {error && (
          <p role="alert" className="mt-3 rounded-lg border border-bad/30 bg-bad/10 px-3 py-2 text-sm text-bad">
            {error}
            {remaining !== null && remaining > 0 && (
              <span className="block text-2xs opacity-80">
                {remaining} {remaining === 1 ? 'try' : 'tries'} left before this code stops working.
              </span>
            )}
          </p>
        )}

        <div className="mt-5 flex items-center justify-between text-xs">
          <button type="button" onClick={onBack} className="text-muted hover:underline">
            Use a different email
          </button>
          <button
            type="button"
            disabled={cooldown > 0 || busy}
            onClick={async () => {
              setCooldown(RESEND_COOLDOWN_S)
              setError(null)
              try {
                await resendOtp(pending.email)
              } catch (err) {
                setError(err instanceof ApiError ? err.message : 'Could not send another code.')
              }
            }}
            className="text-accent disabled:text-muted"
          >
            {cooldown > 0 ? `Send again in ${cooldown}s` : 'Send another code'}
          </button>
        </div>

        <button
          type="button"
          disabled={busy || code.length !== CODE_LENGTH}
          onClick={() => void submit(code)}
          className="mt-4 w-full rounded-lg bg-accent px-4 py-2 text-sm font-medium text-accent-ink disabled:opacity-50"
        >
          {busy ? 'Checking…' : 'Sign in'}
        </button>
      </div>
    </div>
  )
}
