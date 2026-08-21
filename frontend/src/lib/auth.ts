/**
 * Signing in with an email address and a password.
 *
 * Two steps, always: the password, then a code sent to the address. The second
 * step is what actually creates the session — until the code is entered, a wrong
 * password and a correct one have produced exactly the same thing, which is
 * nothing.
 *
 * The session is an httpOnly cookie. Nothing here holds a password, a code or a
 * token in memory beyond the request that carries it.
 */

import { request } from './api'
import type { MeResponse } from './types'

/** What the server says after a password is accepted, or a sign-up is started. */
export interface OtpPending {
  email: string
  otp_required: true
  expires_in: number
  attempts_allowed: number
  /** Present only while the server is in development mode. */
  dev_mode?: boolean
  dev_code?: string
  dev_note?: string
}

export function register(email: string, password: string): Promise<OtpPending> {
  return request<OtpPending>('/auth/register', {
    method: 'POST',
    body: { email, password },
  })
}

export function login(email: string, password: string): Promise<OtpPending> {
  return request<OtpPending>('/auth/login', {
    method: 'POST',
    body: { email, password },
  })
}

/** The step that actually signs you in. Returns the same body as /auth/me. */
export function verifyOtp(email: string, code: string): Promise<MeResponse> {
  return request<MeResponse>('/auth/verify-otp', {
    method: 'POST',
    body: { email, code },
  })
}

export function resendOtp(email: string): Promise<OtpPending> {
  return request<OtpPending>('/auth/resend-otp', {
    method: 'POST',
    body: { email },
  })
}

export function logout(): Promise<unknown> {
  return request('/auth/logout', { method: 'POST', body: {} })
}

/**
 * How many tries are left on a wrong code, if the server said.
 *
 * Worth surfacing: five wrong codes burns the code entirely, and someone who
 * mistypes twice should know that before it happens rather than after.
 */
export function attemptsRemaining(err: unknown): number | null {
  const details = (err as { details?: Record<string, unknown> })?.details
  const n = details?.attempts_remaining
  return typeof n === 'number' ? n : null
}
