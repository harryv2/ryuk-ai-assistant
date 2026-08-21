/**
 * One typed function per endpoint.
 *
 * Everything goes through `request()`, which always sends the session cookie
 * (`credentials: 'include'`), always sends an `X-Request-ID` so a failure can be
 * traced in the server logs, and always turns the error envelope into an
 * `ApiError` carrying `code` and `request_id`.
 *
 * Requests are same-origin in dev because vite.config.ts proxies `/api`.
 */

import type {
  ApiErrorBody,
  CancelResponse,
  ConversationDetail,
  ConversationListResponse,
  ErrorCode,
  Id,
  MeResponse,
  PromptListResponse,
  PromptStatus,
  QueryRequest,
  QueryResponse,
  RespondResponse,
  SearchParams,
  SearchResponse,
  SyncStatus,
  SyncTriggerRequest,
  SyncTriggerResponse,
} from './types'

export const API_BASE = '/api/v1'

/* ------------------------------------------------------------------- errors */

export class ApiError extends Error {
  readonly code: ErrorCode
  readonly status: number
  readonly details: Record<string, unknown>
  readonly requestId: string | null
  readonly retryAfterS: number | null

  constructor(init: {
    code: ErrorCode
    message: string
    status: number
    details?: Record<string, unknown>
    requestId?: string | null
    retryAfterS?: number | null
  }) {
    super(init.message)
    this.name = 'ApiError'
    this.code = init.code
    this.status = init.status
    this.details = init.details ?? {}
    this.requestId = init.requestId ?? null
    this.retryAfterS = init.retryAfterS ?? null
  }

  /** The session is gone. Send the user to sign in. */
  get needsSignIn(): boolean {
    return this.code === 'NOT_AUTHENTICATED'
  }

  /** The session is fine; the Google grant is not. Show the reconnect banner. */
  get needsReauth(): boolean {
    return this.code === 'GOOGLE_REAUTH_REQUIRED'
  }

  /** The card moved on underneath us — refetch it rather than retrying. */
  get promptStale(): boolean {
    return this.code === 'PROMPT_NOT_PENDING'
  }
}

const KNOWN_CODES = new Set<ErrorCode>([
  'VALIDATION_ERROR',
  'NOT_AUTHENTICATED',
  'GOOGLE_REAUTH_REQUIRED',
  'RATE_LIMITED',
  'PROMPT_NOT_PENDING',
  'PROMPT_VALUE_INVALID',
  'ORCHESTRATION_TIMEOUT',
  'GOOGLE_UNAVAILABLE',
  'NOT_FOUND',
  'INTERNAL',
  'STREAM_EXPIRED',
])

function codeFor(status: number, raw: unknown): ErrorCode {
  if (typeof raw === 'string' && KNOWN_CODES.has(raw as ErrorCode)) return raw as ErrorCode
  switch (status) {
    case 401:
      return 'NOT_AUTHENTICATED'
    case 404:
      return 'NOT_FOUND'
    case 409:
      return 'PROMPT_NOT_PENDING'
    case 422:
      return 'VALIDATION_ERROR'
    case 428:
      return 'GOOGLE_REAUTH_REQUIRED'
    case 429:
      return 'RATE_LIMITED'
    case 503:
      return 'GOOGLE_UNAVAILABLE'
    case 504:
      return 'ORCHESTRATION_TIMEOUT'
    default:
      return 'INTERNAL'
  }
}

/** Turn a failed Response into an ApiError. Shared with the ndjson stream. */
export async function toApiError(res: Response): Promise<ApiError> {
  const requestId = res.headers.get('X-Request-ID')
  const retryAfterHeader = res.headers.get('Retry-After')
  const retryAfterS = retryAfterHeader ? Number(retryAfterHeader) : null

  let body: Partial<ApiErrorBody> | null = null
  let text = ''
  try {
    text = await res.text()
    body = text ? (JSON.parse(text) as ApiErrorBody) : null
  } catch {
    body = null
  }

  const envelope = body?.error
  return new ApiError({
    code: codeFor(res.status, envelope?.code),
    message:
      envelope?.message ||
      (text && text.length < 200 ? text : '') ||
      `Request failed with ${res.status}.`,
    status: res.status,
    details: envelope?.details ?? {},
    requestId: envelope?.request_id ?? requestId,
    retryAfterS: Number.isFinite(retryAfterS) ? retryAfterS : null,
  })
}

/** Anything that is not an HTTP answer: offline, DNS, aborted mid-flight. */
function networkError(err: unknown): ApiError {
  const message =
    err instanceof Error && err.name === 'AbortError'
      ? 'Request cancelled.'
      : 'Cannot reach the server.'
  return new ApiError({
    code: 'GOOGLE_UNAVAILABLE',
    message,
    status: 0,
    details: { cause: err instanceof Error ? err.message : String(err) },
  })
}

/* ------------------------------------------------------------------ plumbing */

export function newRequestId(): string {
  const c = globalThis.crypto
  if (c && 'randomUUID' in c) return c.randomUUID()
  return `req-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`
}

type QueryValue = string | number | boolean | undefined | null | string[]

export function buildUrl(path: string, params?: Record<string, QueryValue>): string {
  const url = path.startsWith('/api') ? path : `${API_BASE}${path}`
  if (!params) return url
  const search = new URLSearchParams()
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === '') continue
    search.set(key, Array.isArray(value) ? value.join(',') : String(value))
  }
  const qs = search.toString()
  return qs ? `${url}?${qs}` : url
}

interface RequestInitEx {
  method?: 'GET' | 'POST' | 'PATCH' | 'DELETE'
  body?: unknown
  params?: Record<string, QueryValue>
  signal?: AbortSignal
  requestId?: string
}

export async function request<T>(path: string, init: RequestInitEx = {}): Promise<T> {
  const headers: Record<string, string> = {
    Accept: 'application/json',
    'X-Request-ID': init.requestId ?? newRequestId(),
  }
  if (init.body !== undefined) headers['Content-Type'] = 'application/json'

  let res: Response
  try {
    res = await fetch(buildUrl(path, init.params), {
      method: init.method ?? 'GET',
      // The session is an httpOnly cookie. Without this, every call is a 401.
      credentials: 'include',
      headers,
      body: init.body === undefined ? undefined : JSON.stringify(init.body),
      signal: init.signal,
    })
  } catch (err) {
    throw networkError(err)
  }

  if (!res.ok) throw await toApiError(res)
  if (res.status === 204) return undefined as T

  const text = await res.text()
  if (!text) return undefined as T
  try {
    return JSON.parse(text) as T
  } catch {
    throw new ApiError({
      code: 'INTERNAL',
      message: 'The server sent something that is not JSON.',
      status: res.status,
      details: { body: text.slice(0, 200) },
      requestId: res.headers.get('X-Request-ID'),
    })
  }
}

/* ----------------------------------------------------------------- endpoints */

/* auth */

export function getMe(signal?: AbortSignal): Promise<MeResponse> {
  return request<MeResponse>('/auth/me', { signal })
}

/**
 * A link, never a fetch: the response is a 302 to Google's consent screen and
 * fetch cannot follow that into a top-level navigation.
 */
export function googleAuthUrl(next = '/'): string {
  return buildUrl('/auth/google', { next })
}

export function disconnectGoogle(purge = false, signal?: AbortSignal): Promise<unknown> {
  return request('/auth/google', { method: 'DELETE', params: { purge }, signal })
}

/* the one entry point */

export function postQuery(body: QueryRequest, signal?: AbortSignal): Promise<QueryResponse> {
  return request<QueryResponse>('/query', { method: 'POST', body, signal })
}

/* conversations */

export function getConversations(
  params: { limit?: number; cursor?: string; archived?: boolean } = {},
  signal?: AbortSignal,
): Promise<ConversationListResponse> {
  return request<ConversationListResponse>('/conversations', {
    params: { limit: params.limit ?? 20, cursor: params.cursor, archived: params.archived },
    signal,
  })
}

/** The durable record. Every gap in the stream is repaired from here. */
export function getConversation(
  id: Id,
  params: { include_trace?: boolean; limit?: number; before_seq?: number } = {},
  signal?: AbortSignal,
): Promise<ConversationDetail> {
  return request<ConversationDetail>(`/conversations/${encodeURIComponent(id)}`, {
    params: {
      include_trace: params.include_trace,
      limit: params.limit,
      before_seq: params.before_seq,
    },
    signal,
  })
}

export function renameConversation(id: Id, title: string): Promise<{ id: Id; title: string | null }> {
  return request(`/conversations/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    body: { title },
  })
}

export function deleteConversation(id: Id): Promise<{ id: Id; archived: boolean }> {
  return request(`/conversations/${encodeURIComponent(id)}`, { method: 'DELETE' })
}

/* prompts */

export function getPrompts(
  params: {
    status?: PromptStatus | 'all'
    conversation_id?: Id
    run_id?: Id
    limit?: number
  } = {},
  signal?: AbortSignal,
): Promise<PromptListResponse> {
  return request<PromptListResponse>('/prompts', {
    params: {
      status: params.status ?? 'pending',
      conversation_id: params.conversation_id,
      run_id: params.run_id,
      limit: params.limit,
    },
    signal,
  })
}

/** One endpoint for all six kinds — the row's value_schema is the authority. */
export function respondToPrompt(
  id: Id,
  value: unknown,
  signal?: AbortSignal,
): Promise<RespondResponse> {
  return request<RespondResponse>(`/prompts/${encodeURIComponent(id)}/respond`, {
    method: 'POST',
    body: { value },
    signal,
  })
}

export function cancelPrompt(id: Id, signal?: AbortSignal): Promise<CancelResponse> {
  return request<CancelResponse>(`/prompts/${encodeURIComponent(id)}/cancel`, {
    method: 'POST',
    signal,
  })
}

/* sync */

export function getSyncStatus(signal?: AbortSignal): Promise<SyncStatus> {
  return request<SyncStatus>('/sync/status', { signal })
}

export function triggerSync(
  body: SyncTriggerRequest = {},
  signal?: AbortSignal,
): Promise<SyncTriggerResponse> {
  return request<SyncTriggerResponse>('/sync/trigger', {
    method: 'POST',
    body: { services: body.services, mode: body.mode ?? 'incremental' },
    signal,
  })
}

/* search — debugging and the offline evaluation, not the chat path */

/**
 * Every score component for a query, so a wrong answer can be explained rather
 * than argued about. `explain: true` adds the SQL strategy and the row counts.
 */
export function getSearch(params: SearchParams, signal?: AbortSignal): Promise<SearchResponse> {
  return request<SearchResponse>('/search', {
    params: {
      q: params.q,
      services: params.services,
      limit: params.limit,
      freshness: params.freshness,
      since: params.since,
      until: params.until,
      from: params.from,
      attendee: params.attendee,
      mime: params.mime,
      explain: params.explain,
    },
    signal,
  })
}

/**
 * The whole surface in one object. Components import this; the standalone
 * functions above are the same code for anything that prefers a plain import.
 */
export const api = {
  getMe,
  me: getMe,
  googleAuthUrl,
  disconnectGoogle,
  query: postQuery,
  postQuery,
  getConversations,
  conversations: getConversations,
  getConversation,
  conversation: getConversation,
  renameConversation,
  deleteConversation,
  getPrompts,
  prompts: getPrompts,
  respondToPrompt,
  respondPrompt: respondToPrompt,
  cancelPrompt,
  getSyncStatus,
  syncStatus: getSyncStatus,
  triggerSync,
  syncTrigger: triggerSync,
  getSearch,
  search: getSearch,
}

/** The message to show a person when a call fails. */
export function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message
  if (err instanceof Error) return err.message
  return 'Something went wrong.'
}
