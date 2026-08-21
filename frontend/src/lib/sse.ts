/**
 * The event transport. One event union, two carriers:
 *
 *   RunStream   — EventSource over GET /api/v1/runs/{id}/events. Used to watch
 *                 a resumed run, to reattach after a drop, and to wait for an
 *                 approved write to land.
 *   streamQuery — the same envelopes as ndjson over POST /api/v1/query, which
 *                 is how a brand-new run streams: the run id does not exist
 *                 until the POST is answered, so there is nothing to subscribe
 *                 to beforehand.
 *
 * Both enforce the same seq rule, because it is the only thing standing between
 * a dropped packet and a corrupted answer:
 *
 *   seq == last + 1   apply it, advance
 *   seq <= last       a duplicate from a reconnect — drop it
 *   seq >  last + 1   events were missed — do not guess. Replay from the
 *                     buffer, and tell the caller to refetch the conversation,
 *                     which is the durable record.
 */

import { API_BASE, buildUrl, toApiError } from './api'
import type {
  ActionRecord,
  ConnectionStatus,
  Id,
  PendingInput,
  QueryRequest,
  QueryResponse,
  RunEvent,
  RunEventBody,
  RunEventType,
} from './types'
import { RUN_EVENT_TYPES, TERMINAL_EVENT_TYPES } from './types'

export interface Gap {
  runId: Id
  expected: number
  received: number
}

export type CloseReason = 'complete' | 'error' | 'manual' | 'expired' | 'exhausted' | 'idle'

/** Call it to stop listening. Safe to call twice. */
export type StopFn = () => void

/** The transport gave up. Not a run failure — the run may well have finished. */
export class StreamError extends Error {
  readonly code = 'STREAM_LOST'
  readonly runId: Id | null

  constructor(message: string, runId: Id | null = null) {
    super(message)
    this.name = 'StreamError'
    this.runId = runId
  }
}

export interface StreamHandlers {
  onEvent: (event: RunEvent) => void
  onStatus?: (status: ConnectionStatus) => void
  /** A gap. Refetch GET /conversations/{id} — the stream cannot be trusted. */
  onGap?: (gap: Gap) => void
  /** The replay buffer no longer holds what we missed. Same fallback. */
  onExpired?: (runId: Id) => void
  /** The POST was rejected, or the transport died for good. */
  onError?: (error: unknown) => void
  onClose?: (reason: CloseReason) => void
}

/** seq bookkeeping, shared by both carriers. */
export class SeqTracker {
  private last: number

  constructor(from = 0) {
    this.last = from
  }

  get lastSeq(): number {
    return this.last
  }

  /** What to do with an arriving seq. `apply` advances the cursor. */
  check(seq: number): 'apply' | 'drop' | 'gap' {
    if (!Number.isFinite(seq)) return 'drop'
    if (seq <= this.last) return 'drop'
    if (seq > this.last + 1) return 'gap'
    this.last = seq
    return 'apply'
  }

  reset(to: number): void {
    this.last = to
  }
}

function parseEnvelope(raw: string): RunEvent | null {
  try {
    const parsed = JSON.parse(raw) as Partial<RunEvent>
    if (!parsed || typeof parsed !== 'object') return null
    if (typeof parsed.type !== 'string' || typeof parsed.seq !== 'number') return null
    return parsed as RunEvent
  } catch {
    return null
  }
}

/* --------------------------------------------------------------- EventSource */

export interface RunStreamOptions {
  /** Replay from this seq inclusive. Usually lastSeq + 1. */
  fromSeq?: number
  /** Server-side filter. Omit for the full trace. */
  types?: RunEventType[]
  /** Close the stream when one of these lands. Defaults to run.complete/error. */
  stopOn?: RunEventType[]
  /** Give up if nothing arrives for this long. Used when watching a write land. */
  idleTimeoutMs?: number
  /** Manual reconnect attempts after a transport error. */
  maxRetries?: number
}

export class RunStream {
  private es: EventSource | null = null
  private readonly tracker: SeqTracker
  private readonly stopOn: Set<RunEventType>
  private retries = 0
  private gapRepairs = 0
  private timer: ReturnType<typeof setTimeout> | null = null
  private idleTimer: ReturnType<typeof setTimeout> | null = null
  private closed = false

  constructor(
    readonly runId: Id,
    private readonly handlers: StreamHandlers,
    private readonly options: RunStreamOptions = {},
  ) {
    this.tracker = new SeqTracker(Math.max(0, (options.fromSeq ?? 1) - 1))
    this.stopOn = new Set(options.stopOn ?? TERMINAL_EVENT_TYPES)
  }

  get lastSeq(): number {
    return this.tracker.lastSeq
  }

  start(): this {
    this.open()
    return this
  }

  close(reason: CloseReason = 'manual'): void {
    if (this.closed) return
    this.closed = true
    this.teardown()
    this.handlers.onStatus?.('closed')
    this.handlers.onClose?.(reason)
  }

  private teardown(): void {
    if (this.timer) clearTimeout(this.timer)
    if (this.idleTimer) clearTimeout(this.idleTimer)
    this.timer = null
    this.idleTimer = null
    if (this.es) {
      this.es.close()
      this.es = null
    }
  }

  private url(): string {
    return buildUrl(`${API_BASE}/runs/${encodeURIComponent(this.runId)}/events`, {
      from_seq: this.tracker.lastSeq + 1,
      types: this.options.types,
    })
  }

  private open(): void {
    if (this.closed) return
    this.handlers.onStatus?.(this.retries > 0 ? 'reconnecting' : 'connecting')

    // Same-origin through the vite proxy, so the session cookie rides along.
    const es = new EventSource(this.url(), { withCredentials: true })
    this.es = es

    es.onopen = () => {
      this.retries = 0
      this.handlers.onStatus?.('open')
      this.armIdle()
    }

    const handle = (ev: MessageEvent<string>) => this.receive(ev.data)
    // The server names every event, so `message` never fires — but a proxy that
    // strips `event:` lines would make it the only thing that does.
    es.onmessage = handle
    for (const type of RUN_EVENT_TYPES) es.addEventListener(type, handle as EventListener)

    es.onerror = () => {
      if (this.closed) return
      // EventSource retries on its own using Last-Event-ID, but then the URL
      // still carries the stale from_seq. Own the reconnect instead.
      this.es?.close()
      this.es = null
      this.retry()
    }
  }

  private armIdle(): void {
    if (!this.options.idleTimeoutMs) return
    if (this.idleTimer) clearTimeout(this.idleTimer)
    this.idleTimer = setTimeout(() => this.close('idle'), this.options.idleTimeoutMs)
  }

  private retry(): void {
    const max = this.options.maxRetries ?? 6
    if (this.retries >= max) {
      this.handlers.onStatus?.('error')
      this.handlers.onError?.(
        new StreamError('Lost the live connection to this run.', this.runId),
      )
      this.close('exhausted')
      return
    }
    const wait = Math.min(8000, 400 * 2 ** this.retries)
    this.retries += 1
    this.handlers.onStatus?.('reconnecting')
    this.timer = setTimeout(() => this.open(), wait)
  }

  private receive(raw: string): void {
    if (this.closed) return
    this.armIdle()

    const event = parseEnvelope(raw)
    if (!event) return

    switch (this.tracker.check(event.seq)) {
      case 'drop':
        return
      case 'gap': {
        const gap: Gap = {
          runId: this.runId,
          expected: this.tracker.lastSeq + 1,
          received: event.seq,
        }
        this.handlers.onGap?.(gap)
        this.gapRepairs += 1
        if (this.gapRepairs > 2) {
          // The buffer is not giving us what we missed. Postgres will.
          this.handlers.onExpired?.(this.runId)
          this.close('expired')
          return
        }
        // Reopen from the first event we missed; the server replays it.
        this.es?.close()
        this.es = null
        this.open()
        return
      }
      case 'apply':
        break
    }

    if (event.type === 'error' && event.data?.code === 'STREAM_EXPIRED') {
      this.handlers.onEvent(event)
      this.handlers.onExpired?.(this.runId)
      this.close('expired')
      return
    }

    this.handlers.onEvent(event)

    if (this.stopOn.has(event.type)) {
      this.close(event.type === 'error' ? 'error' : 'complete')
    }
  }
}

/**
 * Watch a run that already exists: a resumed one, a reattach after a drop, or
 * the wait for an approved write to land. Returns the stopper.
 */
export function openRunStream(
  runId: Id,
  handlers: StreamHandlers,
  options: RunStreamOptions = {},
): StopFn {
  const stream = new RunStream(runId, handlers, options).start()
  return () => stream.close('manual')
}

/* --------------------------------------------------------------------- ndjson */

export interface QueryStreamResult {
  runId: Id | null
  lastSeq: number
  /** Set when the server answered with a buffered JSON body instead. */
  final: QueryResponse | null
  transport: 'ndjson' | 'buffered'
  /** True when a gap cut the stream short — reattach and reconcile. */
  gapped: boolean
}

export interface QueryStreamHandlers extends StreamHandlers {
  /** The buffered fallback body, when the server does not stream. */
  onFinal?: (response: QueryResponse) => void
}

/**
 * POST the query and read the run as it happens.
 *
 * If the server answers with plain JSON rather than ndjson, the whole run is in
 * that body — hand it back and let the caller reconcile from it.
 */
export async function streamQuery(
  body: QueryRequest,
  handlers: QueryStreamHandlers,
  signal?: AbortSignal,
): Promise<QueryStreamResult> {
  handlers.onStatus?.('connecting')

  let res: Response
  try {
    res = await fetch(buildUrl(`${API_BASE}/query`, { stream: 'ndjson' }), {
      method: 'POST',
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json',
        Accept: 'application/x-ndjson, application/json',
        'X-Request-ID': body.client_request_id ?? crypto.randomUUID(),
      },
      body: JSON.stringify(body),
      signal,
    })
  } catch (err) {
    handlers.onStatus?.('error')
    if (err instanceof Error && err.name === 'AbortError') {
      handlers.onClose?.('manual')
      return { runId: null, lastSeq: 0, final: null, transport: 'ndjson', gapped: false }
    }
    throw err
  }

  if (!res.ok) {
    handlers.onStatus?.('error')
    throw await toApiError(res)
  }

  const contentType = res.headers.get('content-type') ?? ''
  if (!res.body || !contentType.includes('ndjson')) {
    const final = (await res.json()) as QueryResponse
    handlers.onStatus?.('closed')
    handlers.onFinal?.(final)
    handlers.onClose?.('complete')
    return {
      runId: final?.run_id ?? null,
      lastSeq: 0,
      final,
      transport: 'buffered',
      gapped: false,
    }
  }

  handlers.onStatus?.('open')

  const tracker = new SeqTracker(0)
  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''
  let runId: Id | null = null
  let gapped = false
  let reason: CloseReason = 'complete'

  const consume = (line: string): boolean => {
    const trimmed = line.trim()
    if (!trimmed) return true
    const event = parseEnvelope(trimmed)
    if (!event) return true
    if (!runId && event.run_id) runId = event.run_id

    const verdict = tracker.check(event.seq)
    if (verdict === 'drop') return true
    if (verdict === 'gap') {
      gapped = true
      handlers.onGap?.({
        runId: event.run_id,
        expected: tracker.lastSeq + 1,
        received: event.seq,
      })
      reason = 'expired'
      return false // stop reading; the caller reattaches with from_seq
    }

    handlers.onEvent(event)
    if (TERMINAL_EVENT_TYPES.includes(event.type)) {
      reason = event.type === 'error' ? 'error' : 'complete'
      return false
    }
    return true
  }

  try {
    for (;;) {
      const { done, value } = await reader.read()
      if (done) break
      buffer += decoder.decode(value, { stream: true })
      let newline = buffer.indexOf('\n')
      let keepReading = true
      while (newline !== -1) {
        const line = buffer.slice(0, newline)
        buffer = buffer.slice(newline + 1)
        keepReading = consume(line)
        if (!keepReading) break
        newline = buffer.indexOf('\n')
      }
      if (!keepReading) break
    }
    if (buffer.trim()) consume(buffer)
  } catch (err) {
    if (!(err instanceof Error && err.name === 'AbortError')) {
      handlers.onStatus?.('error')
      throw err
    }
    reason = 'manual'
  } finally {
    try {
      await reader.cancel()
    } catch {
      /* already closed */
    }
  }

  handlers.onStatus?.('closed')
  handlers.onClose?.(reason)
  return { runId, lastSeq: tracker.lastSeq, final: null, transport: 'ndjson', gapped }
}

/** `/api/v1/runs/{id}/events` → `{id}`. The respond() reply hands us this path. */
export function runIdFromWatchPath(path: string | null | undefined): Id | null {
  if (!path) return null
  const match = /\/runs\/([^/]+)\/events/.exec(path)
  return match ? decodeURIComponent(match[1]) : null
}

/* ------------------------------------------------------- buffered → events */

/**
 * Turn a buffered `POST /query` body into the events it would have streamed.
 *
 * Two uses. A deployment that does not stream (a proxy that refuses ndjson, a
 * server with streaming off) still fills the trace panel row by row. And on
 * `run.complete` the same function reconciles: the durable body is replayed
 * over whatever the stream managed to deliver, so a dropped `step.finished`
 * or a missed card lands anyway.
 */
export function eventsFromQueryResponse(res: QueryResponse, query = ''): RunEvent[] {
  const bodies: RunEventBody[] = []
  const actions = new Map<Id, ActionRecord>((res.actions ?? []).map((a) => [a.id, a]))
  const prompts = new Map<Id, PendingInput>((res.pending_inputs ?? []).map((p) => [p.id, p]))
  const seenActions = new Set<Id>()
  const seenPrompts = new Set<Id>()

  bodies.push({
    type: 'run.started',
    data: { conversation_id: res.conversation_id, message_id: res.message_id, query },
  })

  if (res.intent) {
    bodies.push({
      type: 'intent',
      data: {
        ...res.intent,
        answer_style: res.intent.answer_style ?? res.answer_style,
        planner_tier: res.intent.planner_tier ?? res.planner_tier,
      },
    })
  }

  const steps = [...(res.steps ?? [])].sort((a, b) => a.seq - b.seq)
  for (const step of steps) {
    bodies.push({
      type: 'plan.step',
      data: { node_id: step.node_id, op: step.op, depends_on: step.depends_on ?? [] },
    })
  }
  for (const step of steps) {
    if (step.status === 'pending') continue
    bodies.push({
      type: 'step.started',
      data: { node_id: step.node_id, op: step.op, round: step.round },
    })
    if (step.status === 'running') continue
    bodies.push({
      type: 'step.finished',
      data: {
        node_id: step.node_id,
        op: step.op,
        status: step.status,
        round: step.round,
        duration_ms: step.duration_ms ?? 0,
        attempts: step.attempts,
        summary: step.result_summary ?? null,
      },
    })
  }

  const raiseAction = (action: ActionRecord) => {
    if (seenActions.has(action.id)) return
    seenActions.add(action.id)
    bodies.push({
      type: 'action.prepared',
      data: {
        action_id: action.id,
        op: action.op,
        status: action.status,
        requires_input_id: action.requires_input_id ?? null,
        external_ref: action.external_ref ?? null,
        preview: action.preview ?? {},
        payload_fields: action.payload_fields,
        expires_at: action.expires_at ?? null,
      },
    })
  }

  const raisePrompt = (prompt: PendingInput) => {
    if (seenPrompts.has(prompt.id) || prompt.status !== 'pending') return
    seenPrompts.add(prompt.id)
    bodies.push({
      type: 'input.raised',
      data: {
        input_id: prompt.id,
        kind: prompt.kind,
        blocking: prompt.blocking,
        node_id: prompt.node_id ?? undefined,
        message_id: prompt.message_id ?? res.message_id,
        prompt: prompt.prompt,
        value_schema: prompt.value_schema,
        options: prompt.options ?? null,
        expires_at: prompt.expires_at ?? null,
      },
    })
  }

  const blocks = res.content ?? (res.text ? [{ type: 'text' as const, data: { markdown: res.text } }] : [])
  blocks.forEach((block, index) => {
    if (block.type === 'text') {
      if (!block.data?.markdown) return
      bodies.push({
        type: 'content.delta',
        data: { message_id: res.message_id, block_index: index, text: block.data.markdown },
      })
      return
    }
    if (block.type === 'action') {
      const action = actions.get(block.ref)
      // A ref with no object was dropped on read. Never render an empty box.
      if (action) raiseAction(action)
      return
    }
    // A widget carries its own data rather than a ref, so there is nothing
    // here to resolve against the rows behind the message.
    if (block.type !== 'input') return
    const prompt = prompts.get(block.ref)
    if (prompt) raisePrompt(prompt)
  })

  for (const action of actions.values()) raiseAction(action)
  for (const prompt of prompts.values()) raisePrompt(prompt)

  const blocking = [...prompts.values()].find((p) => p.blocking && p.status === 'pending')
  if (res.status === 'awaiting_input' && blocking) {
    bodies.push({
      type: 'run.paused',
      data: {
        reason: 'awaiting_input',
        input_id: blocking.id,
        resumable_until: blocking.expires_at ?? undefined,
      },
    })
  }

  bodies.push({
    type: 'run.complete',
    data: {
      status: res.status,
      message_id: res.message_id,
      answer_style: res.answer_style,
      planner_tier: res.planner_tier,
      usage: res.usage,
      timings: res.timings,
      degraded: res.degraded ?? [],
    },
  })

  const ts = new Date().toISOString()
  return bodies.map(
    (body, index) => ({ v: 1, seq: index + 1, run_id: res.run_id, ts, ...body }) as RunEvent,
  )
}

/* ------------------------------------------------------------------ startRun */

export interface StartRunHandlers extends StreamHandlers {
  /** The buffered body, when the server answered with JSON instead of ndjson. */
  onFinal?: (response: QueryResponse) => void
}

export interface StartRunOptions {
  /** Server-side filter for the reattach leg. Omit for the full trace. */
  types?: RunEventType[]
}

/**
 * Ask a question and read the run as it happens.
 *
 * A new run has no id until the POST is answered, so there is nothing to
 * subscribe to first — the events come back down the POST as ndjson. If that
 * stream gaps or dies before `run.complete`, this reattaches to
 * `GET /runs/{id}/events` from the first event it missed and the server
 * replays. If the server did not stream at all, the buffered body is expanded
 * into the same events so the UI behaves identically.
 *
 * Returns the stopper. Call it before starting the next run.
 */
export function startRun(
  body: QueryRequest,
  handlers: StartRunHandlers,
  options: StartRunOptions = {},
): StopFn {
  const controller = new AbortController()
  let stopped = false
  let follow: StopFn | null = null
  let lastSeq = 0
  let terminal = false

  const stop: StopFn = () => {
    if (stopped) return
    stopped = true
    controller.abort()
    follow?.()
    follow = null
  }

  const onEvent = (event: RunEvent) => {
    lastSeq = Math.max(lastSeq, event.seq)
    if (TERMINAL_EVENT_TYPES.includes(event.type)) terminal = true
    handlers.onEvent(event)
  }

  void (async () => {
    try {
      // onFinal is handled here, once, after the body has been expanded —
      // passing it down as well would report the same settled body twice.
      const result = await streamQuery(
        body,
        { ...handlers, onEvent, onFinal: undefined },
        controller.signal,
      )
      if (stopped) return

      if (result.final) {
        // Buffered: replay the durable body as the events it stands for.
        for (const event of eventsFromQueryResponse(result.final, body.query)) onEvent(event)
        handlers.onFinal?.(result.final)
        return
      }

      if (terminal) return

      const runId = result.runId
      if (!runId) {
        handlers.onError?.(new StreamError('The run ended before it said which run it was.'))
        return
      }
      // Gapped, or the connection dropped mid-run. The run is still going
      // server-side; pick it up from the first event we did not see.
      follow = openRunStream(runId, { ...handlers, onEvent }, {
        fromSeq: lastSeq + 1,
        types: options.types,
      })
    } catch (err) {
      if (stopped) return
      handlers.onError?.(err)
    }
  })()

  return stop
}
