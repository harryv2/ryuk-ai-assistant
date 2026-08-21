/**
 * The event stream, turned into something a component can render.
 *
 * One reducer, one switch, exhaustive over the event union. Every case applies
 * the moment its event lands — the panel is meant to fill in as the run
 * happens, not redraw at the end.
 *
 * Three rules the rest of the file is built around:
 *
 *   1. Steps keep **arrival order**, not plan order. `plan.step` events come in
 *      as the model streams the plan, so the rows appear one by one, and a row
 *      never jumps position afterwards.
 *   2. A gap in `seq` is never guessed past. The stream layer replays from the
 *      buffer; if that fails, `GET /conversations/{id}` is the durable record
 *      and `hydrate` rebuilds from it.
 *   3. `run.complete` is a checkpoint, not the end of the story. The settled
 *      body is reconciled over whatever the stream delivered, so a missed
 *      `step.finished` or a card that never arrived still shows up.
 */

import { useCallback, useEffect, useMemo, useReducer, useRef } from 'react'

import { ApiError, cancelPrompt as apiCancelPrompt, getConversation, respondToPrompt } from './api'
import { openRunStream, runIdFromWatchPath, startRun, type StopFn } from './sse'
import type {
  ActionRecord,
  ConnectionStatus,
  ContentBlock,
  ConversationDetail,
  Degraded,
  Freshness,
  Id,
  LiveBlock,
  PendingInput,
  QueryResponse,
  ResolvedWindow,
  RunEvent,
  RunState,
  ServiceName,
  StepRecord,
  TraceStep,
} from './types'

/* --------------------------------------------------------------- the state */

export const initialRunState: RunState = {
  runId: null,
  conversationId: null,
  messageId: null,
  query: null,
  status: 'idle',
  streaming: false,
  connection: 'idle',
  lastSeq: 0,

  progress: null,
  probe: null,
  intent: null,
  windows: {},
  resolvedWindow: null,

  steps: {},
  stepOrder: [],

  blocks: [],
  answerText: '',
  tokens: '',

  prompts: {},
  promptOrder: [],
  actions: {},
  actionOrder: [],
  promptBusy: {},
  promptErrors: {},

  degraded: [],
  usage: null,
  timings: null,
  answerStyle: null,
  plannerTier: null,
  entities: [],

  paused: null,
  error: null,
  startedAt: null,
  finishedAt: null,
}

export type RunCommand =
  | { kind: 'reset'; keepConversation?: boolean }
  /** The person hit send. Optimistic, before run.started comes back. */
  | { kind: 'asked'; query: string; conversationId: Id | null }
  | { kind: 'event'; event: RunEvent }
  | { kind: 'connection'; status: ConnectionStatus }
  /** The settled `POST /query` body — the authority over the stream. */
  | { kind: 'reconcile'; response: QueryResponse }
  /** The durable record, after a gap or a reload. */
  | { kind: 'hydrate'; conversation: ConversationDetail }
  | { kind: 'prompt-busy'; id: Id; busy: boolean }
  | { kind: 'prompt-error'; id: Id; message: string | null }
  | { kind: 'prompt-merge'; prompt: PendingInput }
  | { kind: 'action-merge'; action: ActionRecord }
  | { kind: 'stream-error'; code: string; message: string }

/* -------------------------------------------------------------- step upkeep */

function stepKey(nodeId: string, round = 0): string {
  // A retry round is its own row: the same node, run again, is new information.
  return round ? `${nodeId}#${round}` : nodeId
}

function blankStep(nodeId: string, op: string, order: number): TraceStep {
  return {
    node_id: nodeId,
    op,
    depends_on: [],
    status: 'pending',
    attempts: 0,
    retries: [],
    order,
  }
}

/** Insert on first sight, patch on every sight after. Order never changes. */
function upsertStep(
  state: RunState,
  key: string,
  nodeId: string,
  op: string,
  patch: Partial<TraceStep>,
): RunState {
  const existing = state.steps[key]
  if (existing) {
    return {
      ...state,
      steps: { ...state.steps, [key]: { ...existing, ...patch } },
    }
  }
  const seeded: TraceStep = {
    ...blankStep(nodeId, op, state.stepOrder.length),
    ...patch,
  }
  return {
    ...state,
    steps: { ...state.steps, [key]: seeded },
    stepOrder: [...state.stepOrder, key],
  }
}

/** "gmail.search_emails" → "gmail". Used to name a degraded service. */
function serviceOf(op: string | undefined): ServiceName | null {
  if (!op) return null
  const head = op.split('.')[0]
  return head === 'gmail' || head === 'gcal' || head === 'gdrive' ? head : null
}

function withDegraded(list: Degraded[], next: Degraded): Degraded[] {
  if (list.some((d) => d.service === next.service && d.reason === next.reason)) return list
  return [...list, next]
}

/**
 * The two carriers spell a degraded entry differently. The durable body sends
 * the whole object; `run.complete` sends the compact form — just the names of
 * the services that were degraded. Take either, so the banner names the service
 * instead of rendering an object that has no fields on it.
 */
function degradedEntries(list: (Degraded | string)[] | undefined): Degraded[] {
  if (!list?.length) return []
  return list.map((entry) =>
    typeof entry === 'string' ? { service: entry, reason: 'unavailable' } : entry,
  )
}

/* ------------------------------------------------------------ block upkeep */

/**
 * Blocks are held in arrival order, and addressed by the server's
 * `block_index`. Those are two different things on purpose: a card can appear
 * at 650ms, long before the block it will eventually occupy exists, so a card
 * the server has not numbered is appended and never claims a number. Only a
 * numbered block can be the target of a delta — which is what makes "a drafted
 * body streams into the confirm card" work without prose ever landing in a card
 * by accident.
 */
function applyDelta(blocks: LiveBlock[], index: number, text: string): LiveBlock[] {
  const at = blocks.findIndex((b) => b.index === index)
  if (at === -1) return [...blocks, { kind: 'text', text, index }]

  const next = [...blocks]
  const target = next[at]
  // Only text and card drafts stream in character by character. A widget
  // arrives whole, so there is nothing here to append to.
  if (target.kind === 'text') next[at] = { ...target, text: target.text + text }
  else if (target.kind === 'action' || target.kind === 'input') {
    next[at] = { ...target, draft: target.draft + text }
  }
  return next
}

function joinText(blocks: LiveBlock[]): string {
  return blocks
    .filter((b): b is Extract<LiveBlock, { kind: 'text' }> => b.kind === 'text')
    .map((b) => b.text)
    .filter(Boolean)
    .join('\n\n')
}

/** One card per id, in the order the cards arrived. */
function placeCard(
  blocks: LiveBlock[],
  block: Extract<LiveBlock, { kind: 'action' | 'input' }>,
): LiveBlock[] {
  if (blocks.some((b) => (b.kind === 'action' || b.kind === 'input') && b.ref === block.ref))
    return blocks
  return [...blocks, block]
}

/* ------------------------------------------------------------- the reducer */

export function runReducer(state: RunState, command: RunCommand): RunState {
  switch (command.kind) {
    case 'reset':
      // "New chat" has to forget which thread this was, or the next question
      // lands back in the old one. Clearing the run but keeping the thread is
      // what "ask that again" wants, so both are available and the caller
      // says which it means.
      return {
        ...initialRunState,
        conversationId: command.keepConversation ? state.conversationId : null,
      }

    case 'asked':
      return {
        ...initialRunState,
        conversationId: command.conversationId,
        query: command.query,
        status: 'running',
        streaming: true,
        connection: 'connecting',
      }

    case 'connection':
      return { ...state, connection: command.status }

    case 'prompt-busy':
      return {
        ...state,
        promptBusy: { ...state.promptBusy, [command.id]: command.busy },
      }

    case 'prompt-error': {
      const errors = { ...state.promptErrors }
      if (command.message) errors[command.id] = command.message
      else delete errors[command.id]
      return { ...state, promptErrors: errors }
    }

    case 'prompt-merge': {
      const id = command.prompt.id
      const merged = { ...(state.prompts[id] ?? {}), ...command.prompt }
      return {
        ...state,
        prompts: { ...state.prompts, [id]: merged },
        promptOrder: state.promptOrder.includes(id)
          ? state.promptOrder
          : [...state.promptOrder, id],
        blocks: placeCard(state.blocks, { kind: 'input', ref: id, draft: '' }),
      }
    }

    case 'action-merge': {
      const id = command.action.id
      const merged = { ...(state.actions[id] ?? {}), ...command.action }
      return {
        ...state,
        actions: { ...state.actions, [id]: merged },
        actionOrder: state.actionOrder.includes(id)
          ? state.actionOrder
          : [...state.actionOrder, id],
      }
    }

    case 'stream-error':
      return {
        ...state,
        connection: 'error',
        streaming: false,
        error: state.error ?? {
          code: 'INTERNAL',
          message: command.message,
          retryable: true,
          details: { code: command.code },
        },
      }

    case 'reconcile':
      return reconcile(state, command.response)

    case 'hydrate':
      return hydrate(state, command.conversation)

    case 'event':
      return applyEvent(state, command.event)
  }
}

function applyEvent(state: RunState, event: RunEvent): RunState {
  const base: RunState = { ...state, lastSeq: Math.max(state.lastSeq, event.seq) }

  switch (event.type) {
    case 'run.started': {
      const d = event.data
      return {
        ...initialRunState,
        runId: event.run_id,
        conversationId: d.conversation_id ?? state.conversationId,
        messageId: d.message_id ?? null,
        query: d.query ?? state.query,
        status: 'running',
        streaming: true,
        connection: 'open',
        lastSeq: event.seq,
        startedAt: event.ts,
      }
    }

    case 'progress':
      return { ...base, progress: event.data }

    case 'probe.done':
      return { ...base, probe: event.data }

    case 'intent': {
      const d = event.data
      const windows = d.windows ?? {}
      const first = Object.values(windows)[0] as ResolvedWindow | undefined
      return {
        ...base,
        intent: d,
        windows,
        resolvedWindow: first ?? base.resolvedWindow,
        answerStyle: d.answer_style ?? base.answerStyle,
        plannerTier: d.planner_tier ?? base.plannerTier,
      }
    }

    case 'plan.step': {
      const d = event.data
      // The plan draws itself: one row per event, in the order they arrive.
      return upsertStep(base, stepKey(d.node_id), d.node_id, d.op, {
        depends_on: d.depends_on ?? [],
        expect: d.expect,
        optional: d.optional,
        freshness: d.freshness,
        is_write: d.is_write,
        needs_confirm: d.needs_confirm,
        status: 'pending',
      })
    }

    case 'step.started': {
      const d = event.data
      const round = d.round ?? 0
      return upsertStep(base, stepKey(d.node_id, round), d.node_id, d.op, {
        status: 'running',
        label: d.label,
        round,
        attempts: Math.max(1, base.steps[stepKey(d.node_id, round)]?.attempts ?? 0),
        started_at: event.ts,
      })
    }

    case 'step.retrying': {
      const d = event.data
      const key = stepKey(d.node_id, 0)
      const current = base.steps[key]
      return upsertStep(base, key, d.node_id, d.op ?? current?.op ?? d.node_id, {
        status: 'running',
        attempts: Math.max(current?.attempts ?? 1, d.attempt + 1),
        retries: [
          ...(current?.retries ?? []),
          {
            attempt: d.attempt,
            of: d.of,
            error_class: d.error_class,
            google_status: d.google_status,
            backoff_ms: d.backoff_ms,
            label: d.label,
            at: event.ts,
          },
        ],
      })
    }

    case 'step.finished': {
      const d = event.data
      const key = stepKey(d.node_id, d.round ?? 0)
      const current = base.steps[key]
      const skipReason =
        d.status === 'succeeded' ? null : (d.outcome?.reason ?? d.outcome?.message ?? d.status)

      const next = upsertStep(base, key, d.node_id, d.op ?? current?.op ?? d.node_id, {
        status: d.status,
        round: d.round ?? current?.round,
        duration_ms: d.duration_ms,
        attempts: d.attempts ?? current?.attempts ?? 1,
        summary: d.summary ?? null,
        outcome: d.outcome ?? null,
        skipReason,
        finished_at: event.ts,
      })

      // A failed step on an optional node is a named gap, not a failure —
      // but only when the service itself failed. A step that died because WE
      // wrote its arguments wrong (INVALID and friends) is our fault, and
      // putting "Gmail — unavailable" over it sends the reader to a status
      // page for a service that was never even called.
      const service = serviceOf(d.op ?? current?.op)
      const ourFault =
        ['INVALID', 'VALIDATION_ERROR', 'NOT_FOUND'].includes(String(d.outcome?.class ?? '')) ||
        ['invalid_arguments', 'bad_args', 'unresolved_reference', 'gate_false', 'plan_invalid'].includes(
          String(d.outcome?.reason ?? ''),
        )
      if (d.status === 'failed' && service && !ourFault) {
        return {
          ...next,
          degraded: withDegraded(next.degraded, {
            service,
            reason: d.outcome?.reason ?? 'step_failed',
            detail: d.outcome?.message,
            until: d.outcome?.until ?? null,
          }),
        }
      }
      return next
    }

    case 'content.delta': {
      const d = event.data
      const index = typeof d.block_index === 'number' ? d.block_index : 0
      // Deltas for one index concatenate. Applying one twice corrupts the text,
      // which is the whole reason for the seq rule.
      const blocks = applyDelta(base.blocks, index, d.text)
      return {
        ...base,
        messageId: base.messageId ?? d.message_id,
        blocks,
        answerText: joinText(blocks),
      }
    }

    case 'token':
      // The pre-render stream. Trimmed — this is a view, not a transcript.
      return { ...base, tokens: (base.tokens + event.data.text).slice(-4000) }

    case 'input.raised': {
      const d = event.data
      const prompt: PendingInput = {
        id: d.input_id,
        run_id: event.run_id,
        conversation_id: base.conversationId ?? undefined,
        message_id: d.message_id ?? base.messageId,
        node_id: d.node_id ?? null,
        kind: d.kind,
        blocking: d.blocking,
        prompt: d.prompt,
        value_schema: d.value_schema,
        options: d.options ?? null,
        status: 'pending',
        response: null,
        expires_at: d.expires_at ?? null,
        answered_at: null,
        created_at: event.ts,
      }
      return {
        ...base,
        prompts: { ...base.prompts, [prompt.id]: { ...base.prompts[prompt.id], ...prompt } },
        promptOrder: base.promptOrder.includes(prompt.id)
          ? base.promptOrder
          : [...base.promptOrder, prompt.id],
        blocks: placeCard(base.blocks, {
          kind: 'input',
          ref: prompt.id,
          draft: '',
          index: d.block_index,
        }),
      }
    }

    case 'action.prepared': {
      const d = event.data
      const action: ActionRecord = {
        id: d.action_id,
        op: d.op,
        status: d.status,
        requires_input_id: d.requires_input_id ?? null,
        external_ref: d.external_ref ?? null,
        preview: d.preview ?? {},
        payload_fields: d.payload_fields ?? [],
        expires_at: d.expires_at ?? null,
        created_at: event.ts,
      }
      return {
        ...base,
        actions: { ...base.actions, [action.id]: { ...base.actions[action.id], ...action } },
        actionOrder: base.actionOrder.includes(action.id)
          ? base.actionOrder
          : [...base.actionOrder, action.id],
        blocks: placeCard(base.blocks, {
          kind: 'action',
          ref: action.id,
          draft: '',
          index: d.block_index,
        }),
      }
    }

    case 'run.paused':
      return {
        ...base,
        paused: event.data,
        status: 'awaiting_input',
        streaming: false,
        finishedAt: event.ts,
      }

    case 'run.complete': {
      const d = event.data
      return {
        ...base,
        status: d.status,
        streaming: false,
        messageId: d.message_id ?? base.messageId,
        answerStyle: d.answer_style ?? base.answerStyle,
        plannerTier: d.planner_tier ?? base.plannerTier,
        usage: d.usage ?? base.usage,
        timings: d.timings ?? base.timings,
        degraded: d.degraded?.length ? degradedEntries(d.degraded) : base.degraded,
        progress: null,
        finishedAt: event.ts,
      }
    }

    case 'error':
      return {
        ...base,
        error: event.data,
        status: event.data.partial ? base.status : 'failed',
        streaming: false,
        progress: null,
        finishedAt: event.ts,
      }

    case 'action.done': {
      const d = event.data
      const current = base.actions[d.action_id]
      return {
        ...base,
        actions: {
          ...base.actions,
          [d.action_id]: {
            ...(current ?? { id: d.action_id, op: d.op, preview: {} }),
            id: d.action_id,
            op: d.op,
            status: d.status ?? 'done',
            result: d.result,
            error: null,
            attempts: d.attempts,
            executed_at: event.ts,
          },
        },
        actionOrder: base.actionOrder.includes(d.action_id)
          ? base.actionOrder
          : [...base.actionOrder, d.action_id],
      }
    }

    case 'action.failed': {
      const d = event.data
      const current = base.actions[d.action_id]
      return {
        ...base,
        actions: {
          ...base.actions,
          [d.action_id]: {
            ...(current ?? { id: d.action_id, op: d.op, preview: {} }),
            id: d.action_id,
            op: d.op,
            status: d.status ?? 'failed',
            error: d.error,
            attempts: d.attempts,
          },
        },
        actionOrder: base.actionOrder.includes(d.action_id)
          ? base.actionOrder
          : [...base.actionOrder, d.action_id],
      }
    }
  }
}

/* -------------------------------------------------------------- reconcilers */

function stepsFromRecords(state: RunState, records: StepRecord[]): RunState {
  let next = state
  for (const record of [...records].sort((a, b) => a.seq - b.seq)) {
    const key = stepKey(record.node_id, record.round ?? 0)
    next = upsertStep(next, key, record.node_id, record.op, {
      depends_on: record.depends_on ?? [],
      status: record.status,
      duration_ms: record.duration_ms ?? null,
      attempts: record.attempts ?? next.steps[key]?.attempts ?? 0,
      summary: record.result_summary ?? null,
      round: record.round,
      started_at: record.started_at ?? undefined,
      finished_at: record.finished_at ?? undefined,
      skipReason:
        record.status === 'succeeded' || record.status === 'pending' ? null : record.status,
    })
  }
  return next
}

function textFromBlocks(content: ContentBlock[] | undefined): string {
  return (content ?? [])
    .filter((b): b is Extract<ContentBlock, { type: 'text' }> => b.type === 'text')
    .map((b) => b.data.markdown)
    .filter(Boolean)
    .join('\n\n')
}

function liveBlocks(content: ContentBlock[] | undefined, previous: LiveBlock[]): LiveBlock[] {
  if (!content?.length) return previous
  const draftOf = (ref: Id) => {
    const found = previous.find(
      (b) => (b.kind === 'action' || b.kind === 'input') && b.ref === ref,
    )
    return found && (found.kind === 'action' || found.kind === 'input') ? found.draft : ''
  }
  // The durable record is where a block's real index comes from.
  return content.map((block, index) =>
    block.type === 'text'
      ? { kind: 'text', text: block.data.markdown, index }
      : block.type === 'widget'
        ? // A widget is self-contained; it renders from its own data, and its
          // text is already on screen in the text block above it.
          { kind: 'widget', block, index }
        : block.type === 'action'
          ? { kind: 'action', ref: block.ref, draft: draftOf(block.ref), index }
          : { kind: 'input', ref: block.ref, draft: draftOf(block.ref), index },
  )
}

/**
 * The settled body wins. Everything the stream built is kept where the body is
 * silent, and replaced where it speaks — a card that arrived twice, a step
 * whose `finished` was lost, a status that changed after the last event.
 */
export function reconcile(state: RunState, res: QueryResponse): RunState {
  let next: RunState = {
    ...state,
    runId: res.run_id ?? state.runId,
    conversationId: res.conversation_id ?? state.conversationId,
    messageId: res.message_id ?? state.messageId,
    status: res.status,
    streaming: false,
    intent: res.intent ?? state.intent,
    windows: res.intent?.windows ?? state.windows,
    resolvedWindow:
      (Object.values(res.intent?.windows ?? {})[0] as ResolvedWindow | undefined) ??
      state.resolvedWindow,
    answerStyle: res.answer_style ?? state.answerStyle,
    plannerTier: res.planner_tier ?? state.plannerTier,
    usage: res.usage ?? state.usage,
    timings: res.timings ?? state.timings,
    degraded: res.degraded?.length ? res.degraded : state.degraded,
    entities: res.entities ?? state.entities,
  }

  next = stepsFromRecords(next, res.steps ?? [])

  for (const action of res.actions ?? []) {
    next = {
      ...next,
      actions: { ...next.actions, [action.id]: { ...next.actions[action.id], ...action } },
      actionOrder: next.actionOrder.includes(action.id)
        ? next.actionOrder
        : [...next.actionOrder, action.id],
    }
  }

  for (const prompt of res.pending_inputs ?? []) {
    next = {
      ...next,
      prompts: { ...next.prompts, [prompt.id]: { ...next.prompts[prompt.id], ...prompt } },
      promptOrder: next.promptOrder.includes(prompt.id)
        ? next.promptOrder
        : [...next.promptOrder, prompt.id],
    }
  }

  const blocks = liveBlocks(res.content, next.blocks)
  return {
    ...next,
    blocks,
    answerText: textFromBlocks(res.content) || res.text || next.answerText,
  }
}

/**
 * Rebuild from Postgres. This is what a `seq` gap costs: one refetch, and the
 * state is exactly what the server has, however many events went missing.
 */
export function hydrate(state: RunState, convo: ConversationDetail): RunState {
  const runId = state.runId
  const run = runId ? convo.runs.find((r) => r.id === runId) : convo.runs[convo.runs.length - 1]
  const messages = convo.messages.filter((m) => m.role === 'assistant' && (!runId || m.run_id === runId))
  const message = messages[messages.length - 1]

  let next: RunState = {
    ...state,
    conversationId: convo.id,
    runId: run?.id ?? state.runId,
    messageId: message?.id ?? state.messageId,
    status: run?.status ?? state.status,
    streaming: run?.status === 'running',
    intent: run?.intent ?? state.intent,
    plannerTier: run?.planner_tier ?? state.plannerTier,
    entities: convo.entities ?? state.entities,
  }

  if (message?.trace?.length) next = stepsFromRecords(next, message.trace)

  for (const action of convo.actions ?? []) {
    next = {
      ...next,
      actions: { ...next.actions, [action.id]: { ...next.actions[action.id], ...action } },
      actionOrder: next.actionOrder.includes(action.id)
        ? next.actionOrder
        : [...next.actionOrder, action.id],
    }
  }

  for (const prompt of convo.pending_inputs ?? []) {
    next = {
      ...next,
      prompts: { ...next.prompts, [prompt.id]: { ...next.prompts[prompt.id], ...prompt } },
      promptOrder: next.promptOrder.includes(prompt.id)
        ? next.promptOrder
        : [...next.promptOrder, prompt.id],
    }
  }

  const blocks = liveBlocks(message?.content, next.blocks)
  return {
    ...next,
    blocks,
    answerText: textFromBlocks(message?.content) || next.answerText,
  }
}

/* ------------------------------------------------------------------- hook */

export interface UseRunOptions {
  conversationId?: Id | null
  /** Overrides the stored zone for this query only. Defaults to the browser's. */
  timezone?: string
  weekStart?: number
  freshness?: Freshness
  /** Pull the durable record when a run settles or the stream gaps. Default on. */
  reconcileFromServer?: boolean
  onConversation?: (id: Id) => void
  /** Every successful read of the durable record, so the shell can hold it too. */
  onHydrated?: (conversation: ConversationDetail) => void
  onReauthRequired?: (error: ApiError) => void
  onError?: (error: unknown) => void
}

export interface UseRunResult {
  state: RunState
  /** Trace rows in arrival order — the order the plan drew itself in. */
  steps: TraceStep[]
  /** Cards still waiting on the person. */
  openPrompts: PendingInput[]
  /** The one holding the run up, if any. */
  blockingPrompt: PendingInput | null
  actions: ActionRecord[]
  busy: boolean
  ask: (query: string) => void
  respond: (inputId: Id, value: unknown) => Promise<void>
  dismissPrompt: (inputId: Id) => Promise<void>
  reattach: (runId: Id, fromSeq?: number) => void
  refetch: (conversationId?: Id | null) => Promise<ConversationDetail | null>
  reset: (options?: { keepConversation?: boolean }) => void
  stop: () => void
}

const browserZone = (): string | undefined => {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || undefined
  } catch {
    return undefined
  }
}

export function useRun(options: UseRunOptions = {}): UseRunResult {
  const [state, dispatch] = useReducer(runReducer, {
    ...initialRunState,
    conversationId: options.conversationId ?? null,
  })

  const stopRef = useRef<StopFn | null>(null)
  const stateRef = useRef(state)
  stateRef.current = state

  const optionsRef = useRef(options)
  optionsRef.current = options

  // The thread this hook was mounted on — a SEED, not a standing default.
  //
  // `options.conversationId` is re-read from props on every render, so using
  // it as a fallback means "New chat" can clear the run state and have the old
  // thread quietly reinstated on the very next question. Holding the seed in a
  // ref that only reset() may clear is what makes forgetting stick.
  const seedConversationRef = useRef<Id | null>(options.conversationId ?? null)

  useEffect(() => () => stopRef.current?.(), [])

  const refetch = useCallback(
    async (conversationId?: Id | null): Promise<ConversationDetail | null> => {
      const id = conversationId ?? stateRef.current.conversationId
      if (!id) return null
      try {
        const convo = await getConversation(id, { include_trace: true })
        dispatch({ kind: 'hydrate', conversation: convo })
        optionsRef.current.onHydrated?.(convo)
        return convo
      } catch (err) {
        if (err instanceof ApiError && err.needsReauth) optionsRef.current.onReauthRequired?.(err)
        optionsRef.current.onError?.(err)
        return null
      }
    },
    [],
  )

  const handlers = useMemo(
    () => ({
      onEvent: (event: RunEvent) => {
        dispatch({ kind: 'event', event })
        if (event.type === 'run.started' && event.data.conversation_id) {
          optionsRef.current.onConversation?.(event.data.conversation_id)
        }
        if (event.type === 'run.complete' || event.type === 'error') {
          // The stream is an accelerator. Settle against the durable record.
          if (optionsRef.current.reconcileFromServer !== false) void refetch()
        }
      },
      onStatus: (status: ConnectionStatus) => dispatch({ kind: 'connection', status }),
      onGap: () => {
        // Never guess across a gap: the conversation is the source of truth.
        if (optionsRef.current.reconcileFromServer !== false) void refetch()
      },
      onExpired: () => {
        void refetch()
      },
      onError: (err: unknown) => {
        if (err instanceof ApiError) {
          if (err.needsReauth) optionsRef.current.onReauthRequired?.(err)
          dispatch({ kind: 'stream-error', code: err.code, message: err.message })
        } else {
          const message = err instanceof Error ? err.message : 'The live connection dropped.'
          dispatch({ kind: 'stream-error', code: 'STREAM_LOST', message })
        }
        optionsRef.current.onError?.(err)
        void refetch()
      },
      onFinal: (response: QueryResponse) => dispatch({ kind: 'reconcile', response }),
    }),
    [refetch],
  )

  const stop = useCallback(() => {
    stopRef.current?.()
    stopRef.current = null
  }, [])

  const ask = useCallback(
    (query: string) => {
      const text = query.trim()
      if (!text) return
      stop()
      const conversationId =
        stateRef.current.conversationId ?? seedConversationRef.current ?? null
      dispatch({ kind: 'asked', query: text, conversationId })
      stopRef.current = startRun(
        {
          query: text,
          conversation_id: conversationId ?? undefined,
          timezone: optionsRef.current.timezone ?? browserZone(),
          week_start: optionsRef.current.weekStart,
          freshness: optionsRef.current.freshness,
        },
        handlers,
      )
    },
    [handlers, stop],
  )

  const reattach = useCallback(
    (runId: Id, fromSeq?: number) => {
      stop()
      dispatch({ kind: 'connection', status: 'connecting' })
      stopRef.current = openRunStream(runId, handlers, {
        fromSeq: fromSeq ?? stateRef.current.lastSeq + 1,
      })
    },
    [handlers, stop],
  )

  const respond = useCallback(
    async (inputId: Id, value: unknown) => {
      dispatch({ kind: 'prompt-busy', id: inputId, busy: true })
      dispatch({ kind: 'prompt-error', id: inputId, message: null })
      try {
        const res = await respondToPrompt(inputId, value)
        const answered = stateRef.current.prompts[inputId]
        if (answered) {
          dispatch({
            kind: 'prompt-merge',
            prompt: {
              ...answered,
              status: res.status ?? 'answered',
              response: value,
              answered_at: res.answered_at ?? new Date().toISOString(),
            },
          })
        }
        if (res.action) dispatch({ kind: 'action-merge', action: res.action })
        // Edit raises a fresh confirm against the revised payload.
        if (res.next_input) dispatch({ kind: 'prompt-merge', prompt: res.next_input })

        const resumed = res.resumed_run_id ?? runIdFromWatchPath(res.watch)
        if (resumed) reattach(resumed)
        else if (optionsRef.current.reconcileFromServer !== false) void refetch()
      } catch (err) {
        const message =
          err instanceof ApiError ? err.message : 'That answer did not go through.'
        dispatch({ kind: 'prompt-error', id: inputId, message })
        if (err instanceof ApiError) {
          if (err.needsReauth) optionsRef.current.onReauthRequired?.(err)
          // The card moved on underneath us. Refetch it rather than retrying.
          if (err.promptStale) void refetch()
        }
        optionsRef.current.onError?.(err)
      } finally {
        dispatch({ kind: 'prompt-busy', id: inputId, busy: false })
      }
    },
    [reattach, refetch],
  )

  const dismissPrompt = useCallback(
    async (inputId: Id) => {
      dispatch({ kind: 'prompt-busy', id: inputId, busy: true })
      try {
        const res = await apiCancelPrompt(inputId)
        const current = stateRef.current.prompts[inputId]
        if (current) {
          dispatch({ kind: 'prompt-merge', prompt: { ...current, status: res.status ?? 'cancelled' } })
        }
        if (res.action) dispatch({ kind: 'action-merge', action: res.action })
      } catch (err) {
        // A prompt that already settled is fine to walk away from.
        if (err instanceof ApiError && !err.promptStale) {
          dispatch({ kind: 'prompt-error', id: inputId, message: err.message })
        }
      } finally {
        dispatch({ kind: 'prompt-busy', id: inputId, busy: false })
        if (optionsRef.current.reconcileFromServer !== false) void refetch()
      }
    },
    [refetch],
  )

  const reset = useCallback(
    (opts?: { keepConversation?: boolean }) => {
      stop()
      // Forget the mount-time thread too, or the next question falls back to
      // it and "New chat" turns into "carry on where you were".
      if (!opts?.keepConversation) seedConversationRef.current = null
      dispatch({ kind: 'reset', keepConversation: opts?.keepConversation })
    },
    [stop],
  )

  const steps = useMemo(
    () => state.stepOrder.map((key) => state.steps[key]).filter(Boolean),
    [state.stepOrder, state.steps],
  )

  const openPrompts = useMemo(
    () =>
      state.promptOrder
        .map((id) => state.prompts[id])
        .filter((p): p is PendingInput => Boolean(p) && p.status === 'pending'),
    [state.promptOrder, state.prompts],
  )

  const actions = useMemo(
    () => state.actionOrder.map((id) => state.actions[id]).filter(Boolean),
    [state.actionOrder, state.actions],
  )

  return {
    state,
    steps,
    openPrompts,
    blockingPrompt: openPrompts.find((p) => p.blocking) ?? null,
    actions,
    busy: state.streaming || state.status === 'running',
    ask,
    respond,
    dismissPrompt,
    reattach,
    refetch,
    reset,
    stop,
  }
}
