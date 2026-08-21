/**
 * The API, in TypeScript. Mirrors docs/API.md and docs/contracts.md.
 *
 * Three groups live here:
 *   1. wire types  — exactly what the server sends
 *   2. SSE union   — discriminated on `type`, so a reducer switch is exhaustive
 *   3. view types  — what the reducer in useRun.ts produces, plus the prop
 *                    contracts the components render against
 */

/* ------------------------------------------------------------------ scalars */

/** RFC 3339 with an explicit offset. The server always emits UTC. */
export type Iso = string

/** nanoid, 21 chars. Opaque — never parse one. */
export type Id = string

export type ServiceName = 'gmail' | 'gcal' | 'gdrive'

export type PromptKind = 'confirm' | 'choice' | 'multi_choice' | 'text' | 'form' | 'date_range'

export type PromptStatus = 'pending' | 'answered' | 'cancelled' | 'expired' | 'superseded'

export type ActionStatus =
  | 'draft'
  | 'approved'
  | 'queued'
  | 'running'
  | 'done'
  | 'failed'
  | 'cancelled'

export type RunStatus =
  | 'running'
  | 'awaiting_input'
  | 'complete'
  | 'failed'
  | 'timeout'
  | 'cancelled'

export type StepStatus =
  | 'pending'
  | 'running'
  | 'succeeded'
  | 'failed'
  | 'skipped'
  | 'timeout'
  | 'cancelled'

export type Phase = 'front_door' | 'prepass' | 'probe' | 'plan' | 'dispatch' | 'render'

export type Freshness = 'auto' | 'cached' | 'live'

export type AnswerStyle = 'card' | 'prose' | (string & {})

/** app.google.retry.ErrorClass */
export type ErrorClass =
  | 'TRANSIENT'
  | 'RATE_LIMITED'
  | 'QUOTA_EXHAUSTED'
  | 'AUTH_EXPIRED'
  | 'AUTH_REVOKED'
  | 'PRECONDITION'
  | 'NOT_FOUND'
  | 'INVALID'
  | 'UNKNOWN'

/** The closed error-code table, plus the one stream-only code. */
export type ErrorCode =
  | 'VALIDATION_ERROR'
  | 'NOT_AUTHENTICATED'
  | 'GOOGLE_REAUTH_REQUIRED'
  | 'RATE_LIMITED'
  | 'PROMPT_NOT_PENDING'
  | 'PROMPT_VALUE_INVALID'
  | 'ORCHESTRATION_TIMEOUT'
  | 'GOOGLE_UNAVAILABLE'
  | 'NOT_FOUND'
  | 'INTERNAL'
  | 'STREAM_EXPIRED'

export interface ApiErrorBody {
  error: {
    code: ErrorCode
    message: string
    details: Record<string, unknown>
    request_id: string
  }
}

/**
 * A JSON Schema node, typed enough for the six card renderers to read without
 * casting. Unknown keywords (`x-widget`, …) survive through the index signature.
 */
export interface SchemaNode {
  $schema?: string
  type?: string | string[]
  title?: string
  description?: string
  enum?: unknown[]
  const?: unknown
  format?: string
  default?: unknown
  examples?: unknown[]
  minLength?: number
  maxLength?: number
  pattern?: string
  minimum?: number
  maximum?: number
  minItems?: number
  maxItems?: number
  uniqueItems?: boolean
  required?: string[]
  properties?: Record<string, SchemaNode>
  items?: SchemaNode
  additionalProperties?: boolean | SchemaNode
  [keyword: string]: unknown
}

/* --------------------------------------------------------------- primitives */

/** app.orchestrator.temporal.Window — half-open [start, end). */
export interface ResolvedWindow {
  start: Iso
  end: Iso
  tz: string
  interpretation: string
}

export interface Intent {
  name: string
  services: ServiceName[]
  has_write: boolean
  confidence: number
  entities?: Record<string, unknown>
  windows?: Record<string, ResolvedWindow>
  answer_style?: AnswerStyle
  planner_tier?: number
}

export interface Usage {
  llm_calls: number
  model?: string
  prompt_tokens?: number
  completion_tokens?: number
  usd?: number
}

export interface Timings {
  front_door_ms?: number
  prepass_ms?: number
  probe_ms?: number
  plan_ms?: number
  dispatch_ms?: number
  render_ms?: number
  total_ms?: number
  [phase: string]: number | undefined
}

/**
 * Something that fell over while the run still answered.
 *
 * `service` is null when the fault is ours rather than Google's — a plan that
 * asked for the impossible, or a step whose dependency never produced anything.
 * Naming a service in that case blames the wrong party, so the UI must read
 * `fault` and say who it was.
 */
export interface Degraded {
  service: ServiceName | string | null
  /** Whose fault: Google's, our plan's, or a step this one was waiting on. */
  fault?: 'service' | 'plan' | 'dependency'
  /** The plan node it happened at, when it happened at one. */
  node?: string | null
  op?: string | null
  reason: string
  class?: ErrorClass
  code?: number
  detail?: string
  until?: Iso | null
}

export interface PromptOption {
  id: string
  label: string
  /** Free-form display extras: emails in 90 days, when you last met, … */
  meta?: Record<string, unknown> | null
}

/**
 * One field of a `form` prompt.
 *
 * The choices for a field live here, on the field — *not* in the prompt's
 * top-level `options`, which is where the single-answer kinds keep theirs.
 */
export interface PromptField {
  name: string
  label?: string | null
  help_text?: string | null
  required?: boolean
  options?: PromptOption[] | null
}

export interface PromptQuestion {
  /**
   * How this card should be presented, when the kind alone is not enough.
   * `reconnect` is a confirm whose run is parked on an expired connection:
   * the buttons are wrong for it, so it renders as its own thing.
   */
  variant?: 'reconnect'
  question: string
  help_text?: string | null
  /** `form` prompts only: the fields to draw, each with its own options. */
  fields?: PromptField[] | null
}

export interface ActionPreview {
  to?: string[]
  cc?: string[]
  bcc?: string[]
  subject?: string
  body_excerpt?: string
  body?: string
  attachments?: unknown[]
  [field: string]: unknown
}

export interface ActionErrorInfo {
  class?: ErrorClass
  code?: number
  message: string
}

/** A prepared write. `draft` means prepared and *not* done. */
export interface ActionRecord {
  id: Id
  op: string
  status: ActionStatus
  requires_input_id?: Id | null
  external_ref?: string | null
  preview: ActionPreview
  /** Which payload keys the confirm card is allowed to edit. */
  payload_fields?: string[]
  result?: Record<string, unknown> | null
  error?: ActionErrorInfo | null
  attempts?: number
  revision?: number
  job_id?: string | null
  queued_at?: Iso | null
  executed_at?: Iso | null
  expires_at?: Iso | null
  created_at?: Iso
  /** Present on the respond() reply: the stream to watch for the outcome. */
  watch?: string | null
}

/** The short name the components render against. Same row, same shape. */
export type Action = ActionRecord

/** A question the system is waiting on. `pending_inputs` in the API. */
export interface PendingInput {
  id: Id
  run_id?: Id
  conversation_id?: Id
  message_id?: Id | null
  node_execution_id?: Id | null
  kind: PromptKind
  /** true → the run is paused on this. false → the run finished; a write waits. */
  blocking: boolean
  prompt: PromptQuestion
  value_schema: SchemaNode
  options?: PromptOption[] | null
  status: PromptStatus
  response?: unknown
  expires_at?: Iso | null
  answered_at?: Iso | null
  created_at?: Iso
  /** Set when an action names this prompt as its `requires_input_id`. */
  action?: ActionRecord | null
  node_id?: string | null
}

/**
 * What a widget button may do. Two verbs, on purpose.
 *
 * `ask` types a sentence for you, so it goes down the ordinary query path and
 * a write still stops at its confirmation card. `open` follows a link that
 * came out of your own data. Neither one executes anything: a button that
 * could would be a second write path around the confirmation gate.
 */
export type WidgetAction =
  | { kind: 'ask'; label: string; query: string }
  | { kind: 'open'; label: string; url: string }

/**
 * A structured answer the UI draws itself. Never markup — the backend sends
 * data and this app owns every pixel.
 *
 * `text` is the markdown this replaced, and it is not optional. It is what a
 * screen reader reads, what a copy-paste copies, and what renders when a
 * client meets a `widget` or a `v` it does not know. Messages are durable, so
 * one written today has to still render in a year.
 */
export interface WidgetBlock {
  type: 'widget'
  v: number
  widget: string
  text: string
  /**
   * True when the widget *is* the answer and the text is only its fallback —
   * a template's rows, drawn instead of printed. False when it decorates prose
   * that says something the widget does not, in which case both are shown.
   */
  replaces_text?: boolean
  data: Record<string, unknown>
}

export type ContentBlock =
  | { type: 'text'; data: { markdown: string } }
  | { type: 'action'; ref: Id }
  | { type: 'input'; ref: Id }
  | WidgetBlock

export interface Message {
  id: Id
  seq: number
  role: 'user' | 'assistant'
  content: ContentBlock[]
  run_id?: Id | null
  created_at: Iso
  trace?: StepRecord[]
}

/** `node_executions`, as returned by /query and by conversations?include_trace. */
export interface StepRecord {
  node_id: string
  op: string
  seq: number
  round: number
  status: StepStatus
  depends_on: string[]
  args?: Record<string, unknown>
  result_summary?: Record<string, unknown> | null
  started_at?: Iso | null
  finished_at?: Iso | null
  duration_ms?: number | null
  attempts?: number
}

export interface EntityChip {
  id?: Id
  entity_type: string
  entity_ref: string
  label: string
  meta?: Record<string, unknown>
  last_seen_at?: Iso
}

/* ------------------------------------------------------------- request/reply */

export interface QueryRequest {
  query: string
  conversation_id?: Id | null
  timezone?: string
  week_start?: number
  freshness?: Freshness
  client_request_id?: string
}

export interface QueryResponse {
  run_id: Id
  conversation_id: Id
  message_id: Id
  status: RunStatus
  planner_tier?: number
  intent?: Intent
  answer_style?: AnswerStyle
  text?: string
  content?: ContentBlock[]
  actions?: ActionRecord[]
  pending_inputs?: PendingInput[]
  steps?: StepRecord[]
  entities?: EntityChip[]
  degraded?: Degraded[]
  usage?: Usage
  timings?: Timings
}

export interface PromptListResponse {
  items: PendingInput[]
  count: number
}

export interface RespondResponse {
  input_id: Id
  status: PromptStatus
  answered_at?: Iso
  value: unknown
  /** Blocking prompts only: the run that just resumed. */
  resumed_run_id?: Id | null
  action?: ActionRecord | null
  /** The Edit flow raises a fresh confirm prompt against the revised payload. */
  next_input?: PendingInput | null
  llm_calls?: number
  watch?: string | null
}

export interface CancelResponse {
  input_id: Id
  status: PromptStatus
  cancelled_at?: Iso
  action?: ActionRecord | null
  run_status?: RunStatus
}

export interface ConversationSummary {
  id: Id
  title: string
  title_is_derived: boolean
  last_message_at: Iso
  created_at: Iso
  archived_at?: Iso | null
  message_count: number
  pending_input_count: number
  last_run?: { id: Id; status: RunStatus; intent?: string } | null
}

export interface ConversationListResponse {
  items: ConversationSummary[]
  next_cursor?: string | null
  has_more: boolean
}

export interface RunRecord {
  id: Id
  trigger_message_id?: Id
  status: RunStatus
  planner_tier?: number
  intent?: Intent | null
  token_usage?: { prompt?: number; completion?: number; model?: string; usd?: number } | null
  error?: Record<string, unknown> | null
  started_at?: Iso
  finished_at?: Iso | null
  duration_ms?: number | null
}

/** The durable record, and the fallback whenever the stream lets us down. */
export interface ConversationDetail {
  id: Id
  title: string
  title_is_derived: boolean
  created_at: Iso
  last_message_at: Iso
  archived_at?: Iso | null
  messages: Message[]
  runs: RunRecord[]
  pending_inputs: PendingInput[]
  actions: ActionRecord[]
  entities: EntityChip[]
  has_more: boolean
}

export interface MeResponse {
  user: {
    id: Id
    email: string
    display_name?: string | null
    has_password?: boolean
    timezone: string
    work_week_start: number
    created_at?: Iso
  }
  google: {
    connected: boolean
    /** The workspace being read, which need not be the sign-in address. */
    account_email?: string | null
    provider_account_id?: string | null
    scopes?: string[]
    expires_at?: Iso | null
    needs_reauth: boolean
  }
  limits?: { queries_per_hour: number; remaining_this_hour: number }
}

export interface SyncServiceState {
  last_synced_at: Iso | null
  last_success_at: Iso | null
  lag_seconds: number | null
  items_indexed: number
  backfill_complete: boolean
  cursor_present: boolean
  consecutive_failures: number
  circuit_open_until: Iso | null
  last_error: { class?: ErrorClass; code?: number; message: string; at?: Iso } | null
  healthy: boolean
}

export interface SyncStatus {
  services: Record<string, SyncServiceState>
  freshness: { worst_lag_seconds: number; target_seconds: number; within_target: boolean }
  dlq?: { open: number; oldest_at?: Iso | null }
  next_scheduled_at?: Iso | null
}

export interface SyncTriggerRequest {
  services?: ServiceName[]
  mode?: 'incremental' | 'backfill' | 'full'
}

export interface SyncTriggerResponse {
  queued: { service: ServiceName; mode: string; task_id: string; queue: string }[]
  skipped: { service: ServiceName; reason: string; until?: Iso; task_id?: string }[]
  poll?: string
}

/* ------------------------------------------------------------------- search */

/**
 * `GET /api/v1/search` — the retrieval layer with the lid off. The chat path
 * never calls it; it is there to explain a bad answer and to run the offline
 * Precision@5 evaluation, which is why every score component is returned.
 */
export interface SearchScores {
  /** Raw pgvector cosine, `1 - (embedding <=> $query)`. */
  cosine: number
  /** Cosine normalised per corpus. The floors and the margin are computed on this. */
  cn: number
  ts_rank?: number
  vec_rank?: number
  fts_rank?: number
  /** Reciprocal rank fusion, k=60. Ordering only — never a relevance number. */
  rrf?: number
  decay?: number
  boost?: number
  /** `rrf × decay × boost`. The sort key. */
  final?: number
}

export interface SearchHit {
  id: Id
  /** Provider identifiers: message_id/thread_id, event_id/calendar_id, file_id. */
  ref: Record<string, unknown>
  title: string
  snippet?: string
  when?: Iso | null
  from?: string
  scores: SearchScores
  /** EXACT_ID · EXACT_SENDER · EXACT_FILENAME · ALIAS_TOKEN_IN_SUBJECT */
  evidence?: string[]
  /** Regex extractors over the excerpt: pnr, flight_no, date, links. */
  extracted?: Record<string, string>
}

export interface SearchServiceResult {
  took_ms: number
  prefiltered?: number
  vector_candidates?: number
  fts_candidates?: number
  returned: number
  hits: SearchHit[]
}

export interface SearchDecision {
  top_cn: number | null
  runner_up_cn: number | null
  margin: number | null
  floor_read: number
  floor_write: number
  margin_threshold: number
  above_read_floor: boolean
  above_write_floor: boolean
  ambiguous: boolean
  verdict: 'confident' | 'ambiguous' | 'absent' | (string & {})
}

export interface SearchResponse {
  q: string
  took_ms: number
  embedding_ms?: number
  embedding_cached?: boolean
  services: Partial<Record<ServiceName, SearchServiceResult>>
  decision: SearchDecision
  /** Only with `explain=true`: the SQL strategy and row counts per stage. */
  plan?: Record<string, unknown>
}

export interface SearchParams {
  q: string
  services?: ServiceName[]
  limit?: number
  freshness?: 'cached' | 'live'
  since?: Iso
  until?: Iso
  /** Gmail only. Exact prefilter on `from_email`. */
  from?: string
  /** Calendar only. GIN index hit on `attendee_emails`. */
  attendee?: string
  /** Drive only, e.g. `application/pdf`. */
  mime?: string
  explain?: boolean
}

/* ------------------------------------------------------------------ SSE union */

export interface Envelope<T extends string, D> {
  v: number
  seq: number
  type: T
  run_id: Id
  ts: Iso
  data: D
}

export interface RunStartedData {
  conversation_id: Id
  message_id: Id
  query: string
  timezone?: string
}

export interface ProgressData {
  phase: Phase
  label: string
  pct?: number
}

export interface ProbeCandidate {
  service: ServiceName
  label: string
  cn: number
  evidence?: string[]
  id?: string
}

export interface ProbeDoneData {
  took_ms: number
  embedding_ms?: number
  candidates: Partial<Record<ServiceName, number>>
  top?: ProbeCandidate[]
  extracted?: Record<string, string[]>
  ambiguous?: boolean
}

export type IntentData = Intent

export interface PlanStepData {
  node_id: string
  op: string
  depends_on: string[]
  expect?: 'one' | 'many'
  optional?: boolean
  freshness?: 'cached' | 'live'
  is_write?: boolean
  needs_confirm?: boolean
  gate?: Record<string, unknown> | null
  defer?: Record<string, unknown> | null
  speculate?: boolean
}

export interface StepStartedData {
  node_id: string
  op: string
  round?: number
  label?: string
}

export interface StepRetryingData {
  node_id: string
  op?: string
  attempt: number
  of: number
  error_class: ErrorClass
  google_status?: number
  backoff_ms?: number
  label?: string
}

export interface StepOutcome {
  reason: string
  class?: ErrorClass
  code?: number
  message?: string
  until?: Iso
}

export interface StepFinishedData {
  node_id: string
  op?: string
  status: Exclude<StepStatus, 'pending' | 'running'>
  round?: number
  duration_ms: number
  attempts?: number
  summary?: Record<string, unknown> | null
  outcome?: StepOutcome | null
}

export interface ContentDeltaData {
  message_id: Id
  block_index: number
  text: string
}

export interface TokenData {
  call: string
  text: string
}

export interface InputRaisedData {
  input_id: Id
  kind: PromptKind
  blocking: boolean
  node_id?: string
  message_id?: Id
  prompt: PromptQuestion
  value_schema: SchemaNode
  options?: PromptOption[] | null
  expires_at?: Iso | null
  /** Where the card sits in the message being written, when the server says. */
  block_index?: number
}

export interface ActionPreparedData {
  action_id: Id
  op: string
  status: ActionStatus
  requires_input_id?: Id | null
  external_ref?: string | null
  preview: ActionPreview
  payload_fields?: string[]
  expires_at?: Iso | null
  /** Set when a drafted body is about to stream into this card's block. */
  block_index?: number
}

export interface RunPausedData {
  reason: string
  input_id: Id
  resumable_until?: Iso
  completed_nodes?: number
  remaining_nodes?: number
}

export interface RunCompleteData {
  status: RunStatus
  message_id: Id
  answer_style?: AnswerStyle
  planner_tier?: number
  usage?: Usage
  timings?: Timings
  /**
   * The compact form on the wire: `run.complete` names the services that were
   * degraded, where the durable body carries the whole entry. Read it through
   * `degradedEntries`, which takes either.
   */
  degraded?: (Degraded | ServiceName | string)[]
}

export interface RunErrorData {
  code: ErrorCode
  message: string
  retryable?: boolean
  details?: Record<string, unknown>
  partial?: boolean
  message_id?: Id
}

export interface ActionDoneData {
  action_id: Id
  op: string
  status: ActionStatus
  result: Record<string, unknown>
  attempts?: number
  audit_id?: number
}

export interface ActionFailedData {
  action_id: Id
  op: string
  status: ActionStatus
  error: ActionErrorInfo
  attempts?: number
  retryable?: boolean
  audit_id?: number
}

export type RunEvent =
  | Envelope<'run.started', RunStartedData>
  | Envelope<'progress', ProgressData>
  | Envelope<'probe.done', ProbeDoneData>
  | Envelope<'intent', IntentData>
  | Envelope<'plan.step', PlanStepData>
  | Envelope<'step.started', StepStartedData>
  | Envelope<'step.retrying', StepRetryingData>
  | Envelope<'step.finished', StepFinishedData>
  | Envelope<'content.delta', ContentDeltaData>
  | Envelope<'token', TokenData>
  | Envelope<'input.raised', InputRaisedData>
  | Envelope<'action.prepared', ActionPreparedData>
  | Envelope<'run.paused', RunPausedData>
  | Envelope<'run.complete', RunCompleteData>
  | Envelope<'error', RunErrorData>
  | Envelope<'action.done', ActionDoneData>
  | Envelope<'action.failed', ActionFailedData>

export type RunEventType = RunEvent['type']

/** One event without the envelope — a `{type, data}` pair, still discriminated. */
export type RunEventBody = {
  [K in RunEventType]: { type: K; data: Extract<RunEvent, { type: K }>['data'] }
}[RunEventType]

/** Every type on a run's channel, in the order the server can send them. */
export const RUN_EVENT_TYPES: RunEventType[] = [
  'run.started',
  'progress',
  'probe.done',
  'intent',
  'plan.step',
  'step.started',
  'step.retrying',
  'step.finished',
  'content.delta',
  'token',
  'input.raised',
  'action.prepared',
  'run.paused',
  'run.complete',
  'error',
  'action.done',
  'action.failed',
]

export const TERMINAL_EVENT_TYPES: RunEventType[] = ['run.complete', 'error']

/* ------------------------------------------------------------- view + props */

export type ConnectionStatus =
  | 'idle'
  | 'connecting'
  | 'open'
  | 'reconnecting'
  | 'closed'
  | 'error'

export interface RetryNote {
  attempt: number
  of: number
  error_class: ErrorClass
  google_status?: number
  backoff_ms?: number
  label?: string
  at: Iso
}

/** One row of the trace panel. Built from plan.step → step.* events. */
export interface TraceStep {
  node_id: string
  op: string
  depends_on: string[]
  status: StepStatus
  /** Op.progress_label(args) — written by the op, safe to show. */
  label?: string
  duration_ms?: number | null
  attempts: number
  retries: RetryNote[]
  summary?: Record<string, unknown> | null
  outcome?: StepOutcome | null
  /** Why a non-succeeded step ended that way, in one phrase. */
  skipReason?: string | null
  optional?: boolean
  expect?: 'one' | 'many'
  freshness?: 'cached' | 'live'
  is_write?: boolean
  needs_confirm?: boolean
  round?: number
  started_at?: Iso
  finished_at?: Iso
  /** Arrival order, not plan order. The panel renders in this order. */
  order: number
}

/**
 * The message being written right now, block by block. A `text` block grows by
 * content.delta; an `action`/`input` block is a card. A delta addressed at a
 * card's block index is the drafted body streaming into that card.
 */
export type LiveBlock =
  | { kind: 'text'; text: string; index?: number }
  | { kind: 'widget'; block: WidgetBlock; index?: number }
  | { kind: 'action'; ref: Id; draft: string; index?: number }
  | { kind: 'input'; ref: Id; draft: string; index?: number }

export interface RunState {
  runId: Id | null
  conversationId: Id | null
  messageId: Id | null
  query: string | null
  status: RunStatus | 'idle'
  /** True between run.started and run.complete/error/paused. */
  streaming: boolean
  connection: ConnectionStatus
  lastSeq: number

  progress: ProgressData | null
  probe: ProbeDoneData | null
  intent: Intent | null
  windows: Record<string, ResolvedWindow>
  /** The window the answer hangs on — the first one the planner resolved. */
  resolvedWindow: ResolvedWindow | null

  steps: Record<string, TraceStep>
  stepOrder: string[]

  blocks: LiveBlock[]
  /** Text blocks joined — the prose answer as it streams. */
  answerText: string
  /** Raw model tokens, for the "watch it think" view. */
  tokens: string

  prompts: Record<Id, PendingInput>
  promptOrder: Id[]
  actions: Record<Id, ActionRecord>
  actionOrder: Id[]
  /** Per-prompt UI state: in-flight answer, and the last server complaint. */
  promptBusy: Record<Id, boolean>
  promptErrors: Record<Id, string>

  degraded: Degraded[]
  usage: Usage | null
  timings: Timings | null
  answerStyle: AnswerStyle | null
  plannerTier: number | null
  entities: EntityChip[]

  paused: RunPausedData | null
  error: RunErrorData | null
  startedAt: Iso | null
  finishedAt: Iso | null
}

/** A message in the chat column: from the thread, or being written live. */
export interface ChatMessage {
  id: Id
  seq: number
  role: 'user' | 'assistant'
  content: ContentBlock[]
  run_id?: Id | null
  created_at: Iso
  /** Assistant message still being written. */
  streaming?: boolean
  /** User message not yet echoed back by the server. */
  optimistic?: boolean
  /** For a streaming assistant message: text drafted into a card, by action id. */
  drafts?: Record<Id, string>
}

/* ---- component prop contracts. Components/*.tsx implement exactly these ---- */

export interface SyncBarProps {
  status: SyncStatus | null
  loading: boolean
  error: string | null
  onSyncNow: () => void
}

export interface ChatProps {
  messages: ChatMessage[]
  run: RunState
  prompts: Record<Id, PendingInput>
  actions: Record<Id, ActionRecord>
  /** Composer is controlled by the shell, so a card can prefill it. */
  draft: string
  onDraftChange: (value: string) => void
  /** Enabled even while a card is open — typing is a valid answer. */
  composerEnabled: boolean
  busy: boolean
  onSend: (text: string) => void
  onRespond: (inputId: Id, value: unknown) => void
  onCancelPrompt: (inputId: Id) => void
  /** "None of these — search differently": cancels and hands the box back. */
  onSearchDifferently: (prompt: PendingInput) => void
  /** Expired card → run the question again. */
  onRedo: (prompt: PendingInput) => void
  notice: string | null
  emptyHint?: string[]
  /** Live steps, shown inline under the answer while a run is going. */
  steps?: TraceStep[]
  /** Hourly allowance, shown beside Send only when it is nearly gone. */
  remaining?: number
  limit?: number
}

export interface MessageListProps {
  messages: ChatMessage[]
  run: RunState
  /** Steps of the run in flight; shown above the turn it belongs to. */
  steps?: TraceStep[]
  prompts: Record<Id, PendingInput>
  actions: Record<Id, ActionRecord>
  onRespond: (inputId: Id, value: unknown) => void
  onCancelPrompt: (inputId: Id) => void
  onSearchDifferently: (prompt: PendingInput) => void
  onRedo: (prompt: PendingInput) => void  /** A widget button asking a follow-up on the person's behalf. */
  onAsk: (question: string) => void
}

export interface ComposerProps {
  value: string
  onChange: (value: string) => void
  onSend: (text: string) => void
  enabled: boolean
  busy: boolean
  /** Set when a card is open, e.g. "Answering: Which John?" */
  answeringLabel?: string | null
  placeholder?: string
  /** Shown beside Send, and only when it starts to matter. */
  remaining?: number
  limit?: number
}

export interface PromptCardProps {
  prompt: PendingInput
  action?: ActionRecord | null
  /** Draft text streaming into this card, if the run is writing one. */
  streamingDraft?: string
  busy?: boolean
  error?: string | null
  onRespond: (value: unknown) => void
  onCancel: () => void
  onSearchDifferently: () => void
  onRedo: () => void
}

export interface ActionCardProps {
  action: ActionRecord
  prompt?: PendingInput | null
  streamingDraft?: string
  busy?: boolean
  error?: string | null
  onRespond: (value: unknown) => void
  onCancel: () => void
}

