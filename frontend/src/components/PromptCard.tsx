import { useState } from 'react'

import { googleAuthUrl } from '../lib/api'

import {
  formatBytes,
  formatCount,
  formatDateTime,
  relativeTime,
  renderMarkdown,
  truncate,
} from '../lib/format'
import type {
  PendingInput,
  PromptCardProps,
  PromptOption,
  SchemaNode,
} from '../lib/types'

/**
 * Every question the system asks, rendered from `prompt.kind`.
 *
 * The six kinds are six branches here and nothing else anywhere: the server
 * validates against `value_schema`, so a new kind is a new branch in this file
 * and no API change. The checks below are a courtesy that catches the obvious
 * mistakes before a round trip — the schema on the row is still the authority.
 */

/* --------------------------------------------------------------- schema bits */

const EMAIL = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/
const ISO_LIKE = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}/

function jsonType(value: unknown): string {
  if (value === null) return 'null'
  if (Array.isArray(value)) return 'array'
  if (typeof value === 'number') return Number.isInteger(value) ? 'integer' : 'number'
  return typeof value
}

function typeMatches(value: unknown, expected: string): boolean {
  const actual = jsonType(value)
  if (expected === 'number') return actual === 'number' || actual === 'integer'
  return actual === expected
}

function sameValue(a: unknown, b: unknown): boolean {
  return JSON.stringify(a) === JSON.stringify(b)
}

/** A readable label for a property name or an option key. */
function humanise(name: string): string {
  return name.replace(/_/g, ' ').replace(/\b\w/, (letter) => letter.toUpperCase())
}

/**
 * The subset of draft 2020-12 worth checking in the browser. Anything it misses
 * the server catches and returns as `PROMPT_VALUE_INVALID`.
 */
function validate(value: unknown, schema: SchemaNode, path = 'This'): string[] {
  const problems: string[] = []
  if (!schema || typeof schema !== 'object') return problems

  const types = schema.type ? (Array.isArray(schema.type) ? schema.type : [schema.type]) : null
  if (types && !types.some((type) => typeMatches(value, type))) {
    problems.push(`${path} should be ${types.join(' or ')}.`)
    return problems
  }

  if (Array.isArray(schema.enum) && !schema.enum.some((option) => sameValue(option, value))) {
    problems.push(`${path} is not one of the choices offered.`)
  }

  if (typeof value === 'string') {
    if (schema.minLength != null && value.length < schema.minLength) {
      problems.push(`${path} needs at least ${schema.minLength} characters.`)
    }
    if (schema.maxLength != null && value.length > schema.maxLength) {
      problems.push(`${path} is longer than ${schema.maxLength} characters.`)
    }
    if (schema.format === 'email' && !EMAIL.test(value)) {
      problems.push(`${path} is not an email address.`)
    }
    if (schema.format === 'date-time' && Number.isNaN(Date.parse(value))) {
      problems.push(`${path} is not a date and time.`)
    }
  }

  if (typeof value === 'number') {
    if (schema.minimum != null && value < schema.minimum) {
      problems.push(`${path} must be at least ${schema.minimum}.`)
    }
    if (schema.maximum != null && value > schema.maximum) {
      problems.push(`${path} must be at most ${schema.maximum}.`)
    }
  }

  if (Array.isArray(value)) {
    if (schema.minItems != null && value.length < schema.minItems) {
      problems.push(
        schema.minItems === 1
          ? 'Pick at least one.'
          : `Pick at least ${schema.minItems}.`,
      )
    }
    if (schema.maxItems != null && value.length > schema.maxItems) {
      problems.push(`Pick no more than ${schema.maxItems}.`)
    }
    if (schema.uniqueItems && new Set(value.map((item) => JSON.stringify(item))).size !== value.length) {
      problems.push('Each choice can only appear once.')
    }
    if (schema.items) {
      for (const item of value) problems.push(...validate(item, schema.items, 'Each entry'))
    }
  }

  if (value && typeof value === 'object' && !Array.isArray(value)) {
    const record = value as Record<string, unknown>
    for (const key of schema.required ?? []) {
      if (record[key] === undefined || record[key] === '' || record[key] === null) {
        problems.push(`${humanise(key)} is needed.`)
      }
    }
    for (const [key, child] of Object.entries(schema.properties ?? {})) {
      if (record[key] !== undefined && record[key] !== '') {
        problems.push(...validate(record[key], child, humanise(key)))
      }
    }
  }

  return Array.from(new Set(problems))
}

/* ------------------------------------------------------------------ datetime */

function pad(value: number): string {
  return String(value).padStart(2, '0')
}

/** RFC 3339 → the local value a `datetime-local` input wants. */
function isoToLocalInput(value: unknown): string {
  if (typeof value !== 'string' || !value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}T${pad(
    date.getHours(),
  )}:${pad(date.getMinutes())}`
}

/** …and back, as UTC with an explicit offset, which is what the server takes. */
function localInputToIso(value: string): string {
  if (!value) return ''
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? '' : date.toISOString()
}

/* --------------------------------------------------------------- option meta */

const META_LABELS: Record<string, string> = {
  last_meeting: 'last met',
  meeting_count_90d: 'meetings, 90d',
  emails_90d: 'emails, 90d',
  next_event: 'next',
  from: 'from',
  date: 'date',
  size: 'size',
}

function metaValue(key: string, value: unknown): string {
  if (value == null) return '—'
  if (typeof value === 'number') {
    return /size|bytes/i.test(key) ? formatBytes(value) : formatCount(value)
  }
  if (typeof value === 'boolean') return value ? 'yes' : 'no'
  if (typeof value === 'object') return JSON.stringify(value)
  const text = String(value)
  if (ISO_LIKE.test(text) && !Number.isNaN(Date.parse(text))) return formatDateTime(text)
  return text
}

function OptionMeta({ meta }: { meta?: Record<string, unknown> | null }) {
  const entries = Object.entries(meta ?? {}).filter(([, value]) => value != null && value !== '')
  if (!entries.length) return null
  return (
    <span className="mt-0.5 flex flex-wrap gap-x-2 gap-y-0.5 text-2xs text-faint">
      {entries.map(([key, value]) => (
        <span key={key}>
          <span className="text-faint">{META_LABELS[key] ?? humanise(key).toLowerCase()}</span>{' '}
          <span className="text-muted">{metaValue(key, value)}</span>
        </span>
      ))}
    </span>
  )
}

/** `options` is display only; a `choice` with no options still has its enum. */
function optionsOf(prompt: PendingInput): PromptOption[] {
  if (prompt.options?.length) return prompt.options
  const schema = prompt.value_schema ?? {}
  const enums = Array.isArray(schema.enum)
    ? schema.enum
    : Array.isArray(schema.items?.enum)
      ? schema.items.enum
      : []
  return enums.map((value) => ({ id: String(value), label: String(value) }))
}

/* ---------------------------------------------------------------- one field */

interface FieldProps {
  name: string
  schema: SchemaNode
  value: unknown
  required: boolean
  disabled: boolean
  onChange: (next: unknown) => void
  /** Two open cards must not fight over the same DOM id. */
  idPrefix: string
  /**
   * The field's own choices, from `prompt.fields[i].options`. A form keeps them
   * per field rather than in the prompt's top-level `options`, so a field with
   * choices would otherwise draw as a free-text box and ask a person to type an
   * id by hand.
   */
  options?: PromptOption[] | null
}

const INPUT =
  'w-full rounded-lg border border-line bg-canvas px-2.5 py-1.5 text-xs text-ink placeholder:text-faint focus:border-accent focus:outline-none focus:ring-1 focus:ring-accent/40 disabled:opacity-60'

function Field({
  name,
  schema,
  value,
  required,
  disabled,
  onChange,
  idPrefix,
  options,
}: FieldProps) {
  const types = schema.type ? (Array.isArray(schema.type) ? schema.type : [schema.type]) : []
  const label = schema.title ?? humanise(name)
  const id = `${idPrefix}-${name}`

  let control: JSX.Element

  if (options?.length) {
    control = (
      <select
        id={id}
        className={INPUT}
        disabled={disabled}
        value={value == null ? '' : String(value)}
        onChange={(event) => onChange(event.target.value)}
      >
        <option value="">Choose…</option>
        {options.map((option) => (
          <option key={option.id} value={option.id}>
            {option.label}
          </option>
        ))}
      </select>
    )
  } else if (Array.isArray(schema.enum) && schema.enum.length) {
    control = (
      <select
        id={id}
        className={INPUT}
        disabled={disabled}
        value={value == null ? '' : String(value)}
        onChange={(event) => onChange(event.target.value)}
      >
        <option value="">Choose…</option>
        {schema.enum.map((option) => (
          <option key={String(option)} value={String(option)}>
            {String(option)}
          </option>
        ))}
      </select>
    )
  } else if (types.includes('boolean')) {
    control = (
      <label className="flex items-center gap-2 text-xs text-ink">
        <input
          id={id}
          type="checkbox"
          className="h-3.5 w-3.5 rounded border-line accent-accent"
          disabled={disabled}
          checked={value === true}
          onChange={(event) => onChange(event.target.checked)}
        />
        {schema.description ?? 'yes'}
      </label>
    )
  } else if (types.includes('integer') || types.includes('number')) {
    control = (
      <input
        id={id}
        type="number"
        className={INPUT}
        disabled={disabled}
        min={schema.minimum}
        max={schema.maximum}
        step={types.includes('integer') ? 1 : 'any'}
        value={value == null ? '' : String(value)}
        onChange={(event) => {
          const raw = event.target.value
          onChange(raw === '' ? undefined : Number(raw))
        }}
      />
    )
  } else if (types.includes('array')) {
    const list = Array.isArray(value) ? value : []
    control = (
      <textarea
        id={id}
        rows={2}
        className={INPUT}
        disabled={disabled}
        placeholder="one per line"
        value={list.map(String).join('\n')}
        onChange={(event) =>
          onChange(
            event.target.value
              .split('\n')
              .map((line) => line.trim())
              .filter(Boolean),
          )
        }
      />
    )
  } else if (schema.format === 'date-time') {
    control = (
      <input
        id={id}
        type="datetime-local"
        className={INPUT}
        disabled={disabled}
        value={isoToLocalInput(value)}
        onChange={(event) => onChange(localInputToIso(event.target.value))}
      />
    )
  } else if (
    schema['x-widget'] === 'textarea' ||
    (schema.maxLength != null && schema.maxLength > 300) ||
    name === 'body'
  ) {
    control = (
      <textarea
        id={id}
        rows={6}
        className={INPUT}
        disabled={disabled}
        maxLength={schema.maxLength}
        value={typeof value === 'string' ? value : ''}
        onChange={(event) => onChange(event.target.value)}
      />
    )
  } else {
    control = (
      <input
        id={id}
        type={schema.format === 'email' ? 'email' : 'text'}
        className={INPUT}
        disabled={disabled}
        maxLength={schema.maxLength}
        value={typeof value === 'string' ? value : value == null ? '' : String(value)}
        onChange={(event) => onChange(event.target.value)}
      />
    )
  }

  return (
    <div>
      <label htmlFor={id} className="mb-1 block text-2xs font-medium text-muted">
        {label}
        {required && <span className="ml-0.5 text-bad">*</span>}
      </label>
      {control}
      {schema.description && !types.includes('boolean') && (
        <p className="mt-0.5 text-2xs text-faint">{schema.description}</p>
      )}
    </div>
  )
}

/* ------------------------------------------------------------ answered value */

function describeValue(prompt: PendingInput, value: unknown): string {
  const options = optionsOf(prompt)
  const labelFor = (id: unknown) =>
    options.find((option) => option.id === String(id))?.label ?? String(id)

  switch (prompt.kind) {
    case 'confirm': {
      const record = (value ?? {}) as { approve?: boolean; patch?: unknown }
      if (record.approve === true) return 'Approved — sent'
      if (record.patch) return 'Edited, and asked again'
      return 'Not now'
    }
    case 'choice':
      return labelFor(value)
    case 'multi_choice':
      return Array.isArray(value) ? value.map(labelFor).join(', ') : String(value)
    case 'date_range': {
      const record = (value ?? {}) as { start?: string; end?: string }
      return `${formatDateTime(record.start)} – ${formatDateTime(record.end)}`
    }
    case 'text':
      return typeof value === 'string' ? truncate(value, 160) : String(value)
    default:
      if (value && typeof value === 'object' && !Array.isArray(value)) {
        return Object.entries(value as Record<string, unknown>)
          .map(([key, item]) => `${humanise(key)}: ${metaValue(key, item)}`)
          .join(' · ')
      }
      return String(value)
  }
}

/* ------------------------------------------------------------------- buttons */

const PRIMARY =
  'rounded-lg bg-accent px-3 py-1.5 text-xs font-semibold text-accent-ink transition-opacity hover:opacity-90 disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50'
const SECONDARY =
  'rounded-lg border border-line px-3 py-1.5 text-xs font-medium text-muted transition-colors hover:bg-elevated hover:text-ink disabled:opacity-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50'
const QUIET =
  'text-2xs text-muted underline underline-offset-2 transition-colors hover:text-ink disabled:opacity-50'

/* --------------------------------------------------------------------- card */

export default function PromptCard({
  prompt,
  action,
  streamingDraft,
  busy = false,
  error = null,
  onRespond,
  onCancel,
  onSearchDifferently,
  onRedo,
}: PromptCardProps) {
  const schema = prompt.value_schema ?? {}
  const options = optionsOf(prompt)

  const [choice, setChoice] = useState<string | null>(null)
  const [picked, setPicked] = useState<string[]>([])
  const [text, setText] = useState('')
  const [range, setRange] = useState<{ start: string; end: string }>({ start: '', end: '' })
  const [form, setForm] = useState<Record<string, unknown>>(() => {
    const initial: Record<string, unknown> = {}
    for (const [key, child] of Object.entries(schema.properties ?? {})) {
      if (child.default !== undefined) initial[key] = child.default
    }
    return initial
  })
  const [editing, setEditing] = useState(false)
  const [patch, setPatch] = useState<Record<string, unknown>>({})
  const [problems, setProblems] = useState<string[]>([])

  const inactive = prompt.status !== 'pending'
  const disabled = busy || inactive

  /**
   * A pick, in the shape the schema asks for.
   *
   * The card shows a list and the person taps one, so the natural value is the
   * id. But the schema is usually an object naming the field it wants —
   * `{"file_id": "..."}`, `{"event_id": "..."}` — because the runner binds the
   * answer straight into a step's arguments. Sending the bare id got back
   * "This should be object", which tells the person nothing they can act on.
   *
   * The property name is read from the schema rather than hard-coded, so the
   * same card works for a file, an event, a message or whatever a future
   * connector adds.
   */
  const shapePick = (value: string | string[]): unknown => {
    if (schema.type !== 'object') return value
    const keys = Object.keys(schema.properties ?? {})
    const key = (schema.required ?? []).find((k: string) => keys.includes(k)) ?? keys[0]
    return key ? { [key]: value } : value
  }

  const submit = (value: unknown) => {
    const found = validate(value, schema)
    if (found.length) {
      setProblems(found)
      return
    }
    setProblems([])
    onRespond(value)
  }

  /* ------------------------------------------------------------ edit fields */

  const patchSchema = (schema.properties?.patch ?? {}) as SchemaNode
  const patchProps = patchSchema.properties ?? {}
  const editable = (
    action?.payload_fields?.length
      ? action.payload_fields.filter(
          (field) => Object.keys(patchProps).length === 0 || field in patchProps,
        )
      : Object.keys(patchProps)
  ).filter((field) => field !== 'reply_to_message_id')

  const startEditing = () => {
    const preview = action?.preview ?? {}
    const initial: Record<string, unknown> = {}
    for (const field of editable) {
      const current = field === 'body' ? (preview.body ?? streamingDraft ?? preview.body_excerpt) : preview[field]
      if (current !== undefined) initial[field] = current
    }
    setPatch(initial)
    setEditing(true)
  }

  /* ------------------------------------------------------------- reconnect */

  // A parked run, not a decision. The run is held so nothing is lost, but the
  // only thing that helps is reconnecting — "Send it / Not now / Edit" under
  // "your connection expired" asks a question that has no answer.
  if (prompt.prompt.variant === 'reconnect') {
    return (
      <section className="overflow-hidden rounded-xl border border-warn/40 bg-warn/5">
        <div className="px-3.5 py-3">
          <p
            className="text-sm text-ink [&_p]:mb-1.5 [&_p:last-child]:mb-0 [&_strong]:font-semibold"
            dangerouslySetInnerHTML={{ __html: renderMarkdown(prompt.prompt.question) }}
          />
          <div className="mt-3 flex flex-wrap gap-2">
            <a
              href={googleAuthUrl('/')}
              className="rounded-lg bg-accent px-3 py-1.5 text-xs font-semibold text-accent-ink transition hover:opacity-90"
            >
              Reconnect Google
            </a>
            <button
              type="button"
              onClick={onCancel}
              className="rounded-lg px-3 py-1.5 text-xs font-medium text-muted transition hover:text-ink"
            >
              Not now
            </button>
          </div>
        </div>
      </section>
    )
  }

  /* ----------------------------------------------------------------- header */

  const header = (
    <header className="border-b border-line bg-elevated/60 px-3 py-2">
      <p className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
        {/* The question is written by the model, so it arrives as markdown.
            Rendered raw it shows its own asterisks, which reads as a glitch. */}
        <span
          className="text-sm font-semibold text-ink [&_code]:rounded [&_code]:bg-elevated [&_code]:px-1 [&_code]:font-medium [&_p]:inline"
          dangerouslySetInnerHTML={{ __html: renderMarkdown(prompt.prompt.question) }}
        />
      </p>
      {prompt.prompt.help_text && (
        <p
          className="mt-1 text-2xs text-muted [&_p]:inline"
          dangerouslySetInnerHTML={{ __html: renderMarkdown(prompt.prompt.help_text) }}
        />
      )}
      <p className="mt-1 flex flex-wrap items-center gap-x-2 text-2xs text-faint">
        <span>
          {prompt.blocking
            ? 'Waiting on your answer.'
            : 'Nothing has happened yet.'}
        </span>
        {prompt.expires_at && prompt.status === 'pending' && (
          <span>expires {relativeTime(prompt.expires_at)}</span>
        )}
      </p>
    </header>
  )

  /* ------------------------------------------------------- inactive states */

  if (inactive) {
    const chosen = prompt.response
    const chips = prompt.kind === 'choice' || prompt.kind === 'multi_choice'
    const isPicked = (id: string) =>
      Array.isArray(chosen) ? chosen.map(String).includes(id) : String(chosen) === id

    return (
      <article className="animate-rise-in overflow-hidden rounded-xl border border-line bg-surface/70 shadow-card">
        {header}
        <div className="space-y-2 px-3 py-2.5">
          {chips && options.length > 0 ? (
            <ul className="space-y-1.5">
              {options.map((option) => (
                <li
                  key={option.id}
                  className={`rounded-lg border px-2.5 py-1.5 ${
                    isPicked(option.id)
                      ? 'border-accent bg-accent-soft'
                      : 'border-line opacity-60'
                  }`}
                >
                  <span className="flex flex-col text-xs text-ink">
                    <span className="font-medium">{option.label}</span>
                    <OptionMeta meta={option.meta} />
                  </span>
                </li>
              ))}
            </ul>
          ) : chosen !== undefined && chosen !== null ? (
            <p className="text-xs text-ink">
              <span className="text-faint">You said </span>
              {describeValue(prompt, chosen)}
            </p>
          ) : null}

          {prompt.status === 'answered' && (
            <p className="text-2xs text-muted">
              Answered {relativeTime(prompt.answered_at)}.
            </p>
          )}
          {prompt.status === 'expired' && (
            <div>
              <p className="text-2xs text-warn">
                This expired {relativeTime(prompt.expires_at)} and can no longer be answered.
              </p>
              <button type="button" onClick={onRedo} className={`mt-1.5 ${SECONDARY}`}>
                Ask again
              </button>
            </div>
          )}
          {prompt.status === 'superseded' && (
            <p className="text-2xs text-faint">Replaced by a newer request.</p>
          )}
          {prompt.status === 'cancelled' && (
            <div>
              <p className="text-2xs text-muted">Cancelled. Nothing was sent.</p>
              <button type="button" onClick={onRedo} className={`mt-1.5 ${SECONDARY}`}>
                Ask again
              </button>
            </div>
          )}
        </div>
      </article>
    )
  }

  /* ---------------------------------------------------------- live branches */

  const searchDifferently = (
    <button type="button" onClick={onSearchDifferently} disabled={busy} className={QUIET}>
      None of these — search differently
    </button>
  )

  const footerProblems = (
    <>
      {problems.length > 0 && (
        <ul className="space-y-0.5">
          {problems.map((problem) => (
            <li key={problem} className="text-2xs text-bad">
              {problem}
            </li>
          ))}
        </ul>
      )}
      {error && <p className="text-2xs text-bad">{error}</p>}
    </>
  )

  let body: JSX.Element

  switch (prompt.kind) {
    /* ------------------------------------------------------------- confirm */
    case 'confirm': {
      const preview = action?.preview ?? {}
      const streamed = streamingDraft ?? ''
      const settled = (preview.body as string | undefined) ?? preview.body_excerpt ?? ''
      const shown = streamed.length >= settled.length ? streamed : settled

      body = editing ? (
        <div className="space-y-2.5">
          {editable.length ? (
            editable.map((field) => (
              <Field
                key={field}
                name={field}
                schema={(patchProps[field] as SchemaNode) ?? { type: 'string' }}
                value={patch[field]}
                required={false}
                disabled={busy}
                idPrefix={prompt.id}
                onChange={(next) => setPatch((current) => ({ ...current, [field]: next }))}
              />
            ))
          ) : (
            <p className="text-2xs text-muted">There is nothing on this one you can edit.</p>
          )}
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              disabled={busy || !editable.length}
              className={PRIMARY}
              onClick={() => {
                const trimmed: Record<string, unknown> = {}
                for (const [key, value] of Object.entries(patch)) {
                  if (value !== undefined && value !== '') trimmed[key] = value
                }
                submit({ approve: false, patch: trimmed })
              }}
            >
              Save changes
            </button>
            <button
              type="button"
              disabled={busy}
              className={SECONDARY}
              onClick={() => setEditing(false)}
            >
              Back
            </button>
          </div>
          <p className="text-2xs text-faint">
            Saving asks again against the edited version. Still nothing sent.
          </p>
          {footerProblems}
        </div>
      ) : (
        <div className="space-y-2.5">
          {!action && shown && (
            <div className="max-h-56 overflow-y-auto whitespace-pre-wrap rounded-lg bg-canvas px-3 py-2 text-xs leading-relaxed text-ink ring-1 ring-inset ring-line">
              {shown}
            </div>
          )}
          <div className="flex flex-wrap gap-2">
            <button
              type="button"
              disabled={busy}
              className={PRIMARY}
              onClick={() => submit({ approve: true })}
            >
              {busy ? 'Sending…' : 'Send it'}
            </button>
            <button
              type="button"
              disabled={busy}
              className={SECONDARY}
              onClick={() => submit({ approve: false })}
            >
              Not now
            </button>
            <button type="button" disabled={busy} className={SECONDARY} onClick={startEditing}>
              Edit
            </button>
          </div>
          {footerProblems}
        </div>
      )
      break
    }

    /* -------------------------------------------------------------- choice */
    case 'choice': {
      body = (
        <div className="space-y-2.5">
          <ul className="space-y-1.5">
            {options.map((option) => {
              const selected = choice === option.id
              return (
                <li key={option.id}>
                  <button
                    type="button"
                    disabled={disabled}
                    onClick={() => setChoice(option.id)}
                    aria-pressed={selected}
                    className={`w-full rounded-lg border px-2.5 py-2 text-left transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-accent/50 ${
                      selected
                        ? 'border-accent bg-accent-soft'
                        : 'border-line bg-canvas hover:border-accent/50 hover:bg-elevated'
                    }`}
                  >
                    <span className="flex flex-col">
                      <span className="text-xs font-medium text-ink">{option.label}</span>
                      <OptionMeta meta={option.meta} />
                    </span>
                  </button>
                </li>
              )
            })}
          </ul>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              disabled={busy || !choice}
              className={PRIMARY}
              onClick={() => choice && submit(shapePick(choice))}
            >
              Continue
            </button>
            {searchDifferently}
          </div>
          {footerProblems}
        </div>
      )
      break
    }

    /* -------------------------------------------------------- multi_choice */
    case 'multi_choice': {
      const toggle = (id: string) =>
        setPicked((current) =>
          current.includes(id) ? current.filter((item) => item !== id) : [...current, id],
        )
      body = (
        <div className="space-y-2.5">
          <div className="flex gap-3">
            <button
              type="button"
              disabled={disabled}
              className={QUIET}
              onClick={() => setPicked(options.map((option) => option.id))}
            >
              Select all
            </button>
            <button
              type="button"
              disabled={disabled}
              className={QUIET}
              onClick={() => setPicked([])}
            >
              Select none
            </button>
          </div>
          <ul className="space-y-1.5">
            {options.map((option) => (
              <li key={option.id}>
                <label
                  className={`flex cursor-pointer items-start gap-2 rounded-lg border px-2.5 py-2 transition-colors ${
                    picked.includes(option.id)
                      ? 'border-accent bg-accent-soft'
                      : 'border-line bg-canvas hover:bg-elevated'
                  }`}
                >
                  <input
                    type="checkbox"
                    className="mt-0.5 h-3.5 w-3.5 shrink-0 rounded border-line accent-accent"
                    disabled={disabled}
                    checked={picked.includes(option.id)}
                    onChange={() => toggle(option.id)}
                  />
                  <span className="flex min-w-0 flex-col">
                    <span className="text-xs font-medium text-ink">{option.label}</span>
                    <OptionMeta meta={option.meta} />
                  </span>
                </label>
              </li>
            ))}
          </ul>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              disabled={busy || picked.length === 0}
              className={PRIMARY}
              onClick={() => submit(shapePick(picked))}
            >
              Continue{picked.length ? ` · ${formatCount(picked.length)}` : ''}
            </button>
            {searchDifferently}
          </div>
          {footerProblems}
        </div>
      )
      break
    }

    /* ---------------------------------------------------------- date_range */
    case 'date_range': {
      const props = schema.properties ?? {}
      body = (
        <div className="space-y-2.5">
          <div className="grid gap-2.5 sm:grid-cols-2">
            <Field
              name="start"
              schema={{ format: 'date-time', title: 'From', ...(props.start ?? {}) }}
              value={range.start}
              required={(schema.required ?? []).includes('start')}
              disabled={disabled}
              idPrefix={prompt.id}
              onChange={(next) =>
                setRange((current) => ({ ...current, start: String(next ?? '') }))
              }
            />
            <Field
              name="end"
              schema={{ format: 'date-time', title: 'To', ...(props.end ?? {}) }}
              value={range.end}
              required={(schema.required ?? []).includes('end')}
              disabled={disabled}
              idPrefix={prompt.id}
              onChange={(next) => setRange((current) => ({ ...current, end: String(next ?? '') }))}
            />
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              disabled={busy}
              className={PRIMARY}
              onClick={() => {
                if (range.start && range.end && Date.parse(range.end) <= Date.parse(range.start)) {
                  setProblems(['The end has to come after the start.'])
                  return
                }
                submit({ start: range.start, end: range.end })
              }}
            >
              Continue
            </button>
            <span className="text-2xs text-faint">
              The window is half-open — the end is not included.
            </span>
          </div>
          {footerProblems}
        </div>
      )
      break
    }

    /* ---------------------------------------------------------------- text */
    case 'text': {
      const long = (schema.maxLength ?? 0) > 200 || schema['x-widget'] === 'textarea'
      body = (
        <div className="space-y-2.5">
          {long ? (
            <textarea
              rows={3}
              className={INPUT}
              disabled={disabled}
              maxLength={schema.maxLength}
              value={text}
              onChange={(event) => setText(event.target.value)}
            />
          ) : (
            <input
              type="text"
              className={INPUT}
              disabled={disabled}
              maxLength={schema.maxLength}
              value={text}
              onChange={(event) => setText(event.target.value)}
            />
          )}
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              disabled={busy || !text.trim()}
              className={PRIMARY}
              onClick={() => submit(text)}
            >
              Continue
            </button>
            {schema.maxLength && (
              <span className="text-2xs text-faint tabular-nums">
                {text.length}/{schema.maxLength}
              </span>
            )}
          </div>
          {footerProblems}
        </div>
      )
      break
    }

    /* ---------------------------------------------------------------- form */
    case 'form': {
      const props = schema.properties ?? {}
      const required = schema.required ?? []
      // A form declares its own fields, and each one carries its own options.
      // The schema is still the authority on what is valid; `fields` is what to
      // draw and in what order. A field the schema does not mention is drawn
      // anyway — the server asked for it.
      const declared = prompt.prompt.fields ?? []
      const names = declared.length
        ? Array.from(new Set([...declared.map((field) => field.name), ...Object.keys(props)]))
        : Object.keys(props)

      body = (
        <div className="space-y-2.5">
          {names.map((name) => {
            const field = declared.find((entry) => entry.name === name)
            const child = props[name] ?? { type: 'string' }
            return (
              <Field
                key={name}
                name={name}
                schema={{
                  ...child,
                  title: field?.label ?? child.title,
                  description: child.description ?? field?.help_text ?? undefined,
                }}
                options={field?.options}
                value={form[name]}
                required={required.includes(name) || field?.required === true}
                disabled={disabled}
                idPrefix={prompt.id}
                onChange={(next) => setForm((current) => ({ ...current, [name]: next }))}
              />
            )
          })}
          <div className="flex flex-wrap items-center gap-2">
            <button
              type="button"
              disabled={busy}
              className={PRIMARY}
              onClick={() => {
                const cleaned: Record<string, unknown> = {}
                for (const [key, value] of Object.entries(form)) {
                  if (value !== undefined && value !== '') cleaned[key] = value
                }
                submit(cleaned)
              }}
            >
              Continue
            </button>
            {searchDifferently}
          </div>
          {footerProblems}
        </div>
      )
      break
    }

    /* ------------------------------------------------- an unshipped kind */
    default: {
      body = (
        <div className="space-y-2.5">
          <p className="text-2xs text-muted">
            This is a kind this build does not draw yet. The schema still decides what is valid.
          </p>
          <textarea
            rows={4}
            className={`${INPUT} font-mono`}
            disabled={disabled}
            value={text}
            placeholder='{"value": …}'
            onChange={(event) => setText(event.target.value)}
          />
          <button
            type="button"
            disabled={busy || !text.trim()}
            className={PRIMARY}
            onClick={() => {
              try {
                submit(JSON.parse(text))
              } catch {
                setProblems(['That is not valid JSON.'])
              }
            }}
          >
            Continue
          </button>
          {footerProblems}
        </div>
      )
    }
  }

  return (
    <article className="animate-rise-in overflow-hidden rounded-xl border border-line bg-surface shadow-card">
      {header}
      <div className="px-3 py-2.5">{body}</div>
      {prompt.blocking && (
        <footer className="border-t border-line px-3 py-1.5">
          <button type="button" onClick={onCancel} disabled={busy} className={QUIET}>
            Cancel this
          </button>
        </footer>
      )}
    </article>
  )
}
