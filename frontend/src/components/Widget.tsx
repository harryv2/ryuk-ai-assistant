/**
 * Structured answers, drawn rather than printed.
 *
 * "Six events, each with a time and some guests" is a set of things with
 * shape. Flattening it to markdown and asking the browser to re-read the
 * markdown throws that shape away at the last possible moment — and it means
 * the answer can never *do* anything.
 *
 * Two rules make this safe. The backend never sends markup, only data, so
 * there is nothing to sanitise and the UI owns every pixel. And a widget it
 * does not recognise is not an error: the block carries the markdown it
 * replaced, and `MessageList` falls back to that. A message written today has
 * to still render in a year.
 *
 * A widget can do exactly two things. `ask` puts a sentence in the composer as
 * if you typed it — so a write still stops at its confirmation card, because
 * it goes down the ordinary path. `open` follows a link that came out of your
 * own data. Nothing here executes anything.
 */

import { useState } from 'react'

import { formatBytes, formatDay, formatTime, truncate } from '../lib/format'
import type { WidgetAction, WidgetBlock } from '../lib/types'

interface WidgetProps {
  block: WidgetBlock
  onAsk: (question: string) => void
}

/* ------------------------------------------------------------------ shared */

function Actions({
  actions,
  onAsk,
}: {
  actions?: WidgetAction[] | null
  onAsk: (q: string) => void
}) {
  if (!actions?.length) return null
  return (
    <div className="mt-2 flex flex-wrap gap-1.5">
      {actions.map((action, i) =>
        action.kind === 'open' ? (
          <a
            key={i}
            href={action.url}
            target="_blank"
            rel="noopener noreferrer nofollow"
            className="rounded-md border border-line px-2 py-1 text-2xs font-medium text-ink transition hover:bg-elevated"
          >
            {action.label}
          </a>
        ) : (
          <button
            key={i}
            type="button"
            onClick={() => onAsk(action.query)}
            className="rounded-md border border-line px-2 py-1 text-2xs font-medium text-ink transition hover:bg-elevated"
          >
            {action.label}
          </button>
        ),
      )}
    </div>
  )
}

function Shell({ children }: { children: React.ReactNode }) {
  return <div className="overflow-hidden rounded-xl border border-line">{children}</div>
}

function Row({ children, last }: { children: React.ReactNode; last: boolean }) {
  return <div className={last ? 'px-3.5 py-2.5' : 'border-b border-line px-3.5 py-2.5'}>{children}</div>
}

/* ------------------------------------------------------------------- typed */

function EventList({ data, onAsk }: { data: any; onAsk: (q: string) => void }) {
  // Grouped by day, because "what is on next week" is read a day at a time.
  const days = new Map<string, any[]>()
  for (const item of data.items ?? []) {
    const key = formatDay(item.starts_at)
    days.set(key, [...(days.get(key) ?? []), item])
  }

  return (
    <Shell>
      {[...days.entries()].map(([day, events], di, all) => (
        <div key={day} className={di === all.length - 1 ? '' : 'border-b border-line'}>
          <p className="bg-elevated/60 px-3.5 py-1.5 text-2xs font-medium text-muted">{day}</p>
          {events.map((event, i) => (
            <Row key={event.id ?? i} last={i === events.length - 1}>
              <div className="flex items-baseline gap-3">
                <span className="shrink-0 tabular-nums text-xs text-muted">
                  {event.all_day
                    ? 'All day'
                    : `${formatTime(event.starts_at)}${event.ends_at ? `–${formatTime(event.ends_at)}` : ''}`}
                </span>
                <span className="min-w-0 flex-1 text-sm text-ink">{event.title}</span>
              </div>
              {(event.location || event.guests?.length > 0) && (
                <p className="mt-0.5 truncate pl-[3.9rem] text-2xs text-muted">
                  {[
                    event.location,
                    event.guests?.length ? `${event.guests.length} guests` : null,
                  ]
                    .filter(Boolean)
                    .join(' · ')}
                </p>
              )}
              <div className="pl-[3.9rem]">
                <Actions
                  onAsk={onAsk}
                  actions={[
                    { kind: 'ask', label: 'Move', query: `Move "${event.title}" to ` },
                    ...(event.url
                      ? [{ kind: 'open' as const, label: 'Open', url: event.url }]
                      : []),
                  ]}
                />
              </div>
            </Row>
          ))}
        </div>
      ))}
    </Shell>
  )
}

function EmailList({ data, onAsk }: { data: any; onAsk: (q: string) => void }) {
  const items = data.items ?? []
  return (
    <Shell>
      {items.map((mail: any, i: number) => (
        <Row key={mail.id ?? i} last={i === items.length - 1}>
          <div className="flex items-baseline justify-between gap-3">
            <p className="min-w-0 flex-1 truncate text-sm font-medium text-ink">{mail.subject}</p>
            <span className="shrink-0 text-2xs text-muted">{formatDay(mail.received_at)}</span>
          </div>
          <p className="mt-0.5 truncate text-xs text-muted">
            {mail.from_name || mail.from_email}
          </p>
          {mail.excerpt && (
            <p className="mt-1 text-2xs leading-relaxed text-faint">
              {truncate(mail.excerpt, 150)}
            </p>
          )}
          <Actions
            onAsk={onAsk}
            actions={[
              { kind: 'ask', label: 'Reply', query: `Draft a reply to "${mail.subject}"` },
              ...(mail.url ? [{ kind: 'open' as const, label: 'Open', url: mail.url }] : []),
            ]}
          />
        </Row>
      ))}
    </Shell>
  )
}

function FileList({ data, onAsk }: { data: any; onAsk: (q: string) => void }) {
  const items = data.items ?? []
  return (
    <Shell>
      {items.map((file: any, i: number) => (
        <Row key={file.id ?? i} last={i === items.length - 1}>
          <div className="flex items-baseline justify-between gap-3">
            <p className="min-w-0 flex-1 truncate text-sm text-ink">{file.name}</p>
            <span className="shrink-0 text-2xs text-muted">{formatDay(file.modified_at)}</span>
          </div>
          <p className="mt-0.5 truncate text-2xs text-muted">
            {[file.owner, file.size_bytes ? formatBytes(file.size_bytes) : null]
              .filter(Boolean)
              .join(' · ')}
          </p>
          <Actions
            onAsk={onAsk}
            actions={[
              { kind: 'ask', label: 'Summarise', query: `Summarise "${file.name}"` },
              ...(file.url ? [{ kind: 'open' as const, label: 'Open', url: file.url }] : []),
            ]}
          />
        </Row>
      ))}
    </Shell>
  )
}

function FreeSlots({ data, onAsk }: { data: any; onAsk: (q: string) => void }) {
  // The one answer markdown cannot express. "You are free 3:00–4:30" is a
  // fact; a button that books it is the thing somebody actually wanted.
  return (
    <div className="space-y-2.5">
      {(data.days ?? []).map((day: any) => (
        <div key={day.date}>
          <p className="mb-1.5 text-2xs font-medium text-muted">{formatDay(day.date)}</p>
          <div className="flex flex-wrap gap-1.5">
            {day.slots.map((slot: any, i: number) => (
              <button
                key={i}
                type="button"
                onClick={() =>
                  onAsk(
                    `Book ${formatDay(slot.start)} ${formatTime(slot.start)}–${formatTime(slot.end)}: `,
                  )
                }
                className="rounded-lg border border-line px-2.5 py-1.5 text-xs tabular-nums text-ink transition hover:border-accent hover:bg-accent-soft"
              >
                {formatTime(slot.start)}–{formatTime(slot.end)}
              </button>
            ))}
          </div>
        </div>
      ))}
    </div>
  )
}

/* ----------------------------------------------------------------- generic */

function Table({ data }: { data: any }) {
  return (
    <div className="overflow-x-auto rounded-xl border border-line">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-line bg-elevated/60">
            {data.columns.map((c: any) => (
              <th
                key={c.key}
                className={`px-3 py-2 text-2xs font-medium text-muted ${
                  c.align === 'right' ? 'text-right' : 'text-left'
                }`}
              >
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.rows.map((row: any, i: number) => (
            <tr key={i} className={i ? 'border-t border-line' : ''}>
              {data.columns.map((c: any) => (
                <td
                  key={c.key}
                  className={`px-3 py-2 text-ink ${c.align === 'right' ? 'text-right tabular-nums' : ''}`}
                >
                  {row[c.key]}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function PlainList({ data, onAsk }: { data: any; onAsk: (q: string) => void }) {
  const items = data.items ?? []
  return (
    <Shell>
      {items.map((item: any, i: number) => (
        <Row key={i} last={i === items.length - 1}>
          <div className="flex items-baseline justify-between gap-3">
            <p className="min-w-0 flex-1 text-sm text-ink">{item.title}</p>
            {item.badge && (
              <span className="shrink-0 rounded-full bg-elevated px-2 py-0.5 text-2xs text-muted">
                {item.badge}
              </span>
            )}
            {item.meta && <span className="shrink-0 text-2xs text-muted">{item.meta}</span>}
          </div>
          {item.subtitle && <p className="mt-0.5 text-xs text-muted">{item.subtitle}</p>}
          <Actions actions={item.actions} onAsk={onAsk} />
        </Row>
      ))}
    </Shell>
  )
}

function Stat({ data }: { data: any }) {
  const tone =
    data.tone === 'good' ? 'text-ok' : data.tone === 'bad' ? 'text-bad' : data.tone === 'warn' ? 'text-warn' : 'text-ink'
  return (
    <div className="rounded-xl border border-line px-4 py-3">
      <p className={`text-2xl font-semibold tabular-nums ${tone}`}>{data.value}</p>
      {data.label && <p className="mt-0.5 text-xs text-muted">{data.label}</p>}
      {data.detail && <p className="mt-1 text-2xs text-faint">{data.detail}</p>}
    </div>
  )
}

function KeyValues({ data }: { data: any }) {
  return (
    <dl className="overflow-hidden rounded-xl border border-line">
      {data.pairs.map((pair: any, i: number) => (
        <div
          key={i}
          className={`flex gap-3 px-3.5 py-2 text-sm ${i ? 'border-t border-line' : ''}`}
        >
          <dt className="w-2/5 shrink-0 text-muted">{pair.label}</dt>
          <dd className="min-w-0 flex-1 text-ink">{pair.value}</dd>
        </div>
      ))}
    </dl>
  )
}

function Timeline({ data }: { data: any }) {
  return (
    <ol className="space-y-3 border-l border-line pl-4">
      {data.entries.map((entry: any, i: number) => (
        <li key={i} className="relative">
          <span className="absolute -left-[1.32rem] top-1.5 h-2 w-2 rounded-full bg-accent" />
          {entry.at && <p className="text-2xs text-muted">{entry.at}</p>}
          <p className="text-sm text-ink">{entry.title}</p>
          {entry.detail && <p className="mt-0.5 text-xs text-muted">{entry.detail}</p>}
        </li>
      ))}
    </ol>
  )
}

function Comparison({ data }: { data: any }) {
  return (
    <div className="grid grid-cols-2 gap-2">
      {[data.left, data.right].map((side: any, i: number) => (
        <div key={i} className="rounded-xl border border-line">
          <p className="border-b border-line bg-elevated/60 px-3 py-1.5 text-2xs font-medium text-muted">
            {side.label}
          </p>
          <dl className="px-3 py-2">
            {side.pairs.map((pair: any, j: number) => (
              <div key={j} className="py-1">
                <dt className="text-2xs text-muted">{pair.label}</dt>
                <dd className="text-sm text-ink">{pair.value}</dd>
              </div>
            ))}
          </dl>
        </div>
      ))}
    </div>
  )
}

function Chips({ data, onAsk }: { data: any; onAsk: (q: string) => void }) {
  return (
    <div className="flex flex-wrap gap-1.5">
      {data.items.map((chip: any, i: number) =>
        chip.action?.kind === 'ask' ? (
          <button
            key={i}
            type="button"
            onClick={() => onAsk(chip.action.query)}
            className="rounded-full border border-line px-3 py-1 text-xs text-ink transition hover:border-accent hover:bg-accent-soft"
          >
            {chip.label}
          </button>
        ) : (
          <span key={i} className="rounded-full bg-elevated px-3 py-1 text-xs text-muted">
            {chip.label}
          </span>
        ),
      )}
    </div>
  )
}

/* ---------------------------------------------------------------- registry */

const REGISTRY: Record<string, (p: { data: any; onAsk: (q: string) => void }) => JSX.Element> = {
  event_list: EventList,
  email_list: EmailList,
  file_list: FileList,
  free_slots: FreeSlots,
  table: ({ data }) => <Table data={data} />,
  list: PlainList,
  stat: ({ data }) => <Stat data={data} />,
  key_values: ({ data }) => <KeyValues data={data} />,
  timeline: ({ data }) => <Timeline data={data} />,
  comparison: ({ data }) => <Comparison data={data} />,
  chips: Chips,
}

/** True when this client can draw the widget; the caller shows text if not. */
export function canRender(block: WidgetBlock): boolean {
  return Boolean(REGISTRY[block.widget])
}

export default function Widget({ block, onAsk }: WidgetProps) {
  const [failed, setFailed] = useState(false)
  const Component = REGISTRY[block.widget]
  if (!Component || failed) return null

  // A widget that throws must not take the message with it. The text block is
  // already on screen above it, so the answer survives either way.
  try {
    return <Component data={block.data} onAsk={onAsk} />
  } catch {
    setFailed(true)
    return null
  }
}
