/**
 * One line about your data, and only when it earns the space.
 *
 * Everything working is the normal case, and the normal case needs no report.
 * So a healthy mirror renders nothing at all: no dots, no counts, no ages. The
 * strip appears when something is behind or broken, says which part in plain
 * words, and offers the one button that helps.
 */

import { api } from '../lib/api'
import { relativeTime } from '../lib/format'
import type { ServiceName, SyncBarProps, SyncServiceState } from '../lib/types'

const SERVICES: ServiceName[] = ['gmail', 'gcal', 'gdrive']

/** What a person calls these, not what the API calls them. */
const PLAIN: Record<ServiceName, string> = {
  gmail: 'mail',
  gcal: 'calendar',
  gdrive: 'files',
}

type Condition = 'fine' | 'stale' | 'broken'

/**
 * How far past the freshness goal before the age is worth saying out loud.
 *
 * Being a few minutes behind is the normal state of anything that syncs on a
 * schedule, and reporting it would put a line on the screen that is almost
 * always there — which is the same as having no line at all.
 */
const STALE_MULTIPLE = 6

function conditionOf(state: SyncServiceState | undefined, targetSeconds: number): Condition {
  if (!state || !state.last_success_at) return 'stale'
  // `healthy` on the wire folds freshness in with working, so it cannot answer
  // "is this broken?" — being behind is not the same as being down, and saying
  // "not responding" about a service that is merely late is a false alarm.
  if (state.circuit_open_until && Date.parse(state.circuit_open_until) > Date.now()) return 'broken'
  if (state.consecutive_failures > 0 || state.last_error) return 'broken'
  if (!state.backfill_complete) return 'stale'
  if (state.lag_seconds != null && state.lag_seconds > targetSeconds * STALE_MULTIPLE) {
    return 'stale'
  }
  return 'fine'
}

/** The oldest last-success across the services that are behind. */
function oldest(states: Array<SyncServiceState | undefined>): string | null {
  const times = states
    .map((s) => s?.last_success_at)
    .filter((t): t is string => Boolean(t))
    .sort()
  return times[0] ?? null
}

/** "mail" · "mail and calendar" · "mail, calendar and files" */
function joinPlain(names: string[]): string {
  if (names.length <= 1) return names[0] ?? ''
  return `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`
}

export interface SyncBarExtraProps {
  /**
   * From `GET /auth/me` — `google.needs_reauth`. The shell owns auth state, so
   * the banner only appears when it says so; deriving it here as well would put
   * two reconnect banners on screen at once.
   */
  needsReauth?: boolean
  /** Where the consent screen lives. A link, never a fetch. */
  reconnectHref?: string
}

export default function SyncBar({
  status,
  loading,
  error,
  onSyncNow,
  needsReauth = false,
  reconnectHref,
}: SyncBarProps & SyncBarExtraProps) {
  const services = status?.services ?? {}
  const target = status?.freshness?.target_seconds ?? 900
  const href = reconnectHref ?? api.googleAuthUrl('/')

  const broken = SERVICES.filter((s) => conditionOf(services[s], target) === 'broken')
  const behind = SERVICES.filter((s) => conditionOf(services[s], target) === 'stale')

  if (needsReauth) {
    return (
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1 border-b border-line bg-bad/10 px-4 py-2 sm:px-6">
        <p className="text-xs text-ink">
          <span className="font-semibold text-bad">Google needs reconnecting.</span> Your mail,
          calendar and files are not reachable until you allow access again.
        </p>
        <a
          href={href}
          className="ml-auto rounded-lg bg-accent px-3 py-1 text-xs font-semibold text-accent-ink transition hover:opacity-90"
        >
          Reconnect
        </a>
      </div>
    )
  }

  // Nothing to report, and no error from a sync somebody asked for. Say nothing.
  if (!broken.length && !behind.length && !error && !loading) return null

  const tone = broken.length ? 'bad' : 'warn'
  const since = oldest(behind.map((s) => services[s]))

  // An error on its own used to fall through to the "behind" sentence with an
  // empty list of services in it — "Your  last updated a while ago." Each
  // branch now only runs when it has something to name.
  let message: string
  if (broken.length) {
    message = `Your ${joinPlain(broken.map((s) => PLAIN[s]))} ${
      broken.length > 1 ? 'are' : 'is'
    } not responding. Answers may be missing recent items.`
  } else if (behind.length) {
    // Stated as a fact, not as activity. Nothing may be running, and "catching
    // up" would promise a recovery that is not actually under way.
    message = `Your ${joinPlain(behind.map((s) => PLAIN[s]))} last updated ${
      since ? relativeTime(since) : 'a while ago'
    }.`
  } else {
    message = 'That last check did not go through.'
  }

  return (
    <div
      className={`border-b border-line px-4 py-2 sm:px-6 ${
        tone === 'bad' ? 'bg-bad/10' : 'bg-warn/10'
      }`}
    >
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <p className={`text-xs ${tone === 'bad' ? 'text-bad' : 'text-warn'}`}>
          {loading ? 'Checking for new items…' : message}
        </p>
        {!loading && (
          <button
            type="button"
            onClick={onSyncNow}
            className="ml-auto rounded-lg border border-line bg-surface px-2.5 py-1 text-xs font-medium text-ink transition hover:bg-elevated"
          >
            Check again
          </button>
        )}
      </div>
      {error && <p className="pt-1 text-2xs text-muted">{error}</p>}
    </div>
  )
}
