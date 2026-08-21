import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react'

import SignIn from './components/SignIn'
import UserMenu from './components/UserMenu'
import { logout as signOut } from './lib/auth'

import Chat from './components/Chat'
import History from './components/History'
import Integrations from './components/Integrations'
import SyncStatusPanel from './components/SyncStatus'
import { ApiError, api, getMe, getSyncStatus, googleAuthUrl } from './lib/api'
import { useRun } from './lib/useRun'
import type {
  ActionRecord,
  ChatMessage,
  ContentBlock,
  ConversationDetail,
  Id,
  LiveBlock,
  MeResponse,
  Message,
  PendingInput,
  ConversationSummary,
  SyncStatus,
} from './lib/types'

/**
 * The shell.
 *
 * It owns three things and delegates the rest: the session, the mirror's
 * freshness, and the thread. The run itself lives in `useRun`, which turns the
 * event stream into render state; the conversation on the left and the trace on
 * the right are both fed from here.
 *
 * Two columns, always: the chat is where you talk, the trace is where you watch
 * it work. On a narrow screen the trace collapses behind a button rather than
 * disappearing from the product.
 */

type AuthState =
  | { phase: 'loading' }
  | { phase: 'signed-out' }
  | { phase: 'unreachable'; message: string }
  | { phase: 'ready'; me: MeResponse }

const AUTH_POLL_MS = 60_000
const SYNC_POLL_MS = 20_000

// Every one of these has to land on real data. A suggested question that
// always answers "nothing found" is a worse first impression than no
// suggestion at all, and it reads as the product being broken.
const EMPTY_HINTS = [
  'What is on my calendar next week?',
  'Find the Acme proposal and tell me what changed',
  'Cancel my Turkish Airlines flight',
  'Find emails from Sarah about the budget',
]

/* --------------------------------------------------------------- utilities */

function readThread(): string | null {
  const hash = window.location.hash.replace(/^#\/?/, '')
  return hash.startsWith('c/') ? hash.slice(2) || null : null
}

function writeThread(id: string): void {
  const next = `#/c/${id}`
  if (window.location.hash !== next) window.history.replaceState(null, '', next)
}

/** Wide enough for two columns side by side. Matches Tailwind's `lg`. */

function readTheme(): 'light' | 'dark' {
  try {
    const saved = window.localStorage.getItem('theme')
    if (saved === 'light' || saved === 'dark') return saved
  } catch {
    /* private mode */
  }
  return document.documentElement.classList.contains('dark') ? 'dark' : 'light'
}

/** The width at which the history column stops covering the conversation. */
const WIDE = '(min-width: 768px)'

/** Whether the history column was last left open. Open by default. */
function readHistoryOpen(): boolean {
  try {
    return window.localStorage.getItem('history-open') !== 'no'
  } catch {
    return true // private mode
  }
}

/** Open on a wide screen, shut on a narrow one where it would cover the chat. */
function initialHistoryOpen(): boolean {
  if (typeof window === 'undefined') return false
  return window.matchMedia(WIDE).matches && readHistoryOpen()
}

/** The show/hide-the-column glyph. Used in both places it can appear. */
function SidebarIcon() {
  return (
    <svg viewBox="0 0 16 16" aria-hidden className="h-4 w-4">
      <rect
        x="1.5"
        y="2.5"
        width="13"
        height="11"
        rx="2"
        fill="none"
        stroke="currentColor"
        strokeWidth="1.5"
      />
      <path d="M6 2.5v11" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  )
}

function messageText(message: Message | ChatMessage): string {
  return message.content
    .filter((block): block is Extract<ContentBlock, { type: 'text' }> => block.type === 'text')
    .map((block) => block.data.markdown)
    .join('\n\n')
    .trim()
}

/** The live blocks, in the shape a message renders from. */
function blocksToContent(blocks: LiveBlock[]): ContentBlock[] {
  return blocks
    .filter((block) => block.kind !== 'text' || block.text.length > 0)
    .map<ContentBlock>((block) => {
      if (block.kind === 'text') return { type: 'text', data: { markdown: block.text } }
      // A widget arrives whole rather than as a ref, so it passes straight
      // through — there is nothing to resolve it against.
      if (block.kind === 'widget') return block.block
      return { type: block.kind, ref: block.ref }
    })
}

/** Drafted bodies streaming into cards, keyed by the card they belong to. */
function draftsFrom(blocks: LiveBlock[]): Record<Id, string> {
  const drafts: Record<Id, string> = {}
  for (const block of blocks) {
    // Named positively rather than "not text": a widget is not text either,
    // and it has no ref to key a draft by.
    if ((block.kind === 'action' || block.kind === 'input') && block.draft) {
      drafts[block.ref] = block.draft
    }
  }
  return drafts
}

function indexById<T extends { id: Id }>(list: T[] | undefined): Record<Id, T> {
  const out: Record<Id, T> = {}
  for (const item of list ?? []) out[item.id] = item
  return out
}

/* -------------------------------------------------------------------- app */

export default function App() {
  const [auth, setAuth] = useState<AuthState>({ phase: 'loading' })
  // Kept only so the sign-in form can prefill it after signing out. Not a
  // session, not a credential — the session is the httpOnly cookie.
  const [lastEmail, setLastEmail] = useState('')
  const [online, setOnline] = useState(() => navigator.onLine)
  const [theme, setTheme] = useState<'light' | 'dark'>(readTheme)

  const [sync, setSync] = useState<SyncStatus | null>(null)

  const [thread, setThread] = useState<ConversationDetail | null>(null)
  const [pendingUser, setPendingUser] = useState<{ id: Id; text: string; at: string } | null>(null)
  const [draft, setDraft] = useState('')
  const [notice, setNotice] = useState<string | null>(null)
  const [needsReauth, setNeedsReauth] = useState(false)
  // Open by default where there is room for two columns; on a phone the trace
  // rides over the chat, so it starts closed and the button brings it in.

  const [historyOpen, setHistoryOpen] = useState(initialHistoryOpen)
  const [history, setHistory] = useState<ConversationSummary[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)
  const [showIntegrations, setShowIntegrations] = useState(false)
  const [showSyncStatus, setShowSyncStatus] = useState(false)

  // Follow the breakpoint rather than sampling the width once. The first
  // measurement can happen while the window is still sizing, and on a real
  // device this is also the rotation and window-drag case: crossing to wide
  // restores the choice, crossing to narrow puts the conversation back.
  useEffect(() => {
    const wide = window.matchMedia(WIDE)
    const apply = () => setHistoryOpen(wide.matches && readHistoryOpen())
    apply()
    wide.addEventListener('change', apply)
    return () => wide.removeEventListener('change', apply)
  }, [])

  const toggleHistory = useCallback(() => {
    setHistoryOpen((open) => {
      const next = !open
      // Only a deliberate choice on a wide screen is remembered — opening the
      // sheet on a phone is a glance at the list, not a layout preference.
      if (window.matchMedia(WIDE).matches) {
        try {
          window.localStorage.setItem('history-open', next ? 'yes' : 'no')
        } catch {
          /* private mode */
        }
      }
      return next
    })
  }, [])

  const initialThread = useRef<string | null>(readThread()).current
  const [threadId, setThreadId] = useState<string | null>(initialThread)
  const pendingRef = useRef(pendingUser)
  pendingRef.current = pendingUser

  /* --------------------------------------------------------------- run --- */

  const onHydrated = useCallback((convo: ConversationDetail) => {
    setThread(convo)
    setThreadId(convo.id)
    writeThread(convo.id)
    // The durable record now carries the message we were holding optimistically.
    const pending = pendingRef.current
    if (pending && convo.messages.some((m) => m.role === 'user' && messageText(m) === pending.text)) {
      setPendingUser(null)
    }
  }, [])

  const run = useRun({
    conversationId: initialThread,
    onConversation: (id) => {
      setThreadId(id)
      writeThread(id)
    },
    onHydrated,
    onReauthRequired: () => setNeedsReauth(true),
    onError: (err) => {
      if (err instanceof ApiError && err.code === 'RATE_LIMITED') {
        const wait = err.retryAfterS ? ` Try again in ${err.retryAfterS}s.` : ''
        setNotice(`That is 100 queries this hour, which is the limit.${wait}`)
      }
    },
  })

  const { state: runState, refetch, ask } = run

  useEffect(() => {
    if (initialThread) void refetch(initialThread)
  }, [initialThread, refetch])

  /* ----------------------------------------------------------- history --- */

  const loadHistory = useCallback(async (signal?: AbortSignal) => {
    setHistoryLoading(true)
    try {
      const page = await api.getConversations({ limit: 50 }, signal)
      setHistory(page.items)
    } catch {
      // A sidebar that cannot load is not worth an error banner over the chat.
    } finally {
      if (!signal?.aborted) setHistoryLoading(false)
    }
  }, [])

  /* ----------------------------------------------------------- session --- */

  const loadMe = useCallback(async (signal?: AbortSignal) => {
    try {
      const me = await getMe(signal)
      setAuth({ phase: 'ready', me })
      setNeedsReauth(!me.google.connected || me.google.needs_reauth)
    } catch (err) {
      if (signal?.aborted) return
      if (err instanceof ApiError && err.needsSignIn) {
        setAuth({ phase: 'signed-out' })
        return
      }
      if (err instanceof ApiError && err.needsReauth) {
        setNeedsReauth(true)
        return
      }
      if (err instanceof ApiError && err.status === 0) {
        setAuth({ phase: 'unreachable', message: err.message })
        return
      }
      setAuth({
        phase: 'unreachable',
        message: err instanceof Error ? err.message : 'The API is not answering.',
      })
    }
  }, [])

  useEffect(() => {
    const controller = new AbortController()
    void loadMe(controller.signal)
    const id = window.setInterval(() => void loadMe(), AUTH_POLL_MS)
    return () => {
      controller.abort()
      window.clearInterval(id)
    }
  }, [loadMe])

  useEffect(() => {
    const onVisible = () => {
      if (document.visibilityState === 'visible') void loadMe()
    }
    const goOnline = () => {
      setOnline(true)
      void loadMe()
    }
    const goOffline = () => setOnline(false)

    document.addEventListener('visibilitychange', onVisible)
    window.addEventListener('online', goOnline)
    window.addEventListener('offline', goOffline)
    return () => {
      document.removeEventListener('visibilitychange', onVisible)
      window.removeEventListener('online', goOnline)
      window.removeEventListener('offline', goOffline)
    }
  }, [loadMe])

  /* -------------------------------------------------------------- sync --- */

  const signedIn = auth.phase === 'ready'

  const loadSync = useCallback(
    async (signal?: AbortSignal) => {
      if (!signedIn) return
      try {
        const status = await getSyncStatus(signal)
        setSync(status)
      } catch (err) {
        if (signal?.aborted) return
        if (err instanceof ApiError && err.needsReauth) setNeedsReauth(true)
      }
    },
    [signedIn],
  )

  useEffect(() => {
    if (!signedIn) return
    const controller = new AbortController()
    void loadSync(controller.signal)
    const id = window.setInterval(() => void loadSync(), SYNC_POLL_MS)
    return () => {
      controller.abort()
      window.clearInterval(id)
    }
  }, [signedIn, loadSync])


  /* ------------------------------------------------------------- trace --- */

  // Widen the window and the second column comes back, unless it was closed on
  // purpose. Narrow it and the overlay gets out of the way.

  /* ------------------------------------------------------------- theme --- */

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
    try {
      window.localStorage.setItem('theme', theme)
    } catch {
      /* private mode: the class is still set for this session */
    }
  }, [theme])

  /* ------------------------------------------------------------ thread --- */

  const prompts = useMemo<Record<Id, PendingInput>>(
    () => ({ ...indexById(thread?.pending_inputs), ...runState.prompts }),
    [thread, runState.prompts],
  )

  const actions = useMemo<Record<Id, ActionRecord>>(
    () => ({ ...indexById(thread?.actions), ...runState.actions }),
    [thread, runState.actions],
  )

  const messages = useMemo<ChatMessage[]>(() => {
    const history: ChatMessage[] = (thread?.messages ?? []).map((m) => ({ ...m }))
    const out = [...history]

    if (pendingUser) {
      out.push({
        id: pendingUser.id,
        seq: (history[history.length - 1]?.seq ?? 0) + 1,
        role: 'user',
        content: [{ type: 'text', data: { markdown: pendingUser.text } }],
        run_id: null,
        created_at: pendingUser.at,
        optimistic: true,
      })
    }

    const liveId = runState.messageId ?? (runState.runId ? `live-${runState.runId}` : null)
    const alreadyStored = liveId ? history.some((m) => m.id === liveId) : false
    const hasLive = runState.blocks.length > 0 || runState.streaming

    if (liveId && hasLive && !alreadyStored) {
      out.push({
        id: liveId,
        seq: (out[out.length - 1]?.seq ?? 0) + 1,
        role: 'assistant',
        content: blocksToContent(runState.blocks),
        run_id: runState.runId,
        created_at: runState.startedAt ?? new Date().toISOString(),
        streaming: runState.streaming,
        drafts: draftsFrom(runState.blocks),
      })
    }

    return out
  }, [thread, pendingUser, runState.messageId, runState.runId, runState.blocks, runState.streaming, runState.startedAt])

  /** The question that produced a prompt, so "ask again" re-asks the real thing. */
  const queryBehind = useCallback(
    (prompt: PendingInput): string => {
      if (prompt.run_id && prompt.run_id === runState.runId && runState.query) return runState.query
      const list = thread?.messages ?? []
      const assistantIndex = list.findIndex((m) => m.run_id && m.run_id === prompt.run_id)
      const from = assistantIndex === -1 ? list.length : assistantIndex
      for (let i = from - 1; i >= 0; i -= 1) {
        const message = list[i]
        if (message.role === 'user') return messageText(message)
      }
      return runState.query ?? ''
    },
    [thread, runState.runId, runState.query],
  )

  const handleSignOut = useCallback(async () => {
    // Clear the screen before the request lands. Signing out should feel
    // immediate, and a failed call must not leave someone looking at a
    // signed-in page they believe they have left.
    setLastEmail(auth.phase === 'ready' ? auth.me.user.email : '')
    setAuth({ phase: 'signed-out' })
    try {
      await signOut()
    } catch {
      // The cookie is cleared server-side on any successful call; if the
      // request failed the next one will 401 and land in the same place.
    }
  }, [auth])

  const send = useCallback(
    (text: string) => {
      const query = text.trim()
      if (!query || runState.streaming) return
      setNotice(null)
      setDraft('')
      setPendingUser({ id: `local-${Date.now()}`, text: query, at: new Date().toISOString() })
      ask(query)
    },
    [ask, runState.streaming],
  )

  const respond = useCallback(
    (inputId: Id, value: unknown) => {
      setNotice(null)
      void run.respond(inputId, value)
    },
    [run],
  )

  const cancelPrompt = useCallback(
    (inputId: Id) => {
      void run.dismissPrompt(inputId)
    },
    [run],
  )

  /** "None of these — search differently": drop the card, hand the box back. */
  const searchDifferently = useCallback(
    (prompt: PendingInput) => {
      void run.dismissPrompt(prompt.id)
      const previous = queryBehind(prompt)
      setDraft(previous ? `${previous} ` : '')
      setNotice('Dropped that question. Say it differently and I will look again.')
    },
    [run, queryBehind],
  )

  const redo = useCallback(
    (prompt: PendingInput) => {
      const previous = queryBehind(prompt)
      if (previous) send(previous)
      else setNotice('Ask that again and I will pick it up from there.')
    },
    [queryBehind, send],
  )

  const startNewThread = useCallback(() => {
    run.stop()
    run.reset()
    setThread(null)
    setThreadId(null)
    setPendingUser(null)
    setDraft('')
    setNotice(null)
    window.history.replaceState(null, '', window.location.pathname)
  }, [run])

  /* --------------------------------------------------- history actions --- */

  // Load the list once signed in, and again whenever a run finishes — that is
  // when a new thread gets its derived title and an old one moves to the top.
  useEffect(() => {
    if (!signedIn) return
    const ctrl = new AbortController()
    void loadHistory(ctrl.signal)
    return () => ctrl.abort()
  }, [signedIn, loadHistory, runState.streaming])

  const openThread = useCallback(
    (id: Id) => {
      if (id === threadId) {
        setHistoryOpen((open) => (window.innerWidth < 768 ? false : open))
        return
      }
      run.stop()
      run.reset()
      setThread(null)
      setPendingUser(null)
      setDraft('')
      setNotice(null)
      setThreadId(id)
      writeThread(id)
      void refetch(id)
      if (window.innerWidth < 768) setHistoryOpen(false)
    },
    [threadId, run, refetch],
  )

  const renameThread = useCallback((id: Id, title: string) => {
    // Move the label first. The request is a formality; if it fails the next
    // load puts the old name back, and nobody was blocked meanwhile.
    setHistory((rows) =>
      rows.map((c) => (c.id === id ? { ...c, title, title_is_derived: false } : c)),
    )
    void api.renameConversation(id, title).catch(() => undefined)
  }, [])

  const deleteThread = useCallback(
    (id: Id) => {
      setHistory((rows) => rows.filter((c) => c.id !== id))
      void api.deleteConversation(id).catch(() => undefined)
      if (id === threadId) startNewThread()
    },
    [threadId, startNewThread],
  )

  /* ------------------------------------------------------------ render --- */

  const me = auth.phase === 'ready' ? auth.me : null

  // A dot on the account row, and nothing else. The detail is one click away
  // in "Your information"; a permanent strip across the top was a line people
  // learned to look past, which is the same as not showing it.
  const syncTrouble = Object.values(sync?.services ?? {}).some(
    (state) =>
      state.consecutive_failures > 0 ||
      Boolean(state.last_error) ||
      Boolean(state.circuit_open_until && Date.parse(state.circuit_open_until) > Date.now()),
  )
  return (
    <div className="flex h-full min-h-0 flex-col bg-canvas text-ink">
      {/* The sign-in card carries its own lockup, so a header above it is the
          same brand twice with nothing to do in between. */}
      {/* ChatGPT-style: with the column open it holds the brand and the
          account, so a header above it would be a second copy of both. It
          comes back when the column is shut, and on a phone where the column
          is a sheet rather than a place. */}
      {auth.phase !== 'signed-out' && (!historyOpen || !signedIn) && (
      <TopBar
        me={me}
        theme={theme}
        onToggleTheme={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
        onNewThread={startNewThread}
        canStartNew={signedIn && Boolean(threadId || messages.length)}
        onSignOut={handleSignOut}
        showHistoryToggle={signedIn}
        historyOpen={historyOpen}
        onToggleHistory={toggleHistory}
        onOpenIntegrations={() => setShowIntegrations(true)}
        onOpenSyncStatus={() => setShowSyncStatus(true)}
      />
      )}

      {auth.phase === 'unreachable' && (
        <Banner tone="bad">
          <span>{auth.message} Nothing can run until it is back.</span>
          <button
            type="button"
            onClick={() => void loadMe()}
            className="rounded-md bg-bad px-2.5 py-1 text-xs font-medium text-white"
          >
            Try again
          </button>
        </Banner>
      )}

      {/* Never connected and expired are different problems. Telling somebody
          who just signed up that their connection "needs renewing" sends them
          looking for something they never had. */}
      {signedIn && needsReauth && !me?.google.connected && (
        <Banner tone="accent">
          <span>No workspace connected yet.</span>
          <button
            type="button"
            onClick={() => setShowIntegrations(true)}
            className="rounded-md bg-accent px-2.5 py-1 text-xs font-medium text-accent-ink"
          >
            Connect
          </button>
        </Banner>
      )}

      {signedIn && needsReauth && me?.google.connected && (
        <Banner tone="warn">
          <span>Your Google connection expired.</span>
          {/* A link, not a fetch: the response is a redirect to the consent screen. */}
          <a
            href={googleAuthUrl('/')}
            className="rounded-md bg-warn px-2.5 py-1 text-xs font-medium text-white"
          >
            Reconnect
          </a>
        </Banner>
      )}

      {signedIn && (
        <>

          {/* Two columns where there is room. Narrower than that, the trace
              slides over the chat instead of squeezing it to nothing. */}
          <main className="relative flex min-h-0 flex-1 overflow-hidden">
            <History
              header={
                <div className="flex items-center gap-2 px-3 pb-1 pt-3">
                  <span
                    aria-hidden
                    className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-accent text-xs font-bold text-accent-ink"
                  >
                    R
                  </span>
                  <span className="text-sm font-semibold text-ink">Ryuk</span>
                  <button
                    type="button"
                    onClick={toggleHistory}
                    aria-label="Hide chats"
                    className="ml-auto rounded-lg p-1.5 text-muted transition hover:bg-elevated hover:text-ink"
                  >
                    <SidebarIcon />
                  </button>
                </div>
              }
              footer={
                me ? (
                  <div className="border-t border-line p-2">
                    <UserMenu
                      email={me.user.email}
                      displayName={me.user.display_name}
                      theme={theme}
                      onToggleTheme={() => setTheme((t) => (t === 'dark' ? 'light' : 'dark'))}
                      onSignOut={handleSignOut}
                      onOpenIntegrations={() => setShowIntegrations(true)}
                      onOpenSyncStatus={() => setShowSyncStatus(true)}
                      needsAttention={syncTrouble}
                      variant="row"
                    />
                  </div>
                ) : null
              }
              items={history}
              activeId={threadId}
              open={historyOpen}
              loading={historyLoading}
              onSelect={openThread}
              onNew={() => {
                startNewThread()
                if (window.innerWidth < 768) setHistoryOpen(false)
              }}
              onRename={renameThread}
              onDelete={deleteThread}
              onClose={() => setHistoryOpen(false)}
            />

            <div className="flex min-w-0 flex-1 flex-col bg-surface">
              <Chat
                messages={messages}
                run={runState}
                steps={run.steps}
                prompts={prompts}
                actions={actions}
                draft={draft}
                onDraftChange={setDraft}
                composerEnabled={!needsReauth && online}
                busy={runState.streaming}
                onSend={send}
                onRespond={respond}
                onCancelPrompt={cancelPrompt}
                onSearchDifferently={searchDifferently}
                onRedo={redo}
                notice={notice}
                emptyHint={EMPTY_HINTS}
                remaining={me?.limits?.remaining_this_hour}
                limit={me?.limits?.queries_per_hour}
              />
            </div>

          </main>
        </>
      )}

      {showSyncStatus && <SyncStatusPanel onClose={() => setShowSyncStatus(false)} />}

      {showIntegrations && me && (
        <Integrations
          connected={me.google.connected}
          needsReauth={needsReauth}
          accountEmail={me.google.account_email}
          onClose={() => setShowIntegrations(false)}
          onDisconnect={async () => {
            setShowIntegrations(false)
            try {
              await api.disconnectGoogle(true)
            } finally {
              await loadMe()
            }
          }}
        />
      )}

      {auth.phase === 'signed-out' && (
        <SignIn
          initialEmail={lastEmail}
          onSignedIn={(me) => {
            setAuth({ phase: 'ready', me })
          }}
        />
      )}

      {auth.phase === 'loading' && (
        <div className="flex flex-1 items-center justify-center">
          <p className="text-sm text-muted">Checking your session…</p>
        </div>
      )}
    </div>
  )
}

/* ------------------------------------------------------------------ top bar */

function TopBar({
  me,
  theme,
  onToggleTheme,
  onNewThread,
  canStartNew,
  onSignOut,
  showHistoryToggle,
  historyOpen,
  onToggleHistory,
  onOpenIntegrations,
  onOpenSyncStatus,
}: {
  me: MeResponse | null
  theme: 'light' | 'dark'
  onToggleTheme: () => void
  onNewThread: () => void
  canStartNew: boolean
  onSignOut?: () => void
  showHistoryToggle?: boolean
  historyOpen?: boolean
  onToggleHistory?: () => void
  onOpenIntegrations?: () => void
  onOpenSyncStatus?: () => void
}) {
  return (
    <header className="flex items-center gap-3 border-b border-line bg-surface px-3 py-2.5 sm:px-4">
      {showHistoryToggle && (
        <button
          type="button"
          onClick={onToggleHistory}
          aria-label={historyOpen ? 'Hide conversations' : 'Show conversations'}
          aria-expanded={historyOpen}
          className="-ml-1 shrink-0 rounded-lg p-2 text-muted transition hover:bg-elevated hover:text-ink"
        >
          <SidebarIcon />
        </button>
      )}
      <span
        aria-hidden
        className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg bg-accent text-sm font-bold text-accent-ink"
      >
        R
      </span>
      <div className="min-w-0 leading-tight">
        <h1 className="truncate text-sm font-semibold text-ink">Ryuk</h1>
        <p className="hidden truncate text-2xs text-muted sm:block">AI Assistant</p>
      </div>

      <div className="ml-auto flex items-center gap-2">
        <button
          type="button"
          onClick={onNewThread}
          disabled={!canStartNew}
          className={`rounded-lg px-2.5 py-1.5 text-xs text-muted transition hover:bg-elevated disabled:opacity-40 ${
            historyOpen ? 'md:hidden' : ''
          }`}
        >
          New chat
        </button>

        {me && (
          <UserMenu
            email={me.user.email}
            displayName={me.user.display_name}
            theme={theme}
            onToggleTheme={onToggleTheme}
            onSignOut={onSignOut ?? (() => {})}
            onOpenIntegrations={onOpenIntegrations}
            onOpenSyncStatus={onOpenSyncStatus}
          />
        )}
      </div>
    </header>
  )
}

const BANNER_SKIN: Record<'bad' | 'warn' | 'accent', string> = {
  bad: 'border-bad/30 bg-bad/10 text-bad',
  warn: 'border-warn/30 bg-warn/10 text-warn',
  accent: 'border-accent/30 bg-accent/10 text-ink',
}

function Banner({ tone, children }: { tone: 'bad' | 'warn' | 'accent'; children: ReactNode }) {
  return (
    <div
      className={`flex flex-wrap items-center gap-2 border-b px-4 py-2 text-sm sm:px-6 ${BANNER_SKIN[tone]}`}
    >
      {children}
    </div>
  )
}

