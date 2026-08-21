import { existsSync } from 'node:fs'
import type { IncomingMessage, ServerResponse } from 'node:http'

import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv } from 'vite'

/**
 * Two things here are load-bearing.
 *
 * 1. The `/api` proxy. It keeps the browser on one origin, so the httpOnly
 *    `alpha_session` cookie is sent with every request without weakening
 *    SameSite just to develop.
 *
 * 2. The SSE carve-out inside `configure()`. http-proxy buffers by default and
 *    a compressed response buffers even harder, so `run.started`, `probe.done`,
 *    `plan.step` … all arrive in one clump when the run ends and the trace panel
 *    looks broken. For anything under `/events` we ask upstream for identity
 *    encoding, strip the buffering headers off the response, and flush the head
 *    to the browser before the first chunk.
 */
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')

  // Inside compose the API is a service name; on a laptop it is localhost.
  // compose passes VITE_API_URL, which is written from the host's point of
  // view — inside the container `localhost` is the container itself, so the
  // host is rewritten to the service name rather than quietly proxying to
  // nothing.
  const inDocker = existsSync('/.dockerenv')
  const configured = env.VITE_API_TARGET || env.VITE_API_URL
  const target = configured
    ? inDocker
      ? configured.replace(/\/\/(localhost|127\.0\.0\.1|0\.0\.0\.0)(:|\/|$)/, '//api$2')
      : configured
    : inDocker
      ? 'http://api:8000'
      : 'http://localhost:8000'

  const isStream = (url: string | undefined) =>
    typeof url === 'string' && (url.includes('/events') || url.includes('stream=ndjson'))

  return {
    plugins: [react()],
    server: {
      // Required, or Vite binds 127.0.0.1 and the port mapping out of the
      // container reaches nothing.
      host: true,
      port: 5173,
      strictPort: true,
      // compose sets CHOKIDAR_USEPOLLING; Vite does not read it on its own.
      watch: env.CHOKIDAR_USEPOLLING ? { usePolling: true, interval: 300 } : undefined,
      proxy: {
        '/api': {
          target,
          changeOrigin: true,
          ws: false,
          // No timeout: an SSE stream is meant to stay open.
          proxyTimeout: 0,
          timeout: 0,
          configure(proxy) {
            proxy.on('proxyReq', (proxyReq, req: IncomingMessage) => {
              if (!isStream(req.url)) return
              // gzip would sit in a buffer waiting for a full window.
              proxyReq.setHeader('Accept-Encoding', 'identity')
              proxyReq.setHeader('Connection', 'keep-alive')
              proxyReq.setHeader('Cache-Control', 'no-cache')
            })

            proxy.on(
              'proxyRes',
              (proxyRes, req: IncomingMessage, res: ServerResponse) => {
                if (!isStream(req.url)) return
                proxyRes.headers['cache-control'] = 'no-cache, no-transform'
                proxyRes.headers['x-accel-buffering'] = 'no'
                proxyRes.headers['connection'] = 'keep-alive'
                delete proxyRes.headers['content-encoding']
                delete proxyRes.headers['content-length']
                // http-proxy writes the head *after* this listener returns, so
                // the flush has to wait a tick or it sends an empty head.
                setImmediate(() => {
                  if (!res.writableEnded && typeof res.flushHeaders === 'function') {
                    res.flushHeaders()
                  }
                })
              },
            )

            proxy.on('error', (err, _req, res) => {
              // A dead API should not take the dev server down with it.
              const out = res as ServerResponse | undefined
              if (out && 'writeHead' in out && !out.headersSent) {
                out.writeHead(502, { 'content-type': 'application/json' })
                out.end(
                  JSON.stringify({
                    error: {
                      code: 'GOOGLE_UNAVAILABLE',
                      message: `The API at ${target} is not answering (${err.message}).`,
                      details: { target },
                      request_id: 'vite-proxy',
                    },
                  }),
                )
              }
            })
          },
        },
      },
    },
    preview: { host: true, port: 5173 },
    build: { outDir: 'dist', sourcemap: true, target: 'es2020' },
  }
})
