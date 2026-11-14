import { render } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { vi } from 'vitest'
import App from '../App'

export interface Call {
  url: string
  method: string
  body?: unknown
}

type Handler = (call: Call) => { status?: number; json?: unknown }

// fetch simulado: busca el primer handler cuyo "METODO /ruta" coincide
export function mockApi(handlers: Record<string, Handler | unknown>) {
  const calls: Call[] = []
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    const method = (init?.method ?? 'GET').toUpperCase()
    let body: unknown = init?.body
    if (typeof body === 'string') body = JSON.parse(body)
    const call: Call = { url, method, body }
    calls.push(call)
    const key = Object.keys(handlers).find((k) => k === `${method} ${url}`)
    if (!key) return new Response(JSON.stringify({ detail: 'no mock' }), { status: 404 })
    const h = handlers[key]
    const res = typeof h === 'function' ? (h as Handler)(call) : (h as { status?: number; json?: unknown })
    const status = res.status ?? 200
    return new Response(status === 204 ? null : JSON.stringify(res.json ?? null), {
      status,
      headers: { 'Content-Type': 'application/json' },
    })
  })
  vi.stubGlobal('fetch', fn)
  return calls
}

export function renderApp(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  )
}

export const reader = { id: 2, username: 'lector', full_name: 'Ana Lectora', role: 'reader' }
