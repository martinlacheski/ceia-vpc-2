// Cliente de la API y tipos del contrato (prefijo /api, sesión por cookie)

export interface User {
  id: number
  username: string
  full_name: string
  role: 'reader' | 'admin'
}

export interface RouteSummary {
  id: number
  name: string
  assignee: { id: number; full_name: string }
  meters_total: number
  meters_read: number
}

export interface LastReading {
  value: string
  taken_at: string
}

export type Service = 'water' | 'energy'

export const SERVICE_LABEL: Record<Service, string> = { water: 'Agua', energy: 'Energía' }

export interface MeterItem {
  id: number
  code: string
  service: Service
  address: string
  lat: number
  lon: number
  status: 'pending' | 'read'
  last_reading: LastReading | null
}

export interface MeterDetail extends MeterItem {
  route_id: number
  has_sample_photo: boolean
}

export interface Analysis {
  analysis_id: string
  display_found: boolean
  reading: string | null
  confidence: number | null
  flipped: boolean
  digits_model_available: boolean
  images: { overlay: string | null; crop: string | null; digits: string | null }
}

export interface ReadingPayload {
  analysis_id: string
  value: string
  lat?: number
  lon?: number
  accuracy_m?: number
}

export interface Reading {
  id: number
  meter_id: number
  value: string
  proposed_value: string | null
  corrected: boolean
  confidence: number | null
  lat: number | null
  lon: number | null
  accuracy_m: number | null
  taken_at: string
  user: { id: number; full_name: string }
  route: { id: number; name: string }
  final_value: string // valor vigente: el de la revisión del admin si la hubo, si no el del lecturista
  review: ReadingReview | null
}

export interface Stats {
  readings_total: number
  accepted_without_correction_pct: number | null
  meters_total: number
  meters_read: number
}

export interface ReadingReview {
  value: string
  reviewed_by: { id: number; full_name: string }
  reviewed_at: string
}

export interface ReadingsPage {
  items: Reading[]
  total: number
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  return (await requestWithHeaders<T>(path, init)).data
}

async function requestWithHeaders<T>(path: string, init: RequestInit = {}): Promise<{ data: T; headers: Headers }> {
  const res = await fetch(path, { credentials: 'include', ...init })
  if (!res.ok) {
    let detail = `Error ${res.status}`
    try {
      const data = await res.json()
      if (typeof data?.detail === 'string') detail = data.detail
      else if (Array.isArray(data?.detail) && data.detail[0]?.msg) detail = data.detail[0].msg
    } catch {
      /* sin cuerpo JSON */
    }
    throw new ApiError(res.status, detail)
  }
  if (res.status === 204) return { data: undefined as T, headers: res.headers }
  return { data: (await res.json()) as T, headers: res.headers }
}

const json = (body: unknown): RequestInit => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
})

export const api = {
  me: () => request<User>('/api/auth/me').then(unwrapUser),
  login: (username: string, password: string) =>
    request<User>('/api/auth/login', json({ username, password })).then(unwrapUser),
  logout: () => request<void>('/api/auth/logout', { method: 'POST' }),
  routes: () => request<RouteSummary[]>('/api/routes'),
  routeMeters: (id: number) => request<MeterItem[]>(`/api/routes/${id}/meters`),
  meter: (id: number) => request<MeterDetail>(`/api/meters/${id}`),
  samplePhoto: async (id: number): Promise<Blob> => {
    const res = await fetch(`/api/meters/${id}/sample-photo`, { credentials: 'include' })
    if (!res.ok) throw new ApiError(res.status, 'No se pudo obtener la foto de ejemplo')
    return res.blob()
  },
  analyze: (id: number, photo: Blob) => {
    const form = new FormData()
    form.append('photo', photo, 'foto.jpg')
    return request<Analysis>(`/api/meters/${id}/analyze`, { method: 'POST', body: form })
  },
  saveReading: (id: number, payload: ReadingPayload) =>
    request<Reading>(`/api/meters/${id}/readings`, json(payload)),
  readings: (params: { route_id?: number; meter_id?: number } = {}) => {
    const q = new URLSearchParams()
    if (params.route_id) q.set('route_id', String(params.route_id))
    if (params.meter_id) q.set('meter_id', String(params.meter_id))
    const s = q.toString()
    return request<Reading[]>(`/api/readings${s ? `?${s}` : ''}`)
  },
  // Una página del historial; el total viene en el encabezado X-Total-Count
  readingsPage: async (params: { route_id?: number; limit: number; offset: number }): Promise<ReadingsPage> => {
    const q = new URLSearchParams()
    if (params.route_id) q.set('route_id', String(params.route_id))
    q.set('limit', String(params.limit))
    q.set('offset', String(params.offset))
    const { data, headers } = await requestWithHeaders<Reading[]>(`/api/readings?${q}`)
    return { items: data, total: Number(headers.get('X-Total-Count') ?? data.length) }
  },
  reviewReading: (id: number, value: string) => request<Reading>(`/api/readings/${id}/review`, json({ value })),
  stats: () => request<Stats>('/api/stats'),
}

// el contrato dice `{user}` pero se tolera también el usuario directo
function unwrapUser(data: unknown): User {
  const d = data as { user?: User }
  return (d.user ?? data) as User
}
