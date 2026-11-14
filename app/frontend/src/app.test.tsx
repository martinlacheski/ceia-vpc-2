import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mockApi, renderApp, reader } from './test/helpers'

// Leaflet no se ejercita en jsdom: se reemplaza el mapa por un stub
vi.mock('./components/MeterMap', () => ({ default: () => <div data-testid="map" /> }))

afterEach(() => {
  vi.unstubAllGlobals()
})

describe('login', () => {
  it('muestra el detalle del 401 al fallar el ingreso', async () => {
    mockApi({
      'GET /api/auth/me': { status: 401, json: { detail: 'No autenticado' } },
      'POST /api/auth/login': {
        status: 401,
        json: { detail: 'Usuario o contraseña incorrectos' },
      },
    })
    renderApp('/')
    const user = userEvent.setup()
    await user.type(await screen.findByLabelText('Usuario'), 'lector')
    await user.type(screen.getByLabelText('Contraseña'), 'mala')
    await user.click(screen.getByRole('button', { name: 'Ingresar' }))
    expect(await screen.findByText('Usuario o contraseña incorrectos')).toBeInTheDocument()
  })
})

describe('vista de ruta', () => {
  it('lista medidores con insignia Pendiente / Leído', async () => {
    mockApi({
      'GET /api/auth/me': { json: reader },
      'GET /api/routes': {
        json: [
          { id: 1, name: 'Ruta Norte', assignee: { id: 2, full_name: 'Ana Lectora' }, meters_total: 2, meters_read: 1 },
        ],
      },
      'GET /api/routes/1/meters': {
        json: [
          { id: 10, code: 'M-010', address: 'Calle 1', lat: -34.6, lon: -58.4, status: 'pending', last_reading: null },
          {
            id: 11, code: 'M-011', address: 'Calle 2', lat: -34.61, lon: -58.41, status: 'read',
            last_reading: { value: '01234', taken_at: '2026-01-01T10:00:00Z' },
          },
        ],
      },
    })
    renderApp('/routes/1')
    expect(await screen.findByText('M-010')).toBeInTheDocument()
    expect(screen.getByText('Pendiente')).toBeInTheDocument()
    expect(screen.getByText('Leído')).toBeInTheDocument()
    expect(screen.getByText(/01234/)).toBeInTheDocument()
  })
})

describe('tomar lectura', () => {
  const meter = {
    id: 5, code: 'M-005', address: 'Av. Siempreviva 742', lat: -34.6, lon: -58.4,
    route_id: 1, status: 'pending', last_reading: null, has_sample_photo: false,
  }
  const analysis = {
    analysis_id: 'abc', display_found: true, reading: '00123', confidence: 0.91, flipped: false,
    digits_model_available: true,
    images: { overlay: 'data:image/jpeg;base64,AA==', crop: 'data:image/jpeg;base64,AA==', digits: 'data:image/jpeg;base64,AA==' },
  }
  const saved = {
    id: 1, meter_id: 5, value: '00125', proposed_value: '00123', corrected: true, confidence: 0.91,
    lat: -34.6, lon: -58.4, accuracy_m: 8, taken_at: '2026-01-01T10:00:00Z',
    user: { id: 2, full_name: 'Ana Lectora' },
  }

  beforeEach(() => {
    vi.stubGlobal('navigator', { ...navigator })
  })

  async function analyzeAndEdit() {
    const user = userEvent.setup()
    const file = new File(['x'], 'foto.jpg', { type: 'image/jpeg' })
    await user.upload(await screen.findByLabelText(/Foto del medidor/), file)
    const input = await screen.findByLabelText('Lectura')
    expect(input).toHaveValue('00123')
    await user.clear(input)
    await user.type(input, '00125')
    await user.click(screen.getByRole('button', { name: 'Confirmar lectura' }))
    return user
  }

  it('envía el valor editado con coordenadas si hay geolocalización', async () => {
    const calls = mockApi({
      'GET /api/auth/me': { json: reader },
      'GET /api/meters/5': { json: meter },
      'POST /api/meters/5/analyze': { json: analysis },
      'POST /api/meters/5/readings': { status: 201, json: saved },
    })
    Object.defineProperty(navigator, 'geolocation', {
      configurable: true,
      value: {
        getCurrentPosition: (ok: PositionCallback) =>
          ok({ coords: { latitude: -34.6, longitude: -58.4, accuracy: 8 } } as GeolocationPosition),
      },
    })
    renderApp('/meters/5/read')
    await analyzeAndEdit()
    await waitFor(() => expect(calls.some((c) => c.url === '/api/meters/5/readings')).toBe(true))
    const post = calls.find((c) => c.url === '/api/meters/5/readings')!
    expect(post.body).toEqual({ analysis_id: 'abc', value: '00125', lat: -34.6, lon: -58.4, accuracy_m: 8 })
    expect(await screen.findByText(/corregida/i)).toBeInTheDocument()
    expect(screen.getByText(/Ubicación de la foto: -34\.60000, -58\.40000 \(±8 m\)/)).toBeInTheDocument()
    expect(screen.queryByText(/lejos del medidor/i)).not.toBeInTheDocument()
  })

  it('envía la lectura sin coordenadas si se deniega la geolocalización', async () => {
    const calls = mockApi({
      'GET /api/auth/me': { json: reader },
      'GET /api/meters/5': { json: meter },
      'POST /api/meters/5/analyze': { json: analysis },
      'POST /api/meters/5/readings': { status: 201, json: { ...saved, lat: null, lon: null, accuracy_m: null } },
    })
    Object.defineProperty(navigator, 'geolocation', {
      configurable: true,
      value: {
        getCurrentPosition: (_ok: PositionCallback, err: PositionErrorCallback) =>
          err({ code: 1, message: 'denied' } as GeolocationPositionError),
      },
    })
    renderApp('/meters/5/read')
    await analyzeAndEdit()
    await waitFor(() => expect(calls.some((c) => c.url === '/api/meters/5/readings')).toBe(true))
    const post = calls.find((c) => c.url === '/api/meters/5/readings')!
    expect(post.body).toEqual({ analysis_id: 'abc', value: '00125' })
    expect(await screen.findByText(/Sin ubicación de la foto/)).toBeInTheDocument()
  })
})
