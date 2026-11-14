import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mockApi, renderApp, reader } from './test/helpers'

// Leaflet no se ejercita en jsdom: se reemplaza el mapa de lecturas por un stub
vi.mock('./components/ReadingsMap', () => ({
  default: ({ readings, onOpen }: { readings: { id: number; value: string }[]; onOpen: (r: unknown) => void }) => (
    <div data-testid="readings-map">
      {readings.map((r) => (
        <button key={r.id} onClick={() => onOpen(r)}>
          {`Abrir lectura ${r.id} del mapa`}
        </button>
      ))}
    </div>
  ),
}))

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

  it('no ofrece mapa de medidores: solo el listado', async () => {
    mockApi({
      'GET /api/auth/me': { json: reader },
      'GET /api/routes': { json: [{ id: 1, name: 'Ruta Norte', assignee: { id: 2, full_name: 'Ana Lectora' }, meters_total: 1, meters_read: 0 }] },
      'GET /api/routes/1/meters': {
        json: [{ id: 10, code: 'M-010', address: 'Calle 1', lat: -34.6, lon: -58.4, status: 'pending', last_reading: null }],
      },
    })
    renderApp('/routes/1')
    expect(await screen.findByText('M-010')).toBeInTheDocument()
    expect(screen.queryByRole('tab', { name: 'Mapa' })).not.toBeInTheDocument()
    expect(screen.queryByText('Mapa')).not.toBeInTheDocument()
    expect(screen.queryByText(/-34\.6/)).not.toBeInTheDocument()
  })
})

const admin = { id: 1, username: 'admin', full_name: 'Admin Uno', role: 'admin' }
const baseReading = {
  id: 7, meter_id: 10, value: '01234', proposed_value: '01233', corrected: true, confidence: 0.9,
  lat: -34.6, lon: -58.4, accuracy_m: 12, taken_at: '2026-01-01T10:00:00Z',
  user: { id: 2, full_name: 'Ana Lectora' },
}
function adminApi(readings: unknown[]) {
  return mockApi({
    'GET /api/auth/me': { json: admin },
    'GET /api/stats': { json: { readings_total: readings.length, accepted_without_correction_pct: 50, meters_total: 2, meters_read: 1 } },
    'GET /api/routes': { json: [] },
    'GET /api/readings': { json: readings },
  })
}

describe('panel de administración', () => {
  it('abre la foto en un modal con las imágenes procesadas y lo cierra con Esc', async () => {
    adminApi([baseReading])
    renderApp('/admin')
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: /Ver foto de la lectura 7/ }))
    const dialog = await screen.findByRole('dialog')
    expect(dialog).toHaveAttribute('aria-modal', 'true')
    expect(within(dialog).getByAltText('Foto de la lectura 7')).toHaveAttribute('src', '/api/readings/7/photo')
    for (const kind of ['overlay', 'crop', 'digits']) {
      expect(dialog.querySelector(`img[src="/api/readings/7/images/${kind}"]`)).not.toBeNull()
    }
    expect(within(dialog).getByText('01234')).toBeInTheDocument()
    expect(within(dialog).getByText('corregida')).toBeInTheDocument()
    expect(within(dialog).getByText(/90%/)).toBeInTheDocument()
    expect(within(dialog).getAllByText(/Ana Lectora/).length).toBeGreaterThan(0)
    expect(within(dialog).getByRole('link', { name: /-34\.60000, -58\.40000 \(±12 m\)/ })).toHaveAttribute(
      'href',
      'https://www.google.com/maps/@?api=1&map_action=map&center=-34.6,-58.4&zoom=19&basemap=satellite',
    )
    expect(within(dialog).getByRole('link', { name: /-34\.60000/ })).toHaveAttribute('target', '_blank')
    expect(within(dialog).getByRole('link', { name: /-34\.60000/ })).toHaveAttribute('rel', 'noreferrer')
    await user.keyboard('{Escape}')
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
  })

  it('oculta una imagen procesada que da 404', async () => {
    adminApi([baseReading])
    renderApp('/admin')
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: /Ver foto de la lectura 7/ }))
    const dialog = await screen.findByRole('dialog')
    fireEvent.error(dialog.querySelector('img[src="/api/readings/7/images/crop"]')!)
    expect(dialog.querySelector('img[src="/api/readings/7/images/crop"]')).toBeNull()
    expect(dialog.querySelector('img[src="/api/readings/7/images/overlay"]')).not.toBeNull()
  })

  it('muestra el mapa de lecturas (sin mapa de medidores) y abre el modal desde el mapa', async () => {
    adminApi([baseReading, { ...baseReading, id: 8, lat: null, lon: null, accuracy_m: null }])
    renderApp('/admin')
    const user = userEvent.setup()
    expect(await screen.findByRole('heading', { name: 'Mapa de lecturas' })).toBeInTheDocument()
    expect(screen.queryByText(/Mapa de medidores/)).not.toBeInTheDocument()
    expect(await screen.findByTestId('readings-map')).toHaveTextContent('Abrir lectura 7 del mapa')
    expect(screen.getByTestId('readings-map')).not.toHaveTextContent('lectura 8')
    await user.click(screen.getByRole('button', { name: 'Abrir lectura 7 del mapa' }))
    expect(await screen.findByRole('dialog')).toBeInTheDocument()
  })

  it('muestra un aviso en vez del mapa si ninguna lectura tiene ubicación', async () => {
    adminApi([{ ...baseReading, lat: null, lon: null, accuracy_m: null }])
    renderApp('/admin')
    expect(await screen.findByText(/Ninguna lectura tiene ubicación/)).toBeInTheDocument()
    expect(screen.queryByTestId('readings-map')).not.toBeInTheDocument()
  })

  it('marca "revisar" solo las lecturas con confianza menor a 0,5', async () => {
    adminApi([
      { ...baseReading, id: 1, confidence: 0.3 },
      { ...baseReading, id: 2, confidence: 0.5 },
      { ...baseReading, id: 3, confidence: null },
    ])
    renderApp('/admin')
    expect(await screen.findAllByText('revisar')).toHaveLength(1)
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

  // GPS simulado: watchPosition entrega las posiciones en orden (o un error) y registra las opciones pedidas
  function mockGps(fixes: { lat: number; lon: number; accuracy: number }[] | 'denied') {
    const options: PositionOptions[] = []
    Object.defineProperty(navigator, 'geolocation', {
      configurable: true,
      value: {
        watchPosition: (ok: PositionCallback, err: PositionErrorCallback, opts: PositionOptions) => {
          options.push(opts)
          if (fixes === 'denied') err({ code: 1, message: 'denied' } as GeolocationPositionError)
          else
            for (const f of fixes)
              ok({ coords: { latitude: f.lat, longitude: f.lon, accuracy: f.accuracy } } as GeolocationPosition)
          return 1
        },
        clearWatch: () => {},
        getCurrentPosition: (_ok: PositionCallback, err: PositionErrorCallback) =>
          err({ code: 2, message: 'unavailable' } as GeolocationPositionError),
      },
    })
    return options
  }

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
    mockGps([{ lat: -34.6, lon: -58.4, accuracy: 8 }])
    renderApp('/meters/5/read')
    await analyzeAndEdit()
    await waitFor(() => expect(calls.some((c) => c.url === '/api/meters/5/readings')).toBe(true))
    const post = calls.find((c) => c.url === '/api/meters/5/readings')!
    expect(post.body).toEqual({ analysis_id: 'abc', value: '00125', lat: -34.6, lon: -58.4, accuracy_m: 8 })
    expect(await screen.findByText(/corregida/i)).toBeInTheDocument()
    expect(screen.getByText(/Ubicación de la foto: -34\.60000, -58\.40000 \(±8 m\)/)).toBeInTheDocument()
    expect(screen.queryByText(/lejos del medidor/i)).not.toBeInTheDocument()
  })

  it.each([
    [0.3, true],
    [0.9, false],
  ])('con confianza %s el aviso de lectura baja se muestra: %s', async (confidence, shown) => {
    mockApi({
      'GET /api/auth/me': { json: reader },
      'GET /api/meters/5': { json: meter },
      'POST /api/meters/5/analyze': { json: { ...analysis, confidence } },
    })
    renderApp('/meters/5/read')
    const user = userEvent.setup()
    await user.upload(await screen.findByLabelText(/Foto del medidor/), new File(['x'], 'foto.jpg', { type: 'image/jpeg' }))
    expect(await screen.findByLabelText('Lectura')).toHaveValue('00123')
    const warning = screen.queryByText(/Lectura con confianza baja/)
    if (shown) expect(warning).toBeInTheDocument()
    else expect(warning).not.toBeInTheDocument()
  })

  it('no muestra mapa de medidores ni coordenadas del medidor', async () => {
    mockApi({ 'GET /api/auth/me': { json: reader }, 'GET /api/meters/5': { json: meter } })
    renderApp('/meters/5/read')
    expect(await screen.findByText('M-005')).toBeInTheDocument()
    expect(document.querySelector('.leaflet-container')).toBeNull()
  })

  it('sigue el GPS desde que se abre la pantalla y envía la posición más precisa', async () => {
    const calls = mockApi({
      'GET /api/auth/me': { json: reader },
      'GET /api/meters/5': { json: meter },
      'POST /api/meters/5/analyze': { json: analysis },
      'POST /api/meters/5/readings': { status: 201, json: { ...saved, accuracy_m: 6 } },
    })
    const options = mockGps([
      { lat: -34.61, lon: -58.41, accuracy: 40 },
      { lat: -34.6, lon: -58.4, accuracy: 6 },
      { lat: -34.62, lon: -58.42, accuracy: 15 },
    ])
    renderApp('/meters/5/read')
    expect(await screen.findByText('GPS: ±6 m')).toBeInTheDocument()
    expect(options[0]).toMatchObject({ enableHighAccuracy: true, maximumAge: 0 })
    await analyzeAndEdit()
    await waitFor(() => expect(calls.some((c) => c.url === '/api/meters/5/readings')).toBe(true))
    const post = calls.find((c) => c.url === '/api/meters/5/readings')!
    expect(post.body).toEqual({ analysis_id: 'abc', value: '00125', lat: -34.6, lon: -58.4, accuracy_m: 6 })
  })

  it('avisa cuando la ubicación no está disponible', async () => {
    mockApi({ 'GET /api/auth/me': { json: reader }, 'GET /api/meters/5': { json: meter } })
    mockGps('denied')
    renderApp('/meters/5/read')
    expect(await screen.findByText(/Ubicación no disponible/)).toBeInTheDocument()
  })

  it('envía la lectura sin coordenadas si se deniega la geolocalización', async () => {
    const calls = mockApi({
      'GET /api/auth/me': { json: reader },
      'GET /api/meters/5': { json: meter },
      'POST /api/meters/5/analyze': { json: analysis },
      'POST /api/meters/5/readings': { status: 201, json: { ...saved, lat: null, lon: null, accuracy_m: null } },
    })
    mockGps('denied')
    renderApp('/meters/5/read')
    await analyzeAndEdit()
    await waitFor(() => expect(calls.some((c) => c.url === '/api/meters/5/readings')).toBe(true))
    const post = calls.find((c) => c.url === '/api/meters/5/readings')!
    expect(post.body).toEqual({ analysis_id: 'abc', value: '00125' })
    expect(await screen.findByText(/Sin ubicación de la foto/)).toBeInTheDocument()
  })
})
