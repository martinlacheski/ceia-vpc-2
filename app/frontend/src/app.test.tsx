import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { mockApi, renderApp, reader } from './test/helpers'

// Leaflet no se ejercita en jsdom: se reemplaza el mapa de lecturas por un stub
vi.mock('./components/ReadingsMap', () => ({
  default: ({ readings, onOpen }: { readings: { id: number; value: string }[]; onOpen?: (r: unknown) => void }) => (
    <div data-testid="readings-map">
      {readings.map((r) =>
        onOpen ? (
          <button key={r.id} onClick={() => onOpen(r)}>
            {`Abrir lectura ${r.id} del mapa`}
          </button>
        ) : (
          <span key={r.id}>{`Lectura ${r.id} en el mapa`}</span>
        ),
      )}
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
          { id: 10, code: 'M-010', service: 'water', address: 'Calle 1', lat: -34.6, lon: -58.4, status: 'pending', last_reading: null },
          {
            id: 11, code: 'E-M-011', service: 'energy', address: 'Calle 2', lat: -34.61, lon: -58.41, status: 'read',
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
    // cada medidor muestra su servicio
    expect(screen.getByText('Agua')).toBeInTheDocument()
    expect(screen.getByText('Energía')).toBeInTheDocument()
  })

  it('no ofrece mapa de medidores: solo el listado', async () => {
    mockApi({
      'GET /api/auth/me': { json: reader },
      'GET /api/routes': { json: [{ id: 1, name: 'Ruta Norte', assignee: { id: 2, full_name: 'Ana Lectora' }, meters_total: 1, meters_read: 0 }] },
      'GET /api/routes/1/meters': {
        json: [{ id: 10, code: 'M-010', service: 'water', address: 'Calle 1', lat: -34.6, lon: -58.4, status: 'pending', last_reading: null }],
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
  route: { id: 1, name: 'Ruta Norte' },
  final_value: '01234',
  review: null,
}
function adminApi(readings: unknown[], extra: Record<string, unknown> = {}) {
  return mockApi({
    ...extra,
    'GET /api/auth/me': { json: admin },
    'GET /api/stats': { json: { readings_total: readings.length, accepted_without_correction_pct: 50, meters_total: 2, meters_read: 1 } },
    'GET /api/routes': { json: [] },
    'GET /api/readings': { json: readings },
    // páginas del historial (20 por página)
    ...Object.fromEntries(
      [0, 20, 40].map((offset) => [
        `GET /api/readings?limit=20&offset=${offset}`,
        { json: readings.slice(offset, offset + 20), headers: { 'X-Total-Count': String(readings.length) } },
      ]),
    ),
  })
}

describe('panel de administración', () => {
  it('muestra en el menú las pestañas Panel, Mapa de lecturas e Historial de lecturas', async () => {
    adminApi([baseReading])
    renderApp('/admin')
    const nav = await screen.findByRole('navigation')
    expect(within(nav).getByRole('link', { name: 'Panel' })).toHaveAttribute('href', '/admin')
    expect(within(nav).getByRole('link', { name: 'Mapa de lecturas' })).toHaveAttribute('href', '/admin/mapa')
    expect(within(nav).getByRole('link', { name: 'Historial de lecturas' })).toHaveAttribute('href', '/admin/historial')
    expect(await screen.findByText('Aceptadas sin corrección')).toBeInTheDocument()
    expect(screen.queryByRole('heading', { name: 'Historial de lecturas' })).not.toBeInTheDocument()
    expect(screen.queryByTestId('readings-map')).not.toBeInTheDocument()
  })

  it('abre la foto en un modal con las imágenes procesadas y lo cierra con Esc', async () => {
    adminApi([baseReading])
    renderApp('/admin/historial')
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: /Ver foto de la lectura 7/ }))
    const dialog = await screen.findByRole('dialog')
    expect(dialog).toHaveAttribute('aria-modal', 'true')
    expect(within(dialog).getByAltText('Foto de la lectura 7')).toHaveAttribute('src', '/api/readings/7/photo')
    expect(within(dialog).getByAltText('Segmentación del visor de la lectura 7')).toBeInTheDocument()
    expect(within(dialog).getByText('Imagen original')).toBeInTheDocument()
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
    expect(within(dialog).getByTestId('readings-map')).toHaveTextContent('Lectura 7 en el mapa')
    await user.keyboard('{Escape}')
    await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
  })

  it('el modal no muestra mapa si la foto no tiene ubicación', async () => {
    adminApi([{ ...baseReading, lat: null, lon: null, accuracy_m: null }])
    renderApp('/admin/historial')
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: /Ver foto de la lectura 7/ }))
    const dialog = await screen.findByRole('dialog')
    expect(within(dialog).queryByTestId('readings-map')).not.toBeInTheDocument()
    expect(within(dialog).getAllByText('sin dato').length).toBeGreaterThan(0)
  })

  it('oculta una imagen procesada que da 404', async () => {
    adminApi([baseReading])
    renderApp('/admin/historial')
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: /Ver foto de la lectura 7/ }))
    const dialog = await screen.findByRole('dialog')
    fireEvent.error(dialog.querySelector('img[src="/api/readings/7/images/crop"]')!)
    expect(dialog.querySelector('img[src="/api/readings/7/images/crop"]')).toBeNull()
    expect(dialog.querySelector('img[src="/api/readings/7/images/overlay"]')).not.toBeNull()
  })

  it('muestra el mapa de lecturas (sin mapa de medidores) y abre el modal desde el mapa', async () => {
    adminApi([baseReading, { ...baseReading, id: 8, lat: null, lon: null, accuracy_m: null }])
    renderApp('/admin/mapa')
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
    renderApp('/admin/mapa')
    expect(await screen.findByText(/Ninguna lectura tiene ubicación/)).toBeInTheDocument()
    expect(screen.queryByTestId('readings-map')).not.toBeInTheDocument()
  })

  it('muestra la ruta de cada lectura antes del medidor', async () => {
    adminApi([baseReading])
    renderApp('/admin/historial')
    const headers = (await screen.findAllByRole('columnheader')).map((h) => h.textContent)
    expect(headers.indexOf('Ruta')).toBe(headers.indexOf('Medidor') - 1)
    const row = (await screen.findByRole('button', { name: /Ver foto de la lectura 7/ })).closest('tr')!
    expect(within(row).getByText('Ruta Norte')).toBeInTheDocument()
  })

  it('pagina el historial de a 20 lecturas', async () => {
    const many = Array.from({ length: 25 }, (_, i) => ({ ...baseReading, id: 100 + i }))
    const calls = adminApi(many)
    renderApp('/admin/historial')
    const user = userEvent.setup()
    expect(await screen.findByText('Mostrando 1–20 de 25')).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: /Ver foto de la lectura/ })).toHaveLength(20)
    expect(screen.getByRole('button', { name: 'Anterior' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: 'Siguiente' }))
    expect(await screen.findByText('Mostrando 21–25 de 25')).toBeInTheDocument()
    expect(screen.getAllByRole('button', { name: /Ver foto de la lectura/ })).toHaveLength(5)
    expect(screen.getByRole('button', { name: 'Siguiente' })).toBeDisabled()
    expect(calls.some((c) => c.url === '/api/readings?limit=20&offset=20')).toBe(true)
  })

  it('muestra una sola observación: "corregida" o, si se aceptó con confianza menor a 0,5, "revisar"', async () => {
    adminApi([
      { ...baseReading, id: 1, corrected: false, confidence: 0.3 },
      { ...baseReading, id: 2, corrected: false, confidence: 0.5 },
      { ...baseReading, id: 3, corrected: false, confidence: null },
      { ...baseReading, id: 4, corrected: true, confidence: 0.25 },
    ])
    renderApp('/admin/historial')
    const row = async (id: number) => (await screen.findByRole('button', { name: `Ver foto de la lectura ${id}` })).closest('tr')!
    expect(within(await row(1)).getByText('revisar')).toBeInTheDocument()
    expect(within(await row(4)).getByText('corregida')).toBeInTheDocument()
    expect(within(await row(4)).queryByText('revisar')).not.toBeInTheDocument()
    expect(screen.getAllByText('revisar')).toHaveLength(1)
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Ver foto de la lectura 4' }))
    const dialog = await screen.findByRole('dialog')
    expect(within(dialog).getByText('corregida')).toBeInTheDocument()
    expect(within(dialog).queryByText('revisar')).not.toBeInTheDocument()
  })

  it('el admin corrige una lectura marcada "revisar" desde el modal', async () => {
    const pending = { ...baseReading, id: 1, corrected: false, confidence: 0.3 }
    const reviewed = {
      ...pending,
      final_value: '01240',
      review: { value: '01240', reviewed_by: { id: 1, full_name: 'Admin Uno' }, reviewed_at: '2026-01-02T10:00:00Z' },
    }
    const calls = adminApi([pending], { 'POST /api/readings/1/review': { json: reviewed } })
    renderApp('/admin/historial')
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: 'Ver foto de la lectura 1' }))
    const dialog = await screen.findByRole('dialog')
    const input = within(dialog).getByLabelText('Valor revisado')
    expect(input).toHaveValue('01234')
    expect(within(dialog).getByRole('button', { name: 'Confirmar valor' })).toBeEnabled()
    await user.clear(input)
    await user.type(input, '12a')
    expect(within(dialog).getByRole('button', { name: 'Guardar corrección' })).toBeDisabled()
    await user.clear(input)
    await user.type(input, '01240')
    await user.click(within(dialog).getByRole('button', { name: 'Guardar corrección' }))
    expect(calls.find((c) => c.method === 'POST')?.body).toEqual({ value: '01240' })
    expect(await within(dialog).findByText('revisada')).toBeInTheDocument()
    expect(within(dialog).queryByLabelText('Valor revisado')).not.toBeInTheDocument()
    expect(within(dialog).getByText(/Admin Uno/)).toBeInTheDocument()
    expect(calls.filter((c) => c.url === '/api/readings?limit=20&offset=0').length).toBeGreaterThan(1)
  })

  it('una lectura revisada muestra el valor final, la marca "revisada" y no ofrece revisar de nuevo', async () => {
    adminApi([
      {
        ...baseReading, id: 1, corrected: false, confidence: 0.3, final_value: '01240',
        review: { value: '01240', reviewed_by: { id: 1, full_name: 'Admin Uno' }, reviewed_at: '2026-01-02T10:00:00Z' },
      },
      { ...baseReading, id: 2 },
    ])
    renderApp('/admin/historial')
    const row = (await screen.findByRole('button', { name: 'Ver foto de la lectura 1' })).closest('tr')!
    expect(within(row).getByText('01240')).toBeInTheDocument()
    expect(within(row).getByText('revisada')).toBeInTheDocument()
    expect(within(row).queryByText('revisar')).not.toBeInTheDocument()
    const user = userEvent.setup()
    await user.click(screen.getByRole('button', { name: 'Ver foto de la lectura 2' }))
    // una lectura corregida por el lecturista tampoco ofrece la revisión
    expect(within(await screen.findByRole('dialog')).queryByLabelText('Valor revisado')).not.toBeInTheDocument()
  })
})

describe('tomar lectura', () => {
  const meter = {
    id: 5, code: 'M-005', service: 'energy', address: 'Av. Siempreviva 742', lat: -34.6, lon: -58.4,
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

  it('muestra el servicio del medidor', async () => {
    mockApi({
      'GET /api/auth/me': { json: reader },
      'GET /api/meters/5': { json: meter },
    })
    renderApp('/meters/5/read')
    expect(await screen.findByText('M-005')).toBeInTheDocument()
    expect(screen.getByText('Energía')).toBeInTheDocument()
  })

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
