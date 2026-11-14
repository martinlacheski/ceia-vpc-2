import { useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../api'
import { useLoad, useWatchPosition } from '../hooks'
import { formatDate } from '../format'
import StatusBadge from '../components/StatusBadge'
import MeterMap from '../components/MeterMap'
import ProgressBar from '../components/ProgressBar'

export default function RouteView() {
  const routeId = Number(useParams().routeId)
  const [tab, setTab] = useState<'list' | 'map'>('list')
  const meters = useLoad(() => api.routeMeters(routeId), [routeId])
  const routes = useLoad(() => api.routes(), [])
  const { position, denied } = useWatchPosition()
  const route = routes.data?.find((r) => r.id === routeId)

  return (
    <>
      <Link to="/" className="back">
        ← Mis rutas
      </Link>
      <h1>{route?.name ?? 'Ruta'}</h1>
      {route && (
        <div className="stack-sm">
          <ProgressBar value={route.meters_read} total={route.meters_total} />
          <span className="muted small">
            {route.meters_read} de {route.meters_total} medidores leídos
          </span>
        </div>
      )}
      <div className="tabs" role="tablist">
        <button role="tab" aria-selected={tab === 'list'} className={tab === 'list' ? 'tab active' : 'tab'} onClick={() => setTab('list')}>
          Listado
        </button>
        <button role="tab" aria-selected={tab === 'map'} className={tab === 'map' ? 'tab active' : 'tab'} onClick={() => setTab('map')}>
          Mapa
        </button>
      </div>
      {meters.loading && <p className="muted">Cargando…</p>}
      {meters.error && <p className="alert alert-error">{meters.error}</p>}
      {meters.data && tab === 'list' && (
        <ul className="stack list-plain">
          {meters.data.map((m) => (
            <li key={m.id}>
              <Link to={`/meters/${m.id}/read`} className="card card-link">
                <div className="row-between">
                  <strong>{m.code}</strong>
                  <StatusBadge status={m.status} />
                </div>
                <span>{m.address}</span>
                {m.last_reading && (
                  <span className="muted small">
                    Última lectura: <b>{m.last_reading.value}</b> · {formatDate(m.last_reading.taken_at)}
                  </span>
                )}
              </Link>
            </li>
          ))}
        </ul>
      )}
      {meters.data && tab === 'map' && (
        <>
          {denied && <p className="alert alert-warn">No se pudo obtener tu ubicación; el mapa muestra solo los medidores.</p>}
          <MeterMap
            meters={meters.data}
            position={position}
            height={440}
            renderPopup={(m) => (
              <div>
                <strong>{m.code}</strong>
                <br />
                {m.address}
                <br />
                <Link to={`/meters/${m.id}/read`}>Tomar lectura</Link>
              </div>
            )}
          />
          <p className="muted small legend">
            <span className="dot dot-pending" /> Pendiente <span className="dot dot-read" /> Leído <span className="dot dot-me" /> Vos
          </p>
        </>
      )}
    </>
  )
}
