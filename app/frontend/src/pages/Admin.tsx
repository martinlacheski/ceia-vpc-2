import { useState } from 'react'
import { api } from '../api'
import { useLoad } from '../hooks'
import { formatDate, formatLocation, mapUrl } from '../format'
import MeterMap from '../components/MeterMap'
import ProgressBar from '../components/ProgressBar'

export default function Admin() {
  const stats = useLoad(() => api.stats(), [])
  const routes = useLoad(() => api.routes(), [])
  const [routeId, setRouteId] = useState<number | ''>('')
  const readings = useLoad(() => api.readings(routeId ? { route_id: routeId } : {}), [routeId])
  // medidores de todas las rutas para el mapa
  const meters = useLoad(async () => {
    const rs = await api.routes()
    return (await Promise.all(rs.map((r) => api.routeMeters(r.id)))).flat()
  }, [])

  const s = stats.data

  return (
    <>
      <h1>Panel de administración</h1>
      {stats.error && <p className="alert alert-error">{stats.error}</p>}
      <div className="cards">
        <Stat label="Lecturas" value={s ? String(s.readings_total) : '…'} />
        <Stat
          label="Aceptadas sin corrección"
          value={s ? (s.accepted_without_correction_pct === null ? '—' : `${Math.round(s.accepted_without_correction_pct)}%`) : '…'}
        />
        <Stat label="Medidores leídos" value={s ? `${s.meters_read}/${s.meters_total}` : '…'} />
      </div>

      <h2>Rutas</h2>
      <div className="stack">
        {routes.data?.map((r) => (
          <div key={r.id} className="card stack-sm">
            <div className="row-between">
              <strong>{r.name}</strong>
              <span className="muted">
                {r.meters_read}/{r.meters_total}
              </span>
            </div>
            <ProgressBar value={r.meters_read} total={r.meters_total} />
            <span className="muted small">Lecturista: {r.assignee.full_name}</span>
          </div>
        ))}
      </div>

      <h2>Mapa de medidores</h2>
      {meters.data && <MeterMap meters={meters.data} height={360} />}

      <h2>Historial de lecturas</h2>
      <label htmlFor="route-filter">Ruta</label>
      <select id="route-filter" value={routeId} onChange={(e) => setRouteId(e.target.value ? Number(e.target.value) : '')}>
        <option value="">Todas</option>
        {routes.data?.map((r) => (
          <option key={r.id} value={r.id}>
            {r.name}
          </option>
        ))}
      </select>
      {readings.error && <p className="alert alert-error">{readings.error}</p>}
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Foto</th>
              <th>Fecha</th>
              <th>Medidor</th>
              <th>Lecturista</th>
              <th>Propuesta</th>
              <th>Final</th>
              <th>Ubicación</th>
            </tr>
          </thead>
          <tbody>
            {readings.data?.map((r) => (
              <tr key={r.id}>
                <td>
                  <a href={`/api/readings/${r.id}/photo`} target="_blank" rel="noreferrer">
                    <img className="thumb" src={`/api/readings/${r.id}/photo`} alt={`Foto de la lectura ${r.id}`} loading="lazy" />
                  </a>
                </td>
                <td>{formatDate(r.taken_at)}</td>
                <td>#{r.meter_id}</td>
                <td>{r.user.full_name}</td>
                <td>{r.proposed_value ?? '—'}</td>
                <td>
                  <b>{r.value}</b> {r.corrected && <span className="badge badge-pending">corregida</span>}
                </td>
                <td>
                  {r.lat !== null && r.lon !== null ? (
                    <a href={mapUrl(r.lat, r.lon)} target="_blank" rel="noreferrer">
                      {formatLocation(r.lat, r.lon, r.accuracy_m)}
                    </a>
                  ) : (
                    <span className="muted">sin dato</span>
                  )}
                </td>
              </tr>
            ))}
            {readings.data?.length === 0 && (
              <tr>
                <td colSpan={7} className="muted">
                  Sin lecturas.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="card stat">
      <span className="stat-value">{value}</span>
      <span className="muted small">{label}</span>
    </div>
  )
}
