import { Link } from 'react-router-dom'
import { api } from '../api'
import { useLoad } from '../hooks'
import ProgressBar from '../components/ProgressBar'

export default function ReaderHome() {
  const { data: routes, error, loading } = useLoad(() => api.routes(), [])
  return (
    <>
      <h1>Mis rutas</h1>
      {loading && <p className="muted">Cargando…</p>}
      {error && <p className="alert alert-error">{error}</p>}
      {routes && routes.length === 0 && <p className="muted">No tenés rutas asignadas.</p>}
      <div className="stack">
        {routes?.map((r) => (
          <Link key={r.id} to={`/routes/${r.id}`} className="card card-link">
            <div className="row-between">
              <strong>{r.name}</strong>
              <span className="muted">
                {r.meters_read}/{r.meters_total}
              </span>
            </div>
            <ProgressBar value={r.meters_read} total={r.meters_total} />
            <span className="muted small">
              {r.meters_total - r.meters_read === 0 ? 'Ruta completa' : `${r.meters_total - r.meters_read} pendientes`}
            </span>
          </Link>
        ))}
      </div>
    </>
  )
}
