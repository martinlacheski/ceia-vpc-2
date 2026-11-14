import { ArrowLeftIcon } from 'lucide-react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../api'
import { useLoad } from '../hooks'
import { formatDate } from '../format'
import PageHeader from '../components/PageHeader'
import StatusBadge from '../components/StatusBadge'
import ProgressBar from '../components/ProgressBar'
import { Alert, AlertDescription } from '../ui/alert'
import { buttonVariants } from '../ui/button'
import { Card, CardContent } from '../ui/card'

// La ruta muestra solo el listado de medidores: no se ubican en un mapa (S12)
export default function RouteView() {
  const routeId = Number(useParams().routeId)
  const meters = useLoad(() => api.routeMeters(routeId), [routeId])
  const routes = useLoad(() => api.routes(), [])
  const route = routes.data?.find((r) => r.id === routeId)

  return (
    <>
      <PageHeader
        title={route?.name ?? 'Ruta'}
        backAction={
          <Link to="/" className={buttonVariants({ variant: 'ghost', size: 'sm', className: 'w-fit' })}>
            <ArrowLeftIcon /> Mis rutas
          </Link>
        }
      />
      {route && (
        <div className="flex flex-col gap-2">
          <ProgressBar value={route.meters_read} total={route.meters_total} />
          <span className="text-sm text-muted-foreground">
            {route.meters_read} de {route.meters_total} medidores leídos
          </span>
        </div>
      )}
      {meters.loading && <p className="text-muted-foreground">Cargando…</p>}
      {meters.error && (
        <Alert variant="destructive">
          <AlertDescription className="text-destructive">{meters.error}</AlertDescription>
        </Alert>
      )}
      {meters.data && (
        <ul className="flex flex-col gap-3">
          {meters.data.map((m) => (
            <li key={m.id}>
              <Link to={`/meters/${m.id}/read`} className="block rounded-xl outline-none transition-shadow hover:shadow-md focus-visible:ring-[3px] focus-visible:ring-ring/50">
                <Card>
                  <CardContent className="flex flex-col gap-1">
                    <div className="flex items-center justify-between gap-2">
                      <strong>{m.code}</strong>
                      <StatusBadge status={m.status} />
                    </div>
                    <span>{m.address}</span>
                    {m.last_reading && (
                      <span className="text-sm text-muted-foreground">
                        Última lectura: <b>{m.last_reading.value}</b> · {formatDate(m.last_reading.taken_at)}
                      </span>
                    )}
                  </CardContent>
                </Card>
              </Link>
            </li>
          ))}
        </ul>
      )}
    </>
  )
}
