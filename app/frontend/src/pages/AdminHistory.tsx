import { useState } from 'react'
import { api, type Reading } from '../api'
import { useLoad } from '../hooks'
import { formatDate, formatLocation, mapUrl } from '../format'
import PageHeader from '../components/PageHeader'
import ObservationBadge from '../components/ObservationBadge'
import ReadingModal from '../components/ReadingModal'
import { Alert, AlertDescription } from '../ui/alert'
import { Button } from '../ui/button'
import { Card } from '../ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../ui/table'

// Historial de lecturas filtrable por ruta; cada foto abre el detalle en un modal
const PAGE_SIZE = 20

export default function AdminHistory() {
  const routes = useLoad(() => api.routes(), [])
  const [routeId, setRouteId] = useState<number | ''>('')
  const [page, setPage] = useState(0)
  const [version, setVersion] = useState(0) // se incrementa para recargar tras una revisión
  const readings = useLoad(
    () => api.readingsPage({ ...(routeId ? { route_id: routeId } : {}), limit: PAGE_SIZE, offset: page * PAGE_SIZE }),
    [routeId, page, version],
  )
  const items = readings.data?.items
  const total = readings.data?.total ?? 0
  const pages = Math.max(1, Math.ceil(total / PAGE_SIZE))
  const [selected, setSelected] = useState<Reading | null>(null)

  return (
    <>
      <PageHeader title="Historial de lecturas" />
      <section className="flex flex-col gap-3">
        <div className="flex flex-col gap-2 sm:w-64">
          <label className="text-sm leading-none font-medium" htmlFor="route-filter">
            Ruta
          </label>
          <select
            id="route-filter"
            className="h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 dark:bg-input/30"
            value={routeId}
            onChange={(e) => {
              setRouteId(e.target.value ? Number(e.target.value) : '')
              setPage(0)
            }}
          >
            <option value="">Todas</option>
            {routes.data?.map((r) => (
              <option key={r.id} value={r.id}>
                {r.name}
              </option>
            ))}
          </select>
        </div>
        {readings.error && (
          <Alert variant="destructive">
            <AlertDescription className="text-destructive">{readings.error}</AlertDescription>
          </Alert>
        )}
        <Card className="py-0">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead className="px-4">Foto</TableHead>
                <TableHead>Fecha</TableHead>
                <TableHead>Ruta</TableHead>
                <TableHead>Medidor</TableHead>
                <TableHead>Lecturista</TableHead>
                <TableHead>Propuesta</TableHead>
                <TableHead>Final</TableHead>
                <TableHead>Ubicación</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {items?.map((r) => (
                <TableRow key={r.id}>
                  <TableCell className="px-4">
                    <button
                      type="button"
                      className="block rounded-md outline-none focus-visible:ring-[3px] focus-visible:ring-ring/50"
                      aria-label={`Ver foto de la lectura ${r.id}`}
                      onClick={() => setSelected(r)}
                    >
                      <img className="h-12 w-16 rounded-md border object-cover" src={`/api/readings/${r.id}/photo`} alt="" loading="lazy" />
                    </button>
                  </TableCell>
                  <TableCell>{formatDate(r.taken_at)}</TableCell>
                  <TableCell>{r.route.name}</TableCell>
                  <TableCell>#{r.meter_id}</TableCell>
                  <TableCell>{r.user.full_name}</TableCell>
                  <TableCell>{r.proposed_value ?? '—'}</TableCell>
                  <TableCell>
                    <span className="flex items-center gap-1">
                      <b>{r.final_value}</b>
                      <ObservationBadge reading={r} />
                    </span>
                  </TableCell>
                  <TableCell>
                    {r.lat !== null && r.lon !== null ? (
                      <a className="underline underline-offset-4" href={mapUrl(r.lat, r.lon)} target="_blank" rel="noreferrer">
                        {formatLocation(r.lat, r.lon, r.accuracy_m)}
                      </a>
                    ) : (
                      <span className="text-muted-foreground">sin dato</span>
                    )}
                  </TableCell>
                </TableRow>
              ))}
              {items?.length === 0 && (
                <TableRow>
                  <TableCell colSpan={8} className="px-4 text-muted-foreground">
                    Sin lecturas.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </Card>
        {total > 0 && (
          <div className="flex flex-wrap items-center justify-between gap-2 text-sm">
            <span className="text-muted-foreground">
              {`Mostrando ${page * PAGE_SIZE + 1}–${Math.min((page + 1) * PAGE_SIZE, total)} de ${total}`}
            </span>
            <div className="flex items-center gap-2">
              <Button variant="outline" size="sm" disabled={page === 0} onClick={() => setPage(page - 1)}>
                Anterior
              </Button>
              <span className="text-muted-foreground">
                Página {page + 1} de {pages}
              </span>
              <Button variant="outline" size="sm" disabled={page + 1 >= pages} onClick={() => setPage(page + 1)}>
                Siguiente
              </Button>
            </div>
          </div>
        )}
      </section>
      <ReadingModal
        reading={selected}
        onClose={() => setSelected(null)}
        onReviewed={(r) => {
          setSelected(r)
          setVersion((v) => v + 1)
        }}
      />
    </>
  )
}
