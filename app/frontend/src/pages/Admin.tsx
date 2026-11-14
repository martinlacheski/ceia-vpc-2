import { useState } from 'react'
import { api, type Reading } from '../api'
import { isLowConfidence } from '../constants'
import { useLoad } from '../hooks'
import { formatDate, formatLocation, mapUrl } from '../format'
import PageHeader from '../components/PageHeader'
import ProgressBar from '../components/ProgressBar'
import ReadingModal from '../components/ReadingModal'
import ReadingsMap from '../components/ReadingsMap'
import { Alert, AlertDescription } from '../ui/alert'
import { Badge } from '../ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '../ui/table'

export default function Admin() {
  const stats = useLoad(() => api.stats(), [])
  const routes = useLoad(() => api.routes(), [])
  const [routeId, setRouteId] = useState<number | ''>('')
  const readings = useLoad(() => api.readings(routeId ? { route_id: routeId } : {}), [routeId])
  const [selected, setSelected] = useState<Reading | null>(null)

  const s = stats.data
  const located = (readings.data ?? []).filter((r) => r.lat !== null && r.lon !== null)

  return (
    <>
      <PageHeader title="Panel de administración" />
      {stats.error && (
        <Alert variant="destructive">
          <AlertDescription className="text-destructive">{stats.error}</AlertDescription>
        </Alert>
      )}
      <div className="grid gap-4 sm:grid-cols-3">
        <Stat label="Lecturas" value={s ? String(s.readings_total) : '…'} />
        <Stat
          label="Aceptadas sin corrección"
          value={s ? (s.accepted_without_correction_pct === null ? '—' : `${Math.round(s.accepted_without_correction_pct)}%`) : '…'}
        />
        <Stat label="Medidores leídos" value={s ? `${s.meters_read}/${s.meters_total}` : '…'} />
      </div>

      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold tracking-tight">Rutas</h2>
        <div className="grid gap-4 sm:grid-cols-2">
          {routes.data?.map((r) => (
            <Card key={r.id}>
              <CardContent className="flex flex-col gap-3">
                <div className="flex items-center justify-between gap-2">
                  <strong>{r.name}</strong>
                  <span className="text-muted-foreground">
                    {r.meters_read}/{r.meters_total}
                  </span>
                </div>
                <ProgressBar value={r.meters_read} total={r.meters_total} />
                <span className="text-sm text-muted-foreground">Lecturista: {r.assignee.full_name}</span>
              </CardContent>
            </Card>
          ))}
        </div>
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold tracking-tight">Mapa de lecturas</h2>
        {readings.data && located.length > 0 && <ReadingsMap readings={located} onOpen={setSelected} height={360} />}
        {readings.data && located.length === 0 && (
          <p className="text-muted-foreground">Ninguna lectura tiene ubicación de la foto todavía.</p>
        )}
      </section>

      <section className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold tracking-tight">Historial de lecturas</h2>
        <div className="flex flex-col gap-2 sm:w-64">
          <label className="text-sm leading-none font-medium" htmlFor="route-filter">
            Ruta
          </label>
          <select
            id="route-filter"
            className="h-9 w-full rounded-md border border-input bg-transparent px-3 py-1 text-sm shadow-xs outline-none focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50 dark:bg-input/30"
            value={routeId}
            onChange={(e) => setRouteId(e.target.value ? Number(e.target.value) : '')}
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
                <TableHead>Medidor</TableHead>
                <TableHead>Lecturista</TableHead>
                <TableHead>Propuesta</TableHead>
                <TableHead>Final</TableHead>
                <TableHead>Ubicación</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {readings.data?.map((r) => (
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
                  <TableCell>#{r.meter_id}</TableCell>
                  <TableCell>{r.user.full_name}</TableCell>
                  <TableCell>{r.proposed_value ?? '—'}</TableCell>
                  <TableCell>
                    <span className="flex items-center gap-1">
                      <b>{r.value}</b>
                      {r.corrected && <Badge variant="warning">corregida</Badge>}
                      {isLowConfidence(r.confidence) && <Badge variant="destructive">revisar</Badge>}
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
              {readings.data?.length === 0 && (
                <TableRow>
                  <TableCell colSpan={7} className="px-4 text-muted-foreground">
                    Sin lecturas.
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </Card>
      </section>
      <ReadingModal reading={selected} onClose={() => setSelected(null)} />
    </>
  )
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-3xl">{value}</CardTitle>
        <span className="text-sm text-muted-foreground">{label}</span>
      </CardHeader>
    </Card>
  )
}
