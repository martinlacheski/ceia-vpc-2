import { api } from '../api'
import { useLoad } from '../hooks'
import PageHeader from '../components/PageHeader'
import ProgressBar from '../components/ProgressBar'
import { Alert, AlertDescription } from '../ui/alert'
import { Card, CardContent, CardHeader, CardTitle } from '../ui/card'

// Panel: indicadores y avance de las rutas. El mapa y el historial tienen su propia pestaña.
export default function Admin() {
  const stats = useLoad(() => api.stats(), [])
  const routes = useLoad(() => api.routes(), [])
  const s = stats.data
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
