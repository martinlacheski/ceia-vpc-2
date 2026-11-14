import { Link } from 'react-router-dom'
import { api } from '../api'
import { useLoad } from '../hooks'
import PageHeader from '../components/PageHeader'
import ProgressBar from '../components/ProgressBar'
import { Alert, AlertDescription } from '../ui/alert'
import { Card, CardContent } from '../ui/card'

export default function ReaderHome() {
  const { data: routes, error, loading } = useLoad(() => api.routes(), [])
  return (
    <>
      <PageHeader title="Mis rutas" />
      {loading && <p className="text-muted-foreground">Cargando…</p>}
      {error && (
        <Alert variant="destructive">
          <AlertDescription className="text-destructive">{error}</AlertDescription>
        </Alert>
      )}
      {routes && routes.length === 0 && <p className="text-muted-foreground">No tenés rutas asignadas.</p>}
      <div className="grid gap-4 sm:grid-cols-2">
        {routes?.map((r) => (
          <Link key={r.id} to={`/routes/${r.id}`} className="rounded-xl outline-none transition-shadow hover:shadow-md focus-visible:ring-[3px] focus-visible:ring-ring/50">
            <Card className="h-full">
              <CardContent className="flex flex-col gap-3">
                <div className="flex items-center justify-between gap-2">
                  <strong>{r.name}</strong>
                  <span className="text-muted-foreground">
                    {r.meters_read}/{r.meters_total}
                  </span>
                </div>
                <ProgressBar value={r.meters_read} total={r.meters_total} />
                <span className="text-sm text-muted-foreground">
                  {r.meters_total - r.meters_read === 0 ? 'Ruta completa' : `${r.meters_total - r.meters_read} pendientes`}
                </span>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>
    </>
  )
}
