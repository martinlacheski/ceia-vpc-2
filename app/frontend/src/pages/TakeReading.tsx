import { useState, type ChangeEvent } from 'react'
import { ArrowLeftIcon, TriangleAlertIcon } from 'lucide-react'
import { Link, useParams } from 'react-router-dom'
import { SERVICE_LABEL, api, type Analysis, type Reading, type ReadingPayload } from '../api'
import { getPosition, useBestPosition, useLoad, type GpsStatus, type Position } from '../hooks'
import { formatLocation } from '../format'
import { isLowConfidence } from '../constants'
import PageHeader from '../components/PageHeader'
import StatusBadge from '../components/StatusBadge'
import { Alert, AlertDescription } from '../ui/alert'
import { Button, buttonVariants } from '../ui/button'
import { Card, CardContent } from '../ui/card'
import { Input } from '../ui/input'

const LABEL = 'text-sm leading-none font-medium'

function ErrorAlert({ children }: { children: React.ReactNode }) {
  return (
    <Alert variant="destructive" role="alert">
      <AlertDescription className="text-destructive">{children}</AlertDescription>
    </Alert>
  )
}

const VALUE_RE = /^\d+(\.\d+)?$/

export default function TakeReading() {
  const meterId = Number(useParams().meterId)
  const meter = useLoad(() => api.meter(meterId), [meterId])
  const [analyzing, setAnalyzing] = useState(false)
  const [analysis, setAnalysis] = useState<Analysis | null>(null)
  const [value, setValue] = useState('')
  const [saving, setSaving] = useState(false)
  const [result, setResult] = useState<Reading | null>(null)
  const [error, setError] = useState<string | null>(null)
  const gps = useBestPosition()

  async function analyze(getPhoto: () => Promise<Blob>) {
    setError(null)
    setResult(null)
    setAnalyzing(true)
    try {
      const res = await api.analyze(meterId, await getPhoto())
      setAnalysis(res)
      setValue(res.reading ?? '')
    } catch (e) {
      setAnalysis(null)
      setError((e as Error).message)
    } finally {
      setAnalyzing(false)
    }
  }

  function onFile(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    e.target.value = ''
    if (file) void analyze(async () => file)
  }

  async function confirm() {
    if (!analysis) return
    const v = value.trim()
    if (!VALUE_RE.test(v)) {
      setError('La lectura debe tener solo dígitos (y opcionalmente un punto decimal).')
      return
    }
    setError(null)
    setSaving(true)
    try {
      // la ubicación es opcional: si se deniega, se guarda igual sin coordenadas
      // la mejor posición seguida desde que se abrió la pantalla; si no hubo ninguna, un último intento
      const pos = gps.best ?? (await getPosition(8000))
      const payload: ReadingPayload = { analysis_id: analysis.analysis_id, value: v }
      if (pos) {
        payload.lat = pos.lat
        payload.lon = pos.lon
        payload.accuracy_m = pos.accuracy_m
      }
      setResult(await api.saveReading(meterId, payload))
    } catch (e) {
      setError((e as Error).message)
    } finally {
      setSaving(false)
    }
  }

  const m = meter.data

  return (
    <>
      <PageHeader
        title="Tomar lectura"
        backAction={
          <Link to={m ? `/routes/${m.route_id}` : '/'} className={buttonVariants({ variant: 'ghost', size: 'sm', className: 'w-fit' })}>
            <ArrowLeftIcon /> Volver a la ruta
          </Link>
        }
      />
      {meter.loading && <p className="text-muted-foreground">Cargando…</p>}
      {meter.error && <ErrorAlert>{meter.error}</ErrorAlert>}
      {m && (
        <Card>
          <CardContent className="flex flex-col gap-1">
            <div className="flex items-center justify-between gap-2">
              <span className="flex items-baseline gap-2">
                <strong>{m.code}</strong>
                <span className="text-sm text-muted-foreground">{SERVICE_LABEL[m.service]}</span>
              </span>
              <StatusBadge status={m.status} />
            </div>
            <span>{m.address}</span>
            {m.last_reading && <span className="text-sm text-muted-foreground">Última lectura: {m.last_reading.value}</span>}
          </CardContent>
        </Card>
      )}

      <GpsIndicator status={gps.status} best={gps.best} />
      {result ? (
        <ResultCard result={result} routeId={m?.route_id} />
      ) : (
        <>
          <Card>
            <CardContent className="flex flex-col gap-3">
              <label className={LABEL} htmlFor="photo">
                Foto del medidor
              </label>
              <Input id="photo" type="file" accept="image/*" capture="environment" onChange={onFile} disabled={analyzing || saving} />
              {m?.has_sample_photo && (
                <Button variant="outline" type="button" disabled={analyzing || saving} onClick={() => analyze(() => api.samplePhoto(meterId))}>
                  Usar foto de ejemplo
                </Button>
              )}
              {analyzing && (
                <p className="flex items-center gap-2 text-muted-foreground" role="status">
                  <span className="size-4 animate-spin rounded-full border-2 border-muted border-t-primary" /> Analizando la foto…
                </p>
              )}
            </CardContent>
          </Card>

          {analysis && !analyzing && (
            <AnalysisPanel analysis={analysis} value={value} onChange={setValue} onConfirm={confirm} saving={saving} />
          )}
        </>
      )}
      {error && <ErrorAlert>{error}</ErrorAlert>}
    </>
  )
}

function AnalysisPanel(props: {
  analysis: Analysis
  value: string
  onChange: (v: string) => void
  onConfirm: () => void
  saving: boolean
}) {
  const { analysis: a, value, onChange, onConfirm, saving } = props
  const typed = !a.display_found || !a.digits_model_available || a.reading === null
  return (
    <Card>
      <CardContent className="flex flex-col gap-4">
        {!a.display_found && (
          <Alert>
            <TriangleAlertIcon />
            <AlertDescription>
              No se detectó el visor del medidor en la foto. Podés sacar otra foto, más cerca y de frente, o escribir la lectura
              a mano.
            </AlertDescription>
          </Alert>
        )}
        {a.display_found && (
          <div className="grid gap-3 sm:grid-cols-3">
            <Figure title="Segmentación del visor" src={a.images.overlay} />
            <Figure title="Visor rectificado" src={a.images.crop} />
            <Figure title="Dígitos detectados" src={a.images.digits} />
          </div>
        )}
        {a.display_found && typed && (
          <Alert>
            <AlertDescription>
              {a.digits_model_available
                ? 'No se pudo leer ningún dígito: escribí la lectura a mano.'
                : 'El modelo de dígitos todavía no está disponible: escribí la lectura que ves en el visor.'}
            </AlertDescription>
          </Alert>
        )}
        {isLowConfidence(a.confidence) && a.reading !== null && (
          <Alert variant="destructive">
            <TriangleAlertIcon />
            <AlertDescription className="text-destructive">
              Lectura con confianza baja: revisá cada dígito con la foto antes de confirmar.
            </AlertDescription>
          </Alert>
        )}
        <div className="flex flex-col gap-2">
          <label className={LABEL} htmlFor="reading">
            Lectura
          </label>
          <Input
            id="reading"
            className="h-12 text-center font-mono text-2xl tracking-widest md:text-2xl"
            inputMode="decimal"
            autoComplete="off"
            value={value}
            placeholder="00000"
            onChange={(e) => onChange(e.target.value.replace(/[^\d.]/g, ''))}
          />
          {a.confidence !== null && a.reading !== null && (
            <span className="text-sm text-muted-foreground">Confianza del modelo: {Math.round(a.confidence * 100)}%</span>
          )}
        </div>
        <Button onClick={onConfirm} disabled={saving || value.trim() === ''}>
          {saving ? 'Guardando…' : 'Confirmar lectura'}
        </Button>
      </CardContent>
    </Card>
  )
}

function Figure({ title, src }: { title: string; src: string | null }) {
  return (
    <figure className="flex flex-col gap-1">
      {src ? (
        <img className="w-full rounded-md border bg-muted object-contain" src={src} alt={title} />
      ) : (
        <div className="flex h-20 items-center justify-center rounded-md border bg-muted text-sm text-muted-foreground">Sin imagen</div>
      )}
      <figcaption className="text-xs text-muted-foreground">{title}</figcaption>
    </figure>
  )
}

function ResultCard({ result, routeId }: { result: Reading; routeId?: number }) {
  const loc = formatLocation(result.lat, result.lon, result.accuracy_m)
  return (
    <Card role="status">
      <CardContent className="flex flex-col gap-3">
        <h2 className="text-lg font-semibold tracking-tight">Lectura guardada</h2>
        <p className="font-mono text-4xl font-bold tracking-widest">{result.value}</p>
        <p>{result.corrected ? `Lectura corregida (propuesta: ${result.proposed_value ?? 'ninguna'}).` : 'Lectura aceptada sin corrección.'}</p>
        <p className="text-muted-foreground">
          {loc === null ? 'Sin ubicación de la foto (no se pudo obtener el GPS).' : `Ubicación de la foto: ${loc}`}
        </p>
        <Link className={buttonVariants()} to={routeId ? `/routes/${routeId}` : '/'}>
          Volver a la ruta
        </Link>
      </CardContent>
    </Card>
  )
}

// Precisión del GPS en vivo: verde hasta 15 m, ámbar hasta 50 m, rojo por encima
function GpsIndicator({ status, best }: { status: GpsStatus; best: Position | null }) {
  if (status === 'unavailable' && best === null) {
    return (
      <p className="text-sm text-muted-foreground">
        Ubicación no disponible: la lectura se guarda sin coordenadas. Revisá el permiso de ubicación del navegador.
      </p>
    )
  }
  if (best === null) return <p className="text-sm text-muted-foreground">Buscando GPS…</p>
  const m = Math.round(best.accuracy_m)
  const tone = m <= 15 ? 'bg-emerald-500' : m <= 50 ? 'bg-amber-500' : 'bg-red-500'
  return (
    <p className="flex items-center gap-2 text-sm">
      <span className={`inline-block size-2.5 rounded-full ${tone}`} aria-hidden="true" />
      <span>{`GPS: ±${m} m`}</span>
      {m > 15 && <span className="text-muted-foreground">(esperá unos segundos para mejorar la precisión)</span>}
    </p>
  )
}
