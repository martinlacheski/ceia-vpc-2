import { useState, type FormEvent } from 'react'
import { api, type Reading } from '../api'
import { observation } from '../constants'
import { formatDate, formatLocation, mapUrl } from '../format'
import { Alert, AlertDescription } from '../ui/alert'
import { Button } from '../ui/button'
import { cn } from '../ui/cn'
import { Input } from '../ui/input'
import ObservationBadge from './ObservationBadge'
import ReadingsMap from './ReadingsMap'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '../ui/dialog'

// Van debajo de la segmentación del visor y el mapa
const RECTIFIED = [
  { kind: 'crop', title: 'Visor rectificado' },
  { kind: 'digits', title: 'Dígitos detectados' },
] as const

// Imagen procesada: si el backend responde 404 (no existe) se oculta
function Processed({ readingId, kind, title, imgClassName }: { readingId: number; kind: string; title: string; imgClassName?: string }) {
  const [missing, setMissing] = useState(false)
  if (missing) return null
  return (
    <figure className="flex flex-col gap-1">
      <img
        className={cn('w-full rounded-md border bg-muted object-contain', imgClassName)}
        src={`/api/readings/${readingId}/images/${kind}`}
        alt={`${title} de la lectura ${readingId}`}
        loading="lazy"
        onError={() => setMissing(true)}
      />
      <figcaption className="text-xs text-muted-foreground">{title}</figcaption>
    </figure>
  )
}

// Modal con la foto de la lectura y sus datos; Esc, botón y clic fuera lo cierran (Radix gestiona foco y aria-modal)
// onReviewed recibe la lectura actualizada después de que el admin la confirma o corrige
export default function ReadingModal({
  reading,
  onClose,
  onReviewed,
}: {
  reading: Reading | null
  onClose: () => void
  onReviewed?: (reading: Reading) => void
}) {
  const loc = reading ? formatLocation(reading.lat, reading.lon, reading.accuracy_m) : null
  const located = reading !== null && reading.lat !== null && reading.lon !== null
  return (
    <Dialog open={reading !== null} onOpenChange={(open) => !open && onClose()}>
      {reading && (
        // En pantallas grandes: foto original a la izquierda; a la derecha datos, segmentación del visor con el mapa y debajo el visor rectificado y los dígitos
        <DialogContent
          aria-modal="true"
          closeLabel="Cerrar"
          className="max-h-[95dvh] w-[calc(100%-2rem)] max-w-[100rem] overflow-y-auto sm:w-[calc(100%-2rem)]"
        >
          <DialogHeader>
            <DialogTitle>Lectura del medidor #{reading.meter_id}</DialogTitle>
            <DialogDescription>
              {reading.user.full_name} · {formatDate(reading.taken_at)}
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 lg:grid-cols-2">
            <figure className="flex min-w-0 flex-col gap-1">
              {/* En lg la foto no aporta alto propio: toma el de la columna derecha y terminan a la par */}
              <div className="relative h-[60dvh] lg:h-auto lg:flex-1">
                <img
                  className="absolute inset-0 h-full w-full rounded-md border bg-muted object-contain"
                  src={`/api/readings/${reading.id}/photo`}
                  alt={`Foto de la lectura ${reading.id}`}
                />
              </div>
              <figcaption className="text-xs text-muted-foreground">Imagen original</figcaption>
            </figure>
            <div className="flex min-w-0 flex-col gap-4">
              <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm sm:grid-cols-3">
                <Item label="Lectura final">
                  <span className="text-lg font-semibold">{reading.final_value}</span> <ObservationBadge reading={reading} />
                </Item>
                <Item label="Lectura propuesta">{reading.proposed_value ?? '—'}</Item>
                <Item label="Confianza">
                  {reading.confidence === null ? '—' : `${Math.round(reading.confidence * 100)}%`}
                </Item>
                <Item label="Lecturista">
                  {reading.user.full_name}
                  {reading.review && reading.review.value !== reading.value && (
                    <span className="block text-xs text-muted-foreground">confirmó {reading.value}</span>
                  )}
                </Item>
                <Item label="Fecha">{formatDate(reading.taken_at)}</Item>
                <Item label="Ubicación de la foto">
                  {located && loc ? (
                    <a className="underline underline-offset-4" href={mapUrl(reading.lat as number, reading.lon as number)} target="_blank" rel="noreferrer">
                      {loc}
                    </a>
                  ) : (
                    <span className="text-muted-foreground">sin dato</span>
                  )}
                </Item>
                {reading.review && (
                  <Item label="Revisión">
                    {reading.review.reviewed_by.full_name} · {formatDate(reading.review.reviewed_at)}
                  </Item>
                )}
              </dl>
              {observation(reading) === 'revisar' && onReviewed && (
                <ReviewForm key={reading.id} reading={reading} onReviewed={onReviewed} />
              )}
              <div className="grid gap-3 sm:grid-cols-2">
                <Processed readingId={reading.id} kind="overlay" title="Segmentación del visor" imgClassName="h-[36dvh]" />
                {located ? (
                  <figure className="flex flex-col gap-1">
                    <ReadingsMap readings={[reading]} height="36dvh" />
                    <figcaption className="text-xs text-muted-foreground">Lugar donde se tomó la foto</figcaption>
                  </figure>
                ) : (
                  <p className="text-sm text-muted-foreground">La foto no tiene ubicación registrada.</p>
                )}
              </div>
              <div className="grid grid-cols-2 gap-3">
                {RECTIFIED.map((k) => (
                  <Processed key={k.kind} readingId={reading.id} kind={k.kind} title={k.title} />
                ))}
              </div>
            </div>
          </div>
        </DialogContent>
      )}
    </Dialog>
  )
}

const VALID_VALUE = /^\d+(\.\d+)?$/

// Revisión del admin: confirma el valor del lecturista o lo corrige
function ReviewForm({ reading, onReviewed }: { reading: Reading; onReviewed: (reading: Reading) => void }) {
  const [value, setValue] = useState(reading.value)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const valid = VALID_VALUE.test(value)
  const changed = value !== reading.value

  async function submit(e: FormEvent) {
    e.preventDefault()
    if (!valid) return
    setSaving(true)
    setError(null)
    try {
      onReviewed(await api.reviewReading(reading.id, value))
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <form onSubmit={submit} className="flex flex-col gap-2 rounded-lg border border-destructive/40 p-3">
      <label htmlFor="review-value" className="text-sm font-medium">
        Valor revisado
      </label>
      <p className="text-xs text-muted-foreground">
        La confianza del modelo fue baja y el lecturista no corrigió el valor. Compará con la foto y confirmalo o corregilo.
      </p>
      <div className="flex flex-wrap gap-2">
        <Input
          id="review-value"
          className="w-40"
          inputMode="decimal"
          value={value}
          onChange={(e) => setValue(e.target.value.trim().replace(/,/g, '.'))}
          aria-invalid={!valid}
        />
        <Button type="submit" disabled={!valid || saving}>
          {changed ? 'Guardar corrección' : 'Confirmar valor'}
        </Button>
      </div>
      {error && (
        <Alert variant="destructive">
          <AlertDescription className="text-destructive">{error}</AlertDescription>
        </Alert>
      )}
    </form>
  )
}

function Item({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-0.5">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd>{children}</dd>
    </div>
  )
}
