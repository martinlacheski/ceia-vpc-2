import { useState } from 'react'
import type { Reading } from '../api'
import { isLowConfidence } from '../constants'
import { formatDate, formatLocation, mapUrl } from '../format'
import { Badge } from '../ui/badge'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '../ui/dialog'

const KINDS = [
  { kind: 'overlay', title: 'Visor detectado' },
  { kind: 'crop', title: 'Visor rectificado' },
  { kind: 'digits', title: 'Dígitos detectados' },
] as const

// Imagen procesada: si el backend responde 404 (no existe) se oculta
function Processed({ readingId, kind, title }: { readingId: number; kind: string; title: string }) {
  const [missing, setMissing] = useState(false)
  if (missing) return null
  return (
    <figure className="flex flex-col gap-1">
      <img
        className="w-full rounded-md border bg-muted object-contain"
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
export default function ReadingModal({ reading, onClose }: { reading: Reading | null; onClose: () => void }) {
  const loc = reading ? formatLocation(reading.lat, reading.lon, reading.accuracy_m) : null
  return (
    <Dialog open={reading !== null} onOpenChange={(open) => !open && onClose()}>
      {reading && (
        // Ancho: en pantallas grandes, foto a la izquierda y datos e imágenes procesadas a la derecha
        <DialogContent aria-modal="true" closeLabel="Cerrar" className="w-[calc(100%-2rem)] max-w-6xl sm:w-[calc(100%-2rem)]">
          <DialogHeader>
            <DialogTitle>Lectura del medidor #{reading.meter_id}</DialogTitle>
            <DialogDescription>
              {reading.user.full_name} · {formatDate(reading.taken_at)}
            </DialogDescription>
          </DialogHeader>
          <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.2fr)]">
          <img
            className="max-h-[75dvh] w-full rounded-md border bg-muted object-contain"
            src={`/api/readings/${reading.id}/photo`}
            alt={`Foto de la lectura ${reading.id}`}
          />
          <div className="flex min-w-0 flex-col gap-4">
          <dl className="grid grid-cols-2 gap-x-4 gap-y-3 text-sm sm:grid-cols-3">
            <Item label="Lectura final">
              <span className="text-lg font-semibold">{reading.value}</span>{' '}
              {reading.corrected && <Badge variant="warning">corregida</Badge>}
            </Item>
            <Item label="Lectura propuesta">{reading.proposed_value ?? '—'}</Item>
            <Item label="Confianza">
              {reading.confidence === null ? '—' : `${Math.round(reading.confidence * 100)}%`}{' '}
              {isLowConfidence(reading.confidence) && <Badge variant="destructive">revisar</Badge>}
            </Item>
            <Item label="Lecturista">{reading.user.full_name}</Item>
            <Item label="Fecha">{formatDate(reading.taken_at)}</Item>
            <Item label="Ubicación de la foto">
              {reading.lat !== null && reading.lon !== null && loc ? (
                <a className="underline underline-offset-4" href={mapUrl(reading.lat, reading.lon)} target="_blank" rel="noreferrer">
                  {loc}
                </a>
              ) : (
                <span className="text-muted-foreground">sin dato</span>
              )}
            </Item>
          </dl>
          <div className="grid grid-cols-2 gap-3">
            {KINDS.map((k) => (
              <div key={k.kind} className={k.kind === 'overlay' ? 'col-span-1 row-span-2' : 'col-span-1'}>
                <Processed readingId={reading.id} kind={k.kind} title={k.title} />
              </div>
            ))}
          </div>
          </div>
          </div>
        </DialogContent>
      )}
    </Dialog>
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
