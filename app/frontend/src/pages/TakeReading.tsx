import { useState, type ChangeEvent } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api, type Analysis, type Reading, type ReadingPayload } from '../api'
import { getPosition, useLoad } from '../hooks'
import { formatLocation } from '../format'
import MeterMap from '../components/MeterMap'
import StatusBadge from '../components/StatusBadge'

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
      const pos = await getPosition(8000)
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
      <Link to={m ? `/routes/${m.route_id}` : '/'} className="back">
        ← Volver a la ruta
      </Link>
      <h1>Tomar lectura</h1>
      {meter.loading && <p className="muted">Cargando…</p>}
      {meter.error && <p className="alert alert-error">{meter.error}</p>}
      {m && (
        <div className="card stack-sm">
          <div className="row-between">
            <strong>{m.code}</strong>
            <StatusBadge status={m.status} />
          </div>
          <span>{m.address}</span>
          {m.last_reading && <span className="muted small">Última lectura: {m.last_reading.value}</span>}
          <MeterMap meters={[m]} height={160} />
        </div>
      )}

      {result ? (
        <ResultCard result={result} routeId={m?.route_id} />
      ) : (
        <>
          <section className="card stack-sm">
            <label htmlFor="photo">Foto del medidor</label>
            <input id="photo" type="file" accept="image/*" capture="environment" onChange={onFile} disabled={analyzing || saving} />
            {m?.has_sample_photo && (
              <button className="btn" type="button" disabled={analyzing || saving} onClick={() => analyze(() => api.samplePhoto(meterId))}>
                Usar foto de ejemplo
              </button>
            )}
            {analyzing && (
              <p className="loading" role="status">
                <span className="spinner" /> Analizando la foto…
              </p>
            )}
          </section>

          {analysis && !analyzing && (
            <AnalysisPanel analysis={analysis} value={value} onChange={setValue} onConfirm={confirm} saving={saving} />
          )}
        </>
      )}
      {error && (
        <p className="alert alert-error" role="alert">
          {error}
        </p>
      )}
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
    <section className="card stack">
      {!a.display_found && (
        <p className="alert alert-warn">
          No se detectó el visor del medidor en la foto. Podés sacar otra foto, más cerca y de frente, o escribir la lectura
          a mano.
        </p>
      )}
      {a.display_found && (
        <div className="images">
          <Figure title="Visor detectado" src={a.images.overlay} />
          <Figure title="Visor rectificado" src={a.images.crop} />
          <Figure title="Dígitos detectados" src={a.images.digits} />
        </div>
      )}
      {a.display_found && typed && (
        <p className="alert alert-info">
          {a.digits_model_available
            ? 'No se pudo leer ningún dígito: escribí la lectura a mano.'
            : 'El modelo de dígitos todavía no está disponible: escribí la lectura que ves en el visor.'}
        </p>
      )}
      <div>
        <label htmlFor="reading">Lectura</label>
        <input
          id="reading"
          className="reading-input"
          inputMode="decimal"
          autoComplete="off"
          value={value}
          placeholder="00000"
          onChange={(e) => onChange(e.target.value.replace(/[^\d.]/g, ''))}
        />
        {a.confidence !== null && a.reading !== null && (
          <span className="muted small">Confianza del modelo: {Math.round(a.confidence * 100)}%</span>
        )}
      </div>
      <button className="btn btn-primary btn-block" onClick={onConfirm} disabled={saving || value.trim() === ''}>
        {saving ? 'Guardando…' : 'Confirmar lectura'}
      </button>
    </section>
  )
}

function Figure({ title, src }: { title: string; src: string | null }) {
  return (
    <figure>
      {src ? <img src={src} alt={title} /> : <div className="img-missing">Sin imagen</div>}
      <figcaption>{title}</figcaption>
    </figure>
  )
}

function ResultCard({ result, routeId }: { result: Reading; routeId?: number }) {
  return (
    <section className="card stack" role="status">
      <h2>Lectura guardada</h2>
      <p className="big-value">{result.value}</p>
      <p>{result.corrected ? `Lectura corregida (propuesta: ${result.proposed_value ?? 'ninguna'}).` : 'Lectura aceptada sin corrección.'}</p>
      <p className="muted">
        {formatLocation(result.lat, result.lon, result.accuracy_m) === null
          ? 'Sin ubicación de la foto (no se pudo obtener el GPS).'
          : `Ubicación de la foto: ${formatLocation(result.lat, result.lon, result.accuracy_m)}`}
      </p>
      <Link className="btn btn-primary btn-block" to={routeId ? `/routes/${routeId}` : '/'}>
        Volver a la ruta
      </Link>
    </section>
  )
}
