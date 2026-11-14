import { useState } from 'react'
import { api, type Reading } from '../api'
import { useLoad } from '../hooks'
import PageHeader from '../components/PageHeader'
import ReadingModal from '../components/ReadingModal'
import ReadingsMap from '../components/ReadingsMap'
import { Alert, AlertDescription } from '../ui/alert'

// Mapa de los lugares donde se tomaron las fotos; cada punto abre el detalle de la lectura
export default function AdminMap() {
  const [version, setVersion] = useState(0) // se incrementa para recargar tras una revisión
  const readings = useLoad(() => api.readings({}), [version])
  const [selected, setSelected] = useState<Reading | null>(null)
  const located = (readings.data ?? []).filter((r) => r.lat !== null && r.lon !== null)

  return (
    <>
      <PageHeader title="Mapa de lecturas" />
      {readings.error && (
        <Alert variant="destructive">
          <AlertDescription className="text-destructive">{readings.error}</AlertDescription>
        </Alert>
      )}
      {readings.data && located.length > 0 && <ReadingsMap readings={located} onOpen={setSelected} height={560} />}
      {readings.data && located.length === 0 && (
        <p className="text-muted-foreground">Ninguna lectura tiene ubicación de la foto todavía.</p>
      )}
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
