import { useEffect } from 'react'
import { CircleMarker, LayersControl, MapContainer, Popup, TileLayer, useMap } from 'react-leaflet'
import type { LatLngBoundsExpression } from 'leaflet'
import type { Reading } from '../api'
import { formatDate } from '../format'
import { Button } from '../ui/button'

interface Props {
  readings: Reading[] // solo lecturas con ubicación
  onOpen: (reading: Reading) => void
  height?: number
}

// Ajusta la vista para que entren todas las fotos
function Fit({ points }: { points: [number, number][] }) {
  const map = useMap()
  const key = JSON.stringify(points)
  useEffect(() => {
    if (points.length === 0) return
    if (points.length === 1) map.setView(points[0], 17)
    else map.fitBounds(points as LatLngBoundsExpression, { padding: [30, 30], maxZoom: 18 })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key, map])
  return null
}

// Mapa de los lugares donde se tomaron las fotos de las lecturas (no de los medidores)
export default function ReadingsMap({ readings, onOpen, height = 360 }: Props) {
  const located = readings.filter((r) => r.lat !== null && r.lon !== null)
  const points = located.map((r) => [r.lat as number, r.lon as number] as [number, number])
  // isolate: Leaflet usa z-index de 400 a 1000; así quedan contenidos y no tapan el modal.
  return (
    <div className="reading-map relative isolate overflow-hidden rounded-xl border" style={{ height }}>
      <MapContainer center={points[0] ?? [-34.6, -58.4]} zoom={15} style={{ height: '100%', width: '100%' }} scrollWheelZoom>
        <LayersControl position="topright">
          <LayersControl.BaseLayer checked name="Satélite">
            <TileLayer
              attribution="Tiles &copy; Esri &mdash; Source: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
              url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
              maxZoom={19}
            />
          </LayersControl.BaseLayer>
          <LayersControl.BaseLayer name="Calles">
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
              url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
          </LayersControl.BaseLayer>
        </LayersControl>
        <Fit points={points} />
        {located.map((r) => (
          <CircleMarker
            key={r.id}
            center={[r.lat as number, r.lon as number]}
            radius={10}
            pathOptions={{ color: '#fff', weight: 2, fillColor: '#0ea5e9', fillOpacity: 0.95 }}
          >
            <Popup>
              <div className="flex flex-col gap-1 text-sm">
                <strong>Medidor #{r.meter_id}</strong>
                <span>
                  Lectura: <b>{r.value}</b>
                </span>
                <span>{r.user.full_name}</span>
                <span>{formatDate(r.taken_at)}</span>
                <Button size="sm" type="button" onClick={() => onOpen(r)}>
                  Ver foto
                </Button>
              </div>
            </Popup>
          </CircleMarker>
        ))}
      </MapContainer>
    </div>
  )
}
