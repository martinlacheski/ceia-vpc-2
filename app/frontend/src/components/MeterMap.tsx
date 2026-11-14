import { useEffect } from 'react'
import { CircleMarker, MapContainer, Popup, TileLayer, useMap } from 'react-leaflet'
import type { LatLngBoundsExpression } from 'leaflet'
import type { Position } from '../hooks'

export interface MapMeter {
  id: number
  code: string
  address: string
  lat: number
  lon: number
  status: 'pending' | 'read'
}

interface Props {
  meters: MapMeter[]
  position?: Position | null
  height?: number
  renderPopup?: (m: MapMeter) => React.ReactNode
}

const COLORS = { pending: '#d97706', read: '#16a34a' }

// Ajusta la vista para que entren los medidores y la posición del usuario
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

export default function MeterMap({ meters, position, height = 360, renderPopup }: Props) {
  const points: [number, number][] = meters.map((m) => [m.lat, m.lon])
  if (position) points.push([position.lat, position.lon])
  const center: [number, number] = points[0] ?? [-34.6, -58.4]

  return (
    <div className="map-wrap" style={{ height }}>
      <MapContainer center={center} zoom={15} style={{ height: '100%', width: '100%' }} scrollWheelZoom>
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        <Fit points={points} />
        {meters.map((m) => (
          <CircleMarker
            key={m.id}
            center={[m.lat, m.lon]}
            radius={10}
            pathOptions={{ color: '#fff', weight: 2, fillColor: COLORS[m.status], fillOpacity: 0.95 }}
          >
            <Popup>{renderPopup ? renderPopup(m) : <strong>{m.code}</strong>}</Popup>
          </CircleMarker>
        ))}
        {position && (
          <CircleMarker
            center={[position.lat, position.lon]}
            radius={8}
            pathOptions={{ color: '#fff', weight: 3, fillColor: '#2563eb', fillOpacity: 1 }}
          >
            <Popup>Tu ubicación</Popup>
          </CircleMarker>
        )}
      </MapContainer>
    </div>
  )
}
