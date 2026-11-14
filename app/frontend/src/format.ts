export function formatDate(iso: string): string {
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return iso
  return d.toLocaleString('es-AR', { dateStyle: 'short', timeStyle: 'short' })
}

// Ubicación donde se tomó la foto, con la precisión informada por el GPS del navegador.
export function formatLocation(lat: number | null, lon: number | null, accuracy: number | null): string | null {
  if (lat === null || lon === null) return null
  const coords = `${lat.toFixed(5)}, ${lon.toFixed(5)}`
  return accuracy === null ? coords : `${coords} (±${Math.round(accuracy)} m)`
}

// Enlace a Google Maps en vista satélite (URLs API, sin clave)
export function mapUrl(lat: number, lon: number): string {
  return `https://www.google.com/maps/@?api=1&map_action=map&center=${lat},${lon}&zoom=19&basemap=satellite`
}
