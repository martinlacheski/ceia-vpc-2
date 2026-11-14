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

export function mapUrl(lat: number, lon: number): string {
  return `https://www.openstreetmap.org/?mlat=${lat}&mlon=${lon}#map=18/${lat}/${lon}`
}
