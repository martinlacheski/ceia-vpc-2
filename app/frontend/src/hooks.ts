import { useEffect, useState } from 'react'

export interface Position {
  lat: number
  lon: number
  accuracy_m: number
}

export type GpsStatus = 'searching' | 'ok' | 'unavailable'

// Sigue el GPS mientras la pantalla está abierta y conserva la posición más precisa recibida.
// El GPS tarda en "engancharse": empezar antes de sacar la foto le da tiempo a mejorar la precisión.
// maximumAge 0 evita que el navegador devuelva una posición vieja (por ejemplo, la estimada por Wi-Fi).
export function useBestPosition(): { best: Position | null; status: GpsStatus } {
  const [best, setBest] = useState<Position | null>(null)
  const [status, setStatus] = useState<GpsStatus>('searching')
  useEffect(() => {
    if (typeof navigator === 'undefined' || !navigator.geolocation) {
      setStatus('unavailable')
      return
    }
    const geo = navigator.geolocation
    const id = geo.watchPosition(
      (p) => {
        const next = { lat: p.coords.latitude, lon: p.coords.longitude, accuracy_m: p.coords.accuracy }
        setBest((prev) => (prev === null || next.accuracy_m < prev.accuracy_m ? next : prev))
        setStatus('ok')
      },
      () => setStatus((s) => (s === 'ok' ? s : 'unavailable')),
      { enableHighAccuracy: true, maximumAge: 0, timeout: 60000 },
    )
    return () => geo.clearWatch(id)
  }, [])
  return { best, status }
}

// Ubicación actual; null si se deniega o no está disponible
export function getPosition(timeoutMs = 8000): Promise<Position | null> {
  return new Promise((resolve) => {
    if (typeof navigator === 'undefined' || !navigator.geolocation) return resolve(null)
    navigator.geolocation.getCurrentPosition(
      (p) => resolve({ lat: p.coords.latitude, lon: p.coords.longitude, accuracy_m: p.coords.accuracy }),
      () => resolve(null),
      { enableHighAccuracy: true, timeout: timeoutMs, maximumAge: 0 },
    )
  })
}

// Carga asíncrona simple con estado de error
export function useLoad<T>(fn: () => Promise<T>, deps: unknown[]) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  useEffect(() => {
    let cancelled = false
    setLoading(true)
    setError(null)
    fn()
      .then((d) => !cancelled && setData(d))
      .catch((e: Error) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoading(false))
    return () => {
      cancelled = true
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)
  return { data, error, loading }
}
