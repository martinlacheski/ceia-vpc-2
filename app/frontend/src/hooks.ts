import { useEffect, useState } from 'react'

export interface Position {
  lat: number
  lon: number
  accuracy_m: number
}

// Ubicación actual; null si se deniega o no está disponible
export function getPosition(timeoutMs = 8000): Promise<Position | null> {
  return new Promise((resolve) => {
    if (typeof navigator === 'undefined' || !navigator.geolocation) return resolve(null)
    navigator.geolocation.getCurrentPosition(
      (p) => resolve({ lat: p.coords.latitude, lon: p.coords.longitude, accuracy_m: p.coords.accuracy }),
      () => resolve(null),
      { enableHighAccuracy: true, timeout: timeoutMs, maximumAge: 10000 },
    )
  })
}

// Seguimiento de la posición para el mapa; `denied` si el usuario la rechaza
export function useWatchPosition(): { position: Position | null; denied: boolean } {
  const [position, setPosition] = useState<Position | null>(null)
  const [denied, setDenied] = useState(false)
  useEffect(() => {
    if (!navigator.geolocation) {
      setDenied(true)
      return
    }
    const id = navigator.geolocation.watchPosition(
      (p) => setPosition({ lat: p.coords.latitude, lon: p.coords.longitude, accuracy_m: p.coords.accuracy }),
      () => setDenied(true),
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 10000 },
    )
    return () => navigator.geolocation.clearWatch(id)
  }, [])
  return { position, denied }
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
