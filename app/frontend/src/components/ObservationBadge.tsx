import { observation } from '../constants'
import { Badge } from '../ui/badge'

const VARIANT = { revisada: 'success', corregida: 'warning', revisar: 'destructive' } as const

// Marca única de la lectura (revisada, corregida o revisar); nada si no hay observación
export default function ObservationBadge({ reading }: { reading: Parameters<typeof observation>[0] }) {
  const obs = observation(reading)
  return obs ? <Badge variant={VARIANT[obs]}>{obs}</Badge> : null
}
