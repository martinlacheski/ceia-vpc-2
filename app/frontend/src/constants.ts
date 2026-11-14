// Por debajo de esta confianza la lectura propuesta se marca para revisar a mano (S13)
export const LOW_CONFIDENCE_THRESHOLD = 0.5

export function isLowConfidence(confidence: number | null | undefined): boolean {
  return confidence !== null && confidence !== undefined && confidence < LOW_CONFIDENCE_THRESHOLD
}

// Una sola observación por lectura, en este orden: la revisó el admin, la corrigió el lecturista,
// o se aceptó tal cual con confianza baja y conviene revisarla.
export type Observation = 'revisada' | 'corregida' | 'revisar'

export function observation(reading: {
  corrected: boolean
  confidence: number | null
  review: unknown | null
}): Observation | null {
  if (reading.review) return 'revisada'
  if (reading.corrected) return 'corregida'
  if (isLowConfidence(reading.confidence)) return 'revisar'
  return null
}
