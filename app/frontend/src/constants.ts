// Por debajo de esta confianza la lectura propuesta se marca para revisar a mano (S13)
export const LOW_CONFIDENCE_THRESHOLD = 0.5

export function isLowConfidence(confidence: number | null | undefined): boolean {
  return confidence !== null && confidence !== undefined && confidence < LOW_CONFIDENCE_THRESHOLD
}
