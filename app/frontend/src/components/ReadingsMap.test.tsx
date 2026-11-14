import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import ReadingsMap from './ReadingsMap'

const reading = {
  id: 1, meter_id: 3, value: '00123', proposed_value: '00123', corrected: false, confidence: 0.9,
  lat: -34.6, lon: -58.4, accuracy_m: 8, taken_at: '2026-01-01T10:00:00Z', user: { id: 2, full_name: 'Ana' }, route: { id: 1, name: 'Ruta Norte' }, final_value: '01234', review: null,
}

describe('mapa de lecturas', () => {
  it('permite elegir entre Calles y Satélite', () => {
    const { container } = render(<ReadingsMap readings={[reading]} onOpen={() => {}} />)
    expect(screen.getByLabelText('Satélite')).toBeChecked()
    expect(screen.getByLabelText('Calles')).not.toBeChecked()
    const tiles = [...container.querySelectorAll('img.leaflet-tile')].map((i) => i.getAttribute('src'))
    expect(tiles.some((t) => t?.includes('server.arcgisonline.com'))).toBe(true)
  })
})
