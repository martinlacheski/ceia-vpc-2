import { MoonIcon, SunIcon } from 'lucide-react'
import { useState } from 'react'
import { Button } from '../ui/button'

// Mismo mecanismo que saas-base: clase `dark` en <html> y preferencia en localStorage ("web-theme")
const KEY = 'web-theme'

export default function ThemeToggle() {
  const [dark, setDark] = useState(() => document.documentElement.classList.contains('dark'))

  function toggle() {
    const next = !dark
    const root = document.documentElement
    root.classList.toggle('dark', next)
    root.dataset.theme = next ? 'dark' : 'light'
    root.style.colorScheme = next ? 'dark' : 'light'
    try {
      window.localStorage.setItem(KEY, next ? 'dark' : 'light')
    } catch {
      /* sin almacenamiento: queda solo en memoria */
    }
    setDark(next)
  }

  const label = dark ? 'Cambiar a tema claro' : 'Cambiar a tema oscuro'
  return (
    <Button variant="ghost" size="icon" type="button" onClick={toggle} aria-label={label} title={label}>
      {dark ? <SunIcon /> : <MoonIcon />}
    </Button>
  )
}
