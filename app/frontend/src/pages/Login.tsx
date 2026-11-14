import { useState, type FormEvent } from 'react'
import { Navigate, useNavigate } from 'react-router-dom'
import { GaugeIcon } from 'lucide-react'
import { useAuth } from '../auth'
import { Alert, AlertDescription } from '../ui/alert'
import { Button } from '../ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../ui/card'
import { Input } from '../ui/input'

export default function Login() {
  const { user, loading, login } = useAuth()
  const navigate = useNavigate()
  const [username, setUsername] = useState('')
  const [password, setPassword] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  if (!loading && user) return <Navigate to="/" replace />

  async function submit(e: FormEvent) {
    e.preventDefault()
    setBusy(true)
    setError(null)
    try {
      await login(username.trim(), password)
      navigate('/', { replace: true })
    } catch (err) {
      setError((err as Error).message)
    } finally {
      setBusy(false)
    }
  }

  const label = 'text-sm leading-none font-medium'
  return (
    <div className="flex min-h-dvh items-center justify-center bg-background p-4 text-foreground">
      <Card className="w-full max-w-md">
        <form className="flex flex-col gap-6" onSubmit={submit}>
          <CardHeader>
            <GaugeIcon className="size-10 text-brand-accent" aria-hidden="true" />
            <CardTitle className="text-xl">Lecturas de medidores</CardTitle>
            <CardDescription>Ingresá con tu usuario para ver tus rutas.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="flex flex-col gap-2">
              <label className={label} htmlFor="username">Usuario</label>
              <Input id="username" autoComplete="username" autoCapitalize="none" value={username} onChange={(e) => setUsername(e.target.value)} required />
            </div>
            <div className="flex flex-col gap-2">
              <label className={label} htmlFor="password">Contraseña</label>
              <Input id="password" type="password" autoComplete="current-password" value={password} onChange={(e) => setPassword(e.target.value)} required />
            </div>
            {error && (
              <Alert variant="destructive">
                <AlertDescription className="text-destructive">{error}</AlertDescription>
              </Alert>
            )}
            <Button type="submit" disabled={busy}>
              {busy ? 'Ingresando…' : 'Ingresar'}
            </Button>
          </CardContent>
        </form>
      </Card>
    </div>
  )
}
