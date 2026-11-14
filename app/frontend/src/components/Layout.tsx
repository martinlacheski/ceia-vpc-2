import { GaugeIcon, LayoutDashboardIcon, LogOutIcon, RouteIcon } from 'lucide-react'
import { Link, NavLink, Outlet, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth'
import { Button } from '../ui/button'
import { cn } from '../ui/cn'
import ThemeToggle from './ThemeToggle'

// Estructura de saas-base: barra lateral compacta (md+), encabezado con controles y contenido con scroll propio
export default function Layout() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()
  const isAdmin = user?.role === 'admin'
  const items = isAdmin
    ? [{ to: '/admin', label: 'Panel', icon: <LayoutDashboardIcon /> }]
    : [{ to: '/rutas', label: 'Mis rutas', icon: <RouteIcon /> }]

  async function exit() {
    await logout()
    navigate('/login', { replace: true })
  }

  const navClass = ({ isActive }: { isActive: boolean }) =>
    cn(
      'flex h-11 items-center gap-3 rounded-lg px-3 text-sm font-medium transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground [&_svg]:size-5',
      isActive && 'bg-sidebar-accent text-sidebar-accent-foreground',
    )

  return (
    <div className="flex h-dvh overflow-hidden bg-background text-foreground">
      <aside className="hidden w-56 shrink-0 flex-col gap-2 border-r border-sidebar-border bg-sidebar p-4 text-sidebar-foreground md:flex">
        <Link to="/" className="mb-2 flex h-11 items-center gap-2 px-3 text-lg font-bold tracking-tight">
          <GaugeIcon className="size-6 text-brand-accent" aria-hidden="true" />
          Lecturas
        </Link>
        <nav aria-label="Navegación principal" className="flex flex-col gap-1">
          {items.map((it) => (
            <NavLink key={it.to} to={it.to} className={navClass}>
              {it.icon}
              {it.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center gap-2 border-b border-border bg-background px-3 py-2 sm:px-6" role="banner">
          <Link to="/" className="flex items-center gap-2 text-lg font-bold tracking-tight md:hidden">
            <GaugeIcon className="size-6 text-brand-accent" aria-hidden="true" />
            Lecturas
          </Link>
          <div className="ml-auto flex items-center gap-2">
            <div className="flex flex-col items-end leading-tight">
              <span className="text-sm font-medium">{user?.full_name}</span>
              <span className="text-xs text-muted-foreground">{isAdmin ? 'Administrador' : 'Lecturista'}</span>
            </div>
            <ThemeToggle />
            <Button variant="outline" size="sm" onClick={exit}>
              <LogOutIcon /> Salir
            </Button>
          </div>
        </header>
        <main className="min-w-0 flex-1 overflow-y-auto p-4 xl:p-5 2xl:p-6">
          <div className="mx-auto flex max-w-5xl flex-col gap-6">
            <Outlet />
          </div>
        </main>
      </div>
    </div>
  )
}
