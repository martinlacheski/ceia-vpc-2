import { Link, Outlet, useNavigate } from 'react-router-dom'
import { useAuth } from '../auth'

export default function Layout() {
  const { user, logout } = useAuth()
  const navigate = useNavigate()

  async function exit() {
    await logout()
    navigate('/login', { replace: true })
  }

  return (
    <div className="app">
      <header className="topbar">
        <Link to="/" className="brand">
          Lecturas
        </Link>
        <div className="who">
          <span className="who-name">{user?.full_name}</span>
          <span className="role">{user?.role === 'admin' ? 'Administrador' : 'Lecturista'}</span>
        </div>
        <button className="btn btn-ghost" onClick={exit}>
          Salir
        </button>
      </header>
      <main className="content">
        <Outlet />
      </main>
    </div>
  )
}
