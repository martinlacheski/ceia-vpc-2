import { Navigate, Route, Routes } from 'react-router-dom'
import { AuthProvider, useAuth } from './auth'
import Layout from './components/Layout'
import Login from './pages/Login'
import ReaderHome from './pages/ReaderHome'
import RouteView from './pages/RouteView'
import TakeReading from './pages/TakeReading'
import Admin from './pages/Admin'

// Guarda de ruta: sin sesión (401 en /api/auth/me) se redirige al login
function Protected({ role }: { role?: 'admin' | 'reader' }) {
  const { user, loading } = useAuth()
  if (loading) return <p className="center muted">Cargando…</p>
  if (!user) return <Navigate to="/login" replace />
  if (role && user.role !== role) return <Navigate to="/" replace />
  return <Layout />
}

function Home() {
  const { user } = useAuth()
  return <Navigate to={user?.role === 'admin' ? '/admin' : '/rutas'} replace />
}

export default function App() {
  return (
    <AuthProvider>
      <Routes>
        <Route path="/login" element={<Login />} />
        <Route element={<Protected />}>
          <Route path="/" element={<Home />} />
          <Route path="/rutas" element={<ReaderHome />} />
          <Route path="/routes/:routeId" element={<RouteView />} />
          <Route path="/meters/:meterId/read" element={<TakeReading />} />
        </Route>
        <Route element={<Protected role="admin" />}>
          <Route path="/admin" element={<Admin />} />
        </Route>
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </AuthProvider>
  )
}
