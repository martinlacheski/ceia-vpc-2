import { Badge } from '../ui/badge'

export default function StatusBadge({ status }: { status: 'pending' | 'read' }) {
  return status === 'read' ? <Badge variant="success">Leído</Badge> : <Badge variant="warning">Pendiente</Badge>
}
