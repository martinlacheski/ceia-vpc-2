export default function StatusBadge({ status }: { status: 'pending' | 'read' }) {
  return (
    <span className={`badge ${status === 'read' ? 'badge-ok' : 'badge-pending'}`}>
      {status === 'read' ? 'Leído' : 'Pendiente'}
    </span>
  )
}
