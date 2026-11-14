import type { ReactNode } from 'react'

// Encabezado de página al estilo de saas-base (título, descripción y acciones)
export default function PageHeader({ title, description, backAction }: { title: ReactNode; description?: ReactNode; backAction?: ReactNode }) {
  return (
    <div className="flex flex-col gap-2">
      {backAction}
      <h1 className="text-xl font-bold tracking-tight text-balance break-words lg:text-2xl xl:text-3xl">{title}</h1>
      {description && <p className="text-muted-foreground">{description}</p>}
    </div>
  )
}
