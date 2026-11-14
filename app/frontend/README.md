# Frontend: lecturas de medidores

SPA en React + TypeScript (Vite) para el lecturista y el administrador. Consume la API bajo `/api`.

## Desarrollo

```bash
npm install
npm run dev        # http://localhost:5173
```

El servidor de desarrollo redirige `/api` y `/health` a `http://localhost:8000`.
Para apuntar a otro backend: `VITE_API_PROXY=http://host:puerto npm run dev`.

## Compilar y probar

```bash
npm run build      # tsc + vite build, salida en dist/
npm test -- --run  # vitest (jsdom, fetch simulado)
```

## Docker

`Dockerfile` compila con Node 22 y sirve `dist/` con nginx. `nginx.conf` hace fallback a
`index.html` y reenvía `/api/` y `/health` al servicio `api:8000` de docker compose
(`client_max_body_size 20m` para las fotos).

## Notas

- La geolocalización requiere HTTPS o `localhost` en el navegador.
- Los mapas usan teselas de OpenStreetMap (sin clave).
