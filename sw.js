// Service worker de VISIONA: modo sin conexión para la interfaz y los modelos.
// Funciona en la raíz ("/") o bajo un prefijo ("/visiona/"): todo se calcula desde su scope.
// - Modelos, WASM y recursos estáticos: primero caché (son inmutables y pesados).
// - Páginas: primero red, caché como respaldo.
// - /api/: nunca se toca (siempre en vivo; nada de respuestas viejas).
const VERSION = 'visiona-v5';
const SHELL = `${VERSION}-shell`;
const ASSETS = `${VERSION}-assets`;
const SCOPE = new URL(self.registration.scope);
const BASE = SCOPE.pathname.replace(/\/$/, '');
const PAGES = ['', 'ver/', 'oir/', 'senas/', 'sonidos/', 'estudio/', 'ajustes/', 'estado/', 'acerca/', 'manifest.webmanifest'].map((p) => new URL(p, SCOPE).toString());

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches
      .open(SHELL)
      .then((cache) => cache.addAll(PAGES))
      .catch(() => undefined),
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => !k.startsWith(VERSION)).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

function isAsset(url) {
  const p = url.pathname;
  return (
    p.startsWith(`${BASE}/models/`) ||
    p.startsWith(`${BASE}/mediapipe/`) ||
    p.startsWith(`${BASE}/_next/static/`) ||
    p.startsWith(`${BASE}/icons/`) ||
    url.hostname === 'cdn.jsdelivr.net' ||
    url.hostname === 'storage.googleapis.com'
  );
}

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  if (url.pathname.includes('/api/')) return;

  if (isAsset(url)) {
    event.respondWith(
      caches.open(ASSETS).then(async (cache) => {
        const hit = await cache.match(req);
        if (hit) return hit;
        const res = await fetch(req);
        if (res.ok) cache.put(req, res.clone());
        return res;
      }),
    );
    return;
  }

  if (req.mode === 'navigate') {
    event.respondWith(
      fetch(req)
        .then((res) => {
          const copy = res.clone();
          caches.open(SHELL).then((cache) => cache.put(req, copy));
          return res;
        })
        .catch(async () => (await caches.match(req)) || (await caches.match(PAGES[0])) || Response.error()),
    );
  }
});
