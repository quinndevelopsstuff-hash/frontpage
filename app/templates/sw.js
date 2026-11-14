// Minimal service worker: caches the app shell only (CSS, JS, icons and an
// offline page). Articles and API calls always go to the network.
const CACHE = "frontpage-shell-{{ version }}";
const SHELL = [
  "{{ url_for('main.offline') }}",
  "{{ url_for('static', filename='app.css', v=version) }}",
  "{{ url_for('static', filename='app.js', v=version) }}",
  "{{ url_for('static', filename='icons/icon-192.png') }}",
  "{{ url_for('static', filename='icons/apple-touch-icon.png') }}",
];

self.addEventListener("install", (event) => {
  event.waitUntil(caches.open(CACHE).then((cache) => cache.addAll(SHELL)).then(() => self.skipWaiting()));
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const request = event.request;
  const url = new URL(request.url);
  if (request.method !== "GET" || url.origin !== self.location.origin) return;

  if (url.pathname.startsWith("/static/")) {
    event.respondWith(caches.match(request).then((hit) => hit || fetch(request)));
    return;
  }
  if (request.mode === "navigate") {
    event.respondWith(fetch(request).catch(() => caches.match("{{ url_for('main.offline') }}")));
  }
});
