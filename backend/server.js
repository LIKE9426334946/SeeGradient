import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';

const root = fileURLToPath(new URL('../public/', import.meta.url));
const assets = new Map([
  ['/', ['index.html', 'text/html; charset=utf-8']],
  ['/index.html', ['index.html', 'text/html; charset=utf-8']],
  ['/styles.css', ['styles.css', 'text/css; charset=utf-8']],
  ['/app.js', ['app.js', 'text/javascript; charset=utf-8']],
  ['/math.js', ['math.js', 'text/javascript; charset=utf-8']],
  ['/charts.js', ['charts.js', 'text/javascript; charset=utf-8']],
  ['/favicon.svg', ['favicon.svg', 'image/svg+xml']]
]);

export function createServer() {
  return http.createServer(async (req, res) => {
    const headers = {
      'X-Content-Type-Options': 'nosniff',
      'Referrer-Policy': 'same-origin',
      'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
      'Cache-Control': 'no-cache'
    };
    if (req.method !== 'GET' && req.method !== 'HEAD') {
      res.writeHead(405, { ...headers, Allow: 'GET, HEAD' });
      res.end('Method not allowed');
      return;
    }
    let path;
    try { path = new URL(req.url, 'http://localhost').pathname; }
    catch { res.writeHead(400, headers); res.end('Bad request'); return; }
    if (path === '/api/health') {
      res.writeHead(200, { ...headers, 'Content-Type': 'application/json; charset=utf-8' });
      res.end(req.method === 'HEAD' ? undefined : JSON.stringify({ status: 'ok', app: 'SeeGradient' }));
      return;
    }
    const asset = assets.get(path);
    if (!asset) { res.writeHead(404, headers); res.end('Not found'); return; }
    try {
      const content = await readFile(resolve(root, asset[0]));
      res.writeHead(200, { ...headers, 'Content-Type': asset[1], 'Content-Length': content.length });
      res.end(req.method === 'HEAD' ? undefined : content);
    } catch {
      res.writeHead(500, headers);
      res.end('Unable to load asset');
    }
  });
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const host = process.env.HOST || '127.0.0.1';
  const port = Number(process.env.PORT || 3048);
  if (host !== '127.0.0.1' && host !== '::1') throw new Error('HOST must be a loopback address. Use Nginx for public access.');
  if (!Number.isInteger(port) || port < 1 || port > 65535) throw new Error('PORT must be a valid port number.');
  const server = createServer();
  server.listen(port, host, () => console.log(`SeeGradient listening on http://${host}:${port}`));
  const shutdown = () => { server.close(() => process.exit(0)); server.closeIdleConnections(); };
  process.on('SIGTERM', shutdown);
  process.on('SIGINT', shutdown);
}
