import test from 'node:test';
import assert from 'node:assert/strict';
import { once } from 'node:events';
import { readFile } from 'node:fs/promises';
import { createServer } from '../backend/server.js';

test('HTTP 服务：页面、全部模块、健康检查、HEAD、白名单与请求方法', async t => {
  const server = createServer();
  server.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(async () => { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); });
  const base = `http://127.0.0.1:${server.address().port}`;
  const health = await fetch(`${base}/api/health`);
  assert.equal(health.status, 200);
  assert.deepEqual(await health.json(), { status: 'ok', app: 'SeeGradient' });
  for (const [path, type] of [['/', 'text/html'], ['/styles.css', 'text/css'], ['/app.js', 'text/javascript'], ['/math.js', 'text/javascript'], ['/charts.js', 'text/javascript'], ['/favicon.svg', 'image/svg+xml']]) {
    const response = await fetch(`${base}${path}`);
    assert.equal(response.status, 200, path);
    assert.ok(response.headers.get('content-type').startsWith(type));
    assert.ok((await response.text()).length > 50);
    assert.equal(response.headers.get('x-content-type-options'), 'nosniff');
    assert.ok(response.headers.get('content-security-policy').includes("script-src 'self'"));
  }
  const head = await fetch(`${base}/`, { method: 'HEAD' });
  assert.equal(head.status, 200);
  assert.equal(await head.text(), '');
  assert.ok(Number(head.headers.get('content-length')) > 0);
  for (const path of ['/package.json', '/backend/server.js', '/.git/config', '/%2e%2e/package.json', '/does-not-exist']) assert.equal((await fetch(base + path)).status, 404, path);
  const post = await fetch(base, { method: 'POST' });
  assert.equal(post.status, 405);
  assert.equal(post.headers.get('allow'), 'GET, HEAD');
});

test('页面中的脚本控件引用都有唯一对应 ID，前端不依赖外部资源', async () => {
  const html = await readFile(new URL('../public/index.html', import.meta.url), 'utf8');
  const app = await readFile(new URL('../public/app.js', import.meta.url), 'utf8');
  const ids = [...html.matchAll(/\bid="([^"]+)"/g)].map(match => match[1]);
  assert.equal(new Set(ids).size, ids.length, 'HTML IDs must be unique');
  for (const [, id] of app.matchAll(/\$\('([^']+)'\)/g)) assert.ok(ids.includes(id), `Missing control: ${id}`);
  assert.ok(!/(?:src|href)="https?:/.test(html));
  assert.ok(html.includes('type="module"'));
});

test('部署端口、root、工作目录和 WebSocket 配置符合要求', async () => {
  const service = await readFile(new URL('../deploy/SeeGradient.service', import.meta.url), 'utf8');
  const nginx = await readFile(new URL('../deploy/SeeGradient.nginx.conf', import.meta.url), 'utf8');
  for (const setting of ['User=root', 'WorkingDirectory=/opt/SeeGradient', 'Environment=HOST=127.0.0.1', 'Environment=PORT=3048', 'ExecStart=/usr/bin/node /opt/SeeGradient/backend/server.js', 'Restart=always']) assert.ok(service.includes(setting), setting);
  for (const setting of ['listen 16048;', 'proxy_pass http://127.0.0.1:3048;', 'proxy_set_header Upgrade $http_upgrade;', 'proxy_set_header Connection $seegradient_connection_upgrade;']) assert.ok(nginx.includes(setting), setting);
});
