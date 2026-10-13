import { dragValue } from './math.js';

export function formatNumber(value, digits = 4) {
  if (!Number.isFinite(value)) return '溢出';
  if (value === 0) return '0';
  const magnitude = Math.abs(value);
  if (magnitude >= 1e5 || magnitude < 10 ** -digits) return value.toExponential(2);
  return value.toFixed(digits).replace(/\.?0+$/, '');
}

function palette() {
  const css = getComputedStyle(document.documentElement);
  const get = name => css.getPropertyValue(name).trim();
  return { ink: get('--ink'), muted: get('--muted'), border: get('--border'), panel: get('--panel'), selected: get('--selected'), accent: get('--accent'), positive: get('--positive'), negative: get('--negative'), grid: get('--grid'), field: get('--field') };
}

function prepare(canvas, width, height) {
  const ratio = Math.min(window.devicePixelRatio || 1, 3);
  const w = Math.round(width * ratio), h = Math.round(height * ratio);
  if (canvas.width !== w || canvas.height !== h) { canvas.width = w; canvas.height = h; }
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  const ctx = canvas.getContext('2d');
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return ctx;
}

function roundRect(ctx, x, y, width, height, radius, fill, stroke) {
  ctx.beginPath();
  ctx.roundRect(x, y, width, height, radius);
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke(); }
}

export class MatrixPlot {
  constructor(canvas, callbacks = {}, editable = false) {
    this.canvas = canvas;
    this.callbacks = callbacks;
    this.editable = editable;
    this.drag = null;
    this.data = null;
    this.observer = new ResizeObserver(() => this.draw());
    this.observer.observe(canvas.closest('.chart-scroll'));
    canvas.addEventListener('pointerdown', event => this.pointerDown(event));
    canvas.addEventListener('pointermove', event => this.pointerMove(event));
    canvas.addEventListener('pointerup', event => this.pointerEnd(event, false));
    canvas.addEventListener('pointercancel', event => this.pointerEnd(event, true));
    canvas.addEventListener('lostpointercapture', event => this.pointerEnd(event, true));
    canvas.addEventListener('keydown', event => this.keyDown(event));
    canvas.addEventListener('dblclick', () => this.callbacks.edit?.());
  }

  setData(data) {
    this.data = { ...data, range: this.drag?.range ?? data.range };
    this.draw();
  }

  draw() {
    if (!this.data) return;
    const { matrix, name, selected, range } = this.data;
    const rows = matrix.length, cols = matrix[0].length;
    const available = this.canvas.closest('.chart-scroll').clientWidth - 10;
    const width = Math.max(available, cols * 104 + (cols - 1) * 8);
    const cellWidth = (width - (cols - 1) * 8) / cols;
    const cellHeight = 146;
    const height = rows * cellHeight + (rows - 1) * 8;
    this.layout = { width, height, cellWidth, cellHeight, rows, cols, halfHeight: 38 };
    const ctx = prepare(this.canvas, width, height);
    const p = palette();
    ctx.textBaseline = 'middle';
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < cols; j++) {
        const x = j * (cellWidth + 8), y = i * (cellHeight + 8);
        const value = matrix[i][j];
        const active = selected.i === i && selected.j === j;
        roundRect(ctx, x + 1, y + 1, cellWidth - 2, cellHeight - 2, 7, active ? p.selected : p.field, active ? p.accent : p.border);
        ctx.font = '9px Consolas, monospace';
        ctx.fillStyle = p.muted;
        ctx.textAlign = 'left';
        ctx.fillText(`${name}[${i},${j}]`, x + 11, y + 17);
        ctx.font = '500 15px Consolas, monospace';
        ctx.fillStyle = p.ink;
        ctx.fillText(formatNumber(value), x + 11, y + 36);
        const baseline = y + 96;
        ctx.strokeStyle = p.grid;
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 3]);
        for (const yy of [baseline - 38, baseline + 38]) {
          ctx.beginPath(); ctx.moveTo(x + 9, yy); ctx.lineTo(x + cellWidth - 9, yy); ctx.stroke();
        }
        ctx.setLineDash([]);
        ctx.beginPath(); ctx.moveTo(x + 9, baseline); ctx.lineTo(x + cellWidth - 9, baseline); ctx.stroke();
        ctx.fillStyle = p.muted;
        ctx.font = '8px Consolas, monospace';
        ctx.fillText('0', x + 8, baseline - 6);
        const barWidth = Math.max(20, Math.min(40, cellWidth * 0.24));
        const barX = x + (cellWidth - barWidth) / 2;
        const barY = baseline - Math.max(-1, Math.min(1, value / range)) * 38;
        const color = value >= 0 ? p.positive : p.negative;
        roundRect(ctx, barX, Math.min(barY, baseline), barWidth, Math.max(1.5, Math.abs(barY - baseline)), 2, color);
        if (this.editable) {
          ctx.beginPath(); ctx.arc(x + cellWidth / 2, barY, 4.5, 0, Math.PI * 2);
          ctx.fillStyle = p.panel; ctx.fill(); ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
        }
        if (active) {
          ctx.beginPath(); ctx.arc(x + cellWidth - 12, y + 16, 2.5, 0, Math.PI * 2); ctx.fillStyle = p.accent; ctx.fill();
        }
      }
    }
    this.canvas.setAttribute('aria-label', `${name}，${rows} 行 ${cols} 列。${matrix.map((row, i) => row.map((v, j) => `[${i},${j}]=${formatNumber(v)}`).join('，')).join('；')}。${this.editable ? '左右方向键选择，上下方向键调整。' : '梯度只读，方向键选择。'}`);
  }

  hit(event) {
    if (!this.layout) return null;
    const bounds = this.canvas.getBoundingClientRect();
    const x = (event.clientX - bounds.left) * this.layout.width / bounds.width;
    const y = (event.clientY - bounds.top) * this.layout.height / bounds.height;
    const j = Math.floor(x / (this.layout.cellWidth + 8));
    const i = Math.floor(y / (this.layout.cellHeight + 8));
    if (i < 0 || j < 0 || i >= this.layout.rows || j >= this.layout.cols) return null;
    if (x - j * (this.layout.cellWidth + 8) > this.layout.cellWidth || y - i * (this.layout.cellHeight + 8) > this.layout.cellHeight) return null;
    return { i, j };
  }

  pointerDown(event) {
    if (this.drag || (event.pointerType === 'mouse' && event.button !== 0)) return;
    const hit = this.hit(event);
    if (!hit) return;
    event.preventDefault();
    this.canvas.focus({ preventScroll: true });
    if (this.editable) {
      this.drag = { ...hit, id: event.pointerId, initial: this.data.matrix[hit.i][hit.j], startY: event.clientY, range: this.data.range, halfHeight: this.layout.halfHeight, moved: false };
      this.canvas.setPointerCapture(event.pointerId);
      this.callbacks.begin?.();
      document.body.classList.add('dragging');
    }
    this.callbacks.select?.(hit.i, hit.j);
  }

  pointerMove(event) {
    if (!this.drag || this.drag.id !== event.pointerId) return;
    const d = this.drag;
    d.moved = d.moved || Math.abs(event.clientY - d.startY) > 0.5;
    if (!d.moved) return;
    this.callbacks.change?.(dragValue(d.initial, d.startY, event.clientY, d.range, d.halfHeight, event.shiftKey));
  }

  pointerEnd(event, cancelled) {
    if (!this.drag || this.drag.id !== event.pointerId) return;
    const d = this.drag;
    this.drag = null;
    if (this.canvas.hasPointerCapture(event.pointerId)) this.canvas.releasePointerCapture(event.pointerId);
    document.body.classList.remove('dragging');
    if (cancelled) this.callbacks.change?.(d.initial);
    this.callbacks.end?.(d.moved && !cancelled);
  }

  keyDown(event) {
    if (event.key === 'Escape' && this.drag) { this.pointerEnd({ pointerId: this.drag.id }, true); return; }
    if (!this.data || this.drag) return;
    const { selected, matrix } = this.data;
    const cols = matrix[0].length;
    const index = selected.i * cols + selected.j;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight' || (!this.editable && ['ArrowUp', 'ArrowDown'].includes(event.key))) {
      event.preventDefault();
      const delta = event.key === 'ArrowLeft' ? -1 : event.key === 'ArrowRight' ? 1 : event.key === 'ArrowUp' ? -cols : cols;
      const next = Math.max(0, Math.min(matrix.length * cols - 1, index + delta));
      this.callbacks.select?.(Math.floor(next / cols), next % cols);
    } else if (this.editable && ['ArrowUp', 'ArrowDown', 'Home'].includes(event.key)) {
      event.preventDefault();
      const step = event.shiftKey ? 0.001 : 0.01;
      const next = event.key === 'Home' ? 0 : matrix[selected.i][selected.j] + (event.key === 'ArrowUp' ? step : -step);
      this.callbacks.begin?.();
      this.callbacks.change?.(Number(next.toFixed(6)));
      this.callbacks.end?.(true);
    } else if (event.key === 'Enter') { event.preventDefault(); this.callbacks.edit?.(); }
  }
}

export class HistoryPlot {
  constructor(canvas) {
    this.canvas = canvas;
    this.values = [];
    this.observer = new ResizeObserver(() => this.draw());
    this.observer.observe(canvas.parentElement);
  }
  setData(values) { this.values = values; this.draw(); }
  draw() {
    const width = Math.max(120, this.canvas.parentElement.clientWidth - 40), height = 159;
    const ctx = prepare(this.canvas, width, height);
    const p = palette();
    const values = this.values.filter(Number.isFinite);
    const max = Math.max(1e-12, ...values) * 1.1;
    const left = 51, right = width - 9, top = 14, bottom = height - 17;
    const toY = value => bottom - value / max * (bottom - top);
    ctx.font = '9px Consolas, monospace';
    for (let tick = 0; tick <= 3; tick++) {
      const value = max * tick / 3;
      const y = toY(value);
      ctx.fillStyle = p.muted; ctx.textAlign = 'left'; ctx.fillText(formatNumber(value, 3), 0, y + 3);
      ctx.strokeStyle = p.grid; ctx.setLineDash([2, 4]); ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(right, y); ctx.stroke();
    }
    ctx.setLineDash([]);
    if (!values.length) return;
    const points = values.map((v, i) => ({ x: left + i / Math.max(1, values.length - 1) * (right - left), y: toY(v) }));
    if (points.length > 1) {
      const gradient = ctx.createLinearGradient(0, top, 0, bottom);
      gradient.addColorStop(0, `${p.positive}25`); gradient.addColorStop(1, `${p.positive}00`);
      ctx.beginPath(); ctx.moveTo(points[0].x, bottom);
      for (const pt of points) ctx.lineTo(pt.x, pt.y);
      ctx.lineTo(points.at(-1).x, bottom); ctx.closePath(); ctx.fillStyle = gradient; ctx.fill();
      ctx.beginPath(); ctx.moveTo(points[0].x, points[0].y);
      for (const pt of points.slice(1)) ctx.lineTo(pt.x, pt.y);
      ctx.strokeStyle = p.accent; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.stroke();
    }
    const last = points.at(-1);
    ctx.beginPath(); ctx.arc(last.x, last.y, 3.5, 0, Math.PI * 2); ctx.fillStyle = p.accent; ctx.fill();
    this.canvas.setAttribute('aria-label', `最近 ${values.length} 次 MSE 记录：初始 ${formatNumber(values[0], 6)}，当前 ${formatNumber(values.at(-1), 6)}。`);
  }
}
