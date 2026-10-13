import { LIMITS, copy, parseMatrix, parseHidden, validateModel, createModel, demoModel, evaluate, gradientStep, parameterCount, niceRange } from './math.js';
import { MatrixPlot, HistoryPlot, formatNumber } from './charts.js';

const $ = id => document.getElementById(id);
const STORAGE_KEY = 'seegradient.experiment.v1';
const THEME_KEY = 'seegradient.theme';
let storageAvailable = true;
let loadWarning = '';
let state = { model: demoModel(), steps: 0, history: [], learningRate: 0.05, speed: 300, layer: 0, tensor: 'w', selected: { i: 0, j: 0 } };
try {
  const raw = localStorage.getItem(STORAGE_KEY);
  if (raw) {
    if (raw.length > 200000) throw new Error('本地实验数据过大');
    const saved = JSON.parse(raw);
    if (saved.version !== 1) throw new Error('存档版本不兼容');
    validateModel(saved.model);
    if (!evaluate(saved.model).finite) throw new Error('存档数值无效');
    state.model = saved.model;
    state.steps = Number.isSafeInteger(saved.steps) && saved.steps >= 0 ? saved.steps : 0;
    state.learningRate = Number.isFinite(saved.learningRate) && saved.learningRate > 0 && saved.learningRate <= 10 ? saved.learningRate : 0.05;
    state.speed = [100, 300, 800].includes(saved.speed) ? saved.speed : 300;
    state.history = Array.isArray(saved.history) ? saved.history.filter(v => typeof v === 'number' && Number.isFinite(v) && v >= 0).slice(-240) : [];
  }
} catch (error) {
  if (error.name === 'SecurityError') storageAvailable = false;
  else loadWarning = '上次保存的数据无法读取，已载入入门示例。';
}

let result = evaluate(state.model);
let previousLoss = null;
let undoState = null;
let timer = null;
let renderFrame = null;
let saveTimer = null;
let lastHistoryTime = 0;
let formDirty = false;
let editHasSnapshot = false;
let lastRenderedSignature = '';
let lastPredictionShape = '';
let predictionCells = [];
let transaction = null;
if (!state.history.length || state.history.at(-1) !== result.loss) state.history.push(result.loss);

const subscripts = n => String(n).replace(/\d/g, d => '₀₁₂₃₄₅₆₇₈₉'[Number(d)]);
const layerSymbol = () => `${state.tensor === 'w' ? 'W' : 'b'}${subscripts(state.layer + 1)}`;
const selectedMatrix = () => state.model.layers[state.layer][state.tensor];
const selectedGradient = () => result.gradients[state.layer][state.tensor];
const selectedValue = () => selectedMatrix()[state.selected.i][state.selected.j];
const snapshot = () => copy(state);

function notice(message = '') { $('notice').textContent = message; $('notice').hidden = !message; }
function checkpoint() { undoState = snapshot(); $('undo-button').disabled = false; }

function saveNow() {
  clearTimeout(saveTimer);
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: 1, model: state.model, steps: state.steps, learningRate: state.learningRate, speed: state.speed, history: state.history }));
    storageAvailable = true;
    $('save-status').textContent = '已保存到此浏览器';
  } catch {
    storageAvailable = false;
    $('save-status').textContent = '本地保存不可用';
    notice('浏览器当前无法保存实验。你仍可继续操作，但刷新后可能丢失；请检查是否禁用了本地存储。');
  }
}

function scheduleSave() {
  clearTimeout(saveTimer);
  $('save-status').textContent = storageAvailable ? '正在保存…' : '本地保存不可用';
  saveTimer = setTimeout(saveNow, 250);
}

function recordHistory(force = false) {
  const now = performance.now();
  if (result.finite && (force || now - lastHistoryTime >= 100)) {
    if (state.history.at(-1) !== result.loss) state.history.push(result.loss);
    state.history = state.history.slice(-240);
    lastHistoryTime = now;
  }
}

function requestRender() {
  if (renderFrame !== null) return;
  renderFrame = requestAnimationFrame(() => { renderFrame = null; render(); });
}

function recalculate(forceHistory = false) {
  const next = evaluate(state.model);
  if (!next.finite) return false;
  previousLoss = result.loss;
  result = next;
  recordHistory(forceHistory);
  requestRender();
  scheduleSave();
  return true;
}

function pause() {
  if (timer !== null) clearInterval(timer);
  timer = null;
  $('play-button').textContent = '▶ 连续训练';
  $('play-button').setAttribute('aria-pressed', 'false');
}

function resetSelection() {
  state.layer = Math.max(0, Math.min(state.layer, state.model.layers.length - 1));
  if (!state.model.useBias) state.tensor = 'w';
  const matrix = selectedMatrix();
  state.selected.i = Math.min(state.selected.i, matrix.length - 1);
  state.selected.j = Math.min(state.selected.j, matrix[0].length - 1);
}

function select(i, j) {
  state.selected = { i, j };
  editHasSnapshot = false;
  render();
}

function beginParameterEdit() {
  pause();
  transaction = { state: snapshot(), previousUndo: undoState };
  notice();
}

function changeParameter(value) {
  if (!Number.isFinite(value) || Math.abs(value) > LIMITS.parameter) {
    notice(`参数需要是绝对值不超过 ${LIMITS.parameter} 的有限数字。`);
    return false;
  }
  const { i, j } = state.selected;
  const old = selectedMatrix()[i][j];
  selectedMatrix()[i][j] = value;
  if (!recalculate()) {
    selectedMatrix()[i][j] = old;
    notice('这个参数让数值溢出，已保留上一个有效值。请减小输入或参数。');
    return false;
  }
  return true;
}

function endParameterEdit(changed) {
  if (transaction && changed) { undoState = transaction.state; $('undo-button').disabled = false; }
  if (transaction && !changed) {
    state.history = transaction.state.history;
    undoState = transaction.previousUndo;
    $('undo-button').disabled = !undoState;
  }
  transaction = null;
  recordHistory(true);
  render();
  saveNow();
}

const plotCallbacks = { select, begin: beginParameterEdit, change: changeParameter, end: endParameterEdit, edit: () => { $('selected-value').focus(); $('selected-value').select(); } };
const weightPlot = new MatrixPlot($('weight-canvas'), plotCallbacks, true);
const gradientPlot = new MatrixPlot($('gradient-canvas'), { select }, false);
const historyPlot = new HistoryPlot($('history-canvas'));

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  const dark = theme === 'dark';
  $('theme-toggle').setAttribute('aria-label', dark ? '切换浅色模式' : '切换深色模式');
  $('theme-toggle').title = dark ? '切换浅色模式' : '切换深色模式';
  try { localStorage.setItem(THEME_KEY, theme); } catch { /* Experiment saving reports storage problems. */ }
  requestRender();
}
let initialTheme;
try { initialTheme = localStorage.getItem(THEME_KEY); } catch { /* Fall back to system preference. */ }
setTheme(['light', 'dark'].includes(initialTheme) ? initialTheme : matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
$('theme-toggle').addEventListener('click', () => setTheme(document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark'));

function syncForm() {
  const model = state.model;
  $('input-x').value = model.x.map(row => row.join(', ')).join('\n');
  $('target-y').value = model.y.map(row => row.join(', ')).join('\n');
  $('hidden-layers').value = model.hidden.join(', ');
  $('activation').value = model.activation;
  $('use-bias').checked = model.useBias;
  $('learning-rate').value = state.learningRate;
  $('training-speed').value = state.speed;
  formDirty = false;
  $('config-message').className = 'config-message';
  $('config-message').textContent = '输出层使用 Linear，损失使用 MSE。';
}

function loadModel(model) {
  pause();
  checkpoint();
  state.model = model;
  state.steps = 0;
  state.layer = 0; state.tensor = 'w'; state.selected = { i: 0, j: 0 };
  result = evaluate(model);
  previousLoss = null;
  state.history = [result.loss];
  editHasSnapshot = false;
  notice();
  syncForm();
  render();
  saveNow();
}

$('config-form').addEventListener('input', () => {
  formDirty = true;
  pause();
  $('config-message').className = 'config-message pending';
  $('config-message').textContent = '设置已修改，点击下方按钮后生效。';
});
$('config-form').addEventListener('submit', event => {
  event.preventDefault();
  try {
    const config = { x: parseMatrix($('input-x').value, '输入 X'), y: parseMatrix($('target-y').value, '目标 Y'), hidden: parseHidden($('hidden-layers').value), activation: $('activation').value, useBias: $('use-bias').checked };
    const candidate = createModel(config);
    const oldShape = [state.model.x[0].length, ...state.model.hidden, state.model.y[0].length].join(',');
    const newShape = [config.x[0].length, ...config.hidden, config.y[0].length].join(',');
    const sameShape = oldShape === newShape && config.useBias === state.model.useBias;
    if (sameShape) candidate.layers = copy(state.model.layers);
    if (!evaluate(candidate).finite) throw new Error('数据使计算溢出，请减小输入或目标值。');
    loadModel(candidate);
    $('config-message').textContent = sameShape ? '设置已应用，现有参数已保留。' : '设置已应用，参数已按新形状初始化。';
  } catch (error) {
    $('config-message').className = 'config-message error';
    $('config-message').textContent = error.message;
  }
});

$('demo-button').addEventListener('click', () => loadModel(demoModel()));
$('linear-button').addEventListener('click', () => loadModel({ x: [[1, 2], [2, 1]], y: [[1, 0], [0, 1]], hidden: [], activation: 'linear', useBias: false, layers: [{ w: [[0.3, -0.2], [0.1, 0.4]], b: [[0, 0]] }] }));
$('randomize-button').addEventListener('click', () => {
  const { x, y, hidden, activation, useBias } = state.model;
  loadModel(createModel({ x, y, hidden, activation, useBias }));
});
$('undo-button').addEventListener('click', () => {
  if (!undoState) return;
  pause();
  state = undoState;
  undoState = null;
  $('undo-button').disabled = true;
  result = evaluate(state.model);
  previousLoss = null;
  editHasSnapshot = false;
  resetSelection();
  syncForm();
  notice();
  render();
  saveNow();
});

function switchTensor(tensor) {
  if (tensor === 'b' && !state.model.useBias) return;
  state.tensor = tensor;
  state.selected = { i: 0, j: 0 };
  editHasSnapshot = false;
  render();
}
$('weight-tab').addEventListener('click', () => switchTensor('w'));
$('bias-tab').addEventListener('click', () => switchTensor('b'));

$('selected-value').addEventListener('focus', () => { editHasSnapshot = false; });
$('selected-value').addEventListener('input', event => {
  pause();
  const value = event.target.valueAsNumber;
  if (event.target.value === '' || !Number.isFinite(value)) return;
  if (!editHasSnapshot) { checkpoint(); editHasSnapshot = true; }
  changeParameter(value);
});
$('selected-value').addEventListener('change', () => {
  $('selected-value').value = selectedValue();
  recordHistory(true); render(); saveNow();
});
$('selected-value').addEventListener('blur', () => {
  $('selected-value').value = selectedValue();
  editHasSnapshot = false;
});

function readLearningRate() {
  const value = $('learning-rate').valueAsNumber;
  if (!Number.isFinite(value) || value <= 0 || value > 10) { notice('学习率需要大于 0 且不超过 10。'); return false; }
  state.learningRate = value;
  return true;
}
$('learning-rate').addEventListener('input', () => {
  pause();
  if (readLearningRate()) { notice(); renderInspector(); scheduleSave(); }
});
$('learning-rate').addEventListener('blur', () => { $('learning-rate').value = state.learningRate; renderInspector(); });
$('training-speed').addEventListener('change', () => {
  state.speed = Number($('training-speed').value);
  if (timer !== null) { clearInterval(timer); timer = setInterval(() => trainStep(false), state.speed); }
  scheduleSave();
});

function trainStep(keepUndo = true) {
  if (formDirty) { pause(); notice('请先应用左侧已修改的网络设置。'); return false; }
  if (!readLearningRate()) { pause(); return false; }
  try {
    const next = gradientStep(state.model, state.learningRate);
    if (keepUndo) checkpoint();
    previousLoss = result.loss;
    state.model = next.model;
    result = next.result;
    state.steps++;
    recordHistory(true);
    render();
    scheduleSave();
    if (result.loss < 1e-12 || result.norm < 1e-12) {
      pause();
      notice(result.loss < 1e-12 ? '损失已小于 10⁻¹²，已暂停连续训练。' : '梯度已接近零，已暂停连续训练；这并不一定代表全局最优。');
    }
    return true;
  } catch (error) { pause(); notice(error.message); return false; }
}
$('step-button').addEventListener('click', () => { pause(); notice(); trainStep(); });
$('play-button').addEventListener('click', () => {
  if (timer !== null) { pause(); saveNow(); return; }
  if (formDirty) { notice('请先应用左侧已修改的网络设置。'); return; }
  if (!readLearningRate()) return;
  checkpoint();
  notice();
  timer = setInterval(() => trainStep(false), state.speed);
  $('play-button').textContent = 'Ⅱ 暂停训练';
  $('play-button').setAttribute('aria-pressed', 'true');
  trainStep(false);
});
$('clear-history').addEventListener('click', () => { state.history = [result.loss]; previousLoss = null; render(); saveNow(); });

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderStructure() {
  const model = state.model;
  const signature = JSON.stringify([model.x.length, model.x[0].length, model.y[0].length, model.hidden, model.activation, model.useBias, state.layer, state.tensor]);
  if (signature === lastRenderedSignature) return;
  lastRenderedSignature = signature;
  const flow = $('network-flow');
  flow.replaceChildren();
  const blocks = [
    { icon: 'X', title: '输入', subtitle: `${model.x.length} 样本 × ${model.x[0].length} 维`, className: '' },
    ...model.hidden.map((n, i) => ({ icon: `h${subscripts(i + 1)}`, title: `隐藏层 ${i + 1}`, subtitle: `${n} 维 · ${model.activation}`, className: 'hidden-layer' })),
    { icon: 'Ŷ', title: '输出', subtitle: `${model.y[0].length} 维 · linear`, className: '' },
    { icon: 'L', title: 'MSE', subtitle: '均方误差', className: 'loss' }
  ];
  blocks.forEach((block, i) => {
    if (i > 0) flow.append(element('span', 'flow-arrow', '→'));
    const node = element('div', `flow-block ${block.className}`);
    node.append(element('span', 'flow-icon', block.icon));
    const text = element('span', 'flow-text');
    text.append(element('strong', '', block.title), document.createTextNode(block.subtitle));
    node.append(text); flow.append(node);
  });
  $('network-equation').textContent = `${model.hidden.length ? `h = ${model.activation}(X @ W + b)  →  ` : ''}Ŷ = ${model.hidden.length ? 'h' : 'X'} @ W${model.useBias ? ' + b' : ''}  →  MSE(Ŷ, Y)`;
  if (!model.useBias) $('network-equation').textContent = $('network-equation').textContent.replaceAll(' + b', '');
  $('parameter-count').textContent = `${parameterCount(model)} 个可训练参数`;
  $('x-shape').textContent = `${model.x.length} × ${model.x[0].length}`;
  $('y-shape').textContent = `${model.y.length} × ${model.y[0].length}`;
  $('output-shape').textContent = `${model.y.length} × ${model.y[0].length}`;
  $('footer-shape').textContent = [model.x[0].length, ...model.hidden, model.y[0].length].join(' → ');
  $('weight-tab').classList.toggle('active', state.tensor === 'w');
  $('bias-tab').classList.toggle('active', state.tensor === 'b');
  $('weight-tab').setAttribute('aria-pressed', state.tensor === 'w');
  $('bias-tab').setAttribute('aria-pressed', state.tensor === 'b');
  $('bias-tab').disabled = !model.useBias;
  const tabs = $('layer-tabs'); tabs.replaceChildren();
  model.layers.forEach((layer, l) => {
    const button = element('button', `layer-tab${l === state.layer ? ' active' : ''}`, `${l === model.layers.length - 1 ? '输出层' : `隐藏层 ${l + 1}`} · ${state.tensor === 'w' ? 'W' : 'b'}${subscripts(l + 1)}`);
    button.type = 'button'; button.setAttribute('aria-pressed', l === state.layer);
    button.append(element('span', '', `${layer[state.tensor].length} × ${layer[state.tensor][0].length}`));
    button.addEventListener('click', () => { state.layer = l; state.selected = { i: 0, j: 0 }; editHasSnapshot = false; render(); });
    tabs.append(button);
  });
}

function renderPredictions() {
  const shape = `${state.model.y.length},${state.model.y[0].length}`;
  if (shape !== lastPredictionShape) {
    lastPredictionShape = shape;
    predictionCells = [];
    const body = $('prediction-body'); body.replaceChildren();
    state.model.y.forEach((row, n) => row.forEach((_, j) => {
      const tr = element('tr');
      const cells = [`[${n}, ${j}]`, '', '', ''].map(value => element('td', '', value));
      tr.append(...cells); body.append(tr); predictionCells.push({ n, j, cells });
    }));
  }
  predictionCells.forEach(({ n, j, cells }) => {
    cells[1].textContent = formatNumber(state.model.y[n][j]);
    cells[2].textContent = formatNumber(result.predictions[n][j], 5);
    cells[3].textContent = formatNumber((result.predictions[n][j] - state.model.y[n][j]) ** 2, 5);
  });
}

function renderInspector() {
  const { i, j } = state.selected;
  const value = selectedValue();
  const gradient = selectedGradient()[i][j];
  $('selected-label').textContent = `${layerSymbol()}[${i}, ${j}]`;
  if (document.activeElement !== $('selected-value')) $('selected-value').value = Number(value.toPrecision(10));
  $('selected-gradient').textContent = formatNumber(gradient, 6);
  $('next-value').textContent = formatNumber(value - state.learningRate * gradient, 6);
  const direction = Math.abs(gradient) < 1e-10 ? 0 : Math.sign(gradient);
  $('gradient-direction').textContent = direction > 0 ? '正梯度 · 更新方向向下 ↓' : direction < 0 ? '负梯度 · 更新方向向上 ↑' : '梯度接近零';
  $('selected-explanation').textContent = direction === 0 ? '这个参数的一阶影响很小。它可能位于驻点，或被激活函数阻断；不代表已经找到最优解。' : `保持其他参数不变，${direction > 0 ? '小幅减小' : '小幅增大'}这个参数，通常会降低损失。跨越较大距离后，梯度方向可能改变。`;
}

function render() {
  resetSelection();
  renderStructure();
  $('loss-value').textContent = formatNumber(result.loss, 6);
  $('gradient-norm').textContent = formatNumber(result.norm, 5);
  $('step-count').replaceChildren(document.createTextNode(String(state.steps)), element('span', 'number-unit', '次'));
  const change = $('loss-change');
  if (previousLoss === null || previousLoss === result.loss) { change.textContent = '调整参数，观察变化'; change.className = ''; }
  else {
    const delta = result.loss - previousLoss;
    change.textContent = `${delta < 0 ? '↓' : '↑'} ${formatNumber(Math.abs(delta), 6)} 较上次计算`;
    change.className = delta < 0 ? 'down' : 'up';
  }
  $('loss-warning').textContent = previousLoss !== null && result.loss > previousLoss * 1.5 ? '变化较大' : '';
  const matrix = selectedMatrix(), gradients = selectedGradient();
  const title = `${matrix.length} × ${matrix[0].length}`;
  $('weight-title').replaceChildren(document.createTextNode(layerSymbol()), element('span', '', title));
  $('gradient-title').replaceChildren(document.createTextNode(`∂L / ∂${layerSymbol()}`), element('span', '', title));
  const range = weightPlot.drag?.range ?? niceRange(matrix.flat().map(v => v * 1.25));
  const gradientRange = niceRange(gradients.flat(), 0.01);
  weightPlot.setData({ matrix, name: layerSymbol(), selected: state.selected, range });
  gradientPlot.setData({ matrix: gradients, name: `∂${layerSymbol()}`, selected: state.selected, range: gradientRange });
  $('weight-scale').textContent = `统一刻度 ±${formatNumber(range)}`;
  $('gradient-scale').textContent = `统一刻度 ±${formatNumber(gradientRange)}`;
  renderInspector();
  renderPredictions();
  historyPlot.setData(state.history);
  $('history-last').textContent = formatNumber(result.loss, 6);
}

window.addEventListener('pagehide', () => { pause(); saveNow(); });
document.addEventListener('visibilitychange', () => { if (document.hidden) { pause(); saveNow(); } });
syncForm();
render();
if (loadWarning) notice(loadWarning);
saveNow();
