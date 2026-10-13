// Row-vector convention throughout: X[B, in] @ W[in, out] + b[1, out].
// The output layer is always linear; MSE is averaged over ALL output elements.
export const LIMITS = Object.freeze({ samples: 16, width: 8, hidden: 4, value: 1000, parameter: 1e6 });
export const ACTIVATIONS = ['tanh', 'relu', 'sigmoid', 'linear'];
export const copy = value => JSON.parse(JSON.stringify(value));
export const zeros = (rows, cols) => Array.from({ length: rows }, () => Array(cols).fill(0));

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

export function validateMatrix(matrix, name = '矩阵', maxRows = LIMITS.samples, maxCols = LIMITS.width, maxValue = LIMITS.value) {
  assert(Array.isArray(matrix) && matrix.length >= 1 && matrix.length <= maxRows, `${name} 需要 1–${maxRows} 行。`);
  const cols = matrix[0]?.length;
  assert(Number.isInteger(cols) && cols >= 1 && cols <= maxCols, `${name} 每行需要 1–${maxCols} 个数字。`);
  for (const row of matrix) {
    assert(Array.isArray(row) && row.length === cols, `${name} 每一行的数字数量必须相同。`);
    assert(row.every(v => typeof v === 'number' && Number.isFinite(v) && Math.abs(v) <= maxValue), `${name} 只能包含绝对值不超过 ${maxValue} 的有限数字。`);
  }
  return matrix;
}

export function parseMatrix(text, name) {
  const source = text.trim();
  assert(source.length > 0 && source.length <= 20000, `${name} 不能为空或过长。`);
  let matrix;
  if (source.startsWith('[')) {
    try { matrix = JSON.parse(source); } catch { throw new Error(`${name} 的 JSON 格式不正确。`); }
    if (Array.isArray(matrix) && matrix.every(v => typeof v === 'number')) matrix = [matrix];
  } else {
    matrix = source.split(/[\n;；]+/).filter(line => line.trim()).map(line =>
      line.trim().split(/[\s,，]+/).map(value => {
        assert(/^[+-]?(?:\d+\.?\d*|\.\d+)(?:e[+-]?\d+)?$/i.test(value), `${name} 包含无效数字：${value.slice(0, 20)}`);
        return Number(value);
      })
    );
  }
  return validateMatrix(matrix, name);
}

export function parseHidden(text) {
  if (!text.trim()) return [];
  const parts = text.trim().split(/[\s,，→]+/);
  assert(parts.every(p => /^\d+$/.test(p)), '隐藏层请填写整数，用逗号分隔，例如 4, 3。');
  const hidden = parts.map(Number);
  validateHidden(hidden);
  return hidden;
}

function validateHidden(hidden) {
  assert(Array.isArray(hidden) && hidden.length <= LIMITS.hidden, `最多支持 ${LIMITS.hidden} 个隐藏层。`);
  assert(hidden.every(n => Number.isInteger(n) && n >= 1 && n <= LIMITS.width), `每个隐藏层需要 1–${LIMITS.width} 个神经元。`);
}

export function validateConfig(config) {
  assert(config && typeof config === 'object', '网络设置无效。');
  validateMatrix(config.x, '输入 X');
  validateMatrix(config.y, '目标 Y');
  assert(config.x.length === config.y.length, '输入 X 和目标 Y 的行数（样本数）必须一致。');
  validateHidden(config.hidden);
  assert(ACTIVATIONS.includes(config.activation), '激活函数无效。');
  assert(typeof config.useBias === 'boolean', '偏置设置无效。');
  return [config.x[0].length, ...config.hidden, config.y[0].length];
}

export function validateModel(model) {
  const sizes = validateConfig(model);
  assert(Array.isArray(model.layers) && model.layers.length === sizes.length - 1, '网络层数不匹配。');
  model.layers.forEach((layer, i) => {
    validateMatrix(layer.w, `W${i + 1}`, LIMITS.width, LIMITS.width, LIMITS.parameter);
    validateMatrix(layer.b, `b${i + 1}`, 1, LIMITS.width, LIMITS.parameter);
    assert(layer.w.length === sizes[i] && layer.w[0].length === sizes[i + 1], `W${i + 1} 形状不匹配。`);
    assert(layer.b[0].length === sizes[i + 1], `b${i + 1} 形状不匹配。`);
  });
  return model;
}

export function createModel(config, random = Math.random) {
  const sizes = validateConfig(config);
  const layers = sizes.slice(0, -1).map((size, index) => {
    const out = sizes[index + 1];
    const bound = Math.sqrt(6 / (size + out));
    return {
      w: Array.from({ length: size }, () => Array.from({ length: out }, () => (random() * 2 - 1) * bound)),
      b: zeros(1, out)
    };
  });
  return { ...copy(config), layers };
}

export function demoModel() {
  return {
    x: [[1, 2], [2, 1]], y: [[1], [0]], hidden: [2], activation: 'tanh', useBias: true,
    layers: [
      { w: [[0.35, -0.25], [0.2, 0.4]], b: [[0.1, -0.1]] },
      { w: [[0.6], [-0.5]], b: [[0.05]] }
    ]
  };
}

function activate(z, kind) {
  if (kind === 'relu') return Math.max(0, z);
  if (kind === 'tanh') return Math.tanh(z);
  if (kind === 'sigmoid') return z >= 0 ? 1 / (1 + Math.exp(-z)) : Math.exp(z) / (1 + Math.exp(z));
  return z;
}

function derivative(z, a, kind) {
  if (kind === 'relu') return z > 0 ? 1 : 0;
  if (kind === 'tanh') return 1 - a * a;
  if (kind === 'sigmoid') return a * (1 - a);
  return 1;
}

export function evaluate(model) {
  const activations = [model.x];
  const preactivations = [];
  let a = model.x;
  model.layers.forEach((layer, l) => {
    const z = zeros(a.length, layer.w[0].length);
    for (let n = 0; n < a.length; n++) {
      for (let j = 0; j < z[0].length; j++) {
        let value = model.useBias ? layer.b[0][j] : 0;
        for (let i = 0; i < a[0].length; i++) value += a[n][i] * layer.w[i][j];
        z[n][j] = value;
      }
    }
    const kind = l === model.layers.length - 1 ? 'linear' : model.activation;
    a = z.map(row => row.map(v => activate(v, kind)));
    preactivations.push(z);
    activations.push(a);
  });
  const count = a.length * a[0].length;
  let loss = 0;
  let upstream = a.map((row, n) => row.map((value, j) => {
    const difference = value - model.y[n][j];
    loss += difference * difference / count;
    return 2 * difference / count;
  }));
  const gradients = Array(model.layers.length);
  for (let l = model.layers.length - 1; l >= 0; l--) {
    const layer = model.layers[l];
    const previous = activations[l];
    const kind = l === model.layers.length - 1 ? 'linear' : model.activation;
    const dz = upstream.map((row, n) => row.map((v, j) => v * derivative(preactivations[l][n][j], activations[l + 1][n][j], kind)));
    const dw = zeros(layer.w.length, layer.w[0].length);
    const db = zeros(1, layer.w[0].length);
    const da = zeros(previous.length, previous[0].length);
    for (let n = 0; n < previous.length; n++) {
      for (let j = 0; j < dw[0].length; j++) {
        if (model.useBias) db[0][j] += dz[n][j];
        for (let i = 0; i < dw.length; i++) {
          dw[i][j] += previous[n][i] * dz[n][j];
          da[n][i] += dz[n][j] * layer.w[i][j];
        }
      }
    }
    gradients[l] = { w: dw, b: db };
    upstream = da;
  }
  const flat = gradients.flatMap(g => [...g.w.flat(), ...g.b.flat()]);
  const norm = Math.hypot(...flat);
  const finite = Number.isFinite(loss) && Number.isFinite(norm) && upstream.flat().every(Number.isFinite) && a.flat().every(Number.isFinite);
  return { loss, predictions: a, gradients, inputGradient: upstream, activations, preactivations, norm, finite };
}

// Calculate all gradients first, then update every parameter simultaneously.
// Reject the entire step on overflow, leaving the caller's model intact.
export function gradientStep(model, learningRate) {
  assert(Number.isFinite(learningRate) && learningRate > 0 && learningRate <= 10, '学习率需要大于 0 且不超过 10。');
  const before = evaluate(model);
  assert(before.finite, '数值已溢出，请减小输入或参数。');
  const next = copy(model);
  next.layers.forEach((layer, l) => {
    for (const key of model.useBias ? ['w', 'b'] : ['w']) {
      layer[key] = layer[key].map((row, i) => row.map((v, j) => {
        const updated = v - learningRate * before.gradients[l][key][i][j];
        assert(Number.isFinite(updated) && Math.abs(updated) <= LIMITS.parameter, '参数变化过大，已停止本次更新。请降低学习率。');
        return updated;
      }));
    }
  });
  const after = evaluate(next);
  assert(after.finite, '数值已溢出，已停止本次更新。请降低学习率。');
  return { model: next, result: after };
}

export function parameterCount(model) {
  return model.layers.reduce((sum, l) => sum + l.w.length * l.w[0].length + (model.useBias ? l.b[0].length : 0), 0);
}

export function niceRange(values, floor = 1) {
  const max = Math.max(floor, ...values.map(Math.abs));
  const magnitude = 10 ** Math.floor(Math.log10(max));
  return [1, 2, 5, 10].map(v => v * magnitude).find(v => v >= max) || 10 * magnitude;
}

export function dragValue(initial, startY, pointerY, range, halfHeight, fine = false) {
  const raw = initial + (startY - pointerY) / halfHeight * range * (fine ? 0.1 : 1);
  return Math.round(Math.max(-range, Math.min(range, raw)) * 10000) / 10000;
}
