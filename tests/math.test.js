import test from 'node:test';
import assert from 'node:assert/strict';
import { ACTIVATIONS, LIMITS, copy, createModel, demoModel, evaluate, gradientStep, parseMatrix, parseHidden, validateModel, parameterCount, niceRange, dragValue } from '../public/math.js';
import { formatNumber } from '../public/charts.js';

function close(actual, expected, tolerance = 1e-8) {
  assert.ok(Math.abs(actual - expected) <= tolerance * Math.max(1, Math.abs(expected)), `${actual} != ${expected}`);
}

function seededRandom(seed = 7) {
  return () => { seed = (1664525 * seed + 1013904223) >>> 0; return seed / 2 ** 32; };
}

function finiteDifferences(model) {
  const analytic = evaluate(model);
  const epsilon = 1e-6;
  assert.equal(analytic.finite, true);
  model.layers.forEach((layer, l) => {
    for (const tensor of ['w', 'b']) layer[tensor].forEach((row, i) => row.forEach((original, j) => {
      layer[tensor][i][j] = original + epsilon;
      const plus = evaluate(model).loss;
      layer[tensor][i][j] = original - epsilon;
      const minus = evaluate(model).loss;
      layer[tensor][i][j] = original;
      close(analytic.gradients[l][tensor][i][j], (plus - minus) / (2 * epsilon), 2e-5);
    }));
  });
  model.x.forEach((row, n) => row.forEach((original, i) => {
    model.x[n][i] = original + epsilon;
    const plus = evaluate(model).loss;
    model.x[n][i] = original - epsilon;
    const minus = evaluate(model).loss;
    model.x[n][i] = original;
    close(analytic.inputGradient[n][i], (plus - minus) / (2 * epsilon), 2e-5);
  }));
}

test('手算：多样本、多输出的 MSE 与 W 梯度', () => {
  const model = {
    x: [[1, 2], [3, 4]], y: [[1, 0], [0, 1]], hidden: [], activation: 'linear', useBias: false,
    layers: [{ w: [[1, 0], [0, 1]], b: [[9, -9]] }]
  };
  const r = evaluate(model);
  assert.deepEqual(r.predictions, [[1, 2], [3, 4]]);
  close(r.loss, 5.5); // (0 + 4 + 9 + 9) / 4
  assert.deepEqual(r.gradients[0].w, [[4.5, 5.5], [6, 8]]);
  assert.deepEqual(r.gradients[0].b, [[0, 0]]);
  assert.deepEqual(r.inputGradient, [[0, 1], [1.5, 1.5]]);
});

test('手算：偏置对 batch 求和，MSE 不额外乘 1/2', () => {
  const model = { x: [[1], [2]], y: [[0], [0]], hidden: [], activation: 'tanh', useBias: true, layers: [{ w: [[2]], b: [[1]] }] };
  const r = evaluate(model);
  assert.deepEqual(r.predictions, [[3], [5]]);
  close(r.loss, 17);
  close(r.gradients[0].w[0][0], 13);
  close(r.gradients[0].b[0][0], 8);
});

for (const activation of ACTIVATIONS) {
  for (const useBias of [true, false]) {
    test(`中心差分：${activation}，偏置 ${useBias}，多层非方阵、多样本、多输出`, () => {
      const model = createModel({ x: [[0.3, -0.5], [0.9, 0.4], [-0.7, 0.8]], y: [[0.2, -0.3], [0.1, 0.7], [0.8, -0.2]], hidden: [3, 4], activation, useBias }, seededRandom(42));
      model.layers.forEach((layer, l) => layer.b[0].forEach((_, i) => { layer.b[0][i] = 0.13 + 0.07 * (l + i); }));
      finiteDifferences(model);
    });
  }
}

test('无隐藏层、多输出的数值梯度', () => {
  finiteDifferences(createModel({ x: [[0.2, -0.4]], y: [[0.2, 0.3, -0.7]], hidden: [], activation: 'sigmoid', useBias: true }, seededRandom()));
});

test('ReLU 在零处采用 0 导数，屏蔽负区间梯度', () => {
  const model = { x: [[0], [-1]], y: [[1], [1]], hidden: [1], activation: 'relu', useBias: false, layers: [{ w: [[1]], b: [[0]] }, { w: [[2]], b: [[0]] }] };
  const r = evaluate(model);
  assert.deepEqual(r.gradients[0].w, [[0]]);
  assert.deepEqual(r.inputGradient, [[0], [0]]);
});

test('Sigmoid 在极大正负数下仍有限', () => {
  const model = { x: [[1000], [-1000]], y: [[0], [1]], hidden: [1], activation: 'sigmoid', useBias: false, layers: [{ w: [[1000]], b: [[0]] }, { w: [[1]], b: [[0]] }] };
  const r = evaluate(model);
  assert.equal(r.finite, true);
  assert.deepEqual(r.predictions, [[1], [0]]);
});

test('SGD 同时更新全部参数，不修改旧模型；更新后重算梯度', () => {
  const model = demoModel();
  const old = copy(model);
  const before = evaluate(model);
  const { model: next, result } = gradientStep(model, 0.05);
  assert.deepEqual(model, old);
  model.layers.forEach((layer, l) => {
    for (const key of ['w', 'b']) layer[key].forEach((row, i) => row.forEach((v, j) => close(next.layers[l][key][i][j], v - 0.05 * before.gradients[l][key][i][j])));
  });
  assert.deepEqual(result, evaluate(next));
  assert.ok(result.loss < before.loss);
});

test('入门示例连续小步训练显著降低损失', () => {
  let model = demoModel();
  const initial = evaluate(model).loss;
  for (let step = 0; step < 250; step++) model = gradientStep(model, 0.05).model;
  assert.ok(evaluate(model).loss < initial * 0.01);
});

test('过大的学习率确实可能让损失升高，不伪造单调下降', () => {
  const model = { x: [[1]], y: [[0]], hidden: [], activation: 'linear', useBias: false, layers: [{ w: [[1]], b: [[0]] }] };
  close(evaluate(model).loss, 1);
  close(gradientStep(model, 2).result.loss, 9);
});

test('禁止的学习率与过大的更新整体拒绝，旧状态保持不变', () => {
  for (const rate of [0, -1, Infinity, NaN, 11]) assert.throws(() => gradientStep(demoModel(), rate));
  const model = { x: [[1000]], y: [[0]], hidden: [], activation: 'linear', useBias: true, layers: [{ w: [[1e6]], b: [[0]] }] };
  const before = copy(model);
  assert.throws(() => gradientStep(model, 10), /参数变化过大/);
  assert.deepEqual(model, before);
});

test('输入解析：文本、中文分隔符、科学计数、JSON 与单样本向量', () => {
  assert.deepEqual(parseMatrix('1, 2\n3 4', 'X'), [[1, 2], [3, 4]]);
  assert.deepEqual(parseMatrix('1，2；3,4', 'X'), [[1, 2], [3, 4]]);
  assert.deepEqual(parseMatrix('.1, -2e-2; +3, 4.', 'X'), [[0.1, -0.02], [3, 4]]);
  assert.deepEqual(parseMatrix('[[1,2],[3,4]]', 'X'), [[1, 2], [3, 4]]);
  assert.deepEqual(parseMatrix('[1,2]', 'X'), [[1, 2]]);
  assert.deepEqual(parseHidden('4, 3'), [4, 3]);
  assert.deepEqual(parseHidden(''), []);
});

test('拒绝不规则、非数字、超范围矩阵与非法网络结构', () => {
  for (const bad of ['', '1,2\n3', '1, hello', '1 Infinity', '1001', '[[null]]', '[["1"]]', '[[true]]', '[]', '{}', '0x10', '[1,2']) assert.throws(() => parseMatrix(bad, 'X'));
  for (const bad of ['0', '9', '2.1', '1,2,3,4,5', '-2', '2abc']) assert.throws(() => parseHidden(bad));
  const model = demoModel(); model.y.push([1]); assert.throws(() => validateModel(model));
  const badWeights = demoModel(); badWeights.layers[0].w = [[1]]; assert.throws(() => validateModel(badWeights));
  const badBias = demoModel(); badBias.layers[0].b = [[1]]; assert.throws(() => validateModel(badBias));
});

test('支持的最大网络可完整计算', () => {
  const model = createModel({ x: Array.from({ length: LIMITS.samples }, () => Array(LIMITS.width).fill(0.1)), y: Array.from({ length: LIMITS.samples }, () => Array(LIMITS.width).fill(0.2)), hidden: [8, 8, 8, 8], activation: 'tanh', useBias: true }, seededRandom());
  validateModel(model);
  assert.equal(evaluate(model).finite, true);
  assert.equal(parameterCount(model), 360);
  assert.equal(parameterCount(demoModel()), 9);
});

test('拖动映射：向上增大、跨零、细调、固定刻度边界', () => {
  close(dragValue(0.2, 100, 81, 1, 38), 0.7);
  close(dragValue(0.2, 100, 119, 1, 38), -0.3);
  close(dragValue(0.2, 100, 81, 1, 38, true), 0.25);
  close(dragValue(0.2, 100, -1000, 1, 38), 1);
  close(dragValue(0.2, 100, 1000, 1, 38), -1);
  assert.equal(niceRange([0, 0]), 1);
  assert.equal(niceRange([-4, 2]), 5);
  assert.equal(niceRange([1 * 1.25]), 2); // 松手后留出余量，允许继续向上拖动。
  assert.equal(niceRange([0.002], 0.01), 0.01);
});

test('显示格式保留整数末尾的零与极小梯度', () => {
  assert.equal(formatNumber(10), '10');
  assert.equal(formatNumber(100), '100');
  assert.equal(formatNumber(-0.0000001), '-1.00e-7');
  assert.equal(formatNumber(0), '0');
  assert.equal(formatNumber(0.125), '0.125');
});
