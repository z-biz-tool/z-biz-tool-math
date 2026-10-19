// nn.ts（神经网络内核：前向 / 反向 / 数据集 / 决策边界栅格）的纯逻辑回归。
// 跑法：node --experimental-strip-types --test tests/nn.test.ts
//
// 为什么测它：
//   README 八种工作模式里的「神经网络（训练 / 决策边界）」整条链路都走 nn.ts，
//   而 nn.ts 此前**零测试**。它的失败形态最阴：梯度方向写反不会报错，
//   只会让「训练到 98%」这个教学卖点变成永远训不上去。
//
// 判据里最硬的一条是**数值梯度校验**（中心差分对比解析梯度）：
//   任何激活函数导数写错、梯度回传方向写反、softmax 归一化漏掉，
//   都会让解析梯度与数值梯度对不上 ⇒ 这条断言有真正的区分力。
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  accuracy,
  activations,
  CLASS_COLORS,
  createModel,
  dataLoss,
  dataset,
  DATASETS,
  decisionRaster,
  decisionRow,
  forward,
  makeCache,
  makeGrad,
  predict,
  predictMargin,
  rng,
  sampleGrad,
  trainBatch,
  trainEpoch,
  zeroGrad,
  type Model,
} from "../src/core/nn.ts";

function closeN(a: number, b: number, tol: number, msg = ""): void {
  assert.ok(
    Number.isFinite(a) && Math.abs(a - b) <= tol,
    `${msg} 期望 ${b}，实得 ${a}（容差 ${tol}）`
  );
}

/** 单样本损失（与 sampleGrad 同一公式），用于数值梯度校验 */
function lossOf(m: Model, x0: number, x1: number, y: number): number {
  const cache = makeCache(m);
  const g = makeGrad(m);
  zeroGrad(g);
  return sampleGrad(m, x0, x1, y, cache, g);
}

/**
 * 中心差分数值梯度 vs 解析梯度。
 * 返回最大绝对误差；解析梯度来自 sampleGrad 累加到 GradPack 的 dw/db。
 */
function gradientError(
  m: Model,
  x0: number,
  x1: number,
  y: number,
  eps = 1e-6
): { maxAbsErr: number; checked: number } {
  const cache = makeCache(m);
  const g = makeGrad(m);
  zeroGrad(g);
  sampleGrad(m, x0, x1, y, cache, g);
  let maxAbsErr = 0;
  let checked = 0;
  for (let l = 0; l < m.layers.length; l++) {
    const L = m.layers[l];
    for (let idx = 0; idx < L.w.length; idx++) {
      const orig = L.w[idx];
      L.w[idx] = orig + eps;
      const lp = lossOf(m, x0, x1, y);
      L.w[idx] = orig - eps;
      const lm = lossOf(m, x0, x1, y);
      L.w[idx] = orig;
      maxAbsErr = Math.max(maxAbsErr, Math.abs((lp - lm) / (2 * eps) - g.dw[l][idx]));
      checked++;
    }
    for (let j = 0; j < L.b.length; j++) {
      const orig = L.b[j];
      L.b[j] = orig + eps;
      const lp = lossOf(m, x0, x1, y);
      L.b[j] = orig - eps;
      const lm = lossOf(m, x0, x1, y);
      L.b[j] = orig;
      maxAbsErr = Math.max(maxAbsErr, Math.abs((lp - lm) / (2 * eps) - g.db[l][j]));
      checked++;
    }
  }
  return { maxAbsErr, checked };
}

test("rng：同种子必定同序列（训练可复现的前提）", () => {
  const a = rng(42);
  const b = rng(42);
  for (let i = 0; i < 8; i++) {
    assert.equal(a(), b(), `第 ${i} 个数必须一致`);
  }
  assert.notEqual(rng(1)(), rng(2)(), "不同种子必须给出不同序列");
});

test("rng：取值落在 [0, 1)", () => {
  const r = rng(7);
  for (let i = 0; i < 500; i++) {
    const v = r();
    assert.ok(v >= 0 && v < 1, `第 ${i} 个数越界：${v}`);
  }
});

test("createModel：非法形状明确报错", () => {
  assert.throws(() => createModel([2]), /至少要有输入层和输出层/);
  assert.throws(() => createModel([0, 2]), /至少要有输入层和输出层/);
  assert.throws(() => createModel([3, 2]), /输入必须是二维/);
});

test("createModel：层数、权重形状、sizes 拷贝", () => {
  const sizes = [2, 5, 3, 2];
  const m = createModel(sizes, "tanh", 11);
  assert.equal(m.layers.length, sizes.length - 1);
  for (let l = 0; l < m.layers.length; l++) {
    const L = m.layers[l];
    assert.equal(L.in, sizes[l]);
    assert.equal(L.out, sizes[l + 1]);
    assert.equal(L.w.length, L.in * L.out);
    assert.equal(L.b.length, L.out);
  }
  // sizes 必须被拷贝：改原数组不影响模型
  sizes[0] = 99;
  assert.equal(m.sizes[0], 2, "sizes 未被拷贝");
  assert.equal(m.layers[0].in, 2);
});

test("createModel：Xavier 初始化，均值≈0、标准差≈sqrt(2/(in+out))", () => {
  const m = createModel([2, 64, 2], "tanh", 7);
  const w = m.layers[0].w;
  const mean = Array.from(w).reduce((s, v) => s + v, 0) / w.length;
  const sd = Math.sqrt(Array.from(w).reduce((s, v) => s + (v - mean) ** 2, 0) / w.length);
  const theory = Math.sqrt(2 / (2 + 64));
  assert.ok(Math.abs(mean) < 0.05, `权重均值应接近 0，实得 ${mean}`);
  assert.ok(
    Math.abs(sd - theory) / theory < 0.2,
    `权重标准差应接近 ${theory.toFixed(6)}，实得 ${sd.toFixed(6)}`
  );
});

test("forward：输出层是概率（非负、和为 1）", () => {
  const m = createModel([2, 8, 3], "tanh", 3);
  const cache = makeCache(m);
  const p = forward(m, 0.5, -0.7, cache);
  assert.equal(p.length, 3);
  let sum = 0;
  for (const v of p) {
    assert.ok(v >= 0 && v <= 1, `概率越界：${v}`);
    sum += v;
  }
  closeN(sum, 1, 1e-12, "softmax 概率之和：");
});

test("forward：logit 极大时不溢出、不出 NaN", () => {
  const m = createModel([2, 4, 2], "tanh", 1);
  // 把输出层偏置拉大，制造 ±1000 量级的 logit
  m.layers[m.layers.length - 1].b[0] = 1000;
  m.layers[m.layers.length - 1].b[1] = -1000;
  const cache = makeCache(m);
  const p = forward(m, 0, 0, cache);
  for (const v of p) assert.ok(Number.isFinite(v), `出现非有限概率：${v}`);
  closeN(Array.from(p).reduce((s, v) => s + v, 0), 1, 1e-12, "溢出后概率之和：");
  assert.ok(p[0] > 0.999, `大 logit 应让第 0 类接近 1，实得 ${p[0]}`);
});

test("activations：第 0 行是输入坐标，行数 = 层数 + 1", () => {
  const m = createModel([2, 4, 2], "relu", 5);
  const cache = makeCache(m);
  const rows = activations(m, 0.25, -1.5, cache);
  assert.equal(rows.length, m.layers.length + 1);
  assert.deepEqual(rows[0], [0.25, -1.5]);
  assert.equal(rows[1].length, 4);
  assert.equal(rows[2].length, 2);
});

test("sampleGrad：解析梯度与中心差分数值梯度一致（tanh）", () => {
  const m = createModel([2, 4, 2], "tanh", 3);
  const { maxAbsErr, checked } = gradientError(m, 0.3, -0.8, 1);
  assert.ok(checked >= 20, `检查的参数太少：${checked}`);
  assert.ok(maxAbsErr < 1e-6, `解析梯度与数值梯度最大误差 ${maxAbsErr} 超过 1e-6`);
});

test("sampleGrad：relu 的导数分支也正确", () => {
  const m = createModel([2, 5, 2], "relu", 9);
  const { maxAbsErr } = gradientError(m, 0.4, 0.6, 0);
  assert.ok(maxAbsErr < 1e-6, `relu 梯度误差 ${maxAbsErr} 超过 1e-6`);
});

test("sampleGrad：sigmoid 的导数分支也正确", () => {
  const m = createModel([2, 5, 2], "sigmoid", 13);
  const { maxAbsErr } = gradientError(m, -0.2, 0.9, 1);
  assert.ok(maxAbsErr < 1e-6, `sigmoid 梯度误差 ${maxAbsErr} 超过 1e-6`);
});

test("sampleGrad：返回的损失等于 −log p[y]", () => {
  const m = createModel([2, 4, 2], "tanh", 2);
  const cache = makeCache(m);
  const g = makeGrad(m);
  zeroGrad(g);
  const loss = sampleGrad(m, 0.1, 0.2, 1, cache, g);
  const p = forward(m, 0.1, 0.2, cache);
  assert.ok(p[1] > 0);
  closeN(loss, -Math.log(p[1]), 1e-12, "sampleGrad 的损失：");
});

test("trainBatch：同一 batch 反复训练，损失单调下降", () => {
  const m = createModel([2, 6, 2], "tanh", 21);
  const data = dataset("xor", 64, 2);
  const cache = makeCache(m);
  const g = makeGrad(m);
  const vel = makeGrad(m);
  const order = Array.from({ length: data.n }, (_, i) => i);
  const opt = { lr: 0.1, momentum: 0, batch: 16 };
  const first = trainBatch(m, data, order, 0, 16, opt, cache, g, vel);
  let last = first;
  for (let i = 0; i < 60; i++) last = trainBatch(m, data, order, 0, 16, opt, cache, g, vel);
  assert.ok(last < first, `损失未下降：首 ${first.toFixed(6)} → 末 ${last.toFixed(6)}`);
});

test("trainBatch：空区间返回 0 且不改动参数", () => {
  const m = createModel([2, 3, 2], "tanh", 4);
  const data = dataset("xor", 8, 1);
  const cache = makeCache(m);
  const g = makeGrad(m);
  const vel = makeGrad(m);
  const before = Array.from(m.layers[0].w);
  const loss = trainBatch(m, data, [0, 1, 2], 0, 0, { lr: 0.1, momentum: 0, batch: 4 }, cache, g, vel);
  assert.equal(loss, 0);
  assert.deepEqual(Array.from(m.layers[0].w), before, "空 batch 不该改参数");
});

test("trainEpoch：步数 = ceil(n / batch)，准确率落在 [0,1]", () => {
  const m = createModel([2, 8, 2], "tanh", 33);
  const data = dataset("xor", 50, 6);
  const cache = makeCache(m);
  const g = makeGrad(m);
  const vel = makeGrad(m);
  const stat = trainEpoch(m, data, { lr: 0.05, momentum: 0.9, batch: 10 }, rng(99), cache, g, vel);
  assert.equal(stat.steps, Math.ceil(50 / 10));
  assert.ok(stat.acc >= 0 && stat.acc <= 1, `准确率越界：${stat.acc}`);
  assert.ok(Number.isFinite(stat.loss), "损失非有限");
});

test("predict / predictMargin：label 是最大概率下标，margin ≥ 0", () => {
  const m = createModel([2, 5, 3], "tanh", 17);
  const cache = makeCache(m);
  for (const [x0, x1] of [
    [0, 0],
    [1, -1],
    [-2, 0.5],
  ]) {
    const p = forward(m, x0, x1, cache);
    const label = predict(m, x0, x1, cache);
    let best = 0;
    for (let j = 1; j < p.length; j++) if (p[j] > p[best]) best = j;
    assert.equal(label, best, `predict 应返回最大概率下标`);
    const { margin } = predictMargin(m, x0, x1, cache);
    assert.ok(margin >= 0, `margin 不得为负：${margin}`);
  }
});

test("predictMargin：margin = 最大概率 − 次大概率", () => {
  const m = createModel([2, 5, 2], "tanh", 23);
  const cache = makeCache(m);
  const p = forward(m, 0.3, 0.7, cache);
  const { label, margin } = predictMargin(m, 0.3, 0.7, cache);
  const sorted = Array.from(p).sort((a, b) => b - a);
  closeN(margin, sorted[0] - sorted[1], 1e-12, "margin：");
  assert.ok(p[label] === sorted[0]);
});

test("accuracy / dataLoss：与逐样本手工计算一致", () => {
  const m = createModel([2, 4, 2], "tanh", 29);
  const data = dataset("circle", 24, 8);
  const cache = makeCache(m);
  let hit = 0;
  let loss = 0;
  for (let i = 0; i < data.n; i++) {
    const p = forward(m, data.xs[2 * i], data.xs[2 * i + 1], cache);
    let best = 0;
    for (let j = 1; j < p.length; j++) if (p[j] > p[best]) best = j;
    if (best === data.ys[i]) hit++;
    loss -= Math.log(Math.max(1e-12, p[data.ys[i]]));
  }
  closeN(accuracy(m, data, cache), hit / data.n, 1e-12, "accuracy：");
  closeN(dataLoss(m, data, cache), loss / data.n, 1e-12, "dataLoss：");
});

test("dataset：四个数据集的规模、类别数、坐标范围", () => {
  const expect: Record<string, { k: number; name: string }> = {
    xor: { k: 2, name: "异或" },
    circle: { k: 2, name: "同心圆" },
    moons: { k: 2, name: "双月" },
    spiral: { k: 3, name: "三臂螺旋" },
  };
  assert.deepEqual(DATASETS, ["xor", "circle", "moons", "spiral"]);
  for (const nm of DATASETS) {
    const d = dataset(nm, 40, 5);
    assert.equal(d.n, 40, `${nm} 的样本数`);
    assert.equal(d.k, expect[nm].k, `${nm} 的类别数`);
    assert.equal(d.name, expect[nm].name, `${nm} 的名称`);
    assert.equal(d.ys.length, d.n);
    assert.equal(d.xs.length, d.n * 2);
    for (let i = 0; i < d.n; i++) {
      assert.ok(d.ys[i] >= 0 && d.ys[i] < d.k, `${nm} 的类别下标越界：${d.ys[i]}`);
      assert.ok(Math.abs(d.xs[2 * i]) <= 3, `${nm} 的 x 越界：${d.xs[2 * i]}`);
      assert.ok(Math.abs(d.xs[2 * i + 1]) <= 3, `${nm} 的 y 越界：${d.xs[2 * i + 1]}`);
    }
    // 类别必须**全部出现**：只查 ys < k 是不够的 ——
    // 把 spiral 的 `i % 3` 改成 `i % 2` 后 k 仍是 3，但第三类永远不出现。
    const seen = new Set(Array.from(d.ys));
    assert.equal(seen.size, d.k, `${nm} 实际只出现了 ${seen.size} 个类别，应为 ${d.k}`);
  }
});

test("dataset：同种子同结果，不同种子不同结果", () => {  const a = dataset("moons", 20, 3);
  const b = dataset("moons", 20, 3);
  const c = dataset("moons", 20, 4);
  assert.deepEqual(Array.from(a.xs), Array.from(b.xs));
  assert.deepEqual(Array.from(a.ys), Array.from(b.ys));
  assert.notDeepEqual(Array.from(a.xs), Array.from(c.xs));
});

test("decisionRow：颜色由 label 与 margin 决定（可反推）", () => {
  const m = createModel([2, 5, 2], "tanh", 41);
  const cache = makeCache(m);
  const w = 8;
  const out = new Uint8ClampedArray(w * 3);
  const left = -3;
  const right = 3;
  const y = 0.5;
  decisionRow(m, left, right, y, w, true, out, 0, cache);
  const dx = (right - left) / (w - 1);
  for (let i = 0; i < w; i++) {
    const { label, margin } = predictMargin(m, left + dx * i, y, cache);
    const c = CLASS_COLORS[label % CLASS_COLORS.length];
    const t = 0.26 + 0.64 * Math.min(1, Math.max(0, margin));
    const bg = [13, 15, 26];
    for (let k = 0; k < 3; k++) {
      const want = bg[k] + (c[k] - bg[k]) * t;
      assert.ok(
        Math.abs(out[i * 3 + k] - want) <= 1,
        `第 ${i} 像素的第 ${k} 通道：期望 ≈${want.toFixed(3)}，实得 ${out[i * 3 + k]}`
      );
    }
  }
});

test("decisionRow：行 0 对应给定的 top 纵坐标（约定：y 随 j 递减）", () => {
  const m = createModel([2, 4, 2], "tanh", 43);
  const cache = makeCache(m);
  const w = 5;
  const a = new Uint8ClampedArray(w * 3);
  const b = new Uint8ClampedArray(w * 3);
  decisionRow(m, -2, 2, 1.5, w, false, a, 0, cache);
  decisionRow(m, -2, 2, 1.5, w, false, b, 0, cache);
  assert.deepEqual(Array.from(a), Array.from(b), "同一 y 必须给出同一行");
});

test("decisionRaster：与逐行 decisionRow 逐字节相同", () => {
  // 实现注释明确承诺「逐行合成必定与整幅一次采样逐字节同解」——这条把它钉住。
  const m = createModel([2, 6, 2], "tanh", 47);
  const w = 9;
  const h = 7;
  const left = -3;
  const right = 3;
  const top = 2;
  const bottom = -2;
  const raster = decisionRaster(m, left, right, top, bottom, w, h, true);
  const cache = makeCache(m);
  const manual = new Uint8ClampedArray(w * h * 3);
  const dy = (bottom - top) / (h - 1);
  for (let j = 0; j < h; j++) {
    decisionRow(m, left, right, top + dy * j, w, true, manual, j * w * 3, cache);
  }
  assert.deepEqual(Array.from(raster), Array.from(manual), "整幅与逐行合成结果不一致");
});

test("decisionRaster：首行对应 top、末行对应 bottom", () => {
  const m = createModel([2, 4, 2], "tanh", 53);
  const w = 6;
  const h = 5;
  const raster = decisionRaster(m, -3, 3, 2, -2, w, h, false);
  const cache = makeCache(m);
  const first = new Uint8ClampedArray(w * 3);
  const last = new Uint8ClampedArray(w * 3);
  decisionRow(m, -3, 3, 2, w, false, first, 0, cache);
  decisionRow(m, -3, 3, -2, w, false, last, 0, cache);
  assert.deepEqual(Array.from(raster.slice(0, w * 3)), Array.from(first));
  assert.deepEqual(Array.from(raster.slice((h - 1) * w * 3)), Array.from(last));
});

test("decisionRaster：非法尺寸返回空数组而不是抛错", () => {
  const m = createModel([2, 3, 2], "tanh", 59);
  assert.equal(decisionRaster(m, -1, 1, 1, -1, 0, 5, true).length, 0);
  assert.equal(decisionRaster(m, -1, 1, 1, -1, 5, 0, true).length, 0);
});
