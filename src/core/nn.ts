/**
 * 神经网络内核：小型全连接 MLP 的前向 / 反向传播、合成数据集、决策边界采样。
 *
 * 教学工具要把每个神经元的激活值、每条边的权重都画出来，所以不封装成黑盒：
 * 层次结构 sizes、逐层权重 w/b 全部显式暴露。中间结果放在可复用的
 * Float64Array 缓存里（决策边界一个像素一次前向，靠分配对象会把帧预算吃光）。
 */

export type Act = "tanh" | "relu" | "sigmoid";

export interface Layer {
  /** out×in 行优先权重 */
  w: Float64Array;
  b: Float64Array;
  in: number;
  out: number;
}

export interface Model {
  sizes: number[];
  layers: Layer[];
  act: Act;
}

export interface TrainOpt {
  lr: number;
  /** 动量，0 即纯 SGD */
  momentum: number;
  /** 每次更新的样本数 */
  batch: number;
}

export interface Dataset {
  /** 2n 个分量：x0,y0,x1,y1,… */
  xs: Float64Array;
  /** 类别下标 */
  ys: Int32Array;
  n: number;
  /** 类别数 */
  k: number;
  name: string;
}

/** mulberry32：同一个种子必定给出同一张网络和同一批样本，否则"训练到 98%"无法复现 */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** 标准正态（Box–Muller） */
function normal(r: () => number): number {
  let u = 0;
  while (u === 0) u = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * r());
}

/** Xavier 初始化：方差按两端维度归一，隐藏层深了也不会一上来就饱和 */
export function createModel(sizes: number[], act: Act = "tanh", seed = 1): Model {
  if (sizes.length < 2 || sizes.some((n) => n < 1)) throw new Error("网络形状至少要有输入层和输出层");
  if (sizes[0] !== 2) throw new Error("输入必须是二维平面上的点");
  const r = rng(seed);
  const layers: Layer[] = [];
  for (let l = 0; l + 1 < sizes.length; l++) {
    const inn = sizes[l];
    const out = sizes[l + 1];
    const w = new Float64Array(inn * out);
    const s = Math.sqrt(2 / (inn + out));
    for (let i = 0; i < w.length; i++) w[i] = normal(r) * s;
    layers.push({ w, b: new Float64Array(out), in: inn, out });
  }
  return { sizes: sizes.slice(), layers, act };
}

/**
 * 前向 + 反向共享的工作区。
 * acts[l]/zs[l] 是每层的激活与预激活；ds[l] 是第 l 层回传到的误差向量。
 */
export interface Cache {
  acts: Float64Array[];
  zs: Float64Array[];
  ds: Float64Array[];
}

export function makeCache(m: Model): Cache {
  const acts: Float64Array[] = [];
  const zs: Float64Array[] = [];
  const ds: Float64Array[] = [];
  for (const L of m.layers) {
    acts.push(new Float64Array(L.out));
    zs.push(new Float64Array(L.out));
    ds.push(new Float64Array(L.out));
  }
  return { acts, zs, ds };
}

function applyAct(a: Act, z: number): number {
  if (a === "relu") return z > 0 ? z : 0;
  if (a === "sigmoid") return 1 / (1 + Math.exp(-z));
  return Math.tanh(z);
}

function actDeriv(a: Act, z: number, act: number): number {
  if (a === "relu") return z > 0 ? 1 : 0;
  if (a === "sigmoid") return act * (1 - act);
  return 1 - act * act;
}

/** 输出层就地改成 softmax 概率；先减最大值，指数再大也不会溢出 */
function softmaxInPlace(v: Float64Array): void {
  let mx = -Infinity;
  for (let i = 0; i < v.length; i++) if (v[i] > mx) mx = v[i];
  let s = 0;
  for (let i = 0; i < v.length; i++) {
    const e = Math.exp(v[i] - mx);
    v[i] = e;
    s += e;
  }
  const inv = s > 0 ? 1 / s : 0;
  for (let i = 0; i < v.length; i++) v[i] *= inv;
}

/** 一次前向，中间量写进 cache，返回输出层的概率向量 */
export function forward(m: Model, x0: number, x1: number, cache: Cache): Float64Array {
  const { layers, act } = m;
  const last = layers.length - 1;
  for (let l = 0; l <= last; l++) {
    const L = layers[l];
    const z = cache.zs[l];
    const a = cache.acts[l];
    const prev = l === 0 ? null : cache.acts[l - 1];
    for (let j = 0; j < L.out; j++) {
      const row = j * L.in;
      let s = L.b[j];
      if (l === 0) s += L.w[row] * x0 + L.w[row + 1] * x1;
      else for (let i = 0; i < L.in; i++) s += L.w[row + i] * prev![i];
      z[j] = s;
      a[j] = l === last ? s : applyAct(act, s);
    }
    if (l === last) softmaxInPlace(a);
  }
  return cache.acts[last];
}

/** 逐层激活值，第 0 行就是输入坐标；画网络流程图用 */
export function activations(m: Model, x0: number, x1: number, cache: Cache): number[][] {
  forward(m, x0, x1, cache);
  const rows: number[][] = [[x0, x1]];
  for (const a of cache.acts) rows.push(Array.from(a));
  return rows;
}

export interface GradPack {
  dw: Float64Array[];
  db: Float64Array[];
}

export function makeGrad(m: Model): GradPack {
  return {
    dw: m.layers.map((L) => new Float64Array(L.w.length)),
    db: m.layers.map((L) => new Float64Array(L.out)),
  };
}

export function zeroGrad(g: GradPack): void {
  for (const v of g.dw) v.fill(0);
  for (const v of g.db) v.fill(0);
}

/**
 * 单样本的损失与梯度累加（softmax + 交叉熵，输出层误差就是 p − onehot，
 * 这条组合的梯度最干净，所以二分类也用两个输出端而不走 sigmoid+BCE）。
 * 返回 −log p[y]；g 不清零，由调用方按 batch 平均。
 */
export function sampleGrad(m: Model, x0: number, x1: number, y: number, cache: Cache, g: GradPack): number {
  const p = forward(m, x0, x1, cache);
  const last = m.layers.length - 1;
  const d0 = cache.ds[last];
  for (let j = 0; j < p.length; j++) d0[j] = p[j] - (j === y ? 1 : 0);
  for (let l = last; l >= 0; l--) {
    const L = m.layers[l];
    const d = cache.ds[l];
    const dw = g.dw[l];
    const db = g.db[l];
    for (let j = 0; j < L.out; j++) {
      const dj = d[j];
      if (dj === 0) continue;
      db[j] += dj;
      const row = j * L.in;
      for (let i = 0; i < L.in; i++) dw[row + i] += dj * (l === 0 ? (i === 0 ? x0 : x1) : cache.acts[l - 1][i]);
    }
    if (l === 0) break;
    const up = cache.ds[l - 1];
    const a = cache.acts[l - 1];
    const z = cache.zs[l - 1];
    for (let i = 0; i < L.in; i++) {
      let s = 0;
      for (let j = 0; j < L.out; j++) s += d[j] * L.w[j * L.in + i];
      up[i] = s * actDeriv(m.act, z[i], a[i]);
    }
  }
  return -Math.log(Math.max(1e-12, p[y]));
}

/** 一个 batch 的平均损失 + 一次带动量的参数更新 */
export function trainBatch(
  m: Model,
  data: Dataset,
  order: ArrayLike<number>,
  from: number,
  to: number,
  opt: TrainOpt,
  cache: Cache,
  g: GradPack,
  vel: GradPack
): number {
  zeroGrad(g);
  let sum = 0;
  let cnt = 0;
  for (let t = from; t < to; t++) {
    const i = order[t];
    sum += sampleGrad(m, data.xs[2 * i], data.xs[2 * i + 1], data.ys[i], cache, g);
    cnt++;
  }
  if (!cnt) return 0;
  const inv = 1 / cnt;
  for (let l = 0; l < m.layers.length; l++) {
    const L = m.layers[l];
    const dw = g.dw[l];
    const vw = vel.dw[l];
    for (let i = 0; i < dw.length; i++) {
      const s = opt.momentum * vw[i] + dw[i] * inv;
      vw[i] = s;
      L.w[i] -= opt.lr * s;
    }
    const db = g.db[l];
    const vb = vel.db[l];
    for (let j = 0; j < db.length; j++) {
      const s = opt.momentum * vb[j] + db[j] * inv;
      vb[j] = s;
      L.b[j] -= opt.lr * s;
    }
  }
  return sum * inv;
}

export interface EpochStat {
  loss: number;
  acc: number;
  steps: number;
}

/** 跑一轮：随机打乱后按 batch 更新 */
export function trainEpoch(
  m: Model,
  data: Dataset,
  opt: TrainOpt,
  r: () => number,
  cache: Cache,
  g: GradPack,
  vel: GradPack
): EpochStat {
  const order = Array.from({ length: data.n }, (_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(r() * (i + 1));
    const t = order[i];
    order[i] = order[j];
    order[j] = t;
  }
  const bs = Math.max(1, Math.min(opt.batch, data.n));
  let loss = 0;
  let steps = 0;
  for (let from = 0; from < data.n; from += bs) {
    loss += trainBatch(m, data, order, from, Math.min(from + bs, data.n), opt, cache, g, vel);
    steps++;
  }
  return { loss, acc: accuracy(m, data), steps };
}

export function predict(m: Model, x0: number, x1: number, cache: Cache): number {
  const p = forward(m, x0, x1, cache);
  let best = 0;
  for (let j = 1; j < p.length; j++) if (p[j] > p[best]) best = j;
  return best;
}

/** 预测类别 + 该类比第二名高出来的置信差（决策边界的明度由它给） */
export function predictMargin(m: Model, x0: number, x1: number, cache: Cache): { label: number; margin: number } {
  const p = forward(m, x0, x1, cache);
  let best = 0;
  let second = -Infinity;
  for (let j = 1; j < p.length; j++) {
    if (p[j] > p[best]) {
      second = p[best];
      best = j;
    } else if (p[j] > second) second = p[j];
  }
  return { label: best, margin: p.length > 1 ? p[best] - second : p[best] };
}

export function accuracy(m: Model, data: Dataset, cache = makeCache(m)): number {
  let hit = 0;
  for (let i = 0; i < data.n; i++)
    if (predict(m, data.xs[2 * i], data.xs[2 * i + 1], cache) === data.ys[i]) hit++;
  return hit / data.n;
}

/** 全数据集平均损失：训练曲线用它，不受 batch 抽样影响 */
export function dataLoss(m: Model, data: Dataset, cache = makeCache(m)): number {
  let s = 0;
  for (let i = 0; i < data.n; i++) {
    const p = forward(m, data.xs[2 * i], data.xs[2 * i + 1], cache);
    s -= Math.log(Math.max(1e-12, p[data.ys[i]]));
  }
  return s / data.n;
}

/* ------------------------------------------------------------------ 数据集 */

function pack(name: string, k: number, pts: [number, number, number][]): Dataset {
  const xs = new Float64Array(pts.length * 2);
  const ys = new Int32Array(pts.length);
  pts.forEach((p, i) => {
    xs[2 * i] = p[0];
    xs[2 * i + 1] = p[1];
    ys[i] = p[2];
  });
  return { xs, ys, n: pts.length, k, name };
}

export type DatasetName = "xor" | "circle" | "moons" | "spiral";

export const DATASETS: DatasetName[] = ["xor", "circle", "moons", "spiral"];

/**
 * 教学用二维数据集都铺在 [-3,3]² 里，和复平面模式的默认视口同一尺度，
 * 换数据集不必重新缩放画布。
 */
export function dataset(name: DatasetName, n: number, seed: number): Dataset {
  const r = rng(seed);
  const pts: [number, number, number][] = [];
  if (name === "xor") {
    for (let i = 0; i < n; i++) {
      const x = (r() * 2 - 1) * 2.6;
      const y = (r() * 2 - 1) * 2.6;
      pts.push([x, y, x * y > 0 ? 1 : 0]);
    }
    return pack("异或", 2, pts);
  }
  if (name === "circle") {
    for (let i = 0; i < n; i++) {
      const inner = i % 2 === 0;
      const rad = inner ? 0.95 * Math.sqrt(r()) : 1.75 + 0.75 * r();
      const th = r() * Math.PI * 2;
      pts.push([rad * Math.cos(th), rad * Math.sin(th), inner ? 0 : 1]);
    }
    return pack("同心圆", 2, pts);
  }
  if (name === "moons") {
    for (let i = 0; i < n; i++) {
      const cls = i % 2;
      const th = r() * Math.PI;
      const x = Math.cos(th) * 1.9 + (cls ? 0.95 : -0.95);
      const y = Math.sin(th) * 1.6 * (cls ? -1 : 1) + (r() - 0.5) * 0.3;
      pts.push([x, y + (cls ? -0.55 : 0.55), cls]);
    }
    return pack("双月", 2, pts);
  }
  for (let i = 0; i < n; i++) {
    const cls = i % 3;
    const t = 0.3 + 2.5 * Math.sqrt(r());
    const th = t * 1.9 + (cls * 2 * Math.PI) / 3 + (r() - 0.5) * 0.3;
    pts.push([t * Math.cos(th), t * Math.sin(th), cls]);
  }
  return pack("三臂螺旋", 3, pts);
}

/* ------------------------------------------------------------ 决策边界栅格 */

/** 类别配色取外壳紫色渐变那一族，避免界面出现两套色感 */
export const CLASS_COLORS: [number, number, number][] = [
  [102, 126, 234],
  [118, 75, 162],
  [245, 87, 108],
  [67, 233, 123],
  [250, 112, 154],
  [79, 172, 254],
];

/**
 * 单行采样：行 0 对应世界纵坐标 top，和 domainColor 同一条约定。
 * 分带渐进细化只认这一条路径，所以逐行合成必定与整幅一次采样逐字节同解。
 */
export function decisionRow(
  m: Model,
  left: number,
  right: number,
  y: number,
  w: number,
  dark: boolean,
  out: Uint8ClampedArray,
  off: number,
  cache: Cache
): void {
  const dx = w > 1 ? (right - left) / (w - 1) : 0;
  const bg = dark ? DARK_BG : LIGHT_BG;
  for (let i = 0; i < w; i++) {
    const { label, margin } = predictMargin(m, left + dx * i, y, cache);
    const c = CLASS_COLORS[label % CLASS_COLORS.length];
    const t = 0.26 + 0.64 * Math.min(1, Math.max(0, margin));
    const o = off + i * 3;
    out[o] = bg[0] + (c[0] - bg[0]) * t;
    out[o + 1] = bg[1] + (c[1] - bg[1]) * t;
    out[o + 2] = bg[2] + (c[2] - bg[2]) * t;
  }
}

const DARK_BG = [13, 15, 26];
const LIGHT_BG = [248, 249, 255];

/** 整幅决策边界：逐行调用 decisionRow，与分带续算共用同一段着色代码 */
export function decisionRaster(
  m: Model,
  left: number,
  right: number,
  top: number,
  bottom: number,
  w: number,
  h: number,
  dark: boolean
): Uint8ClampedArray {
  const out = new Uint8ClampedArray(w * h * 3);
  if (w < 1 || h < 1) return out;
  const cache = makeCache(m);
  const dy = h > 1 ? (bottom - top) / (h - 1) : 0;
  for (let j = 0; j < h; j++) decisionRow(m, left, right, top + dy * j, w, dark, out, j * w * 3, cache);
  return out;
}
