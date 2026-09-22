/**
 * GeoLab 求值机：把 AST 编译成闭包，支持实数 / 复数 / 向量 / 矩阵 / 逻辑值 / 函数值
 *
 * 两条求值路径：
 *  - Engine.eval：通用路径，处理复平面、向量、矩阵、惰性数值分析形式
 *  - compileReal：静态推断为实数的表达式走 (Float64Array) => number 快路径，
 *    供曲面采样、共形着色等每帧上万次的热循环使用
 */

import { VK, type Val, type Node, type FunValue, type BinOp } from "./types.ts";
import * as CN from "./cnum.ts";
import { BUILTIN_NAMES, freeNames, parseDefinition } from "./parser.ts";
import type { Definition } from "./parser.ts";

export const nan = Number.NaN;
const PI = Math.PI;

/* ------------------------------------------------------------------ */
/* 值的构造与转换                                                     */
/* ------------------------------------------------------------------ */

export function num(re: number, im = 0): Val {
  return { k: VK.Num, re, im, b: false, v: null, m: null, fn: null, s: null };
}
export function bool(b: boolean): Val {
  return { k: VK.Bool, re: b ? 1 : 0, im: 0, b, v: null, m: null, fn: null, s: null };
}
export function vecVal(v: number[]): Val {
  return { k: VK.Vec, re: NaN, im: 0, b: false, v, m: null, fn: null, s: null };
}
export function matVal(m: number[][]): Val {
  return { k: VK.Mat, re: NaN, im: 0, b: false, v: null, m, fn: null, s: null };
}
export function funVal(f: FunValue): Val {
  return { k: VK.Fun, re: NaN, im: 0, b: false, v: null, m: null, fn: f, s: null };
}
export function strVal(s: string): Val {
  return { k: VK.Str, re: NaN, im: 0, b: false, v: null, m: null, fn: null, s };
}
export const ZERO = num(0);

export function kindName(x: Val): string {
  switch (x.k) {
    case VK.Num:
      return x.im !== 0 ? "复数" : "实数";
    case VK.Bool:
      return "逻辑值";
    case VK.Vec:
      return `向量(${x.v!.length})`;
    case VK.Mat:
      return `矩阵${x.m!.length}x${x.m![0].length}`;
    case VK.Fun:
      return "函数";
    case VK.Str:
      return "文本";
    default:
      return "值";
  }
}

export function asReal(x: Val, what = "值"): number {
  if (x.k === VK.Bool) return x.b ? 1 : 0;
  if (x.k !== VK.Num) throw new Error(`${what}需要实数，实际是 ${kindName(x)}`);
  if (x.im !== 0) throw new Error(`${what}需要实数，实际是复数 ${CN.fmtC(x.re, x.im)}`);
  return x.re;
}
export function asVec(x: Val, what = "值"): number[] {
  if (x.k === VK.Vec) return x.v!;
  if (x.k === VK.Num && x.im !== 0) return [x.re, x.im];
  if (x.k === VK.Num) return [x.re];
  if (x.k === VK.Mat) return x.m!.flat();
  throw new Error(`${what}需要向量，实际是 ${kindName(x)}`);
}
export function asMat(x: Val, what = "值"): number[][] {
  if (x.k === VK.Mat) return x.m!;
  if (x.k === VK.Vec) return [x.v!.slice()];
  if (x.k === VK.Num) return [[x.re]];
  throw new Error(`${what}需要矩阵，实际是 ${kindName(x)}`);
}
export function asFun(x: Val, what = "值"): FunValue {
  if (x.k === VK.Fun) return x.fn!;
  throw new Error(`${what}需要函数，实际是 ${kindName(x)}`);
}
export function truthy(v: Val): boolean {
  if (v.k === VK.Bool) return v.b;
  if (v.k === VK.Num) return v.re !== 0;
  if (v.k === VK.Vec) return v.v!.some((x) => x !== 0);
  if (v.k === VK.Str) return v.s!.length > 0;
  return false;
}
/** 复数平面读取：任意值 → [re, im] */
export function reIm(x: Val): [number, number] {
  if (x.k === VK.Num) return [x.re, x.im];
  if (x.k === VK.Bool) return [x.b ? 1 : 0, 0];
  if (x.k === VK.Vec) return [x.v![0] ?? 0, x.v![1] ?? 0];
  throw new Error("需要数值（实数或复数）");
}

/** list 字面量：一维数值 → 向量；等长行 → 矩阵 */
export function listVal(items: Val[]): Val {
  if (items.length > 1 && items.every((i) => i.k === VK.Vec)) {
    const len = items[0].v!.length;
    if (len > 1 && items.every((r) => r.v!.length === len)) return matVal(items.map((r) => r.v!.slice()));
  }
  if (items.length > 1 && items.every((i) => i.k === VK.Mat)) {
    const rows: number[][] = [];
    for (const m of items) rows.push(...m.m!.map((r) => r.slice()));
    return matVal(rows);
  }
  const out: number[] = [];
  for (const it of items) {
    if (it.k === VK.Num) {
      out.push(it.re);
      if (it.im !== 0) out.push(it.im);
    } else if (it.k === VK.Vec) out.push(...it.v!);
    else if (it.k === VK.Bool) out.push(it.b ? 1 : 0);
    else throw new Error(`列表元素需要是数值，收到 ${kindName(it)}`);
  }
  return vecVal(out);
}

/* ------------------------------------------------------------------ */
/* 内置函数                                                           */
/* ------------------------------------------------------------------ */

export interface EvalCtx {
  eng: Engine;
  scope: Scope;
  depth: number;
}
export type Scope = { vars: Map<string, Val>; up?: Scope } | undefined;
export type StrictFn = (args: Val[], ctx: EvalCtx) => Val;
export type LazyFn = (args: Node[], ctx: EvalCtx) => Val;

export const STRICT: Record<string, StrictFn> = {};

/** 复数可提升的一元函数 */
const C1: Record<string, (re: number, im: number, o: CN.C) => CN.C> = {
  sin: CN.csin, cos: CN.ccos, tan: CN.ctan,
  asin: CN.casin, acos: CN.cacos, atan: CN.catan,
  sinh: CN.csinh, cosh: CN.ccosh, tanh: CN.ctanh,
  asinh: CN.casinh, acosh: CN.cacosh, atanh: CN.catanh,
  exp: CN.cexp, log: CN.clog, ln: CN.clog, sqrt: CN.csqrt,
};
/** 仅实数定义的一元函数 */
const F1: Record<string, (x: number) => number> = {
  sec: (x) => 1 / Math.cos(x),
  csc: (x) => 1 / Math.sin(x),
  cot: (x) => Math.cos(x) / Math.sin(x),
  coth: (x) => Math.cosh(x) / Math.sinh(x),
  asec: (x) => Math.acos(1 / x),
  acsc: (x) => Math.asin(1 / x),
  acot: (x) => Math.atan(1 / x),
  exp2: (x) => Math.pow(2, x),
  exp10: (x) => Math.pow(10, x),
  log2: (x) => Math.log2(x),
  log10: (x) => Math.log10(x),
  lg: (x) => Math.log10(x),
  cbrt: (x) => Math.cbrt(x),
  sign: (x) => Math.sign(x),
  floor: (x) => Math.floor(x),
  ceil: (x) => Math.ceil(x),
  round: (x) => Math.round(x),
  trunc: (x) => Math.trunc(x),
  frac: (x) => x - Math.floor(x),
  sinc: (x) => (x === 0 ? 1 : Math.sin(x) / x),
  sinpi: (x) => Math.sin(PI * x),
  cospi: (x) => Math.cos(PI * x),
  step: (x) => (x < 0 ? 0 : 1),
  dirac: (x) => (x === 0 ? Infinity : 0),
  erf: (x) => erf(x),
  gamma: (x) => CN.gamma(x),
  lgamma: (x) => lgamma(x),
};

const SC = { re: 0, im: 0 };
for (const name of Object.keys(C1)) {
  const f = C1[name];
  STRICT[name] = (a) => {
    const x = a[0];
    if (!x) throw new Error(`${name} 需要参数`);
    if (x.k === VK.Vec) return vecVal(x.v!.map((v) => f(v, 0, SC).re));
    if (x.k !== VK.Num) throw new Error(`${name} 需要数值参数，收到 ${kindName(x)}`);
    const o = f(x.re, x.im, SC);
    return num(o.re, o.im);
  };
}
for (const name of Object.keys(F1)) {
  const f = F1[name];
  STRICT[name] = (a) => {
    const x = a[0];
    if (x.k === VK.Vec) return vecVal(x.v!.map(f));
    return num(f(asReal(x, `${name} 的参数`)));
  };
}

function nums(a: Val[], name: string): number[] {
  const out: number[] = [];
  for (const v of a) {
    if (v.k === VK.Vec) out.push(...v.v!);
    else if (v.k === VK.Num) {
      if (v.im !== 0) throw new Error(`${name} 只接受实数`);
      out.push(v.re);
    } else if (v.k === VK.Bool) out.push(v.b ? 1 : 0);
    else throw new Error(`${name} 不接受 ${kindName(v)}`);
  }
  return out;
}
function flattenArg(a: Val[], name: string): number[] {
  if (a.length === 1) return nums([a[0]], name);
  return nums(a, name);
}

STRICT.atan2 = (a) => num(Math.atan2(asReal(a[0], "atan2 的 y"), asReal(a[1], "atan2 的 x")));
STRICT.re = STRICT.real = (a) => num(reIm(a[0])[0]);
STRICT.im = STRICT.imag = (a) => num(reIm(a[0])[1]);
STRICT.conj = (a) => {
  const [r, i] = reIm(a[0]);
  return num(r, -i);
};
STRICT.abs = (a) => {
  const x = a[0];
  if (x.k === VK.Vec) return vecVal(x.v!.map(Math.abs));
  if (x.k === VK.Mat) return matVal(x.m!.map((row) => row.map(Math.abs)));
  const [r, i] = reIm(x);
  return num(Math.hypot(r, i));
};
STRICT.norm = (a) => {
  const x = a[0];
  if (x.k === VK.Vec) {
    const p = a[1] ? asReal(a[1], "范数阶") : 2;
    if (!Number.isFinite(p)) return num(Math.max(...x.v!.map(Math.abs)));
    let s = 0;
    for (const v of x.v!) s += Math.pow(Math.abs(v), p);
    return num(Math.pow(s, 1 / p));
  }
  if (x.k === VK.Mat) {
    // Frobenius 范数
    let s = 0;
    for (const row of x.m!) for (const v of row) s += v * v;
    return num(Math.sqrt(s));
  }
  const [r, i] = reIm(x);
  return num(Math.hypot(r, i));
};
STRICT.abs2 = (a) => {
  const [r, i] = reIm(a[0]);
  return num(r * r + i * i);
};
STRICT.phase = STRICT.arg = STRICT.angle0 = (a) => {
  const [r, i] = reIm(a[0]);
  // 负实轴统一取 +π：一元负号会把 0 变成 -0，否则 arg(-1) 会得到 -π
  return num(Math.atan2(i === 0 ? 0 : i, r));
};
STRICT.complex = (a) => num(asReal(a[0], "complex 实部"), asReal(a[1] ?? ZERO, "complex 虚部"));
STRICT.polar = (a) => {
  const m = asReal(a[0], "polar 模");
  const th = asReal(a[1] ?? ZERO, "polar 辐角");
  return num(m * Math.cos(th), m * Math.sin(th));
};
STRICT.__imag = () => num(0, 1);
STRICT.mod = (a) => {
  if (a[0].k === VK.Vec && a[1].k === VK.Num) {
    const m = asReal(a[1]);
    return vecVal(a[0].v!.map((x) => ((x % m) + m) % m));
  }
  const x = asReal(a[0], "mod 的参数");
  const y = asReal(a[1], "mod 的参数");
  return num(((x % y) + y) % y);
};
STRICT.rem = (a) => num(asReal(a[0]) % asReal(a[1]));
STRICT.min = (a) => num(Math.min(...flattenArg(a, "min")));
STRICT.max = (a) => num(Math.max(...flattenArg(a, "max")));
STRICT.clamp = (a) => num(Math.min(Math.max(asReal(a[0]), asReal(a[1])), asReal(a[2])));
STRICT.hypot = (a) => num(Math.hypot(...flattenArg(a, "hypot")));
STRICT.gcd = (a) => {
  const g = (x: number, y: number): number => (y === 0 ? Math.abs(x) : g(y, x % y));
  return num(nums(a, "gcd").reduce((p, q) => g(p, q)));
};
STRICT.lcm = (a) => {
  const g = (x: number, y: number): number => (y === 0 ? Math.abs(x) : g(y, x % y));
  return num(nums(a, "lcm").reduce((p, q) => Math.abs(p * q) / (g(p, q) || 1)));
};
STRICT.factorial = (a) => num(CN.factorial(asReal(a[0], "阶乘参数")));
STRICT["!"] = STRICT.factorial;
STRICT.nCr = (a) => {
  const n = asReal(a[0]), r = Math.round(asReal(a[1]));
  let out = 1;
  for (let k = 0; k < r; k++) out = (out * (n - k)) / (k + 1);
  return num(out);
};
STRICT.binomial = STRICT.nCr;
STRICT.nPr = (a) => {
  const n = asReal(a[0]), r = Math.round(asReal(a[1]));
  let out = 1;
  for (let k = 0; k < r; k++) out *= n - k;
  return num(out);
};
STRICT.beta = (a) => {
  const x = asReal(a[0]), y = asReal(a[1]);
  return num((CN.gamma(x) * CN.gamma(y)) / CN.gamma(x + y));
};
STRICT.isprime = (a) => {
  const n = Math.round(asReal(a[0]));
  if (n < 2) return bool(false);
  for (let k = 2; k * k <= n; k++) if (n % k === 0) return bool(false);
  return bool(true);
};
STRICT.root = (a) => {
  const n = asReal(a[0]), x = asReal(a[1]);
  if (x < 0 && Math.abs(Math.round(n) % 2) === 1) return num(-Math.pow(-x, 1 / n));
  return num(Math.pow(x, 1 / n));
};
STRICT.logb = (a) => num(Math.log(asReal(a[1])) / Math.log(asReal(a[0])));
STRICT.deg = (a) => num((asReal(a[0]) * 180) / PI);
STRICT.rad = (a) => num((asReal(a[0]) * PI) / 180);
STRICT.polyval = (a) => {
  const cs = asVec(a[0], "polyval 系数");
  const x = asReal(a[1], "polyval 自变量");
  let s = 0;
  for (const co of cs) s = s * x + co;
  return num(s);
};
STRICT.component = (a) => num(asVec(a[0], "component 的向量")[Math.round(asReal(a[1], "分量下标")) - 1] ?? nan);

/* --- 向量 / 矩阵 --- */
STRICT.dot = (a) => {
  const x = asVec(a[0], "dot"), y = asVec(a[1], "dot");
  let s = 0;
  for (let i = 0; i < Math.min(x.length, y.length); i++) s += x[i] * y[i];
  return num(s);
};
STRICT.cross = (a) => {
  const x = asVec(a[0], "cross"), y = asVec(a[1], "cross");
  if (x.length === 2 && y.length === 2) return num(x[0] * y[1] - x[1] * y[0]);
  const p = (v: number[], i: number) => v[i] ?? 0;
  return vecVal([
    p(x, 1) * p(y, 2) - p(x, 2) * p(y, 1),
    p(x, 2) * p(y, 0) - p(x, 0) * p(y, 2),
    p(x, 0) * p(y, 1) - p(x, 1) * p(y, 0),
  ]);
};
STRICT.magnitude = (a) => num(Math.hypot(...asVec(a[0], "magnitude")));
STRICT.len = (a) => {
  const x = a[0];
  if (x.k === VK.Str) return num(x.s!.length);
  if (x.k === VK.Mat) return vecVal([x.m!.length, x.m![0].length]);
  if (x.k === VK.Num) return num(1);
  return num(Math.hypot(...asVec(x, "len")));
};
STRICT.angle = (a) => {
  const x = a[0];
  if (x.k === VK.Vec) {
    const v = x.v!;
    if (v.length <= 2) return num(Math.atan2(v[1] ?? 0, v[0] ?? 0));
    const m = Math.hypot(...v);
    return num(m === 0 ? 0 : Math.acos(CN.clamp((v[2] ?? 0) / m, -1, 1)));
  }
  if (a.length >= 2) {
    const u = asVec(x), v = asVec(a[1]);
    const d =
      u.reduce((s, q, i) => s + q * (v[i] ?? 0), 0) / ((Math.hypot(...u) || 1) * (Math.hypot(...v) || 1));
    return num(Math.acos(CN.clamp(d, -1, 1)));
  }
  const [r, i] = reIm(x);
  return num(Math.atan2(i === 0 ? 0 : i, r));
};
STRICT.unit = (a) => {
  const v = asVec(a[0], "unit");
  const m = Math.hypot(...v) || 1;
  return vecVal(v.map((x) => x / m));
};
STRICT.proj = (a) => {
  const u = asVec(a[0], "proj"), v = asVec(a[1], "proj");
  const d = v.reduce((s, x) => s + x * x, 0) || 1;
  const k = u.reduce((s, x, i) => s + x * (v[i] ?? 0), 0) / d;
  return vecVal(v.map((x) => x * k));
};
STRICT.orthogonal = (a) => {
  const v = asVec(a[0]);
  if (v.length === 2) return vecVal([-v[1], v[0]]);
  const w = a[1] ? asVec(a[1]) : v[2] === 0 ? [0, 0, 1] : [1, 0, 0];
  const d = w.reduce((s, x) => s + x * x, 0) || 1;
  const k = v.reduce((s, x, i) => s + x * (w[i] ?? 0), 0) / d;
  return vecVal(w.map((x, i) => v[i] - x * k));
};
STRICT.transpose = (a) => {
  const M = asMat(a[0], "transpose");
  return matVal(M[0].map((_, c) => M.map((r) => r[c])));
};
STRICT.det = (a) => num(det(asMat(a[0], "det")));
STRICT.identity = STRICT.eye = (a) => {
  const n = Math.round(asReal(a[0], "identity 维度"));
  return matVal(Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))));
};
STRICT.inv = (a) => {
  const I = inv(asMat(a[0], "inv"));
  return I ? matVal(I) : num(nan);
};
STRICT.matmul = (a, ctx) => ctx.eng.binaryVal("*", a[0], a[1]);
STRICT.rows = (a) => num(asMat(a[0]).length);
STRICT.cols = (a) => num(asMat(a[0])[0].length);
STRICT.size = (a) => {
  const x = a[0];
  if (x.k === VK.Mat) return vecVal([x.m!.length, x.m![0].length]);
  if (x.k === VK.Vec) return vecVal([x.v!.length]);
  return vecVal([1]);
};
STRICT.solve = (a) => {
  const s = solveLin(asMat(a[0], "solve"), asVec(a[1], "solve"));
  return s ? vecVal(s) : num(nan);
};

/* --- 统计 --- */
STRICT.sort = (a) => vecVal(flattenArg(a, "sort").slice().sort((p, q) => p - q));
STRICT.mean = (a) => {
  const xs = flattenArg(a, "mean");
  return num(xs.reduce((p, q) => p + q, 0) / (xs.length || 1));
};
STRICT.median = (a) => {
  const xs = flattenArg(a, "median").slice().sort((p, q) => p - q);
  const m = xs.length >> 1;
  return num(xs.length % 2 ? xs[m] : (xs[m - 1] + xs[m]) / 2);
};
STRICT.var = (a) => {
  const xs = flattenArg(a, "var");
  const mu = xs.reduce((p, q) => p + q, 0) / (xs.length || 1);
  return num(xs.reduce((s, x) => s + (x - mu) ** 2, 0) / Math.max(1, xs.length - 1));
};
STRICT.std = (a, ctx) => num(Math.sqrt(asReal(STRICT.var!(a, ctx), "方差")));
STRICT.rand = (a) => {
  if (!a.length) return num(Math.random());
  const n = Math.round(asReal(a[0], "rand"));
  const lo = a[1] ? asReal(a[1]) : 0;
  const hi = a[2] ? asReal(a[2]) : 1;
  return vecVal(Array.from({ length: n }, () => lo + Math.random() * (hi - lo)));
};
STRICT.randn = () => {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return num(Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * PI * v));
};
STRICT.primes = (a) => {
  const n = Math.round(asReal(a[0], "primes"));
  const out: number[] = [];
  for (let k = 2; out.length < n; k++) {
    let isp = true;
    for (let d = 2; d * d <= k; d++)
      if (k % d === 0) {
        isp = false;
        break;
      }
    if (isp) out.push(k);
  }
  return vecVal(out);
};

/* --- 线性代数 --- */
export function det(M: number[][]): number {
  const n = M.length;
  if (!n) return NaN;
  if (M.some((r) => r.length !== n)) return NaN;
  if (n === 1) return M[0][0];
  if (n === 2) return M[0][0] * M[1][1] - M[0][1] * M[1][0];
  const A = M.map((r) => r.slice());
  let d = 1;
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    if (Math.abs(A[piv][c]) < 1e-16) return 0;
    if (piv !== c) {
      const t = A[piv];
      A[piv] = A[c];
      A[c] = t;
      d = -d;
    }
    d *= A[c][c];
    for (let r = c + 1; r < n; r++) {
      const f = A[r][c] / A[c][c];
      if (f === 0) continue;
      for (let k = c; k < n; k++) A[r][k] -= f * A[c][k];
    }
  }
  return d;
}

export function inv(M: number[][]): number[][] | null {
  const n = M.length;
  if (!n || M.some((r) => r.length !== n)) return null;
  const A = M.map((r, i) => [...r, ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    if (Math.abs(A[piv][c]) < 1e-14) return null;
    const t = A[piv];
    A[piv] = A[c];
    A[c] = t;
    const p = A[c][c];
    for (let k = 0; k < 2 * n; k++) A[c][k] /= p;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = A[r][c];
      if (f === 0) continue;
      for (let k = 0; k < 2 * n; k++) A[r][k] -= f * A[c][k];
    }
  }
  return A.map((r) => r.slice(n));
}

export function mulMat(A: number[][], B: number[][]): number[][] {
  const n = A.length, k = B.length, m = B[0].length;
  const out: number[][] = [];
  for (let i = 0; i < n; i++) {
    const row = new Array(m).fill(0);
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let t = 0; t < k; t++) s += A[i][t] * B[t][j];
      row[j] = s;
    }
    out.push(row);
  }
  return out;
}

export function solveLin(A: number[][], b: number[]): number[] | null {
  const n = A.length;
  if (!n || A.some((r) => r.length !== n)) return null;
  const M = A.map((r, i) => [...r, b[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (Math.abs(M[piv][c]) < 1e-14) return null;
    const t = M[piv];
    M[piv] = M[c];
    M[c] = t;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / M[c][c];
      if (f === 0) continue;
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((r, i) => r[n] / r[i]);
}

function erf(x: number): number {
  const s = Math.sign(x);
  const ax = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * ax);
  const y =
    1 -
    ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) *
      t *
      Math.exp(-ax * ax);
  return s * y;
}
function lgamma(x: number): number {
  if (x <= 0 && Number.isInteger(x)) return Infinity;
  return Math.log(Math.abs(CN.gamma(x)));
}

/* ------------------------------------------------------------------ */
/* 引擎                                                               */
/* ------------------------------------------------------------------ */

export const CONSTANTS: Record<string, number> = {
  pi: PI,
  tau: CN.TAU,
  e: Math.E,
  phi: (1 + Math.sqrt(5)) / 2,
  euler: 0.5772156649015329,
  inf: Infinity,
  infinity: Infinity,
  true: 1,
  false: 0,
};

export class Engine {
  globals = new Map<string, Val>();
  fns = new Map<string, FunValue>();
  lazy = new Map<string, LazyFn>();
  evals = 0;

  constructor() {
    registerLazy(this);
  }

  define(src: string): Definition {
    const d = parseDefinition(src);
    const ctx: EvalCtx = { eng: this, scope: undefined, depth: 0 };
    if (d.kind === "fn") this.fns.set(d.name, { name: d.name, params: d.params, body: d.node });
    else if (d.kind === "let") this.globals.set(d.name, this.eval(d.node, ctx));
    return d;
  }

  set(name: string, v: Val): void {
    this.globals.set(name, v);
  }
  setNum(name: string, re: number, im = 0): void {
    this.globals.set(name, num(re, im));
  }
  remove(name: string): void {
    this.globals.delete(name);
    this.fns.delete(name);
  }

  lookup(name: string, scope: Scope): Val | null {
    let s: Scope = scope;
    while (s) {
      const v = s.vars.get(name);
      if (v) return v;
      s = s.up;
    }
    const g = this.globals.get(name);
    if (g) return g;
    if (name === "i" || name === "j" || name === "I") return num(0, 1);
    const f = this.fns.get(name);
    if (f) return funVal(f);
    const c = CONSTANTS[name];
    if (c !== undefined) return num(c);
    return null;
  }

  eval(nd: Node, ctx: EvalCtx): Val {
    this.evals++;
    switch (nd.type) {
      case "num":
        return num(nd.value);
      case "str":
        return strVal(nd.value);
      case "ident": {
        const v = this.lookup(nd.name, ctx.scope);
        if (!v) throw new Error(`未定义的名称 “${nd.name}”`);
        return v;
      }
      case "un": {
        const a = this.eval(nd.e, ctx);
        if (nd.op === "!") return bool(!truthy(a));
        if (a.k === VK.Num) return num(nd.op === "-" ? -a.re : a.re, nd.op === "-" ? -a.im : a.im);
        if (a.k === VK.Vec) return vecVal(a.v!.map((x) => (nd.op === "-" ? -x : x)));
        throw new Error("正负号只适用于数值或向量");
      }
      case "post": {
        const a = this.eval(nd.e, ctx);
        const fn = nd.op === "!" ? STRICT.factorial : STRICT.rad;
        return fn([a], ctx);
      }
      case "bin":
        return this.binary(nd.op, nd.l, nd.r, ctx);
      case "cond":
        return this.eval(truthy(this.eval(nd.c, ctx)) ? nd.a : nd.b, ctx);
      case "list":
        return listVal(nd.items.map((x) => this.eval(x, ctx)));
      case "idx": {
        const base = this.eval(nd.e, ctx);
        const i = Math.round(asReal(this.eval(nd.i, ctx), "下标"));
        if (base.k === VK.Vec) return num(base.v![i - 1] ?? nan);
        if (base.k === VK.Mat) {
          const row = base.m![i - 1];
          return row ? vecVal(row.slice()) : num(nan);
        }
        if (base.k === VK.Num) return num(i === 1 ? base.re : base.im);
        throw new Error("该类型不能使用 [] 索引");
      }
      case "call":
        return this.call(nd.name, nd.args, ctx);
      default:
        throw new Error("未知 AST 节点");
    }
  }

  private binary(op: BinOp, l: Node, r: Node, ctx: EvalCtx): Val {
    if (op === "&&" || op === "||") {
      const a = truthy(this.eval(l, ctx));
      if (op === "&&" && !a) return bool(false);
      if (op === "||" && a) return bool(true);
      return bool(truthy(this.eval(r, ctx)));
    }
    return this.binaryVal(op, this.eval(l, ctx), this.eval(r, ctx));
  }

  binaryVal(op: string, A: Val, B: Val): Val {
    if (A.k === VK.Mat || B.k === VK.Mat) {
      if (op === "*" || op === "@") {
        /* 矩阵作用在向量上按线性映射处理（MATLAB 约定），结果仍是向量 */
        if (A.k === VK.Mat && B.k === VK.Vec) return vecVal(mulMat(A.m!, B.v!.map((x) => [x])).map((r) => r[0]));
        if (A.k === VK.Vec && B.k === VK.Mat) return vecVal(mulMat([A.v!], B.m!)[0]);
        return matVal(mulMat(asMat(A, "矩阵乘法"), asMat(B, "矩阵乘法")));
      }
      if (op === "+" || op === "-") {
        const other = A.k === VK.Mat ? B : A;
        if (other.k === VK.Num && other.im === 0) {
          const k = other.re;
          const M = asMat(A.k === VK.Mat ? A : B, "矩阵运算");
          const flip = A.k !== VK.Mat;
          return matVal(
            M.map((r) => r.map((x) => (flip && op === "-" ? k - x : op === "+" ? x + k : x - k))),
          );
        }
        const P = asMat(A, "矩阵加减"), Q = asMat(B, "矩阵加减");
        if (P.length !== Q.length || P.some((r, i) => r.length !== Q[i].length))
          throw new Error(`矩阵尺寸不一致：${P.length}×${P[0].length} 与 ${Q.length}×${Q[0].length}`);
        return matVal(P.map((r, i) => r.map((x, j) => (op === "+" ? x + Q[i][j] : x - Q[i][j]))));
      }
      if (op === "\\") {
        const s = solveLin(asMat(A, "\\ 的系数矩阵"), asVec(B, "\\ 的右端向量"));
        return s ? vecVal(s) : num(nan);
      }
      if (op === "^") {
        let M = asMat(A, "矩阵幂");
        const k = Math.round(asReal(B, "矩阵幂"));
        if (k < 0) {
          const q = inv(M);
          if (!q) return num(nan);
          M = q;
        }
        const n = M.length;
        let out: number[][] = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
        for (let t = 0; t < Math.abs(k); t++) out = mulMat(out, M);
        return matVal(out);
      }
    }
    if ((A.k === VK.Vec || B.k === VK.Vec) && op !== "^") {
      const u = asVec(A, "向量运算"), v = asVec(B, "向量运算");
      const n = Math.max(u.length, v.length);
      const at = (x: number[], i: number) => x[i] ?? 0;
      switch (op) {
        case "+":
          return vecVal(Array.from({ length: n }, (_, i) => at(u, i) + at(v, i)));
        case "-":
          return vecVal(Array.from({ length: n }, (_, i) => at(u, i) - at(v, i)));
        case "*": {
          if (A.k === VK.Vec && B.k === VK.Vec) {
            let s = 0;
            for (let i = 0; i < n; i++) s += at(u, i) * at(v, i);
            return num(s);
          }
          const w = A.k === VK.Vec ? u : v;
          const k = asReal(A.k === VK.Vec ? B : A, "数乘");
          return vecVal(w.map((x) => x * k));
        }
        case "/":
          if (A.k === VK.Vec && B.k === VK.Num) {
            const d = asReal(B);
            return vecVal(u.map((x) => x / d));
          }
          return vecVal(Array.from({ length: n }, (_, i) => at(u, i) / at(v, i)));
        default:
          throw new Error(`向量不支持运算符 ${op}`);
      }
    }
    if (A.k === VK.Num && B.k === VK.Num) {
      const ar = A.re, ai = A.im, br = B.re, bi = B.im;
      switch (op) {
        case "+":
          return num(ar + br, ai + bi);
        case "-":
          return num(ar - br, ai - bi);
        case "*":
          return num(ar * br - ai * bi, ar * bi + ai * br);
        case "/": {
          const o = CN.cdiv(ar, ai, br, bi, SC);
          return num(o.re, o.im);
        }
        case "%":
          return num(((ar % br) + br) % br);
        case "^": {
          const o = CN.cpow(ar, ai, br, bi, SC);
          return num(o.re, o.im);
        }
        case "<":
          return bool(ar < br);
        case "<=":
          return bool(ar <= br);
        case ">":
          return bool(ar > br);
        case ">=":
          return bool(ar >= br);
        case "==":
          return bool(ar === br && ai === bi);
        case "!=":
          return bool(ar !== br || ai !== bi);
        default:
          throw new Error(`不支持的运算符 ${op}`);
      }
    }
    if (A.k === VK.Bool || B.k === VK.Bool) {
      const a = truthy(A), b = truthy(B);
      switch (op) {
        case "==":
          return bool(a === b);
        case "!=":
          return bool(a !== b);
        case "&&":
          return bool(a && b);
        case "||":
          return bool(a || b);
        default:
          break;
      }
    }
    if (A.k === VK.Str || B.k === VK.Str) {
      if (op === "+") return strVal(String(A.s ?? A.re) + String(B.s ?? B.re));
      if (op === "==") return bool(String(A.s ?? A.re) === String(B.s ?? B.re));
    }
    const ar = asReal(A, "左操作数"), br = asReal(B, "右操作数");
    switch (op) {
      case "+":
        return num(ar + br);
      case "-":
        return num(ar - br);
      case "*":
        return num(ar * br);
      case "/":
        return num(ar / br);
      case "^":
        return num(CN.realPow(ar, br));
      case "%":
        return num(((ar % br) + br) % br);
      case "<":
        return bool(ar < br);
      case "<=":
        return bool(ar <= br);
      case ">":
        return bool(ar > br);
      case ">=":
        return bool(ar >= br);
      case "==":
        return bool(ar === br);
      case "!=":
        return bool(ar !== br);
      default:
        throw new Error(`不支持的运算符 ${op}`);
    }
  }

  call(name: string, args: Node[], ctx: EvalCtx): Val {
    const lz = this.lazy.get(name);
    if (lz) return lz(args, ctx);
    const fv = this.fns.get(name);
    if (fv) {
      if (ctx.depth > 128) throw new Error(`递归过深：${name}`);
      const vars = new Map<string, Val>();
      for (let i = 0; i < fv.params.length; i++) vars.set(fv.params[i], this.eval(args[i], ctx));
      return this.eval(fv.body, { eng: this, scope: { vars }, depth: ctx.depth + 1 });
    }
    const st = STRICT[name];
    if (st) return st(args.map((a) => this.eval(a, ctx)), ctx);
    const bound = this.lookup(name, ctx.scope);
    if (bound && bound.k === VK.Fun) {
      const vars = new Map<string, Val>();
      for (let i = 0; i < bound.fn!.params.length; i++) vars.set(bound.fn!.params[i], this.eval(args[i], ctx));
      return this.eval(bound.fn!.body, { eng: this, scope: { vars }, depth: ctx.depth + 1 });
    }
    // 名称是普通数值却带括号：按隐式乘法解释，例如已定义 a=2 后写 a(x+1)
    if (bound && (bound.k === VK.Num || bound.k === VK.Vec)) {
      const rhs = this.eval(args[0], ctx);
      return this.binaryVal("*", bound, rhs);
    }
    throw new Error(BUILTIN_NAMES.has(name) ? `函数 “${name}” 参数不足` : `未知函数 “${name}”`);
  }

  callVal(f: FunValue, args: Val[], scope?: Scope): Val {
    const vars = new Map<string, Val>();
    for (let i = 0; i < f.params.length; i++) vars.set(f.params[i], args[i] ?? ZERO);
    return this.eval(f.body, { eng: this, scope: { vars, up: scope }, depth: 0 });
  }
}

/* ------------------------------------------------------------------ */
/* 惰性形式：数值分析、求和、列表生成                                   */
/* ------------------------------------------------------------------ */

export const SLOT_NAMES = ["x", "y", "z", "t", "u", "v", "w", "s"];
/** 自动挑选自变量时的优先顺序（由 UI 按当前场景注入） */
let CONTEXT_VARS = ["x", "y", "z", "t", "u", "v", "theta", "r"];
export function setContextVars(v: string[]): void {
  CONTEXT_VARS = v;
}

export function autoParams(node: Node, expect: number, ctx: EvalCtx): string[] {
  const eng = ctx.eng;
  const names = [...freeNames(node)].filter(
    (n) => !CONSTANTS[n] && !eng.globals.has(n) && !eng.fns.has(n),
  );
  const params: string[] = [];
  for (const cand of CONTEXT_VARS) {
    if (params.length >= expect) break;
    if (names.includes(cand) && !params.includes(cand)) params.push(cand);
  }
  for (const n of names) {
    if (params.length >= expect) break;
    if (!params.includes(n)) params.push(n);
  }
  while (params.length < expect) {
    const s = SLOT_NAMES[params.length];
    if (params.includes(s)) params.push("_" + params.length);
    else params.push(s);
  }
  return params;
}

/** 把参数（函数值 / 表达式）转成 FunValue */
export function toFun(a: Node, ctx: EvalCtx, expectParams: number, hint = "f"): FunValue {
  const eng = ctx.eng;
  if (a.type === "ident") {
    const f = eng.fns.get(a.name);
    if (f) return f;
    const v = eng.lookup(a.name, ctx.scope);
    if (v && v.k === VK.Fun) return v.fn!;
  }
  const val = a.type === "call" ? eng.lookup(a.name, ctx.scope) : null;
  if (val && val.k === VK.Fun) return val.fn!;
  // 裸的内置函数名：sin / cos / tan … 要包装成 λx. sin(x)，否则会退化成恒等函数
  if (a.type === "ident" && BUILTIN_NAMES.has(a.name) && !eng.fns.has(a.name) && !eng.globals.has(a.name)) {
    const params = SLOT_NAMES.slice(0, Math.max(1, expectParams));
    return {
      name: a.name,
      params,
      body: { type: "call", name: a.name, args: params.map((p) => ({ type: "ident", name: p } as Node)) },
    };
  }
  if (a.type === "ident" && !eng.fns.has(a.name) && BUILTIN_NAMES.has(a.name)) {
    return { name: a.name, params: ["x"], body: { type: "call", name: a.name, args: [{ type: "ident", name: "x" }] } };
  }
  return { name: hint, params: autoParams(a, expectParams, ctx), body: a };
}

function realAt(eng: Engine, f: FunValue, args: number[], scope?: Scope): number {
  return asReal(eng.callVal(f, args.map((a) => num(a)), scope), `${f.name} 的返回值`);
}

function comb(n: number, k: number): number {
  let r = 1;
  for (let i = 0; i < k; i++) r = (r * (n - i)) / (i + 1);
  return r;
}

/** 求和/列表生成的循环变量：必须是裸标识符（i 会被词法解析成虚数单位） */
function loopVar(nd: Node, kind: string): string {
  if (nd.type !== "ident") throw new Error(`${kind} 的第一个参数需要循环变量名（如 k、n），不能用 i —— 它是虚数单位`);
  return nd.name;
}

function registerLazy(eng: Engine): void {
  const L = (name: string, f: LazyFn) => {
    eng.lazy.set(name, f);
  };

  L("if", (a, ctx) =>
    truthy(ctx.eng.eval(a[0], ctx))
      ? ctx.eng.eval(a[1], ctx)
      : a[2]
        ? ctx.eng.eval(a[2], ctx)
        : num(nan),
  );
  L("when", (a, ctx) =>
    truthy(ctx.eng.eval(a[0], ctx))
      ? ctx.eng.eval(a[1], ctx)
      : a[2]
        ? ctx.eng.eval(a[2], ctx)
        : num(nan),
  );
  L("piecewise", (a, ctx) => {
    for (let i = 0; i + 1 < a.length; i += 2) if (truthy(ctx.eng.eval(a[i], ctx))) return ctx.eng.eval(a[i + 1], ctx);
    return num(nan);
  });

  /** diff(f, x) / diff(f, x, n)：复步法优先，退化时用中心差分 */
  L("diff", (a, ctx) => {
    const eng = ctx.eng;
    const f = toFun(a[0], ctx, 1);
    const x = asReal(eng.eval(a[1], ctx), "diff 的自变量");
    const order = a[2] ? Math.round(asReal(eng.eval(a[2], ctx))) : 1;
    if (order === 1) {
      const h = Math.max(1e-25, Math.abs(x) * 1e-18);
      const v = eng.callVal(f, [num(x, h)], ctx.scope);
      if (v.k === VK.Num && Number.isFinite(v.im)) {
        const d = v.im / h;
        if (Number.isFinite(d)) return num(d);
      }
      if (v.k === VK.Vec) {
        const dd = v.v!.map((q) => q / h);
        if (dd.every(Number.isFinite)) return vecVal(dd);
      }
      const hh = 1.4901161193847656e-8 * Math.max(1, Math.abs(x));
      const up = realAt(eng, f, [x + hh], ctx.scope);
      const dn = realAt(eng, f, [x - hh], ctx.scope);
      return num((up - dn) / (2 * hh));
    }
    const h = Math.pow(1e-4, 1 / (order + 1)) * Math.max(1, Math.abs(x));
    let s = 0;
    for (let k = 0; k <= order; k++) {
      s += (k % 2 ? -1 : 1) * comb(order, k) * realAt(eng, f, [x + (k - order / 2) * h], ctx.scope);
    }
    return num(s / Math.pow(h, order));
  });
  L("derivative", (a, ctx) => {
    const eng = ctx.eng;
    const f = toFun(a[0], ctx, 1);
    const order = a[1] ? Math.round(asReal(eng.eval(a[1], ctx))) : 1;
    const body: Node = {
      type: "call",
      name: "diff",
      args: [
        a[0],
        { type: "ident", name: f.params[0] ?? "x" },
        order > 1 ? { type: "num", value: order } : { type: "num", value: 1 },
      ],
    };
    return funVal({ name: `${f.name}′`, params: [f.params[0] ?? "x"], body });
  });
  L("limit", (a, ctx) => {
    const eng = ctx.eng;
    const f = toFun(a[0], ctx, 1);
    const p = asReal(eng.eval(a[1], ctx));
    const v = eng.callVal(f, [num(p)], ctx.scope);
    const usable = (q: number) => Number.isFinite(q) || q === Infinity || q === -Infinity;
    if (v.k === VK.Vec) return v;
    const [vr, vi] = reIm(v);
    if (usable(vr) && (vi === 0 || usable(vi)) && !(Number.isNaN(vr) && Number.isNaN(vi))) return v;
    // 侧向逼近：取第一个收敛的 ε
    for (const eps of [1e-4, 1e-6, 1e-8, 1e-10]) {
      try {
        const L1 = realAt(eng, f, [p + eps], ctx.scope);
        const L0 = realAt(eng, f, [p - eps], ctx.scope);
        if (usable(L1) && usable(L0) && Math.abs(L1 - L0) < 1e-3 * Math.max(1, Math.abs(L1))) return num((L1 + L0) / 2);
      } catch {
        /* 继续尝试下一个 ε */
      }
    }
    const L1 = realAt(eng, f, [p + 1e-7], ctx.scope);
    const L0 = realAt(eng, f, [p - 1e-7], ctx.scope);
    return num((L1 + L0) / 2);
  });

  const integrate: LazyFn = (a, ctx) => {
    const eng = ctx.eng;
    const f = toFun(a[0], ctx, 1);
    const lo = asReal(eng.eval(a[1], ctx), "积分下限");
    const hi = asReal(eng.eval(a[2], ctx), "积分上限");
    const tol = a[3] ? asReal(eng.eval(a[3], ctx)) : 1e-9;
    const g = (x: number) => realAt(eng, f, [x], ctx.scope);
    return num(simpson(g, lo, hi, tol));
  };
  L("integrate", integrate);
  L("quad", integrate);

  L("fzero", (a, ctx) => {
    const eng = ctx.eng;
    const f = toFun(a[0], ctx, 1);
    const second = eng.eval(a[1], ctx);
    if (second.k === VK.Vec && second.v!.length === 2) {
      return num(bisect(eng, f, second.v![0], second.v![1], ctx.scope));
    }
    return num(newton(eng, f, asReal(second, "fzero 初值"), ctx.scope));
  });
  L("roots", (a, ctx) => {
    const eng = ctx.eng;
    const f = toFun(a[0], ctx, 1);
    const seed = a[1] ? reIm(eng.eval(a[1], ctx)) : [1, 1];
    const r = newtonC(eng, f, seed[0], seed[1], ctx.scope);
    return vecVal(r);
  });

  /** 复平面保角映射的像曲线：map(z -> f(z)) 由绘图层调用，这里提供采样工具 */
  L("mapAt", (a, ctx) => {
    const eng = ctx.eng;
    const f = toFun(a[0], ctx, 1);
    const z = a[1] ? reIm(eng.eval(a[1], ctx)) : [0, 0];
    const o = eng.callVal(f, [num(z[0], z[1])], ctx.scope);
    const [r, i] = reIm(o);
    return vecVal([r, i]);
  });

  /** sum / product：sum(i, from, to, expr) 或 sum(向量) */
  const agg = (kind: "+" | "*"): LazyFn => (a, ctx) => {
    const eng = ctx.eng;
    if (a.length === 1) {
      const v = eng.eval(a[0], ctx);
      const xs = asVec(v, kind === "+" ? "sum" : "product");
      let acc = kind === "+" ? 0 : 1;
      for (const x of xs) acc = kind === "+" ? acc + x : acc * x;
      return num(acc);
    }
    const varName = loopVar(a[0], kind === "+" ? "sum" : "product");
    const from = asReal(eng.eval(a[1], ctx));
    const to = asReal(eng.eval(a[2], ctx));
    const vars = new Map<string, Val>();
    const sctx: EvalCtx = { eng, scope: { vars, up: ctx.scope }, depth: ctx.depth + 1 };
    let acc = kind === "+" ? 0 : 1;
    const step = to >= from ? 1 : -1;
    for (let k = from; step > 0 ? k <= to : k >= to; k += step) {
      vars.set(varName, num(k));
      const v = asReal(eng.eval(a[3], sctx), kind === "+" ? "sum 通项" : "product 通项");
      acc = kind === "+" ? acc + v : acc * v;
    }
    return num(acc);
  };
  L("sum", agg("+"));
  L("product", agg("*"));
  L("prod", agg("*"));

  /** 列表生成：seq(i, from, to, expr) / seq(i, from, to, step, expr) */
  L("seq", (a, ctx) => {
    const eng = ctx.eng;
    const varName = loopVar(a[0], "seq");
    const from = asReal(eng.eval(a[1], ctx));
    const to = asReal(eng.eval(a[2], ctx));
    const hasStep = a.length >= 5;
    const step = hasStep ? asReal(eng.eval(a[3], ctx)) : 1;
    const body = a[a.length - 1];
    const vars = new Map<string, Val>();
    const sctx: EvalCtx = { eng, scope: { vars, up: ctx.scope }, depth: ctx.depth + 1 };
    const out: Val[] = [];
    const sgn = step >= 0 ? 1 : -1;
    for (let k = from; sgn > 0 ? k <= to + 1e-12 : k >= to - 1e-12; k += sgn * Math.abs(step)) {
      vars.set(varName, num(k));
      out.push(eng.eval(body, sctx));
    }
    return listVal(out);
  });
  L("list", (a, ctx) => listVal(a.map((x) => ctx.eng.eval(x, ctx))));
  L("vec", (a, ctx) => vecVal(a.map((x) => asReal(ctx.eng.eval(x, ctx), "vec 参数"))));
  L("mat", (a, ctx) => {
    const rows = a.map((r) => asVec(ctx.eng.eval(r, ctx), "mat 行"));
    return matVal(rows);
  });
  const rng = (a: Val[]): Val => {
    const x0 = asReal(a[0], "range 起点");
    const x1 = asReal(a[1], "range 终点");
    const st = a[2] ? asReal(a[2]) : a[1] && a[0] && x1 === x0 ? 1 : 1;
    const out: number[] = [];
    if (st === 0) return vecVal([x0]);
    const sgn = st > 0 ? 1 : -1;
    for (let v = x0; sgn > 0 ? v <= x1 + Math.abs(st) * 1e-9 : v >= x1 - Math.abs(st) * 1e-9; v += st) out.push(v);
    return vecVal(out);
  };
  L("range", (a, ctx) => rng(a.map((x) => ctx.eng.eval(x, ctx))));
  L("linspace", (a, ctx) => {
    const eng = ctx.eng;
    const x0 = asReal(eng.eval(a[0], ctx));
    const x1 = asReal(eng.eval(a[1], ctx));
    const n = Math.max(2, Math.round(a[2] ? asReal(eng.eval(a[2], ctx)) : 100));
    return vecVal(Array.from({ length: n }, (_, i) => x0 + ((x1 - x0) * i) / (n - 1)));
  });

  L("polyfit", (a, ctx) => {
    const eng = ctx.eng;
    const xs = asVec(eng.eval(a[0], ctx), "polyfit 的 x");
    const ys = asVec(eng.eval(a[1], ctx), "polyfit 的 y");
    const deg = Math.round(asReal(eng.eval(a[2], ctx), "polyfit 阶数"));
    return vecVal(polyfit(xs, ys, deg));
  });
  L("interp1", (a, ctx) => {
    const eng = ctx.eng;
    const xs = asVec(eng.eval(a[0], ctx), "interp1 的 x");
    const ys = asVec(eng.eval(a[1], ctx), "interp1 的 y");
    const x = asReal(eng.eval(a[2], ctx));
    return num(interp1(xs, ys, x));
  });

  L("div", (a, ctx) => differentialOp(ctx, a, "div"));
  L("curl", (a, ctx) => differentialOp(ctx, a, "curl"));
  L("grad", (a, ctx) => differentialOp(ctx, a, "grad"));

  /** 数值优化（一维黄金分割），供几何最值问题使用 */
  L("minimize", (a, ctx) => {
    const eng = ctx.eng;
    const f = toFun(a[0], ctx, 1);
    const lo = asReal(eng.eval(a[1], ctx));
    const hi = asReal(eng.eval(a[2], ctx));
    return num(goldenMin((x) => realAt(eng, f, [x], ctx.scope), lo, hi));
  });
  L("label", (a, ctx) => strVal(CN.fmt(asReal(ctx.eng.eval(a[0], ctx), "label"))));
  L("text", (a, ctx) => strVal(a.map((x) => CN.fmt(asReal(ctx.eng.eval(x, ctx), "text"))).join(", ")));
}

/** 梯度 / 散度 / 旋度（中心差分） */
function differentialOp(ctx: EvalCtx, a: Node[], kind: "div" | "curl" | "grad"): Val {
  const eng = ctx.eng;
  const pt: number[] = [];
  for (let i = 1; i < a.length; i++) pt.push(asReal(eng.eval(a[i], ctx), "坐标"));
  if (!pt.length) pt.push(0, 0);
  const h = 1e-5 * Math.max(1, ...pt.map(Math.abs));
  const n = pt.length;
  if (kind === "grad") {
    const f = toFun(a[0], ctx, Math.max(1, n));
    const g = (p: number[]) => reIm(eng.callVal(f, p.map((x) => num(x)), ctx.scope))[0];
    const out: number[] = [];
    for (let i = 0; i < n; i++) {
      const pp = pt.slice(), pm = pt.slice();
      pp[i] += h;
      pm[i] -= h;
      out.push((g(pp) - g(pm)) / (2 * h));
    }
    return vecVal(out);
  }
  const f = toFun(a[0], ctx, Math.max(1, n));
  const F = (p: number[]) => asVec(eng.callVal(f, p.map((x) => num(x)), ctx.scope));
  const J: number[][] = [];
  for (let i = 0; i < n; i++) {
    const pp = pt.slice(), pm = pt.slice();
    pp[i] += h;
    pm[i] -= h;
    const fp = F(pp), fm = F(pm);
    J.push(fp.map((q, k) => (q - (fm[k] ?? 0)) / (2 * h)));
  }
  // J[i][k] = ∂F_k/∂x_i
  if (kind === "div") {
    let s = 0;
    for (let i = 0; i < Math.min(n, J[0] ? J[0].length : 0); i++) s += J[i][i];
    return num(s);
  }
  if (n === 2) {
    // 2D 旋度的 z 分量：∂F₂/∂x − ∂F₁/∂y
    return num((J[0]?.[1] ?? 0) - (J[1]?.[0] ?? 0));
  }
  const c = (i: number, j: number) => (J[i] && J[i][j] !== undefined ? J[i][j] : 0);
  return vecVal([c(1, 2) - c(2, 1), c(2, 0) - c(0, 2), c(0, 1) - c(1, 0)]);
}

function goldenMin(g: (x: number) => number, lo: number, hi: number): number {
  const inv = (Math.sqrt(5) - 1) / 2;
  let a = lo, b = hi;
  let c = b - inv * (b - a), d = a + inv * (b - a);
  let fc = g(c), fd = g(d);
  for (let k = 0; k < 100 && Math.abs(b - a) > 1e-10; k++) {
    if (fc < fd) {
      b = d;
      d = c;
      fd = fc;
      c = b - inv * (b - a);
      fc = g(c);
    } else {
      a = c;
      c = d;
      fc = fd;
      d = a + inv * (b - a);
      fd = g(d);
    }
  }
  return g((a + b) / 2);
}

/* ------------------------------------------------------------------ */
/* 数值分析工具（也供绘图层直接调用）                                  */
/* ------------------------------------------------------------------ */

export function simpson(g: (x: number) => number, a: number, b: number, tol = 1e-9): number {
  const fa = g(a), fb = g(b), fm = g((a + b) / 2);
  const whole = ((b - a) / 6) * (fa + 4 * fm + fb);
  const rec = (
    l: number,
    r: number,
    fl: number,
    fr: number,
    fmid: number,
    wholeV: number,
    e: number,
    depth: number,
  ): number => {
    const m = (l + r) / 2;
    const flm = g((l + m) / 2), frm = g((m + r) / 2);
    const L = ((m - l) / 6) * (fl + 4 * flm + fmid);
    const Rr = ((r - m) / 6) * (fmid + 4 * frm + fr);
    const s2 = L + Rr;
    if (depth >= 26 || Math.abs(s2 - wholeV) <= 15 * e) return s2 + (s2 - wholeV) / 15;
    return rec(l, m, fl, fmid, flm, L, e / 2, depth + 1) + rec(m, r, fmid, fr, frm, Rr, e / 2, depth + 1);
  };
  return rec(a, b, fa, fb, fm, whole, tol, 0);
}

export function bisect(eng: Engine, f: FunValue, lo: number, hi: number, scope?: Scope): number {
  let a = lo, b = hi;
  let fa = realAt(eng, f, [a], scope), fb = realAt(eng, f, [b], scope);
  if (!Number.isFinite(fa) || !Number.isFinite(fb)) return NaN;
  if (fa === 0) return a;
  if (fb === 0) return b;
  if (fa * fb > 0) {
    let found = false;
    for (let k = 1; k <= 40 && !found; k++) {
      for (const nb of [b + (b - a) * 0.2 * k, a - (b - a) * 0.2 * k]) {
        const fnb = realAt(eng, f, [nb], scope);
        if (!Number.isFinite(fnb)) continue;
        if (fa * fnb <= 0) {
          b = nb;
          fb = fnb;
          found = true;
          break;
        }
        if (fnb * fb <= 0) {
          a = nb;
          fa = fnb;
          found = true;
          break;
        }
      }
    }
    if (!found) return NaN;
  }
  for (let k = 0; k < 200; k++) {
    const m = (a + b) / 2;
    const fm = realAt(eng, f, [m], scope);
    if (fm === 0 || Math.abs(b - a) < 1e-14 * Math.max(1, Math.abs(m))) return m;
    if (fa * fm < 0) {
      b = m;
      fb = fm;
    } else {
      a = m;
      fa = fm;
    }
  }
  return (a + b) / 2;
}

export function newton(eng: Engine, f: FunValue, x0: number, scope?: Scope): number {
  let x = x0;
  let best = x, bestAbs = Infinity;
  for (let k = 0; k < 100; k++) {
    const fx = realAt(eng, f, [x], scope);
    if (!Number.isFinite(fx)) {
      x = x * 1.31 + 0.07;
      continue;
    }
    if (Math.abs(fx) < bestAbs) {
      bestAbs = Math.abs(fx);
      best = x;
    }
    if (bestAbs < 1e-14) return best;
    const h = 1e-6 * Math.max(1, Math.abs(x));
    const d = (realAt(eng, f, [x + h], scope) - fx) / h;
    if (d === 0 || !Number.isFinite(d)) break;
    const nx = x - fx / d;
    if (!Number.isFinite(nx)) break;
    if (Math.abs(nx - x) < 1e-13 * Math.max(1, Math.abs(nx))) return nx;
    x = nx;
  }
  return best;
}

/** 复牛顿法：返回根与收敛步数（Newton 分形用） */
export function newtonC(
  eng: Engine,
  f: FunValue,
  zr: number,
  zi: number,
  scope?: Scope,
  maxIter = 60,
): [number, number, number] {
  let re = zr, im = zi;
  const z = [num(0, 0)];
  for (let k = 0; k < maxIter; k++) {
    z[0] = num(re, im);
    const fv = eng.callVal(f, z, scope);
    const [fr, fi] = reIm(fv);
    if (!Number.isFinite(fr) || !Number.isFinite(fi)) return [NaN, NaN, k];
    if (Math.hypot(fr, fi) < 1e-11) return [re, im, k];
    z[0] = num(re + 1e-7, im);
    const d1 = reIm(eng.callVal(f, z, scope));
    const dr = (d1[0] - fr) / 1e-7;
    const di = (d1[1] - fi) / 1e-7;
    const den = dr * dr + di * di;
    if (den === 0 || !Number.isFinite(den)) break;
    const qr = (fr * dr + fi * di) / den;
    const qi = (fi * dr - fr * di) / den;
    const nr = re - qr, ni = im - qi;
    if (!Number.isFinite(nr) || !Number.isFinite(ni)) break;
    if (Math.hypot(nr - re, ni - im) < 1e-12) return [nr, ni, k];
    re = nr;
    im = ni;
  }
  return [re, im, maxIter];
}

export function polyfit(xs: number[], ys: number[], deg: number): number[] {
  const n = deg + 1;
  const A: number[][] = Array.from({ length: n }, () => new Array(n).fill(0));
  const b = new Array(n).fill(0);
  for (let i = 0; i < Math.min(xs.length, ys.length); i++) {
    const pw = new Array(n);
    let p = 1;
    for (let k = 0; k < n; k++) {
      pw[k] = p;
      p *= xs[i];
    }
    for (let r = 0; r < n; r++) {
      b[r] += ys[i] * pw[r];
      for (let c = 0; c < n; c++) A[r][c] += pw[r] * pw[c];
    }
  }
  const sol = solveLin(A, b);
  if (!sol) return new Array(n).fill(nan);
  return sol.slice().reverse();
}

export function interp1(xs: number[], ys: number[], x: number): number {
  const order = xs.map((v, i) => [v, i] as const).sort((p, q) => p[0] - q[0]);
  if (!order.length) return NaN;
  if (x <= order[0][0]) return ys[order[0][1]];
  if (x >= order[order.length - 1][0]) return ys[order[order.length - 1][1]];
  for (let k = 1; k < order.length; k++) {
    if (x <= order[k][0]) {
      const [x0, i0] = order[k - 1];
      const [x1, i1] = order[k];
      const t = (x - x0) / (x1 - x0 || 1);
      return ys[i0] * (1 - t) + ys[i1] * t;
    }
  }
  return NaN;
}

/* ------------------------------------------------------------------ */
/* 编译：通用槽位函数 + 实数快路径                                      */
/* ------------------------------------------------------------------ */

export interface Compiled {
  names: string[];
  /** 参数以 (re, im) 成对放入 slots；返回共享的 Val */
  run: (slots: Float64Array) => Val;
  arity: number;
  real: ((slots: Float64Array) => number) | null;
}

export function compile(eng: Engine, node: Node, names: string[]): Compiled {
  const idx = new Map<string, number>();
  names.forEach((n, i) => idx.set(n, i));
  const cells: Val[] = names.map(() => num(0, 0));
  const vars = new Map<string, Val>();
  names.forEach((n, i) => vars.set(n, cells[i]));
  const scope: Scope = { vars, up: undefined };
  const ctx: EvalCtx = { eng, scope, depth: 0 };
  const run = (s: Float64Array): Val => {
    for (let i = 0; i < names.length; i++) {
      cells[i].re = s[i * 2];
      cells[i].im = s[i * 2 + 1];
    }
    return eng.eval(node, ctx);
  };
  return { names, run, arity: names.length, real: compileReal(eng, node, names) };
}

/** 表达式中出现的自变量名（按优先级排序），用于 UI 自动补全参数 */
export function suggestParams(node: Node, eng: Engine, take: number): string[] {
  // 已经是全局量（参数滑块、控制台里定义的常量）的名字不能占自变量的槽位，
  // 否则求值时会被视口坐标悄悄覆盖，滑块就再也拧不动了
  const names = [...freeNames(node)].filter(
    (n) => !CONSTANTS[n] && !eng.fns.has(n) && !eng.globals.has(n),
  );
  const out: string[] = [];
  for (const cand of CONTEXT_VARS) if (names.includes(cand) && !out.includes(cand)) out.push(cand);
  for (const n of names) if (!out.includes(n)) out.push(n);
  while (out.length < take) out.push(SLOT_NAMES[out.length]);
  return out.slice(0, Math.max(take, 0));
}

type RealFn = (s: Float64Array) => number;

/** 静态推断为实数的表达式 → 生成无分配的快路径 */
export function compileReal(eng: Engine, node: Node, names: string[]): RealFn | null {
  if (!isRealStatic(node, names, eng)) return null;
  const idx = new Map<string, number>();
  names.forEach((n, i) => idx.set(n, i));
  const build = (n: Node): RealFn | null => {
    switch (n.type) {
      case "num": {
        const v = n.value;
        return () => v;
      }
      case "ident": {
        const i = idx.get(n.name);
        if (i !== undefined) return (s) => s[i * 2];
        const g = eng.globals.get(n.name);
        if (g && g.k === VK.Num) return () => g.re;
        const c = CONSTANTS[n.name];
        if (c !== undefined) return () => c;
        return null;
      }
      case "un": {
        const e = build(n.e);
        if (!e) return null;
        return n.op === "-" ? (s) => -e(s) : e;
      }
      case "post": {
        const e = build(n.e);
        if (!e) return null;
        return n.op === "!" ? (s) => CN.factorial(e(s)) : (s) => (e(s) * PI) / 180;
      }
      case "cond": {
        const c = build(n.c), a = build(n.a), b = build(n.b);
        return c && a && b ? (s) => (c(s) !== 0 ? a(s) : b(s)) : null;
      }
      case "bin": {
        const l = build(n.l), r = build(n.r);
        if (!l || !r) return null;
        switch (n.op) {
          case "+":
            return (s) => l(s) + r(s);
          case "-":
            return (s) => l(s) - r(s);
          case "*":
            return (s) => l(s) * r(s);
          case "/":
            return (s) => l(s) / r(s);
          case "%":
            return (s) => {
              const b = r(s);
              const a = l(s);
              return ((a % b) + b) % b;
            };
          case "^": {
            const e = n.r.type === "num" ? n.r.value : null;
            if (e === 2) return (s) => { const a = l(s); return a * a; };
            if (e === 3) return (s) => { const a = l(s); return a * a * a; };
            if (e === 0.5) return (s) => Math.sqrt(l(s));
            if (e === -1) return (s) => 1 / l(s);
            if (e !== null && Number.isInteger(e)) {
              const k = e;
              return (s) => {
                let a = l(s), out = 1;
                if (k < 0) { a = 1 / a; }
                for (let t = 0; t < Math.abs(k); t++) out *= a;
                return out;
              };
            }
            return (s) => CN.realPow(l(s), r(s));
          }
          case "<":
            return (s) => (l(s) < r(s) ? 1 : 0);
          case "<=":
            return (s) => (l(s) <= r(s) ? 1 : 0);
          case ">":
            return (s) => (l(s) > r(s) ? 1 : 0);
          case ">=":
            return (s) => (l(s) >= r(s) ? 1 : 0);
          case "==":
            return (s) => (l(s) === r(s) ? 1 : 0);
          case "!=":
            return (s) => (l(s) !== r(s) ? 1 : 0);
          case "&&":
            return (s) => (l(s) !== 0 && r(s) !== 0 ? 1 : 0);
          case "||":
            return (s) => (l(s) !== 0 || r(s) !== 0 ? 1 : 0);
          default:
            return null;
        }
      }
      case "call": {
        if (n.name === "diff") {
          const target = n.args[0];
          const varNode = n.args[1];
          if (!varNode || varNode.type !== "ident") return null;
          const slot = idx.get(varNode.name);
          if (slot === undefined) return null;
          const fn = build(target);
          if (!fn) return null;
          return (s) => {
            const x = s[slot * 2];
            const h = 1.4901161193847656e-8 * Math.max(1, Math.abs(x));
            const p = s[slot * 2];
            s[slot * 2] = p + h;
            const up = fn(s);
            s[slot * 2] = p - h;
            const dn = fn(s);
            s[slot * 2] = p;
            return (up - dn) / (2 * h);
          };
        }
        const as: RealFn[] = [];
        for (const arg of n.args) {
          const f = build(arg);
          if (!f) return null;
          as.push(f);
        }
        const rf = REAL_FAST[n.name];
        if (rf) {
          if (rf.length === 1 && as.length === 1) {
            const a0 = as[0];
            return (s) => rf(a0(s));
          }
          return (s) => rf(...as.map((f) => f(s)));
        }
        const uf = eng.fns.get(n.name);
        if (uf && uf.params.length === as.length) {
          const inlined = inlineCall(eng, uf, as, idx);
          if (inlined) return inlined;
        }
        return null;
      }
      default:
        return null;
    }
  };
  return build(node);
}

function inlineCall(
  eng: Engine,
  f: FunValue,
  args: RealFn[],
  idx: Map<string, number>,
): RealFn | null {
  const sub = new Map<string, RealFn>();
  f.params.forEach((p, i) => sub.set(p, args[i]));
  const walk = (n: Node): RealFn | null => {
    if (n.type === "ident") {
      const own = sub.get(n.name);
      if (own) return own;
      const slot = idx.get(n.name);
      if (slot !== undefined) return (s) => s[slot * 2];
      const g = eng.globals.get(n.name);
      if (g && g.k === VK.Num) return () => g.re;
      const c = CONSTANTS[n.name];
      if (c !== undefined) return () => c;
      return null;
    }
    return buildVia(n);
  };
  const buildVia = (n: Node): RealFn | null => {
    if (n.type === "num") {
      const v = n.value;
      return () => v;
    }
    if (n.type === "un" && n.op === "-") {
      const e = walk(n.e);
      return e ? (s) => -e(s) : null;
    }
    if (n.type === "bin") {
      const l = walk(n.l), r = walk(n.r);
      if (!l || !r) return null;
      switch (n.op) {
        case "+": return (s) => l(s) + r(s);
        case "-": return (s) => l(s) - r(s);
        case "*": return (s) => l(s) * r(s);
        case "/": return (s) => l(s) / r(s);
        case "^": return (s) => CN.realPow(l(s), r(s));
        default: return null;
      }
    }
    if (n.type === "call") {
      const as: RealFn[] = [];
      for (const a of n.args) {
        const f = walk(a);
        if (!f) return null;
        as.push(f);
      }
      const rf = REAL_FAST[n.name];
      if (rf) return (s) => rf(...as.map((f) => f(s)));
      const uf = eng.fns.get(n.name);
      if (uf && uf.params.length === as.length) return inlineCall(eng, uf, as, idx);
      return null;
    }
    return null;
  };
  return walk(f.body);
}

const REAL_FAST: Record<string, (...a: number[]) => number> = {
  sin: (x) => Math.sin(x),
  cos: (x) => Math.cos(x),
  tan: (x) => Math.tan(x),
  sec: (x) => 1 / Math.cos(x),
  csc: (x) => 1 / Math.sin(x),
  cot: (x) => Math.cos(x) / Math.sin(x),
  asin: (x) => Math.asin(x),
  acos: (x) => Math.acos(x),
  atan: (x) => Math.atan(x),
  atan2: (y, x) => Math.atan2(y, x),
  sinh: (x) => Math.sinh(x),
  cosh: (x) => Math.cosh(x),
  tanh: (x) => Math.tanh(x),
  asinh: (x) => Math.asinh(x),
  acosh: (x) => Math.acosh(x),
  atanh: (x) => Math.atanh(x),
  exp: (x) => Math.exp(x),
  exp2: (x) => Math.pow(2, x),
  exp10: (x) => Math.pow(10, x),
  log: (a, b) => (b === undefined ? Math.log(a) : Math.log(b) / Math.log(a)),
  ln: (x) => Math.log(x),
  log2: (x) => Math.log2(x),
  log10: (x) => Math.log10(x),
  lg: (x) => Math.log10(x),
  cbrt: (x) => Math.cbrt(x),
  sqrt: (x) => Math.sqrt(x),
  abs: (x) => Math.abs(x),
  sign: (x) => Math.sign(x),
  floor: (x) => Math.floor(x),
  ceil: (x) => Math.ceil(x),
  round: (x) => Math.round(x),
  trunc: (x) => Math.trunc(x),
  frac: (x) => x - Math.floor(x),
  mod: (a, b) => ((a % b) + b) % b,
  min: (...a) => Math.min(...a),
  max: (...a) => Math.max(...a),
  clamp: (x, a, b) => Math.min(Math.max(x, a), b),
  hypot: (...a) => Math.hypot(...a),
  step: (x) => (x < 0 ? 0 : 1),
  sinc: (x) => (x === 0 ? 1 : Math.sin(x) / x),
  sinpi: (x) => Math.sin(PI * x),
  cospi: (x) => Math.cos(PI * x),
  gamma: (x) => CN.gamma(x),
  erf: (x) => erf(x),
  re: (x) => x,
  deg: (x) => (x * 180) / PI,
  rad: (x) => (x * PI) / 180,
  factorial: (x) => CN.factorial(x),
  if: (c, a, b) => (c !== 0 ? a : b === undefined ? NaN : b),
};

/**
 * 静态判定表达式是否恒为实数。
 * 任何可能引入虚部、向量或惰性语义的构造都会让判定失败，从而回退到通用路径。
 */
export function isRealStatic(node: Node, names: string[], eng: Engine): boolean {
  const nameSet = new Set(names);
  const UNSAFE_CALLS = new Set([
    "complex", "polar", "vec", "mat", "list", "seq", "range", "linspace", "dot", "cross",
    "unit", "proj", "transpose", "det", "inv", "identity", "eye", "solve", "integrate",
    "quad", "fzero", "roots", "polyfit", "interp1", "sum", "product", "prod", "sort",
    "mean", "median", "var", "std", "rand", "randn", "orthogonal", "div", "curl", "grad",
    "piecewise", "component", "size", "rows", "cols", "label", "text", "mapAt", "minimize",
    "derivative", "primes", "matmul", "abs2", "conj", "phase", "arg", "angle", "im", "imag",
    "norm", "len", "magnitude", "gcd", "lcm", "nCr", "nPr", "binomial", "beta", "isprime",
    "logb", "root", "limit",
  ]);
  const ok = (n: Node): boolean => {
    switch (n.type) {
      case "num":
        return true;
      case "ident":
        if (nameSet.has(n.name)) return true;
        if (CONSTANTS[n.name] !== undefined) return true;
        return eng.globals.get(n.name)?.k === VK.Num;
      case "str":
        return false;
      case "un":
        return n.op !== "!" && ok(n.e);
      case "post":
        return ok(n.e);
      case "idx":
      case "list":
        return false;
      case "cond":
        return ok(n.c) && ok(n.a) && ok(n.b);
      case "bin": {
        if (!ok(n.l) || !ok(n.r)) return false;
        if (n.op === "^") {
          const e = n.r;
          if (e.type !== "num") return false;
          const v = e.value;
          return Number.isInteger(v) || v === 0.5 || v === 2 || v === 3 || v === -1;
        }
        return true;
      }
      case "call": {
        if (UNSAFE_CALLS.has(n.name)) return false;
        // sqrt/log 在实数快路径中按定义域外返回 NaN，正是绘图需要的语义
        if (!n.args.every(ok)) return false;
        const uf = eng.fns.get(n.name);
        if (uf) return isRealStatic(uf.body, [...uf.params, ...names], eng);
        return !!REAL_FAST[n.name];
      }
      default:
        return false;
    }
  };
  return ok(node);
}

/* ------------------------------------------------------------------ */
/* 复数快路径：逐像素着色用的无分配求值                                  */
/* ------------------------------------------------------------------ */

/** 复变求值闭包：把 z = re + im·i 的结果写进出参，全程不新建对象 */
export type CplxFn = (re: number, im: number, o: CN.C) => void;

/**
 * 复变表达式 → 无分配闭包。调用的仍是 cnum 里那批 (re, im, o) 原语，
 * 因此与 AST 路径逐位一致；任何无法等价提升的构造（条件、比较、向量、
 * 用户自定义函数、下标）都返回 null，由调用方回退通用路径。
 */
export function compileCplx(eng: Engine, node: Node, name = "z"): CplxFn | null {
  const konst = (r: number, i: number): CplxFn => {
    return (_re, _im, o) => {
      o.re = r;
      o.im = i;
    };
  };
  // 不含自变量与全局量的子树在编译期算一次；参数滑块一动就重新编译，
  // 所以这里绝不能把 eng.globals 的当前值折进常量，否则缓存会过期。
  const pure = (n: Node): boolean => {
    switch (n.type) {
      case "num":
        return true;
      case "ident":
        return n.name !== name && !eng.globals.has(n.name) && isImmutIdent(n.name);
      case "un":
        return n.op !== "!" && pure(n.e);
      case "bin":
        return pure(n.l) && pure(n.r);
      case "cond":
        return pure(n.c) && pure(n.a) && pure(n.b);
      case "call":
        return !eng.fns.has(n.name) && n.args.every(pure);
      default:
        return false;
    }
  };
  const fold = (n: Node): CN.C | null => {
    try {
      const v = eng.eval(n, { eng, scope: undefined, depth: 0 });
      return v.k === VK.Num ? { re: v.re, im: v.im } : null;
    } catch {
      return null;
    }
  };
  const build = (n: Node): CplxFn | null => {
    if (n.type === "bin" || n.type === "call" || n.type === "cond" || n.type === "un") {
      if (pure(n)) {
        const c = fold(n);
        if (c) return konst(c.re, c.im);
      }
    }
    switch (n.type) {
      case "num":
        return konst(n.value, 0);
      case "ident":
        return identFn(eng, n.name, name, konst);
      case "un": {
        const e = build(n.e);
        if (!e) return null;
        if (n.op === "+") return e;
        if (n.op !== "-") return null;
        return (re, im, o) => {
          e(re, im, o);
          o.re = -o.re;
          o.im = -o.im;
        };
      }
      case "bin": {
        const l = build(n.l);
        const r = build(n.r);
        if (!l || !r) return null;
        const f = BIN_C[n.op];
        if (!f) return null;
        // 每个二元节点在编译期独占一块右操作数暂存，递归时互不覆盖
        const t: CN.C = { re: 0, im: 0 };
        return (re, im, o) => {
          l(re, im, o);
          r(re, im, t);
          f(o.re, o.im, t.re, t.im, o);
        };
      }
      case "call": {
        if (n.args.length !== 1 || eng.fns.has(n.name)) return null;
        const a = build(n.args[0]);
        if (!a) return null;
        const c1 = C1[n.name];
        if (c1) {
          return (re, im, o) => {
            a(re, im, o);
            c1(o.re, o.im, o);
          };
        }
        const g = SCALAR_C[n.name];
        if (g) {
          return (re, im, o) => {
            a(re, im, o);
            o.re = g(o.re, o.im);
            o.im = 0;
          };
        }
        if (n.name === "conj") {
          return (re, im, o) => {
            a(re, im, o);
            o.im = -o.im;
          };
        }
        return null;
      }
      default:
        return null;
    }
  };
  return build(node);
}

const IMMAT = new Set(["i", "j", "I"]);
function isImmutIdent(nm: string): boolean {
  return IMMAT.has(nm) || CONSTANTS[nm] !== undefined;
}

function identFn(
  eng: Engine,
  nm: string,
  zName: string,
  konst: (r: number, i: number) => CplxFn,
): CplxFn | null {
  if (nm === zName) {
    return (re, im, o) => {
      o.re = re;
      o.im = im;
    };
  }
  const g = eng.globals.get(nm);
  if (g) {
    if (g.k !== VK.Num) return null;
    // 全局量每次调用现取：滑块改的是同一个 Map，编译结果可以留着用
    return (_re, _im, o) => {
      const v = eng.globals.get(nm);
      if (!v || v.k !== VK.Num) throw new Error(`未定义的名称 “${nm}”`);
      o.re = v.re;
      o.im = v.im;
    };
  }
  if (IMMAT.has(nm)) return konst(0, 1);
  if (eng.fns.has(nm)) return null;
  const c = CONSTANTS[nm];
  if (c !== undefined) return konst(c, 0);
  return null;
}

/** 复数二元原语；运算符归一化后查表 */
const BIN_C: Record<string, (ar: number, ai: number, br: number, bi: number, o: CN.C) => CN.C> = {
  "+": CN.cadd,
  "-": CN.csub,
  "*": CN.cmul,
  "/": CN.cdiv,
  "^": CN.cpow,
};

/** 取实数的复函数（模、辐角、实部虚部…），与 STRICT 里的定义逐字对应 */
const SCALAR_C: Record<string, (re: number, im: number) => number> = {
  abs: Math.hypot,
  norm: Math.hypot,
  abs2: (re, im) => re * re + im * im,
  re: (re) => re,
  real: (re) => re,
  im: (_re, im) => im,
  imag: (_re, im) => im,
  phase: (re, im) => Math.atan2(im === 0 ? 0 : im, re),
  arg: (re, im) => Math.atan2(im === 0 ? 0 : im, re),
  angle0: (re, im) => Math.atan2(im === 0 ? 0 : im, re),
};

export function evalString(eng: Engine, src: string): Val {
  const d = parseDefinition(src);
  const ctx: EvalCtx = { eng, scope: undefined, depth: 0 };
  if (d.kind === "fn") return funVal({ name: d.name, params: d.params, body: d.node });
  if (d.kind === "let") {
    const v = eng.eval(d.node, ctx);
    eng.globals.set(d.name, v);
    return v;
  }
  return eng.eval(d.node, ctx);
}

/** 把 Val 转成适合显示的形式 */
export function show(x: Val, digits = 6): string {
  switch (x.k) {
    case VK.Num:
      return CN.fmtC(x.re, x.im, digits);
    case VK.Bool:
      return x.b ? "真" : "假";
    case VK.Vec:
      return `[${x.v!.map((v) => CN.fmt(v, digits)).join(", ")}]`;
    case VK.Mat:
      return x.m!.map((r) => `[${r.map((v) => CN.fmt(v, digits)).join(", ")}]`).join("\n");
    case VK.Fun:
      return `${x.fn!.name}(${x.fn!.params.join(", ")})`;
    case VK.Str:
      return x.s ?? "";
    default:
      return "?";
  }
}
