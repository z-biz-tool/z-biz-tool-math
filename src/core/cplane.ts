/**
 * 复平面可视化：共形着色（Needham 配色）、Newton 分形、映射雅可比与像曲线拆分
 *
 * 复数运算全部复用 cnum，输出为绘图层可直接消费的扁平数组；不含 DOM 依赖。
 */
import * as CN from "./cnum.ts";

/** 复数（与 cnum.C 结构一致） */
export interface C {
  re: number;
  im: number;
}

/** 复变函数 */
export type CFast = (re: number, im: number, o: C) => void;

/** 复变函数：cf 为可选的无分配快路径（machine.compileCplx 编出的闭包） */
export interface CFn {
  (z: C): C;
  cf?: CFast;
}

const TAU = CN.TAU;

/** 复步导数步长：f'(z) 由 f(z+ih) 的一次求值取出，无差分相消 */
const CSTEP = 1e-6;

function fin(v: number): boolean {
  return Number.isFinite(v);
}

function norm(o: C): C {
  if (!fin(o.re)) o.re = NaN;
  if (!fin(o.im)) o.im = NaN;
  return o;
}

/** 调用用户函数，抛错视为奇点 */
function call(f: CFn, re: number, im: number): C {
  if (f.cf) {
    const o: C = { re: 0, im: 0 };
    try {
      f.cf(re, im, o);
    } catch {
      return { re: NaN, im: NaN };
    }
    return norm(o);
  }
  try {
    return norm(f({ re, im }));
  } catch {
    return { re: NaN, im: NaN };
  }
}

/** 逐像素热路径的求值槽：调用方必须在下一次求值前把数值取进局部变量 */
const EV: C = { re: 0, im: 0 };

function evalInto(f: CFn, re: number, im: number): boolean {
  if (f.cf) {
    try {
      f.cf(re, im, EV);
    } catch {
      return false;
    }
    return fin(EV.re) && fin(EV.im);
  }
  const v = call(f, re, im);
  EV.re = v.re;
  EV.im = v.im;
  return fin(v.re) && fin(v.im);
}

function okC(v: C): boolean {
  return fin(v.re) && fin(v.im);
}

function stepOf(a: number, b: number, n: number): number {
  return n > 1 ? (b - a) / (n - 1) : 0;
}

/** 等距栅格采样复变函数，行优先（x 为快维），shape = [h, w] */
export function sampleComplex(
  f: CFn,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  w: number,
  h: number,
): { re: Float64Array; im: Float64Array; shape: [number, number] } {
  const W = Math.max(1, Math.floor(w));
  const H = Math.max(1, Math.floor(h));
  const re = new Float64Array(W * H);
  const im = new Float64Array(W * H);
  const dx = stepOf(x0, x1, W);
  const dy = stepOf(y0, y1, H);
  for (let j = 0; j < H; j++) {
    const y = y0 + dy * j;
    for (let i = 0; i < W; i++) {
      const v = call(f, x0 + dx * i, y);
      re[j * W + i] = v.re;
      im[j * W + i] = v.im;
    }
  }
  return { re, im, shape: [H, W] };
}

/** 到最近整数线的距离（0 = 正在线上），用于画 |f| 与 arg 的细等值线 */
function lineDist(t: number): number {
  return Math.abs(t - Math.round(t));
}

/** HSV → RGB 写入三字节 */
function hsvToRgb(hue: number, sat: number, val: number, out: Uint8ClampedArray, o: number): void {
  const hh = (((hue % 1) + 1) % 1) * 6;
  const sect = Math.min(5, Math.max(0, Math.floor(hh)));
  const frac = hh - Math.floor(hh);
  const p = val * (1 - sat);
  const q = val * (1 - sat * frac);
  const t = val * (1 - sat * (1 - frac));
  let r: number;
  let g: number;
  let b: number;
  switch (sect) {
    case 0:
      r = val;
      g = t;
      b = p;
      break;
    case 1:
      r = q;
      g = val;
      b = p;
      break;
    case 2:
      r = p;
      g = val;
      b = t;
      break;
    case 3:
      r = p;
      g = q;
      b = val;
      break;
    case 4:
      r = t;
      g = p;
      b = val;
      break;
    default:
      r = val;
      g = p;
      b = q;
      break;
  }
  out[o] = CN.clamp(r * 255, 0, 255);
  out[o + 1] = CN.clamp(g * 255, 0, 255);
  out[o + 2] = CN.clamp(b * 255, 0, 255);
}

/**
 * 共形着色：色相 = 辐角，明度 = 模（log1p 平滑，跨数量级不失真），
 * 叠加 |f| / arg 的细等值线网格；非有限像素画成黑色（奇点标记）。
 * iterFn 用于多值/隐式函数：以 f(z) 为初值做不动点迭代后再着色。
 */
export function domainColor(
  f: CFn,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  w: number,
  h: number,
  opts: {
    iterFn?: CFn;
    saturation?: number;
    exponent?: number;
    levelStep?: number;
    dark?: boolean;
    alpha?: boolean;
  } = {},
): Uint8ClampedArray {
  const W = Math.max(1, Math.floor(w));
  const H = Math.max(1, Math.floor(h));
  const out = new Uint8ClampedArray(W * H * 3);
  const satRaw = opts.saturation;
  const stepRaw = opts.levelStep;
  const sat = CN.clamp(satRaw === undefined || !fin(satRaw) ? 1 : satRaw, 0, 1);
  const expo = opts.exponent && fin(opts.exponent) && opts.exponent > 0 ? opts.exponent : 1;
  // levelStep 默认 1（|f| 等值线间距，log1p 尺度），置 0 关闭网格
  const step = stepRaw === undefined || !fin(stepRaw) ? 1 : stepRaw;
  const argStep = Math.PI / (4 * Math.max(step, 1e-3));
  const dx = stepOf(x0, x1, W);
  const dy = stepOf(y0, y1, H);
  for (let j = 0; j < H; j++) {
    const y = y0 + dy * j;
    for (let i = 0; i < W; i++) {
      const o = (j * W + i) * 3;
      const x = x0 + dx * i;
      let vr = 0;
      let vi = 0;
      if (opts.iterFn) {
        if (!evalInto(f, x, y)) continue;
        vr = EV.re;
        vi = EV.im;
        for (let k = 0; k < 8; k++) {
          const pr = vr;
          const pi = vi;
          if (!evalInto(opts.iterFn, vr, vi)) break;
          const moved = Math.hypot(EV.re - pr, EV.im - pi);
          vr = EV.re;
          vi = EV.im;
          if (moved < 1e-13) break;
        }
      } else if (!evalInto(f, x, y)) {
        continue; // 奇点：保持全黑
      } else {
        vr = EV.re;
        vi = EV.im;
      }
      const m = Math.hypot(vr, vi);
      if (!fin(m)) continue; // 奇点：保持全黑
      const ang = Math.atan2(vi, vr);
      const hue = ang / TAU + 0.5;
      const lm = Math.log1p(m);
      const lum = Math.pow(lm / (1 + lm), expo);
      let s = sat;
      let val = 0.18 + 0.82 * CN.clamp(lum, 0, 1);
      if (step > 0) {
        const d = Math.min(lineDist(lm / step), lineDist(ang / argStep));
        const wide = 0.05;
        const on = d < wide ? (opts.alpha ? 1 - d / wide : 1) : 0;
        if (opts.dark) {
          // 暗底：等值线提亮成白色网格
          s *= 1 - 0.9 * on;
          val += (1 - val) * 0.95 * on;
        } else {
          s *= 1 - 0.6 * on;
          val *= 1 - 0.85 * on;
        }
      }
      hsvToRgb(hue, CN.clamp(s, 0, 1), CN.clamp(val, 0, 1), out, o);
    }
  }
  return out;
}

/**
 * 复 Newton 迭代：导数用复步 f'(z) = (Im[f(z+ih)] - Im[f(z)])/h - i(Re[...])/h（解析、免符号微分），
 * 带步长限制与回溯阻尼。
 */
function newtonRun(
  f: CFn,
  re: number,
  im: number,
  maxIter: number,
  tol: number,
  limit: number,
): { re: number; im: number; it: number; ok: boolean } {
  let zr = re;
  let zi = im;
  const out: C = { re: 0, im: 0 };
  for (let k = 0; k < maxIter; k++) {
    if (!fin(zr) || !fin(zi)) return { re: 0, im: 0, it: k, ok: false };
    if (!evalInto(f, zr, zi)) return { re: zr, im: zi, it: k, ok: false };
    const vr = EV.re;
    const vi = EV.im;
    const nv = Math.hypot(vr, vi);
    if (nv < tol) return { re: zr, im: zi, it: k, ok: true };
    if (!evalInto(f, zr, zi + CSTEP)) return { re: zr, im: zi, it: k, ok: false };
    const dr = (EV.im - vi) / CSTEP;
    const di = -(EV.re - vr) / CSTEP;
    if (!fin(dr) || !fin(di)) return { re: zr, im: zi, it: k, ok: false };
    if (dr === 0 && di === 0) return { re: zr, im: zi, it: k, ok: false };
    const q = CN.cdiv(vr, vi, dr, di, out);
    // 步长限幅：远离区域时先约束在框内，保证有界
    let sr = q.re;
    let si = q.im;
    const sm = Math.hypot(sr, si);
    if (!fin(sm)) return { re: zr, im: zi, it: k, ok: false };
    if (sm > limit) {
      sr *= limit / sm;
      si *= limit / sm;
    }
    let lam = 1;
    let moved = false;
    for (let b = 0; b < 5; b++) {
      const nr = zr - lam * sr;
      const ni = zi - lam * si;
      const cm = evalInto(f, nr, ni) ? Math.hypot(EV.re, EV.im) : NaN;
      if (fin(cm) && (cm < nv || cm < tol)) {
        zr = nr;
        zi = ni;
        moved = true;
        break;
      }
      lam *= 0.5;
    }
    if (!moved) {
      // 阻尼失败（吸引盆边界的震荡点）：直走一步，让后续像素自然发散
      zr -= sr;
      zi -= si;
      if (!fin(zr) || !fin(zi)) return { re: 0, im: 0, it: k, ok: false };
    }
  }
  return { re: zr, im: zi, it: maxIter, ok: false };
}

/** 收敛结果按 1e-6 聚为根类 */
function addCluster(list: C[], re: number, im: number): void {
  for (const r of list) if (Math.hypot(r.re - re, r.im - im) < 1e-6) return;
  list.push({ re, im });
}

/**
 * 一次整幅采样前定好的共用量：根清单、Newton 步长限幅、像素归属判据。
 * 三者都随包围盒尺寸变化，分带续算必须复用整幅算出来的这一份，否则逐带结果和整幅结果不一致。
 */
export interface NewtonPlan {
  roots: C[];
  limit: number;
  match: number;
}

/** 根发现：三层同心散点，覆盖各吸引盆（含包围区域外的根） */
export function newtonPlan(
  f: CFn,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  tol?: number,
): NewtonPlan {
  const t = tol && fin(tol) ? tol : 1e-9;
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  const rx = Math.abs(x1 - x0) / 2 || 1;
  const ry = Math.abs(y1 - y0) / 2 || 1;
  const limit = 2.2 * Math.max(rx, ry);
  const roots: C[] = [];
  for (const ring of [0.55, 1.25, 2]) {
    for (let m = 0; m < 24; m++) {
      if (roots.length > 64) break;
      const th = ring * 0.7 + (TAU * m) / 24;
      const r = newtonRun(
        f,
        cx + rx * ring * Math.cos(th),
        cy + ry * ring * Math.sin(th),
        100,
        t,
        limit,
      );
      if (r.ok) addCluster(roots, r.re, r.im);
    }
  }
  roots.sort((a, b) => a.re - b.re || a.im - b.im);
  return { roots, limit, match: Math.max(1e-5, 1e-6 * Math.max(rx, ry)) };
}

/**
 * Newton 分形：先自动发现全部相异根（环形散点 + 阻尼 Newton + 聚类），
 * 再逐像素记录归属根序号（-1 = 未收敛）与迭代次数。传入 plan 可跳过根发现。
 */
export function newtonFractal(
  f: CFn,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  w: number,
  h: number,
  opts?: { maxIter?: number; tol?: number; plan?: NewtonPlan },
): { iter: Uint16Array; root: Int16Array; roots: C[] } {
  const W = Math.max(1, Math.floor(w));
  const H = Math.max(1, Math.floor(h));
  const maxIter = Math.max(3, Math.min(65535, Math.floor(opts?.maxIter ?? 60)));
  const tol = opts?.tol && fin(opts.tol) ? opts.tol : 1e-9;
  const { roots, limit, match } = opts?.plan ?? newtonPlan(f, x0, x1, y0, y1, tol);

  const iter = new Uint16Array(W * H);
  const root = new Int16Array(W * H);
  const dx = stepOf(x0, x1, W);
  const dy = stepOf(y0, y1, H);
  for (let j = 0; j < H; j++) {
    const y = y0 + dy * j;
    for (let i = 0; i < W; i++) {
      const k = j * W + i;
      const r = newtonRun(f, x0 + dx * i, y, maxIter, tol, limit);
      iter[k] = r.it;
      let idx = -1;
      if (r.ok) {
        let best = Infinity;
        for (let q = 0; q < roots.length; q++) {
          const d = Math.hypot(roots[q].re - r.re, roots[q].im - r.im);
          if (d < best) {
            best = d;
            idx = q;
          }
        }
        if (!(best <= match)) idx = -1;
      }
      root[k] = idx;
    }
  }
  return { iter, root, roots };
}

/**
 * 实二维映射 (x,y) → (u,v) 的中心差分雅可比。
 * singular: |det| 过小；conformal: CR 方程在尺度容差内成立（保角处可着色）。
 */
export function jacobianOfMap(
  f: (z: C) => [number, number],
  x: number,
  y: number,
): { j: number[][]; det: number; singular: boolean; conformal: boolean } {
  const h = 1e-5 * Math.max(1, Math.abs(x), Math.abs(y));
  const val = (px: number, py: number): [number, number] | null => {
    try {
      const r = f({ re: px, im: py });
      return fin(r[0]) && fin(r[1]) ? [r[0], r[1]] : null;
    } catch {
      return null;
    }
  };
  const fp = val(x + h, y);
  const fm = val(x - h, y);
  const gp = val(x, y + h);
  const gm = val(x, y - h);
  const zero = [0, 0];
  const a = fp && fm ? [(fp[0] - fm[0]) / (2 * h), (fp[1] - fm[1]) / (2 * h)] : zero;
  const b = gp && gm ? [(gp[0] - gm[0]) / (2 * h), (gp[1] - gm[1]) / (2 * h)] : zero;
  const j = [
    [a[0], b[0]],
    [a[1], b[1]],
  ];
  const det = a[0] * b[1] - b[0] * a[1];
  const scale = Math.max(Math.abs(a[0]), Math.abs(a[1]), Math.abs(b[0]), Math.abs(b[1]), 1e-12);
  const tol = 1e-5 * scale + 1e-9;
  const conformal = Math.abs(a[0] - b[1]) < tol && Math.abs(a[1] + b[0]) < tol;
  return { j, det, singular: !fin(det) || Math.abs(det) < 1e-9, conformal };
}

/**
 * 映射后的像曲线：遇到非有限值或相邻间距超过 splitOnPole 时拆断，
 * 避免 Möbius / Joukowski 映射把奇点两侧连成长线。
 */
export function mapCurve(
  f: CFn,
  pts: [number, number][],
  opts?: { splitOnPole?: number },
): [number, number][][] {
  const lim = opts?.splitOnPole && fin(opts.splitOnPole) ? opts.splitOnPole : 1e3;
  const out: [number, number][][] = [];
  let cur: [number, number][] = [];
  const flush = () => {
    if (cur.length > 1) out.push(cur);
    cur = [];
  };
  let prev: C | null = null;
  for (const p of pts) {
    const v = call(f, p[0], p[1]);
    if (!okC(v)) {
      flush();
      prev = null;
      continue;
    }
    if (prev && Math.hypot(v.re - prev.re, v.im - prev.im) > lim) flush();
    cur.push([v.re, v.im]);
    prev = v;
  }
  flush();
  return out;
}

/** f'(z) 的中心差分（解析函数下 f' = ∂u/∂x + i·∂v/∂x） */
function derivAt(f: CFn, re: number, im: number): [number, number] | null {
  const h = 1e-5 * Math.max(1, Math.abs(re), Math.abs(im));
  const fp = call(f, re + h, im);
  const fm = call(f, re - h, im);
  if (!okC(fp) || !okC(fm)) return null;
  const sr = (fp.re - fm.re) / (2 * h);
  const si = (fp.im - fm.im) / (2 * h);
  return fin(sr) && fin(si) ? [sr, si] : null;
}

/** 标记保角映射的折叠处（f'(z) = 0）：栅格变号 + 阻尼 Newton，失败返回空表 */
export function criticalPoints(
  f: CFn,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  res: number,
  tol = 1e-7,
): C[] {
  try {
    const n = Math.max(2, Math.floor(res));
    const dx = stepOf(x0, x1, n);
    const dy = stepOf(y0, y1, n);
    const g: ([number, number] | null)[] = new Array(n * n).fill(null);
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) g[j * n + i] = derivAt(f, x0 + dx * i, y0 + dy * j);

    const found: C[] = [];
    // 去重半径随栅格步长走：粗网格本就无法分辨更近的临界点（重根易分裂成两点）
    const dedupe = Math.max(1e-6, 0.01 * Math.max(Math.abs(dx), Math.abs(dy)));
    for (let j = 0; j < n - 1; j++) {
      for (let i = 0; i < n - 1; i++) {
        // 单元四角上 Re f' 与 Im f' 均需同时出现正负
        let hit = true;
        for (let m = 0; m < 2 && hit; m++) {
          let pos = false;
          let neg = false;
          for (let bit = 0; bit < 4; bit++) {
            const gi = i + (bit & 1);
            const gj = j + ((bit >> 1) & 1);
            const q = g[gj * n + gi];
            const val = q ? q[m] : NaN;
            if (val > 0) pos = true;
            else if (val < 0) neg = true;
          }
          if (!pos || !neg) hit = false;
        }
        if (!hit) continue;
        let zr = x0 + dx * (i + 0.5);
        let zi = y0 + dy * (j + 0.5);
        let err = Infinity;
        for (let k = 0; k < 40; k++) {
          const rv = derivAt(f, zr, zi);
          if (!rv) break;
          err = Math.hypot(rv[0], rv[1]);
          if (err < 1e-12) break;
          const hr = 1e-5 * Math.max(1, Math.abs(zr));
          const gr = derivAt(f, zr + hr, zi);
          const gl = derivAt(f, zr - hr, zi);
          const hc = 1e-5 * Math.max(1, Math.abs(zi));
          const gu = derivAt(f, zr, zi + hc);
          const gd = derivAt(f, zr, zi - hc);
          if (!gr || !gl || !gu || !gd) break;
          const a = (gr[0] - gl[0]) / (2 * hr);
          const b = (gu[0] - gd[0]) / (2 * hc);
          const c = (gr[1] - gl[1]) / (2 * hr);
          const d = (gu[1] - gd[1]) / (2 * hc);
          const det = a * d - b * c;
          if (!fin(det) || Math.abs(det) < 1e-14) break;
          // 2×2 求解 J·Δ = -r
          let lam = 1;
          let improved = false;
          const sr = (-rv[0] * d + rv[1] * b) / det;
          const si = (-rv[1] * a + rv[0] * c) / det;
          if (!fin(sr) || !fin(si)) break;
          for (let t = 0; t < 8 && !improved; t++) {
            const nr = zr + lam * sr;
            const ni = zi + lam * si;
            const nv = derivAt(f, nr, ni);
            if (nv && Math.hypot(nv[0], nv[1]) < err) {
              zr = nr;
              zi = ni;
              improved = true;
            }
            lam *= 0.5;
          }
          if (!improved) break;
        }
        if (!(err <= tol)) continue;
        if (!fin(zr) || !fin(zi)) continue;
        if (zr < x0 || zr > x1 || zi < y0 || zi > y1) continue;
        if (found.some((q) => Math.hypot(q.re - zr, q.im - zi) < dedupe)) continue;
        found.push({ re: zr, im: zi });
      }
    }
    return found;
  } catch {
    return [];
  }
}
