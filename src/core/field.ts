/**
 * 向量场与 ODE：定步长 RK4、数值雅可比、平衡点搜索与本征分类
 *
 * 供几何画板的「微分方程方向场 / 相平面 / 向量场箭头」使用，纯数值无 DOM。
 */
import * as CN from "./cnum.ts";

/** 微分方程右端：dy/dt = f(t, y) */
export type RHS = (t: number, y: number[]) => number[];

/** 向量场（二维）与方向场 */
export type VecField = (x: number, y: number) => [number, number];

type VecFn = (v: number[]) => number[];

function finite(v: number[]): boolean {
  for (let i = 0; i < v.length; i++) if (!Number.isFinite(v[i])) return false;
  return true;
}

/** 默认步数上限：请求过小的 dt 时用它兜底，避免循环爆炸 */
function maxStepsOf(limit?: number): number {
  return !limit || !Number.isFinite(limit) || limit < 1 ? 20000 : Math.floor(limit);
}

/** 调用右端并校验维度/有限性；异常一律返回 null，让积分干净终止而不是抛错 */
function evalRhs(f: RHS, t: number, y: number[], n: number): number[] | null {
  let d: number[] | null;
  try {
    d = f(t, y);
  } catch {
    return null;
  }
  if (!d || d.length !== n) return null;
  return finite(d) ? d : null;
}

/**
 * 定步长经典 RK4。初值行与结果行同格式：[t0, y1, y2, ...]（只给一维时可省略 t0，视为 0）。
 * 每行形如 [t, y1, y2, ...]，首行即初值；有效步长 max(dt, |tEnd-t0|/maxSteps)，末点严格落在 tEnd。
 */
export function integrateSystem(
  rhs: RHS,
  y0: number[],
  opts: [tEnd: number, dt: number, maxSteps?: number],
): number[][] {
  const sol: number[][] = [y0.slice()];
  // 约定：入参与返回行同构，首元为初始时刻；单元素数组退化为「只有状态」
  const hasTime = y0.length > 1;
  const t0 = hasTime ? y0[0] : 0;
  const y = hasTime ? y0.slice(1) : y0.slice();
  const n = y.length;
  const tEnd = opts[0];
  if (n === 0 || !finite(y0) || !Number.isFinite(tEnd) || tEnd === t0) return sol;

  const limit = maxStepsOf(opts[2]);
  const total = Math.abs(tEnd - t0);
  const wanted = Number.isFinite(opts[1]) ? Math.abs(opts[1]) : 0;
  const h = Math.max(wanted, total / limit);
  const steps = Math.max(1, Math.min(limit, Math.ceil(total / h)));
  const step = (tEnd - t0) / steps;

  const tmp = new Array<number>(n);
  for (let k = 1; k <= steps; k++) {
    const ta = t0 + (k - 1) * step;
    const k1 = evalRhs(rhs, ta, y, n);
    if (!k1) break;
    for (let i = 0; i < n; i++) tmp[i] = y[i] + 0.5 * step * k1[i];
    const k2 = evalRhs(rhs, ta + 0.5 * step, tmp, n);
    if (!k2) break;
    for (let i = 0; i < n; i++) tmp[i] = y[i] + 0.5 * step * k2[i];
    const k3 = evalRhs(rhs, ta + 0.5 * step, tmp, n);
    if (!k3) break;
    for (let i = 0; i < n; i++) tmp[i] = y[i] + step * k3[i];
    const k4 = evalRhs(rhs, ta + step, tmp, n);
    if (!k4) break;
    for (let i = 0; i < n; i++) y[i] += (step / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    // 出现 NaN/Inf（数值爆炸、穿越奇点）时丢弃该步，返回已收集结果
    if (!finite(y)) break;
    sol.push([k === steps ? tEnd : t0 + k * step, ...y]);
  }
  return sol;
}

/** 调用向量函数；抛错或空输出返回 null */
function evalVec(f: VecFn, v: number[]): number[] | null {
  let r: number[] | null;
  try {
    r = f(v);
  } catch {
    return null;
  }
  return r && r.length > 0 ? r : null;
}

/** 中心差分雅可比：J[i][j] = ∂f_i/∂x_j，步长 1e-5·max(1,|p_j|)，不可导处降级为 0 */
export function jacobianAt(f: VecFn, p: number[]): number[][] {
  const n = p.length;
  const probe = evalVec(f, p) ?? evalVec(f, p.map((q) => q + 1e-3)) ?? [];
  const m = probe.length;
  const J: number[][] = Array.from({ length: m }, () => new Array<number>(n).fill(0));
  const x = p.slice();
  for (let j = 0; j < n; j++) {
    const h = 1e-5 * Math.max(1, Math.abs(p[j]));
    x[j] = p[j] + h;
    const fp = evalVec(f, x);
    x[j] = p[j] - h;
    const fm = evalVec(f, x);
    x[j] = p[j];
    for (let i = 0; i < m; i++) {
      const a = fp && i < fp.length ? fp[i] : NaN;
      const b = fm && i < fm.length ? fm[i] : NaN;
      J[i][j] = Number.isFinite(a) && Number.isFinite(b) ? (a - b) / (2 * h) : 0;
    }
  }
  return J;
}

/** 小线性方程组求解（列主元 Gauss），奇异返回 null */
function solveSmall(A: number[][], b: number[]): number[] | null {
  const n = b.length;
  if (n === 0 || A.length < n) return null;
  const M = A.slice(0, n).map((row, i) => [...row.slice(0, n), b[i]]);
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (Math.abs(M[piv][c]) < 1e-14) return null;
    if (piv !== c) {
      const t = M[c];
      M[c] = M[piv];
      M[piv] = t;
    }
    const d = M[c][c];
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / d;
      if (f === 0) continue;
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const out = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    out[i] = M[i][n] / M[i][i];
    if (!Number.isFinite(out[i])) return null;
  }
  return out;
}

/** 2×2（或 1×1）解析本征值与相平面分类；复根借 cnum 的复平方根 */
export function classifyEquilibrium(J: number[][]): {
  type: "node" | "saddle" | "center" | "spiral" | "degenerate";
  stable: boolean;
  eigs: { re: number; im: number }[];
} {
  const eigs: { re: number; im: number }[] = [];
  const n = J.length;
  if (n === 1 && J[0].length >= 1 && Number.isFinite(J[0][0])) {
    eigs.push({ re: J[0][0], im: 0 });
  } else if (n === 2 && J[0].length === 2 && J[1].length === 2) {
    const a = J[0][0],
      b = J[0][1],
      c = J[1][0],
      d = J[1][1];
    if (Number.isFinite(a + b + c + d)) {
      const half = (a + d) / 2;
      const disc = half * half - (a * d - b * c);
      // disc<0 时 csqrt 给出纯虚部，即共轭复根
      const sq = CN.csqrt(disc, 0, CN.TMP[0]);
      eigs.push({ re: half + sq.re, im: sq.im });
      eigs.push({ re: half - sq.re, im: -sq.im });
    }
  }
  if (!eigs.length) return { type: "degenerate", stable: false, eigs };

  let scale = 1;
  for (const e of eigs) scale = Math.max(scale, Math.abs(e.re), Math.abs(e.im));
  const tol = 1e-9 * scale;
  const maxMod = Math.max(...eigs.map((e) => Math.hypot(e.re, e.im)));
  let type: "node" | "saddle" | "center" | "spiral" | "degenerate";
  if (maxMod <= tol) type = "degenerate";
  else if (!eigs.every((e) => Math.abs(e.im) <= tol))
    // 复根：实部为零 → 中心，否则螺旋
    type = eigs.every((e) => Math.abs(e.re) <= tol) ? "center" : "spiral";
  else {
    const pos = eigs.some((e) => e.re > tol);
    const neg = eigs.some((e) => e.re < -tol);
    if (pos && neg) type = "saddle";
    else if (eigs.some((e) => Math.abs(e.re) <= tol)) type = "degenerate";
    else type = "node";
  }
  return { type, stable: eigs.every((e) => e.re < -1e-9), eigs };
}

/** 阻尼 Newton（数值雅可比）精修零点，收敛判据 res·|f| */
function refineZero(f: VecFn, x0: number[], n: number, tol: number): number[] | null {
  let x = x0.slice(0, n);
  let fv = evalVec(f, x);
  if (!fv || fv.length < n || !finite(fv.slice(0, n))) return null;
  let err = Math.hypot(...fv.slice(0, n));
  for (let it = 0; it < 60 && err > 1e-14; it++) {
    const J = jacobianAt(f, x);
    if (J.length < n) break;
    const neg = new Array<number>(n);
    for (let i = 0; i < n; i++) neg[i] = -fv[i];
    const dx = solveSmall(J, neg);
    if (!dx) break;
    let lam = 1;
    let improved = false;
    for (let k = 0; k < 14 && !improved; k++) {
      const cand = new Array<number>(n);
      for (let i = 0; i < n; i++) cand[i] = x[i] + lam * dx[i];
      const nf = evalVec(f, cand);
      if (nf && nf.length >= n && finite(nf.slice(0, n))) {
        const ne = Math.hypot(...nf.slice(0, n));
        if (ne < err) {
          x = cand;
          fv = nf;
          err = ne;
          improved = true;
        }
      }
      lam *= 0.5;
    }
    if (!improved) break;
  }
  return err <= tol ? x : null;
}

/**
 * 平衡点搜索：栅格 |f| 局部极小 + 各分量变号单元作多源初值 → 阻尼 Newton，
 * 结果按 1e-6 去重，丢弃非有限或越界者。
 */
export function equilibria(f: VecFn, ranges: [number, number][], grid = 40): number[][] {
  const n = ranges.length;
  if (n === 0) return [];
  const g = Math.max(2, Math.floor(grid) || 40);
  const lo: number[] = [];
  const hi: number[] = [];
  const st: number[] = [];
  for (let d = 0; d < n; d++) {
    const a = ranges[d][0];
    const b = ranges[d][1];
    if (!Number.isFinite(a) || !Number.isFinite(b) || b <= a) return [];
    lo.push(a);
    hi.push(b);
    st.push((b - a) / (g - 1));
  }

  const total = Math.pow(g, n);
  const vals: (number[] | null)[] = new Array(total).fill(null);
  const idxOf = (k: number, out: number[]): void => {
    let rest = k;
    for (let d = n - 1; d >= 0; d--) {
      out[d] = rest % g;
      rest = Math.floor(rest / g);
    }
  };
  const flatOf = (idx: number[]): number => {
    let k = 0;
    for (let d = 0; d < n; d++) k = k * g + idx[d];
    return k;
  };
  const coordOf = (idx: number[]): number[] => {
    const v = new Array<number>(n);
    for (let d = 0; d < n; d++) v[d] = lo[d] + idx[d] * st[d];
    return v;
  };
  const normOf = (fv: number[] | null): number => (fv ? Math.hypot(...fv.slice(0, n)) : Infinity);

  let scaleF = 1;
  const idx = new Array<number>(n);
  for (let k = 0; k < total; k++) {
    idxOf(k, idx);
    const fv = evalVec(f, coordOf(idx));
    if (!fv || fv.length < n || !finite(fv.slice(0, n))) continue;
    vals[k] = fv;
    scaleF = Math.max(scaleF, normOf(fv));
  }

  const seeds: number[][] = [];
  const corners = 1 << n;
  const corner = new Array<number>(n);
  for (let k = 0; k < total; k++) {
    idxOf(k, idx);
    let inside = true;
    for (let d = 0; d < n; d++) if (idx[d] >= g - 1) inside = false;
    if (!inside) continue;
    // 分量变号：零点穿过该单元
    let hit = true;
    for (let m = 0; m < n && hit; m++) {
      let pos = false;
      let neg = false;
      for (let bit = 0; bit < corners; bit++) {
        for (let d = 0; d < n; d++) corner[d] = idx[d] + ((bit >> d) & 1);
        const fv = vals[flatOf(corner)];
        const q = fv ? fv[m] : NaN;
        if (q > 0) pos = true;
        else if (q < 0) neg = true;
      }
      if (!pos || !neg) hit = false;
    }
    if (hit) seeds.push(coordOf(idx));
  }
  // |f| 局部极小：兜住相切型零点（分量不变号）
  for (let k = 0; k < total; k++) {
    if (!vals[k]) continue;
    idxOf(k, idx);
    const self = normOf(vals[k]);
    let isMin = true;
    for (let d = 0; d < n && isMin; d++) {
      for (const sgn of [-1, 1]) {
        const m = idx[d] + sgn;
        if (m < 0 || m >= g) continue;
        const nb = idx.slice();
        nb[d] = m;
        if (normOf(vals[flatOf(nb)]) < self) isMin = false;
      }
    }
    if (isMin) seeds.push(coordOf(idx));
  }

  const tol = 1e-9 * scaleF;
  const found: number[][] = [];
  for (const s of seeds) {
    const r = refineZero(f, s, n, tol);
    if (!r) continue;
    let good = true;
    for (let d = 0; d < n; d++) {
      if (!Number.isFinite(r[d]) || r[d] < lo[d] - 1e-9 || r[d] > hi[d] + 1e-9) good = false;
    }
    if (!good) continue;
    if (found.some((q) => Math.hypot(...q.map((v, d) => v - r[d])) < 1e-6)) continue;
    found.push(r.map((v) => (Math.abs(v) < 1e-10 ? 0 : v)));
  }
  found.sort((a, b) => {
    for (let d = 0; d < n; d++) if (a[d] !== b[d]) return a[d] - b[d];
    return 0;
  });
  return found;
}

/**
 * 箭头场的采样点。normalize 给出单位方向；logScale 用 log1p 压缩长度，
 * 便于跨越多个数量级的场（奇点附近）仍看得清。mag 为绘制长度。
 */
export function quiverField(
  f: VecField,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  nx: number,
  ny: number,
  opts: { normalize?: boolean; logScale?: boolean } = {},
): { x: number; y: number; u: number; v: number; mag: number }[] {
  const out: { x: number; y: number; u: number; v: number; mag: number }[] = [];
  const ix = Math.max(1, Math.floor(nx));
  const iy = Math.max(1, Math.floor(ny));
  const dx = ix > 1 ? (x1 - x0) / (ix - 1) : 0;
  const dy = iy > 1 ? (y1 - y0) / (iy - 1) : 0;
  for (let j = 0; j < iy; j++) {
    const y = y0 + dy * j;
    for (let i = 0; i < ix; i++) {
      const x = x0 + dx * i;
      let uv: [number, number];
      try {
        uv = f(x, y);
      } catch {
        continue;
      }
      const u0 = uv[0];
      const v0 = uv[1];
      if (!Number.isFinite(u0) || !Number.isFinite(v0)) continue;
      const raw = Math.hypot(u0, v0);
      // mag 与实际画出的线段长度一致（hypot(u,v) === mag），便于直接按比例绘制
      const len = opts.logScale ? Math.log1p(raw) : opts.normalize ? 1 : raw;
      const s = raw > 0 ? len / raw : 0;
      out.push({ x, y, u: u0 * s, v: v0 * s, mag: len });
    }
  }
  return out;
}

/** 向量场单步（RK4，弧长参数），非法时返回 null */
function flowStep(
  f: VecField,
  x: number,
  y: number,
  h: number,
): [number, number] | null {
  const at = (px: number, py: number): [number, number] | null => {
    try {
      const r = f(px, py);
      return Number.isFinite(r[0]) && Number.isFinite(r[1]) ? r : null;
    } catch {
      return null;
    }
  };
  const k1 = at(x, y);
  if (!k1) return null;
  const k2 = at(x + 0.5 * h * k1[0], y + 0.5 * h * k1[1]);
  if (!k2) return null;
  const k3 = at(x + 0.5 * h * k2[0], y + 0.5 * h * k2[1]);
  if (!k3) return null;
  const k4 = at(x + h * k3[0], y + h * k3[1]);
  if (!k4) return null;
  const nx = x + (h / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]);
  const ny = y + (h / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]);
  if (!Number.isFinite(nx) || !Number.isFinite(ny)) return null;
  return [nx, ny];
}

/**
 * 流线：每条种子向前、向后各积分 steps 步，拼成一条连续折线并展平为
 * [x0,y0,x1,y1,...]。NaN、越界或进入驻点（|f| 极小）即停止。
 */
export function streamlines(
  seeds: number[][],
  f: (x: number, y: number) => [number, number],
  opts: { dt: number; steps: number; bounds: [number, number, number, number] },
): number[][] {
  const out: number[][] = [];
  const { dt, bounds } = opts;
  const steps = Math.max(1, Math.floor(opts.steps) || 1);
  const [bx0, bx1, by0, by1] = bounds;
  const inside = (x: number, y: number) =>
    Number.isFinite(x) && Number.isFinite(y) && x >= bx0 && x <= bx1 && y >= by0 && y <= by1;
  const speedAt = (x: number, y: number): number => {
    try {
      const r = f(x, y);
      return Math.hypot(r[0], r[1]);
    } catch {
      return NaN;
    }
  };
  const span = Math.max(Math.abs(bx1 - bx0), Math.abs(by1 - by0)) || 1;

  for (const s of seeds) {
    if (s.length < 2 || !inside(s[0], s[1])) continue;
    // 驻点上的种子画不出方向，直接舍弃
    if (!(speedAt(s[0], s[1]) > 1e-12 * span)) continue;
    const back: number[] = [];
    const fwd: number[] = [];
    for (const dir of [-1, 1]) {
      const buf = dir < 0 ? back : fwd;
      let x = s[0];
      let y = s[1];
      for (let k = 0; k < steps; k++) {
        const nx = flowStep(f, x, y, dir * dt);
        if (!nx || !inside(nx[0], nx[1])) break;
        // 接近驻点则截断，避免折线穿过后抖动
        if (!(speedAt(nx[0], nx[1]) > 1e-12 * span)) break;
        x = nx[0];
        y = nx[1];
        buf.push(x, y);
      }
    }
    const line: number[] = [];
    for (let k = back.length - 2; k >= 0; k -= 2) line.push(back[k], back[k + 1]);
    line.push(s[0], s[1]);
    for (let k = 0; k < fwd.length; k += 2) line.push(fwd[k], fwd[k + 1]);
    if (line.length >= 4) out.push(line);
  }
  return out;
}

/**
 * 方向场（dy/dx = f(x,y)）：给出单位长度线段方向，竖直处（f=±∞）自动收敛为 (0,±1)。
 * 对应几何画板「微分方程方向场」。
 */
export function slopeField(
  df: (x: number, y: number) => number,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  nx: number,
  ny: number,
): { x: number; y: number; dx: number; dy: number }[] {
  const out: { x: number; y: number; dx: number; dy: number }[] = [];
  const ix = Math.max(1, Math.floor(nx));
  const iy = Math.max(1, Math.floor(ny));
  const dx = ix > 1 ? (x1 - x0) / (ix - 1) : 0;
  const dy = iy > 1 ? (y1 - y0) / (iy - 1) : 0;
  for (let j = 0; j < iy; j++) {
    const y = y0 + dy * j;
    for (let i = 0; i < ix; i++) {
      const x = x0 + dx * i;
      let s: number;
      try {
        s = df(x, y);
      } catch {
        continue;
      }
      if (Number.isNaN(s)) continue;
      // 用 atan 参数化方向，天然处理 ±Infinity
      const th = Math.atan(s);
      out.push({ x, y, dx: Math.cos(th), dy: Math.sin(th) });
    }
  }
  return out;
}
