/**
 * 线性代数内核：矩阵运算、Gauss 消元、特征值（对称 Jacobi / 2×2 闭式 / 特征多项式求根）、SVD。
 *
 * 约定：矩阵是行优先的 number[][]；复数以 {re, im} 表示。
 * 教学工具里的矩阵都是 4×4 以内，所以优先要「和课本算出来一模一样」，不是渐近最快。
 */

export interface C2 {
  re: number;
  im: number;
}

export type Mat = number[][];

const EPS = 1e-12;

export function zeros(r: number, c: number): Mat {
  return Array.from({ length: r }, () => new Array<number>(c).fill(0));
}

export function identity(n: number): Mat {
  const m = zeros(n, n);
  for (let i = 0; i < n; i++) m[i][i] = 1;
  return m;
}

export function shape(m: Mat): [number, number] {
  const r = m.length;
  const c = r ? m[0].length : 0;
  if (m.some((row) => row.length !== c)) throw new Error("矩阵各行长度不一致");
  return [r, c];
}

export function matAdd(a: Mat, b: Mat): Mat {
  const [ar, ac] = shape(a);
  const [br, bc] = shape(b);
  if (ar !== br || ac !== bc) throw new Error(`加减要求同尺寸：${ar}×${ac} 与 ${br}×${bc}`);
  return a.map((row, i) => row.map((v, j) => v + b[i][j]));
}

export function matSub(a: Mat, b: Mat): Mat {
  const [ar, ac] = shape(a);
  const [br, bc] = shape(b);
  if (ar !== br || ac !== bc) throw new Error(`加减要求同尺寸：${ar}×${ac} 与 ${br}×${bc}`);
  return a.map((row, i) => row.map((v, j) => v - b[i][j]));
}

export function matScale(a: Mat, k: number): Mat {
  return a.map((row) => row.map((v) => v * k));
}

export function matMul(a: Mat, b: Mat): Mat {
  const [ar, ac] = shape(a);
  const [br, bc] = shape(b);
  if (ac !== br) throw new Error(`乘法尺寸不匹配：${ar}×${ac} 乘 ${br}×${bc}`);
  const out = zeros(ar, bc);
  for (let i = 0; i < ar; i++) {
    const ai = a[i];
    const oi = out[i];
    for (let k = 0; k < ac; k++) accum(ai, k, oi, b);
  }
  return out;
}

/** 内层循环单独提出来，V8 会把 a 行、b 行、out 行都保持在同一条隐藏类上 */
function accum(row: number[], k: number, out: number[], b: Mat): void {
  const aik = row[k];
  if (aik === 0) return;
  const brow = b[k];
  for (let j = 0; j < out.length; j++) out[j] += aik * brow[j];
}

export function transpose(a: Mat): Mat {
  const [r, c] = shape(a);
  const out = zeros(c, r);
  for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) out[j][i] = a[i][j];
  return out;
}

/**
 * 带部分主元的 LU 分解（原地写进一个副本）。
 * 行列式、秩、解方程、求逆都从这一份分解出来，所以只实现一次高斯消元。
 */
export interface Lu {
  lu: Mat;
  n: number;
  swaps: number;
  /** 行置换：PA = LU，右端项要按 piv[i] 取行才能对上 */
  piv: number[];
  /** 主元绝对值小于此值即视为降秩 */
  singular: boolean;
}

export function luDecomp(a: Mat): Lu {
  const [r, c] = shape(a);
  if (r !== c) throw new Error("LU 分解要求方阵");
  const lu = a.map((row) => row.slice());
  const piv = Array.from({ length: r }, (_, i) => i);
  let swaps = 0;
  for (let k = 0; k < r; k++) {
    let p = k;
    for (let i = k + 1; i < r; i++) if (Math.abs(lu[i][k]) > Math.abs(lu[p][k])) p = i;
    if (Math.abs(lu[p][k]) < EPS) {
      lu[k][k] = 0;
      continue;
    }
    if (p !== k) {
      const t = lu[p];
      lu[p] = lu[k];
      lu[k] = t;
      const q = piv[p];
      piv[p] = piv[k];
      piv[k] = q;
      swaps++;
    }
    for (let i = k + 1; i < r; i++) {
      const f = lu[i][k] / lu[k][k];
      lu[i][k] = f;
      if (f !== 0) for (let j = k + 1; j < r; j++) lu[i][j] -= f * lu[k][j];
    }
  }
  return { lu, n: r, swaps, piv, singular: lu.some((row, i) => Math.abs(row[i]) < EPS) };
}

/** 用已分解的 LU 回代解一个右端项（不复制 lu） */
function backSubstitute(lu: Mat, piv: number[], n: number, b: number[]): number[] {
  const x = Array.from({ length: n }, (_, i) => b[piv[i]]);
  for (let i = 1; i < n; i++) for (let j = 0; j < i; j++) x[i] -= lu[i][j] * x[j];
  for (let i = n - 1; i >= 0; i--) {
    for (let j = i + 1; j < n; j++) x[i] -= lu[i][j] * x[j];
    x[i] /= lu[i][i];
  }
  return x;
}

export function det(a: Mat): number {
  const [r] = shape(a);
  if (r <= 3) return detSmall(a, r);
  const { lu, swaps } = luDecomp(a);
  let d = swaps % 2 ? -1 : 1;
  for (let i = 0; i < r; i++) d *= lu[i][i];
  return d;
}

/** 2×2 / 3×3 直接按课本展开式：教学场景下这个结果必须和学生手算的一致 */
function detSmall(a: Mat, n: number): number {
  if (n === 1) return a[0][0];
  if (n === 2) return a[0][0] * a[1][1] - a[0][1] * a[1][0];
  return (
    a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) -
    a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
    a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
  );
}

export function solve(a: Mat, b: number[]): number[] {
  const { lu, n, piv, singular } = luDecomp(a);
  if (singular) throw new Error("矩阵奇异，无法唯一求解");
  if (b.length !== n) throw new Error(`右端向量长度应为 ${n}`);
  return backSubstitute(lu, piv, n, b);
}

export function inverse(a: Mat): Mat {
  const { lu, n, piv, singular } = luDecomp(a);
  if (singular) throw new Error("矩阵奇异，不存在逆矩阵");
  const out = zeros(n, n);
  for (let c = 0; c < n; c++) {
    const e = new Array<number>(n).fill(0);
    e[c] = 1;
    const x = backSubstitute(lu, piv, n, e);
    for (let i = 0; i < n; i++) out[i][c] = x[i];
  }
  return out;
}

export interface Rref {
  m: Mat;
  /** 每列是否为主元列 */
  pivots: number[];
  rank: number;
}

/** 简化行阶梯形：秩、零空间、线性相关性的教学展示都靠它 */
export function rref(a: Mat): Rref {
  const m = a.map((row) => row.slice());
  const [rows, cols] = shape(m);
  const pivots: number[] = [];
  let r = 0;
  for (let c = 0; c < cols && r < rows; c++) {
    let piv = r;
    for (let i = r + 1; i < rows; i++) if (Math.abs(m[i][c]) > Math.abs(m[piv][c])) piv = i;
    if (Math.abs(m[piv][c]) < EPS) continue;
    if (piv !== r) {
      const t = m[piv];
      m[piv] = m[r];
      m[r] = t;
    }
    const d = m[r][c];
    for (let j = c; j < cols; j++) m[r][j] /= d;
    for (let i = 0; i < rows; i++) {
      if (i === r) continue;
      const f = m[i][c];
      if (f === 0) continue;
      for (let j = c; j < cols; j++) m[i][j] -= f * m[r][j];
    }
    pivots.push(c);
    r++;
  }
  return { m, pivots, rank: pivots.length };
}

/* --------------------------------------------------------- 特征值分解 */

const cadd = (a: C2, b: C2): C2 => ({ re: a.re + b.re, im: a.im + b.im });
const cmul = (a: C2, b: C2): C2 => ({
  re: a.re * b.re - a.im * b.im,
  im: a.re * b.im + a.im * b.re,
});
const cdiv = (a: C2, b: C2): C2 => {
  const d = b.re * b.re + b.im * b.im || 1;
  return { re: (a.re * b.re + a.im * b.im) / d, im: (a.im * b.re - a.re * b.im) / d };
};
const cabs = (a: C2): number => Math.hypot(a.re, a.im);

function cbrt(v: number): number {
  return v < 0 ? -Math.pow(-v, 1 / 3) : Math.pow(v, 1 / 3);
}

/** 一般三次方程 x³ + b x² + c x + d = 0 的全部根（Cardano，含共轭复根） */
export function cubicRoots(b: number, c: number, d: number): C2[] {
  const p = c - (b * b) / 3;
  const q = (2 * b * b * b) / 27 - (b * c) / 3 + d;
  const disc = (q * q) / 4 + (p * p * p) / 27;
  const shift = -b / 3;
  const tol = 1e-12 * (q * q + Math.abs(p * p * p) + 1);
  if (disc > tol) {
    const s = Math.sqrt(disc);
    const u = cbrt(-q / 2 + s);
    const v = cbrt(-q / 2 - s);
    const re = -(u + v) / 2 + shift;
    const im = (Math.sqrt(3) * Math.abs(u - v)) / 2;
    return [
      { re: u + v + shift, im: 0 },
      { re, im },
      { re, im: -im },
    ];
  }
  if (disc < -tol) {
    const r = Math.sqrt(-(p * p * p) / 27);
    const th = Math.acos(Math.max(-1, Math.min(1, -q / (2 * r))));
    const m = 2 * Math.cbrt(r);
    return [0, 1, 2].map((k) => ({ re: m * Math.cos((th + 2 * Math.PI * k) / 3) + shift, im: 0 }));
  }
  const u = cbrt(-q / 2);
  return [
    { re: 2 * u + shift, im: 0 },
    { re: -u + shift, im: 0 },
    { re: -u + shift, im: 0 },
  ];
}

/** 2×2：迹/行列式闭式解，判别式为负时给出共轭复根（旋转+缩放） */
export function eigen2(a: Mat): C2[] {
  const tr = a[0][0] + a[1][1];
  const dd = a[0][0] * a[1][1] - a[0][1] * a[1][0];
  const disc = tr * tr - 4 * dd;
  if (disc >= 0) {
    const s = Math.sqrt(disc);
    return [
      { re: (tr + s) / 2, im: 0 },
      { re: (tr - s) / 2, im: 0 },
    ];
  }
  const s = Math.sqrt(-disc);
  return [
    { re: tr / 2, im: s / 2 },
    { re: tr / 2, im: -s / 2 },
  ];
}

/** 3×3：按 |A − λI| = 0 展开成三次方程后求根 */
export function eigen3(a: Mat): C2[] {
  const t = a[0][0] + a[1][1] + a[2][2];
  const m11 = a[0][0] * a[1][1] - a[0][1] * a[1][0];
  const m12 = a[0][0] * a[2][2] - a[0][2] * a[2][0];
  const m22 = a[1][1] * a[2][2] - a[1][2] * a[2][1];
  const c2 = m11 + m12 + m22;
  return cubicRoots(-t, c2, -det(a));
}

/* ------------------------------------------------- 特征多项式与求根 */

const csub = (a: C2, b: C2): C2 => ({ re: a.re - b.re, im: a.im - b.im });

/**
 * 特征多项式系数 p(λ)=λⁿ+c₁λⁿ⁻¹+…+cₙ = det(λI−A)，Faddeev–LeVerrier 递推：
 * M₁=I，M_k = A·M_{k−1} + c_{k−1}I，c_k = −tr(A·M_k)/k。
 * 只用矩阵乘法和取迹，比按第一行展开行列式稳，而且 Σλ=−c₁、Πλ=(−1)ⁿcₙ 天然是校验线索。
 */
export function charCoeffs(a: Mat): number[] {
  const n = shape(a)[0];
  const cs: number[] = [];
  let m = identity(n);
  for (let k = 1; k <= n; k++) {
    if (k > 1) {
      m = matMul(a, m);
      for (let i = 0; i < n; i++) m[i][i] += cs[k - 2];
    }
    const am = matMul(a, m);
    cs.push(-am.reduce((s, row, i) => s + row[i], 0) / k);
  }
  return [1, ...cs];
}

/** 复数 Horner 求值，cs 从最高次到常数项 */
function evalPoly(cs: number[], z: C2): C2 {
  let acc: C2 = { re: 0, im: 0 };
  for (const c of cs) acc = cadd(cmul(acc, z), { re: c, im: 0 });
  return acc;
}

function deriv(cs: number[]): number[] {
  const n = cs.length - 1;
  return cs.slice(0, n).map((c, i) => c * (n - i));
}

/** 牛顿磨根；重根处导数趋于 0，此时收工保留迭代前的位置 */
function polishRoot(cs: number[], z: C2): C2 {
  const d = deriv(cs);
  let p = z;
  for (let i = 0; i < 30; i++) {
    const dv = evalPoly(d, p);
    if (cabs(dv) < 1e-16) break;
    const step = cdiv(evalPoly(cs, p), dv);
    p = csub(p, step);
    if (cabs(step) < 1e-15 * (1 + cabs(p))) break;
  }
  return p;
}

/**
 * Durand–Kerner：n 个根同时迭代 zⱼ ← zⱼ − p(zⱼ)/Π_{k≠j}(zⱼ−zₖ)。
 * 不需要化 Hessenberg，也不依赖 deflation，实根复根一视同仁，
 * 教学尺寸的多项式（几次而已）必定在几百步内到位。
 */
function polyRoots(cs: number[]): C2[] {
  const n = cs.length - 1;
  const bound = 1 + Math.max(...cs.slice(1).map(Math.abs));
  const rho = bound * 0.5;
  let z: C2[] = Array.from({ length: n }, (_, k) => {
    const th = (2 * Math.PI * k) / n + 0.4;
    return { re: rho * Math.cos(th), im: rho * Math.sin(th) };
  });
  for (let it = 0; it < 400; it++) {
    const old = z;
    let moved = 0;
    z = old.map((zj, j) => {
      let den: C2 = { re: 1, im: 0 };
      for (let k = 0; k < n; k++) if (k !== j) den = cmul(den, csub(zj, old[k]));
      const step = cdiv(evalPoly(cs, zj), den);
      moved = Math.max(moved, cabs(step));
      return csub(zj, step);
    });
    if (moved < 1e-14 * bound) break;
  }
  return z.map((r) => polishRoot(cs, r));
}

/** 实系数多项式的共轭对尾数不会严格相反；虚部足够小就当实根，再统一排序 */
function snapRoots(roots: C2[]): C2[] {
  return roots
    .map((z) => {
      const tol = 1e-9 * (1 + cabs(z));
      return { re: Math.abs(z.re) < tol ? 0 : z.re, im: Math.abs(z.im) < tol ? 0 : z.im };
    })
    .sort((x, y) => x.re - y.re || x.im - y.im);
}

/**
 * 对称矩阵特征值：Jacobi 旋转。有限步内必定收敛，且给出正交的特征向量，
 * 是二次型主轴（以及 SVD）的基础，所以单独实现而不是走一般 QR。
 */
export interface SymEigen {
  values: number[];
  /** 列向量：vecs[i][k] 是第 i 个特征向量的第 k 个分量 */
  vecs: number[][];
}

export function jacobiEigen(a: Mat, sweeps = 64): SymEigen {
  const n = a.length;
  const m = a.map((r) => r.slice());
  const v = identity(n);
  for (let it = 0; it < sweeps; it++) {
    let off = 0;
    for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) if (i !== j) off += m[i][j] * m[i][j];
    if (off < 1e-24) break;
    for (let p = 0; p < n - 1; p++) {
      for (let q = p + 1; q < n; q++) {
        if (Math.abs(m[p][q]) < 1e-18) continue;
        const theta = (m[q][q] - m[p][p]) / (2 * m[p][q]);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1);
        const s = t * c;
        for (let k = 0; k < n; k++) {
          const mkp = m[k][p];
          const mkq = m[k][q];
          m[k][p] = c * mkp - s * mkq;
          m[k][q] = s * mkp + c * mkq;
        }
        for (let k = 0; k < n; k++) {
          const mpk = m[p][k];
          const mqk = m[q][k];
          m[p][k] = c * mpk - s * mqk;
          m[q][k] = s * mpk + c * mqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = v[k][p];
          const vkq = v[k][q];
          v[k][p] = c * vkp - s * vkq;
          v[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }
  const values = m.map((row, i) => row[i]);
  const order = values.map((_, i) => i).sort((x, y) => values[y] - values[x]);
  return {
    values: order.map((i) => values[i]),
    vecs: order.map((i) => Array.from({ length: n }, (_, k) => v[k][i])),
  };
}

/**
 * 一般方阵特征值：对称阵走 Jacobi（保证全实、且和二次型主轴共用一套实现），
 * 2×2/3×3 用闭式解，更大尺寸先 Hessenberg 再带位移 QR。
 */
export function eigen(a: Mat): C2[] {
  const [r, c] = shape(a);
  if (r !== c) throw new Error("特征值只针对方阵");
  if (r === 1) return [{ re: a[0][0], im: 0 }];
  if (isSymmetric(a)) return jacobiEigen(a).values.map((v) => ({ re: v, im: 0 }));
  if (r === 2) return eigen2(a);
  const cs = charCoeffs(a);
  const raw = r === 3 ? eigen3(a) : polyRoots(cs);
  return snapRoots(raw.map((z) => polishRoot(cs, z)));
}

export function isSymmetric(a: Mat): boolean {
  const [r, c] = shape(a);
  if (r !== c) return false;
  return a.every((row, i) => row.every((v, j) => Math.abs(v - a[j][i]) < 1e-12));
}

/**
 * 对应某个特征值的实特征向量（λ 为实数时）。
 * 复特征值没有实特征向量，此时返回 null，界面上改画不变子空间。
 */
export function eigenVector(a: Mat, lam: number): number[] | null {
  const n = a.length;
  const b = a.map((row, i) => row.map((v, j) => (i === j ? v - lam : v)));
  const { m, pivots, rank } = rref(b);
  if (rank === n) return null;
  const free = Array.from({ length: n }, (_, j) => !pivots.includes(j));
  const i0 = free.indexOf(true);
  if (i0 < 0) return null;
  const x = new Array<number>(n).fill(0);
  x[i0] = 1;
  pivots.forEach((pc, k) => {
    x[pc] = -m[k][i0];
  });
  const len = Math.hypot(...x);
  return len < EPS ? null : x.map((v) => v / len);
}

/* ---------------------------------------------------------------- SVD */

export interface Svd {
  u: Mat;
  /** 奇异值，降序 */
  s: number[];
  v: Mat;
  /** 面积放大率 = 全部奇异值之积（方阵时等于 |det|） */
  area: number;
}

/**
 * 一侧 Jacobi：对 A 的列做旋转把 AᵀA 对角化，直接得到 V 与奇异值。
 * 数值上比先组成 AᵀA 再分解稳（后者条件数平方），而且顺手给出主轴。
 */
export function svd(a: Mat): Svd {
  const [m, n] = shape(a);
  const k = Math.min(m, n);
  const u = a.map((row) => row.slice());
  const v = identity(n);
  for (let sweep = 0; sweep < 30; sweep++) {
    let off = 0;
    for (let p = 0; p < n - 1; p++)
      for (let q = p + 1; q < n; q++) {
        let dp = 0;
        let dq = 0;
        let cross = 0;
        for (let i = 0; i < m; i++) {
          dp += u[i][p] * u[i][p];
          dq += u[i][q] * u[i][q];
          cross += u[i][p] * u[i][q];
        }
        off += cross * cross;
        if (Math.abs(cross) < 1e-15 * Math.sqrt(dp * dq || 1)) continue;
        const zeta = (dq - dp) / (2 * cross);
        const t = Math.sign(zeta || 1) / (Math.abs(zeta) + Math.sqrt(1 + zeta * zeta));
        const c = 1 / Math.sqrt(1 + t * t);
        const s = t * c;
        for (let i = 0; i < m; i++) {
          const x = u[i][p];
          const y = u[i][q];
          u[i][p] = c * x - s * y;
          u[i][q] = s * x + c * y;
        }
        for (let i = 0; i < n; i++) {
          const x = v[i][p];
          const y = v[i][q];
          v[i][p] = c * x - s * y;
          v[i][q] = s * x + c * y;
        }
      }
    if (off < 1e-26) break;
  }
  const colLen = Array.from({ length: n }, (_, j) => Math.hypot(...u.map((row) => row[j])));
  const order = colLen.map((_, i) => i).sort((x, y) => colLen[y] - colLen[x]);
  const s = order.slice(0, k).map((j) => colLen[j]);
  const uu = zeros(m, k);
  order.slice(0, k).forEach((j, c2) => {
    const len = colLen[j] || 1;
    for (let i = 0; i < m; i++) uu[i][c2] = u[i][j] / len;
  });
  const vv = zeros(n, k);
  order.slice(0, k).forEach((j, c2) => {
    for (let i = 0; i < n; i++) vv[i][c2] = v[i][j];
  });
  return { u: uu, s, v: vv, area: s.reduce((x, y) => x * y, 1) };
}

/* ------------------------------------------------------------- 解析辅助 */

/**
 * 矩阵指数 e^{At}（2×2 闭式，用于线性系统相图的轨迹）。
 * 按 Cayley-Hamilton，B = A − (tr/2)I 满足 B² = (disc/4)I，
 * 于是 e^{At} = e^{tr·t/2}[cosh(√disc·t/2)I + (2/√disc)sinh(√disc·t/2)B]，
 * disc<0 时双曲函数自动退化成三角函数。
 */
export function expMat2(a: Mat, t: number): Mat {
  const tr = a[0][0] + a[1][1];
  const disc = tr * tr - 4 * det(a);
  const e = Math.exp((tr / 2) * t);
  const B: Mat = [
    [a[0][0] - tr / 2, a[0][1]],
    [a[1][0], a[1][1] - tr / 2],
  ];
  if (disc > EPS) {
    const s = Math.sqrt(disc);
    return matAdd(matScale(identity(2), e * Math.cosh((s * t) / 2)), matScale(B, (e * 2 * Math.sinh((s * t) / 2)) / s));
  }
  if (Math.abs(disc) <= EPS) return matAdd(matScale(identity(2), e), matScale(B, e * t));
  const w = Math.sqrt(-disc) / 2;
  return matAdd(matScale(identity(2), e * Math.cos(w * t)), matScale(B, (e * Math.sin(w * t)) / w));
}

export function matVec(a: Mat, x: number[]): number[] {
  const [r, c] = shape(a);
  if (x.length !== c) throw new Error(`向量长度应为 ${c}`);
  return Array.from({ length: r }, (_, i) => a[i].reduce((s, v, j) => s + v * x[j], 0));
}
