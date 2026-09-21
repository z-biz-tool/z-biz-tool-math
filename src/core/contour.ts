/**
 * 隐函数等值线：Marching Squares + 折线链接 + 不等式区域掩码
 *
 * 设计要点（供渲染层使用）：
 *  - 输出全部是世界坐标，线段为 Seg = [x1,y1,x2,y2]；
 *  - 边上做线性插值（不是取格心），因此曲线足够平滑；
 *  - 有理式 / tan 的极点是二元函数的常态：含非有限采样的格子整体跳过，
 *    插值交点再做一次"二分收缩体检"，剔除跳变间断造成的伪线段；
 *  - 二元函数求值昂贵，故一层网格只采样一次（contourLevels / implicitFill 复用）。
 *  约定 f 自身不抛异常（允许返回 NaN / ±Infinity 表示无定义）。
 */

import { clamp, isNum } from "./cnum.ts";

/** 世界坐标线段：(x1,y1) → (x2,y2) */
export type Seg = [number, number, number, number];

/** 二元标量场（允许返回 NaN / ±Infinity，表示该点无定义） */
type Fn2 = (x: number, y: number) => number;

/** 顶点处的场值（内外由与 level 的比较给出） */
type Corner = number;

/** 一次采样、多次扫掠的网格：v 按行优先存放 (res+1)² 个顶点值 */
interface Grid {
  xs: number[];
  ys: number[];
  v: Float64Array;
  n: number;
}

/** 采样 (res+1)×(res+1) 网格；参数非法时返回 null */
function sampleField(
  f: Fn2,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  res: number,
): Grid | null {
  const n = Math.floor(res);
  if (!(n >= 1)) return null;
  if (!isNum(x0) || !isNum(x1) || !isNum(y0) || !isNum(y1)) return null;
  const xs = new Array<number>(n + 1);
  const ys = new Array<number>(n + 1);
  for (let i = 0; i <= n; i++) {
    xs[i] = x0 + ((x1 - x0) * i) / n;
    ys[i] = y0 + ((y1 - y0) * i) / n;
  }
  const v = new Float64Array((n + 1) * (n + 1));
  for (let j = 0; j <= n; j++) {
    const y = ys[j];
    const row = j * (n + 1);
    for (let i = 0; i <= n; i++) v[row + i] = f(xs[i], y);
  }
  return { xs, ys, v, n };
}

/** 边上线性插值交点；返回 null 表示该交点不可信（极点等跳变间断） */
function edgeCross(
  f: Fn2,
  xa: number,
  ya: number,
  va: Corner,
  xb: number,
  yb: number,
  vb: Corner,
  level: number,
): [number, number] | null {
  // (va>level)!==(vb>level) 保证 vb!==va，除数非零
  const raw = (level - va) / (vb - va);
  if (!isNum(raw)) return null;
  const t = clamp(raw, 0, 1);
  const px = xa + (xb - xa) * t;
  const py = ya + (yb - ya) * t;
  if (!isNum(px) || !isNum(py)) return null;
  const devA = Math.abs(va - level);
  const devB = Math.abs(vb - level);
  // 端点恰好落在等值线上：插值点就是该端点，残差恒为 0
  if (devA === 0 || devB === 0) return [px, py];
  // 跳变间断体检：真零点在边上二分收缩时残差跟着下降；极点引起的符号翻转
  // 越收缩函数值越发散，残差反而超过较小的端点偏差 → 判为伪交点丢弃。
  let lo = 0;
  let hi = 1;
  let vlo = va - level;
  let vhi = vb - level;
  for (let k = 0; k < 3; k++) {
    const m = (lo + hi) / 2;
    const vm = f(xa + (xb - xa) * m, ya + (yb - ya) * m) - level;
    if (!isNum(vm)) return null;
    if (vm === 0) {
      lo = m;
      hi = m;
      break;
    }
    if ((vlo < 0) !== (vm < 0)) {
      hi = m;
      vhi = vm;
    } else {
      lo = m;
      vlo = vm;
    }
  }
  if (Math.min(Math.abs(vlo), Math.abs(vhi)) > 2 * Math.min(devA, devB)) return null;
  return [px, py];
}

/** 在已采样网格上扫掠一条等值线 */
function sweep(g: Grid, level: number, f: Fn2): Seg[] {
  const out: Seg[] = [];
  if (!isNum(level)) return out;
  const { xs, ys, v, n } = g;
  const N = n + 1;

  const push = (p: [number, number] | null, q: [number, number] | null) => {
    if (!p || !q) return;
    // 退化线段：角点恰在等值线上时两侧交点重合，曲线只是"擦过"该格，丢弃
    const sc = Math.max(1, Math.abs(p[0]), Math.abs(q[0]), Math.abs(p[1]), Math.abs(q[1]));
    if (Math.abs(p[0] - q[0]) < 1e-12 * sc && Math.abs(p[1] - q[1]) < 1e-12 * sc) return;
    out.push([p[0], p[1], q[0], q[1]]);
  };

  for (let j = 0; j < n; j++) {
    const ya = ys[j];
    const yb = ys[j + 1];
    const r0 = j * N;
    const r1 = (j + 1) * N;
    for (let i = 0; i < n; i++) {
      const xa = xs[i];
      const xb = xs[i + 1];
      // 角点：a=左下 b=右下 c=右上 d=左上（逆时针，位序与 case 表一致）
      const va = v[r0 + i];
      const vb = v[r0 + i + 1];
      const vc = v[r1 + i + 1];
      const vd = v[r1 + i];
      if (!isNum(va) || !isNum(vb) || !isNum(vc) || !isNum(vd)) continue;
      const ba = va > level ? 1 : 0;
      const bb = vb > level ? 1 : 0;
      const bc = vc > level ? 1 : 0;
      const bd = vd > level ? 1 : 0;
      const cell = ba | (bb << 1) | (bc << 2) | (bd << 3);
      if (cell === 0 || cell === 15) continue;

      let pb: [number, number] | null = null;
      let pr: [number, number] | null = null;
      let pt: [number, number] | null = null;
      let pl: [number, number] | null = null;
      if (ba !== bb) pb = edgeCross(f, xa, ya, va, xb, ya, vb, level);
      if (bb !== bc) pr = edgeCross(f, xb, ya, vb, xb, yb, vc, level);
      if (bd !== bc) pt = edgeCross(f, xa, yb, vd, xb, yb, vc, level);
      if (ba !== bd) pl = edgeCross(f, xa, ya, va, xa, yb, vd, level);

      if (cell === 5 || cell === 10) {
        // 对角同侧的歧义构型（本位序下 case=5/10，等价于常见记法 0110/1001）：
        // 用格心均值（四角算术平均）决定哪对对角被"连通"。
        const inside = (va + vb + vc + vd) / 4 > level;
        // cell 5：a(左下) 与 c(右上) 在内 → 心在内时切掉 b/d 两角
        // cell 10：b(右下) 与 d(左上) 在内 → 心在内时切掉 a/c 两角
        const joinBR = cell === 5 ? inside : !inside;
        if (joinBR) {
          push(pb, pr);
          push(pt, pl);
        } else {
          push(pb, pl);
          push(pr, pt);
        }
        continue;
      }
      // 非歧义构型：合法交点恰有两个（体检可能剔除，故按有效点数通用配对）
      const hits: [number, number][] = [];
      if (pb) hits.push(pb);
      if (pr) hits.push(pr);
      if (pt) hits.push(pt);
      if (pl) hits.push(pl);
      if (hits.length === 2) push(hits[0], hits[1]);
    }
  }
  return out;
}

/**
 * 单条等值线 f(x,y)=level。
 * res 为每个轴的格子数：网格含 (res+1)² 个采样点，第 i 个采样对应 x0+(x1-x0)*i/res。
 */
export function marchingSquares(
  f: Fn2,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  res: number,
  level: number,
): Seg[] {
  const g = sampleField(f, x0, x1, y0, y1, res);
  return g ? sweep(g, level, f) : [];
}

/** 多条等值线：一次采样、逐 level 扫掠，顺序与 levels 一致 */
export function contourLevels(
  f: Fn2,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  res: number,
  levels: number[],
): { level: number; segments: Seg[] }[] {
  const g = sampleField(f, x0, x1, y0, y1, res);
  const out: { level: number; segments: Seg[] }[] = [];
  for (const level of levels) out.push({ level, segments: g ? sweep(g, level, f) : [] });
  return out;
}

/**
 * 不等式分带：以 level 为中心给出若干条等值线（含 level 本身，位于返回数组正中），
 * 带宽由场值偏差的中位数估计，供渲染层叠半透明色带；需要精确边界仍用 marchingSquares。
 */
export function implicitFill(
  f: Fn2,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  res: number,
  level: number,
): Seg[][] {
  const g = sampleField(f, x0, x1, y0, y1, res);
  if (!g || !isNum(level)) return [];
  // 粗采样估计 |f-level| 的中位数作为带宽尺度（避免硬编码）
  const stride = Math.max(1, Math.floor(g.n / 16));
  const dev: number[] = [];
  for (let j = 0; j <= g.n; j += stride) {
    for (let i = 0; i <= g.n; i += stride) {
      const d = Math.abs(g.v[j * (g.n + 1) + i] - level);
      if (isNum(d)) dev.push(d);
    }
  }
  dev.sort((p, q) => p - q);
  const half = dev.length ? dev[dev.length >> 1] : 0;
  const band = isNum(half) && half > 0 ? half * 0.5 : 0;
  const levels = band > 0 ? [level - 2 * band, level - band, level, level + band, level + 2 * band] : [level];
  const out: Seg[][] = [];
  for (const l of levels) out.push(sweep(g, l, f));
  return out;
}

/** 逐像素采样判定区域内外（1=在内），NaN/无穷一律视为外；行优先，y 随 j 递增 */
export function signedRegionMask(
  f: Fn2,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  w: number,
  h: number,
  level: number,
  want: ">" | "<",
): Uint8Array {
  const W = Math.max(0, Math.floor(w));
  const H = Math.max(0, Math.floor(h));
  const mask = new Uint8Array(W * H);
  if (!isNum(x0) || !isNum(x1) || !isNum(y0) || !isNum(y1) || !isNum(level)) return mask;
  const above = want === ">";
  for (let j = 0; j < H; j++) {
    const y = y0 + ((y1 - y0) * (j + 0.5)) / H;
    const row = j * W;
    for (let i = 0; i < W; i++) {
      const x = x0 + ((x1 - x0) * (i + 0.5)) / W;
      const val = f(x, y);
      if (!isNum(val)) continue;
      if (above ? val > level : val < level) mask[row + i] = 1;
    }
  }
  return mask;
}

/* ------------------------------------------------------------------ */
/* 折线链接：把散线段串成连通折线（渲染时才能用 line join 平滑）        */
/* ------------------------------------------------------------------ */

/**
 * 端点量化步长：相邻格子共享的交点由同一算式算出（位级相同），量化只为归一 -0；
 * 世界坐标量级在 1e6 以内时，Math.round(x/1e-9) 仍落在 2^53 内，键唯一。
 */
const QUANT = 1e-9;

function qkey(x: number, y: number): string {
  return `${Math.round(x / QUANT)},${Math.round(y / QUANT)}`;
}

/** 线段串链：hash 表按量化端点聚合，整体 O(n)；链内线段按行进方向定向 */
function chainSegments(segs: Seg[], minSegments: number): Seg[][] {
  const n = segs.length;
  if (n === 0) return [];
  const keys = new Array<string>(2 * n);
  const at = new Map<string, number[]>();
  for (let s = 0; s < n; s++) {
    const sg = segs[s];
    keys[2 * s] = qkey(sg[0], sg[1]);
    keys[2 * s + 1] = qkey(sg[2], sg[3]);
    for (let e = 0; e < 2; e++) {
      const k = keys[2 * s + e];
      const bucket = at.get(k);
      if (bucket) bucket.push(s, e);
      else at.set(k, [s, e]);
    }
  }
  const used = new Uint8Array(n);

  /** 从 (s, side) 出发沿未消费的线段走到底，返回定向后的链（side 为进入端） */
  const walk = (s: number, side: number): Seg[] => {
    const chain: Seg[] = [];
    let cur = s;
    let from = side;
    for (;;) {
      used[cur] = 1;
      const sg = segs[cur];
      // 按行进方向输出，渲染端可直接 polyline + line join
      chain.push(from === 0 ? sg : [sg[2], sg[3], sg[0], sg[1]]);
      const exit = 1 - from;
      const bucket = at.get(keys[2 * cur + exit]);
      let nxt = -1;
      let enter = 0;
      if (bucket) {
        for (let k = 0; k < bucket.length; k += 2) {
          const t = bucket[k];
          const e = bucket[k + 1];
          if (t === cur || used[t] === 1) continue;
          nxt = t;
          enter = e;
          break;
        }
      }
      if (nxt < 0) break;
      cur = nxt;
      from = enter;
    }
    return chain;
  };

  const chains: Seg[][] = [];
  // 先走开放链（端点无邻居，bucket 只含自己的一段），保证折线顺序稳定
  for (let s = 0; s < n; s++) {
    if (used[s]) continue;
    for (let e = 0; e < 2; e++) {
      const bucket = at.get(keys[2 * s + e]);
      if (bucket !== undefined && bucket.length === 2) {
        chains.push(walk(s, e));
        break;
      }
    }
  }
  // 剩下的都是闭合环
  for (let s = 0; s < n; s++) if (!used[s]) chains.push(walk(s, 0));
  return chains.filter((c) => c.length >= minSegments);
}

/** 追踪等值线：原始线段串成连通折线，长度不足 minSegments 的碎链被丢弃 */
export function traceContours(
  f: Fn2,
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  res: number,
  level: number,
  minSegments = 4,
): Seg[][] {
  return chainSegments(marchingSquares(f, x0, x1, y0, y1, res, level), minSegments);
}
