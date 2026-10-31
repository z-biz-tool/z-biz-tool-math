/** 分形内核：Mandelbrot / Julia / Burning Ship 的逃逸迭代与逐行着色（纯逻辑，Node 可测） */

import { buildLUT, type RGB } from "./colormap.ts";

export type FractalKind = "mandelbrot" | "julia" | "burning";

export const FRACTAL_KINDS: { value: FractalKind; label: string }[] = [
  { value: "mandelbrot", label: "Mandelbrot" },
  { value: "julia", label: "Julia" },
  { value: "burning", label: "Burning Ship" },
];

/** 逃逸半径平方：取 16（|z|>4）让平滑着色的 log-log 折算更准 */
const BAILOUT2 = 16;

export interface Escape {
  /** 完成的迭代步数；未逃逸时 = maxIter */
  n: number;
  /** 平滑迭代数（含小数），未逃逸时 = n */
  mu: number;
  inside: boolean;
}

/** 一点的逃逸迭代。Mandelbrot/Burning 从 z=0 出发，Julia 从 z=(x,y) 出发 */
export function escapeAt(kind: FractalKind, x: number, y: number, cr: number, ci: number, maxIter: number): Escape {
  let zx: number;
  let zy: number;
  let cx: number;
  let cy: number;
  if (kind === "julia") {
    zx = x;
    zy = y;
    cx = cr;
    cy = ci;
  } else {
    zx = 0;
    zy = 0;
    cx = x;
    cy = y;
  }
  let n = 0;
  let r2 = 0;
  while (n < maxIter) {
    // Burning Ship：每步先取绝对值再平方
    if (kind === "burning") {
      zx = Math.abs(zx);
      zy = Math.abs(zy);
    }
    const nx = zx * zx - zy * zy + cx;
    const ny = 2 * zx * zy + cy;
    zx = nx;
    zy = ny;
    n++;
    r2 = zx * zx + zy * zy;
    if (r2 > BAILOUT2) {
      const logAbs = Math.log(r2) / 2;
      const mu = n + 1 - Math.log2(Math.max(1e-12, logAbs));
      return { n, mu, inside: false };
    }
  }
  return { n: maxIter, mu: maxIter, inside: true };
}

export interface FracRowOpt {
  kind: FractalKind;
  cr: number;
  ci: number;
  maxIter: number;
  colormap: string;
  /** 平滑着色（连续色相）；关掉按整数步数取色 */
  smooth: boolean;
  dark: boolean;
  left: number;
  right: number;
  y: number;
  w: number;
}

/**
 * 一行像素的逃逸着色，写入 rgb[off..]。与栅格续算的逐行约定一致：
 * 行内 x 按 left→right 均匀取 w 列，与整幅一次采样逐字节相同。
 */
export function fracRow(o: FracRowOpt, rgb: Uint8ClampedArray, off: number): void {
  const lut = buildLUT(o.colormap);
  const dx = (o.right - o.left) / (o.w - 1);
  for (let i = 0; i < o.w; i++) {
    const e = escapeAt(o.kind, o.left + dx * i, o.y, o.cr, o.ci, o.maxIter);
    let c: RGB;
    if (e.inside) {
      // 集合内部：两种主题都用近黑，经典观感
      c = [5, 6, 12];
    } else {
      const k = Math.min(255, Math.max(0, Math.round(((o.smooth ? e.mu : e.n) / o.maxIter) * 255)));
      c = [lut[k * 3], lut[k * 3 + 1], lut[k * 3 + 2]];
    }
    const p = off + i * 3;
    rgb[p] = c[0];
    rgb[p + 1] = c[1];
    rgb[p + 2] = c[2];
  }
}

/** 供读数板用：单点的迭代数文案 */
export function escapeText(kind: FractalKind, x: number, y: number, cr: number, ci: number, maxIter: number): string {
  const e = escapeAt(kind, x, y, cr, ci, maxIter);
  if (e.inside) return `未逃逸（迭代满 ${maxIter} 步）`;
  return `第 ${e.n} 步逃逸`;
}
