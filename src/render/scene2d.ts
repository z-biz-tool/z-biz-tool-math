/**
 * 2D 场景绘制：把 store 状态画到画布（函数 / 复平面 / 向量场 / 线性映射 / 神经网络 / 动态几何）
 *
 * 约定：
 *  - 每帧现场编译表达式（一次 AST 遍历仅数微秒），因此参数滑块 a/b、控制台里
 *    新定义的用户函数都会立即生效；但跨帧续算的栅格结果一旦被缓存，
 *    身份就必须把参数值一起算进去（见 globalsSig），否则滑块拖了图不变；
 *  - 栅格图（共形着色、不等式掩码、Newton 分形、决策边界）按屏幕行序采样：图像第 0 行必须
 *    对应世界 y 的上界，所以采样时传 (top → bottom)，blit 时传 (bottom, top)。
 */
import { VK } from "../core/types.ts";
import type { Val } from "../core/types.ts";
import {
  Engine,
  compile,
  compileCplx,
  compileReal,
  evalString,
  numericGlobals,
  suggestParams,
} from "../core/machine.ts";
import { parseExpr } from "../core/parser.ts";
import { signedRegionMask, traceContours, type Seg } from "../core/contour.ts";
import {
  classifyEquilibrium,
  equilibria,
  jacobianAt,
  quiverField,
  slopeField,
  streamlines,
} from "../core/field.ts";
import {
  shadeNewton,
  criticalPoints,
  domainColor,
  mapCurve,
  newtonFractal,
  newtonPlan,
  type C,
  type CFn,
  type NewtonPlan,
} from "../core/cplane.ts";
import { colormap, rgbCss } from "../core/colormap.ts";
import { niceTicks } from "../core/view.ts";
import type { Viewport } from "../core/view.ts";
import { releaseRaster, takeRaster } from "./rasterPool.ts";
import type { GeoLabState, Layer, NnState, ToolKind } from "../state.ts";
import type { GeometryDoc } from "../core/geometry.ts";
import {
  det,
  eigen,
  eigenVector,
  expMat2,
  identity,
  isSymmetric,
  matMul,
  matVec,
  rref,
  solve,
  svd,
} from "../core/linalg.ts";
import type { C2, Mat } from "../core/linalg.ts";
import { parseMatrix, parseVector } from "../core/parsemat.ts";
import { CLASS_COLORS, decisionRaster, decisionRow, makeCache, predictMargin } from "../core/nn.ts";
import type { Cache, Model } from "../core/nn.ts";
import * as P from "./plot2d.ts";

const TAU = Math.PI * 2;

export interface SceneOut {
  errors: string[];
  info: string[];
}

/**
 * draft：拖动 / 缩放 / 参数动画进行中的降质档。
 * 复平面栅格按屏幕像素采样，一次全质量重采样可达数百毫秒，交互中必须先给出一帧
 * 粗略图像，停手后由画布再补一次全质量绘制。
 */
export interface SceneOpt {
  draft?: boolean;
  /** 跨帧续算的栅格采样任务；画完的行可直接充当本帧栅格，省掉一次几百毫秒的整幅重采样 */
  raster?: RasterJob | null;
}

function msg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function isNum(v: number): boolean {
  return Number.isFinite(v);
}

function valToReal(v: Val): number {
  if (v.k === VK.Num) return v.re;
  if (v.k === VK.Bool) return v.b ? 1 : 0;
  if (v.k === VK.Vec && v.v && v.v.length) return v.v[0];
  return NaN;
}

/**
 * 表达式 → 可重复求值的实值函数。
 * 优先走 compileReal 的无分配快路径；含复数/向量时退回通用求值并取实部。
 */
export class F1 {
  readonly names: string[];
  private readonly fast: ((s: Float64Array) => number) | null;
  private readonly slow: ((s: Float64Array) => Val) | null;
  private readonly slots: Float64Array;

  constructor(eng: Engine, src: string, take = 1) {
    const node = parseExpr(src);
    this.names = suggestParams(node, eng, take);
    this.slots = new Float64Array(this.names.length * 2);
    const fast = compileReal(eng, node, this.names);
    if (fast) {
      this.fast = fast;
      this.slow = null;
    } else {
      const c = compile(eng, node, this.names);
      this.fast = null;
      this.slow = (s) => c.run(s);
    }
  }

  at(x: number, y = 0, z = 0): number {
    /* contour、场采样这类接口约定求值函数不抛异常：奇点与非法值一律当作无定义 */
    try {
      return this.raw(x, y, z);
    } catch {
      return NaN;
    }
  }

  private raw(x: number, y: number, z: number): number {
    const s = this.slots;
    s[0] = x;
    s[2] = y;
    s[4] = z;
    if (this.fast) return this.fast(s);
    return valToReal(this.slow!(s));
  }

  /** 试算一个点，只为把拼错的函数名/漏写的常量挑出来 —— 采样循环会把它们统统吞掉 */
  badName(): string | null {
    try {
      this.raw(1, 0, 0);
      return null;
    } catch (e) {
      const m = msg(e);
      return /未知函数|未定义的名称/.test(m) ? m : null;
    }
  }
}

function fn(eng: Engine, src: string, take: number, out: SceneOut, tag?: string): F1 | null {
  if (!src || !src.trim()) return null;
  let f: F1;
  try {
    f = new F1(eng, src, take);
  } catch (e) {
    out.errors.push(`${tag || src}：${msg(e)}`);
    return null;
  }
  const bad = f.badName();
  if (bad) {
    out.errors.push(`${tag || src}：${bad}`);
    return null;
  }
  return f;
}

/** 复变函数：以 z 为唯一自变量。传 out 时把失败原因写进去，省略则静默（悬停读数用） */
function cfn(eng: Engine, src: string, out?: SceneOut, tag?: string): CFn | null {
  let run: (s: Float64Array) => Val;
  let cf: ((re: number, im: number, o: C) => void) | null = null;
  try {
    const ast = parseExpr(src);
    const c = compile(eng, ast, ["z"]);
    run = (s) => c.run(s);
    cf = compileCplx(eng, ast, "z");
  } catch (e) {
    if (out) out.errors.push(`${tag ?? src}：${msg(e)}`);
    return null;
  }
  const slots = new Float64Array(2);
  slots[0] = 1;
  /* 试算一次：拼错的函数名要等逐像素着色才炸，那时屏幕上只剩一片空白 */
  try {
    run(slots);
  } catch (e) {
    const m = msg(e);
    if (/未知函数|未定义的名称/.test(m)) {
      if (out) out.errors.push(`${tag ?? src}：${m}`);
      return null;
    }
  }
  slots[0] = 0;
  slots[1] = 0;
  const g = (z: C) => {
    slots[0] = z.re;
    slots[1] = z.im;
    try {
      const v = run(slots);
      return v.k === VK.Num ? { re: v.re, im: v.im } : { re: NaN, im: NaN };
    } catch {
      return { re: NaN, im: NaN };
    }
  };
  // 逐像素着色改走无分配闭包；编不出来的表达式仍用上面的通用路径
  if (cf) g.cf = cf;
  return g;
}

/* ============================================================== 函数模式 */

/**
 * 等值线段从轴空间映回世界坐标。
 * 网格在轴空间等距（对数轴上即等倍率），线性轴下这是恒等映射，逐位不变。
 */
function fromAxis(vp: Viewport, chains: Seg[][]): Seg[][] {
  if (!vp.logX && !vp.logY) return chains;
  return chains.map((ch) =>
    ch.map((s) => [vp.fromAxisX(s[0]), vp.fromAxisY(s[1]), vp.fromAxisX(s[2]), vp.fromAxisY(s[3])] as Seg),
  );
}

function drawFuncMode(p: P.Paper, s: GeoLabState, out: SceneOut): void {
  const eng = s.engine;
  const { vp } = p;
  for (const l of s.layers) {
    if (!l.visible) continue;
    const st: P.StrokeStyle & { samples?: number } = {
      color: l.color,
      width: l.width,
      dashed: l.dashed,
      glow: true,
      samples: l.samples,
    };
    switch (l.kind) {
      case "cartesian": {
        const f = fn(eng, l.expr, 1, out, l.label || l.expr);
        if (f) P.plotCartesian(p, (x) => f.at(x), { ...st, domain: l.domain });
        break;
      }
      case "derivative": {
        const f = fn(eng, l.expr, 1, out, l.label || l.expr);
        if (!f) break;
        /* 中心差分步长随视口尺度自适应，兼顾精度与噪声；
           对数轴上跨度是十倍频程，绝对步长会在小值端把导数抹平，改用相对步长 */
        const slope = vp.logX
          ? (x: number) => {
              const h = Math.max(1e-12, Math.abs(x) * 1e-5);
              return (f.at(x + h) - f.at(x - h)) / (2 * h);
            }
          : (() => {
              const h = Math.max(1e-9, (vp.right - vp.left) * 2.5e-5);
              return (x: number) => (f.at(x + h) - f.at(x - h)) / (2 * h);
            })();
        P.plotCartesian(p, slope, { ...st, dashed: true, glow: false });
        break;
      }
      case "integral": {
        const f = fn(eng, l.expr, 1, out, l.label || l.expr);
        if (!f) break;
        const a = l.domain?.[0] ?? vp.left;
        let b = l.domain?.[1] ?? vp.right;
        /* expr2 为积分上限表达式：可用参数（b、a+1 等），每帧只求值一次 */
        if (l.expr2 && l.expr2.trim()) {
          try {
            const v = evalString(eng, l.expr2);
            if (isNum(v.re)) b = v.re;
            else out.errors.push(`${l.label || l.expr}：上限应为实数`);
          } catch (e) {
            out.errors.push(`${l.label || l.expr} 上限：${msg(e)}`);
          }
        }
        P.plotCartesian(p, (x) => f.at(x), { ...st, fillTo: 0, domain: [a, b], glow: false });
        const n = 420;
        /* 累积面积按轴空间等距推进：对数轴上这样才有足够的首端分辨率 */
        const u0 = vp.toAxisX(a > 0 || !vp.logX ? a : vp.left);
        const u1 = vp.toAxisX(b > 0 || !vp.logX ? b : vp.right);
        const xs: number[] = [];
        const acc: number[] = [];
        let sum = 0;
        let px = vp.fromAxisX(u0);
        let py = f.at(px);
        for (let i = 0; i <= n; i++) {
          const x = vp.fromAxisX(u0 + ((u1 - u0) * i) / n);
          const y = f.at(x);
          if (i > 0 && isNum(y) && isNum(py)) sum += ((x - px) * (y + py)) / 2;
          px = x;
          py = y;
          xs.push(x);
          acc.push(sum);
        }
        const pts = xs.map((x, i) => P.vpScreen(p, x, acc[i]));
        P.strokeChains(p, P.splitChains(pts, p.h * 2), { ...st, dashed: true, width: 1.9, glow: false });
        break;
      }
      case "polar": {
        const f = fn(eng, l.expr, 1, out, l.label || l.expr);
        if (!f) break;
        const [t0, t1] = l.domain ?? [0, TAU];
        P.plotPolar(p, (t) => f.at(t), t0, t1, st);
        break;
      }
      case "param": {
        const fx = fn(eng, l.expr, 1, out, l.label || l.expr);
        const fy = fn(eng, l.expr2 ?? "", 1, out, l.expr2 ?? "y(t)");
        if (!fx || !fy) break;
        const [t0, t1] = l.domain ?? [0, TAU];
        P.plotParam(p, (t) => fx.at(t), (t) => fy.at(t), t0, t1, st);
        break;
      }
      case "implicit": {
        const f = fn(eng, l.expr, 2, out, l.label || l.expr);
        if (!f) break;
        const res = Math.max(70, Math.min(420, Math.round(l.samples / 5)));
        const level = l.level ?? 0;
        const chains = traceContours(
          (u, v) => f.at(vp.fromAxisX(u), vp.fromAxisY(v)) - level,
          vp.axisLeft,
          vp.axisRight,
          vp.axisBottom,
          vp.axisTop,
          res,
          0,
          3,
        );
        P.drawChains(p, fromAxis(vp, chains), { ...st, glow: false });
        break;
      }
      case "inequality": {
        const f = fn(eng, l.expr, 2, out, l.label || l.expr);
        if (!f) break;
        const level = l.level ?? 0;
        const W = Math.max(48, Math.round(p.w / 3));
        const H = Math.max(48, Math.round(p.h / 3));
        /* 掩码按像素铺满画布，所以采样也必须走在轴空间（屏幕 ↔ 轴空间是线性的） */
        const mask = signedRegionMask(
          (u, v) => f.at(vp.fromAxisX(u), vp.fromAxisY(v)) - level,
          vp.axisLeft,
          vp.axisRight,
          vp.axisTop,
          vp.axisBottom,
          W,
          H,
          0,
          l.rel === "<" || l.rel === "<=" ? "<" : ">",
        );
        P.fillMask(p, mask, W, H, l.color, 0.22);
        const chains = traceContours(
          (u, v) => f.at(vp.fromAxisX(u), vp.fromAxisY(v)) - level,
          vp.axisLeft,
          vp.axisRight,
          vp.axisBottom,
          vp.axisTop,
          200,
          0,
          3,
        );
        P.drawChains(p, fromAxis(vp, chains), { ...st, width: 1.8, glow: false });
        break;
      }
      case "sequence": {
        const f = fn(eng, l.expr, 1, out, l.label || l.expr);
        if (!f) break;
        const a = Math.ceil(l.domain?.[0] ?? vp.left);
        const b = Math.floor(l.domain?.[1] ?? vp.right);
        const xs: number[] = [];
        const ys: number[] = [];
        for (let n = a; n <= b && n < a + 5000; n++) {
          const y = f.at(n);
          if (isNum(y)) {
            xs.push(n);
            ys.push(y);
          }
        }
        const stepPx = Math.abs(P.vpScreen(p, 1, 0)[0] - P.vpScreen(p, 0, 0)[0]);
        P.drawSteps(p, xs, ys, { ...st, barWidth: Math.max(2, Math.min(7, stepPx * 0.14)) });
        break;
      }
    }
  }
  drawLegend(p, s.layers);
}

/** 右上角图例：与图层列表同色同名 */
function drawLegend(p: P.Paper, layers: Layer[]): void {
  const items = layers.filter((l) => l.visible && (l.label || l.expr));
  if (!items.length) return;
  const { ctx } = p;
  const th = P.themeOf(p.dark);
  ctx.save();
  ctx.font = "12px 'SF Mono', Menlo, monospace";
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  let y = 18;
  for (const l of items.slice(0, 9)) {
    const text = l.label || l.expr;
    const short = text.length > 34 ? `${text.slice(0, 33)}…` : text;
    const w = ctx.measureText(short).width;
    ctx.strokeStyle = l.color;
    ctx.lineWidth = 2.4;
    ctx.setLineDash(l.dashed ? [5, 4] : []);
    ctx.beginPath();
    ctx.moveTo(p.w - w - 34, y);
    ctx.lineTo(p.w - w - 14, y);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = th.text;
    ctx.fillText(short, p.w - 12, y);
    y += 18;
  }
  ctx.restore();
}

/* ============================================================= 复平面模式 */

/** 交互中的采样步长：约把栅格降到三万像素，换取一帧内画得完 */
const DRAFT_STEP = 5;
/** Newton 分形每像素要迭代到收敛，交互中降得更狠 */
const DRAFT_NEWTON_STEP = 12;
/** 交互中 Newton 分形只迭代到粗判收敛即止（没有可复用的上一次采样时的兜底） */
const DRAFT_NEWTON_ITER = 14;

/** 最近一次全质量采样的固化结果，交互期间按当前视口映射贴回（Newton 与共形着色共用） */
let cplxShot: {
  basis: string;
  buf: Uint8ClampedArray;
  tile: HTMLCanvasElement;
  world: [number, number, number, number];
  roots: C[];
} | null = null;

/** 把一份 rgb 栅格（每像素 3 字节）固化成画布：重复贴同一次采样时只需一次 drawImage */
function rasterTile(rgb: Uint8ClampedArray, w: number, h: number): HTMLCanvasElement | null {
  if (typeof document === "undefined") return null;
  const cv = document.createElement("canvas");
  cv.width = w;
  cv.height = h;
  const ctx = cv.getContext("2d");
  if (!ctx) return null;
  const img = ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const o = i * 4;
    img.data[o] = rgb[i * 3];
    img.data[o + 1] = rgb[i * 3 + 1];
    img.data[o + 2] = rgb[i * 3 + 2];
    img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return cv;
}

/** 把这一帧的全质量栅格固化成交互期间的贴图源；同一块缓冲反复画不必重做 tile */
function keepShot(
  s: GeoLabState,
  rgb: Uint8ClampedArray,
  w: number,
  h: number,
  world: [number, number, number, number],
  roots: C[],
): void {
  const basis = cplxBasis(s);
  if (cplxShot && cplxShot.buf === rgb && cplxShot.basis === basis) return;
  const tile = rasterTile(rgb, w, h);
  cplxShot = tile ? { basis, buf: rgb, tile, world, roots } : null;
}

/* ------------------------------------ 栅格（共形着色 / 决策边界）的分带续算 */

/** 采样身份里的参数指纹：同一表达式配不同参数值是两张图 */
export function globalsSig(eng: Engine): string {
  return numericGlobals(eng)
    .map(([k, re, im]) => `${k}=${re},${im}`)
    .join(",");
}

/** 一帧画不完、要跨帧逐行续算的栅格来自哪套采样器 */
type RasterSource = "cplx" | "nn";

/** Newton 全质量迭代上限（降质档另有 DRAFT_NEWTON_ITER） */
const NEWTON_FULL_ITER = 64;

/**
 * 一次全质量栅格采样任务。整幅要几百毫秒，一帧干不完，
 * 于是按屏幕行自上逐行采样，用完一帧预算就把控制权交还出去。
 */
export interface RasterJob {
  /** 采样结果的全部决定因素；任一变化都会让已算的行作废 */
  key: string;
  /** 采样器归属，同时作为 key 的首字段，让两套栅格互不冒领 */
  source: RasterSource;
  mode: string;
  step: number;
  w: number;
  h: number;
  /** [left, right, bottom, top]，与 blitPixels 同约定 */
  world: [number, number, number, number];
  rgb: Uint8ClampedArray;
  /** 已完成的行数（自屏幕顶部连续） */
  rows: number;
  /** newton：根清单与由整幅包围盒导出的限幅、判据，逐带复用才能和整幅一次算完全一致 */
  plan: NewtonPlan | null;
  cols: number[][];
  /** nn：逐行前向复用的中间量缓存，随任务一起作废 */
  cache: Cache | null;
}

/** 只有逐像素着色的复平面子模式一帧画不完，其余模式不必走续算 */
function isRasterMode(mode: string): boolean {
  return mode === "domain" || mode === "log" || mode === "newton";
}

/** 本帧有没有一份逐像素栅格要走续算 */
function rasterSource(s: GeoLabState): RasterSource | null {
  if (s.mode === "nn") return s.nn.model && s.nn.data && s.nn.showBoundary ? "nn" : null;
  if (s.mode === "complex" && isRasterMode(s.cplx.mode)) return "cplx";
  return null;
}

function gridStep(c: GeoLabState["cplx"], draft: boolean): number {
  const k = Math.max(1, Math.min(4, c.resolution));
  return draft ? Math.max(k, c.mode === "newton" ? DRAFT_NEWTON_STEP : DRAFT_STEP) : k;
}

/** 决策边界：水平采样数由面板给定，换算成「多少屏幕像素一个采样点」；交互中再疏一档 */
function nnStep(nn: NnState, pw: number, draft: boolean): number {
  const base = pw / Math.max(16, Math.min(600, nn.boundaryRes));
  return draft ? base * 3 : base;
}

/** 本帧栅格的采样几何：像素步长与行列数 */
function rasterGeom(
  s: GeoLabState,
  src: RasterSource,
  pw: number,
  ph: number,
  draft: boolean,
): { step: number; w: number; h: number } {
  if (src === "cplx") {
    const step = gridStep(s.cplx, draft);
    return { step, w: gridDim(pw, step), h: gridDim(ph, step) };
  }
  const step = nnStep(s.nn, pw, draft);
  /* 边界本身是几段光滑曲线，几十列就够读出拓扑，不必按像素采样 */
  return { step, w: Math.max(16, Math.round(pw / step)), h: Math.max(16, Math.round(ph / step)) };
}

/**
 * 采样结果的身份：栅格来源、它的全部参数、档位、视口包围盒、栅格尺寸、明暗。
 * 数值直接进字符串，浮点往返是精确的，所以相等即逐像素同解。
 */
function rasterKey(s: GeoLabState, src: RasterSource, vp: Viewport, step: number, w: number, h: number): string {
  const geom = [step, w, h, vp.left, vp.right, vp.bottom, vp.top];
  if (src === "nn") {
    const n = s.nn;
    /** model/data 都是原地训练的可变对象：身份里必须带 epochs 与结构，否则会把上一轮的边界当成这一帧 */
    const model = n.model ? `${n.model.act}:${n.model.sizes.join("x")}` : "-";
    return ["nn", n.dataset, n.samples, n.seed, n.epochs, model, ...geom, s.settings.dark ? 1 : 0].join("\u0000");
  }
  return [cplxBasis(s), ...geom].join("\u0000");
}

/**
 * 采样身份中与视口、栅格尺寸无关的一截：表达式、参数值、着色设置、明暗。
 * 交互期间贴的是「上一次在别的视口算出的像素」，几何一动就得作废；
 * 这些设置一动同样换一张图，所以它与 rasterKey 共用同一份定义，不各写一遍。
 */
function cplxBasis(s: GeoLabState): string {
  const c = s.cplx;
  return [
    "cplx",
    c.f,
    c.mode,
    c.colormap,
    c.levelStep,
    c.iterative ? 1 : 0,
    globalsSig(s.engine),
    s.settings.dark ? 1 : 0,
  ].join("\u0000");
}

/** 交互期间能不能沿用上一次全质量采样（Newton 与共形着色共用这份缓存） */
function shotOf(s: GeoLabState): typeof cplxShot {
  return cplxShot && cplxShot.basis === cplxBasis(s) ? cplxShot : null;
}

/** 本帧若按全质量画，采样身份是什么；无栅格可续算时返回 null */
export function rasterKeyOf(s: GeoLabState, pw: number, ph: number): string | null {
  const src = rasterSource(s);
  if (!src) return null;
  const vp = s.views[s.mode];
  const { step, w, h } = rasterGeom(s, src, pw, ph, false);
  return rasterKey(s, src, vp, step, w, h);
}

function gridDim(px: number, step: number): number {
  return Math.max(64, Math.round(px / step));
}

/** 本帧若按全质量画，对应的续算任务长什么样（无栅格可续算时返回 null） */
export function rasterJobFor(s: GeoLabState, pw: number, ph: number): RasterJob | null {
  const src = rasterSource(s);
  if (!src) return null;
  const vp = s.views[s.mode];
  const { step, w, h } = rasterGeom(s, src, pw, ph, false);
  const job: RasterJob = {
    key: rasterKey(s, src, vp, step, w, h),
    source: src,
    mode: src === "cplx" ? s.cplx.mode : "boundary",
    step,
    w,
    h,
    world: [vp.left, vp.right, vp.bottom, vp.top],
    rgb: new Uint8ClampedArray(w * h * 3),
    rows: 0,
    plan: null,
    cols: [],
    cache: null,
  };
  if (src === "nn") {
    if (!s.nn.model) return null;
    job.cache = makeCache(s.nn.model);
    return job;
  }
  const c = s.cplx;
  const f = cfn(s.engine, c.f);
  if (!f) return null;
  if (c.mode === "newton") {
    // 限幅与根配对判据都由整幅包围盒导出，必须先按全幅算好再逐带复用
    job.plan = newtonPlan(f, vp.left, vp.right, vp.top, vp.bottom);
    const nr = Math.max(1, job.plan.roots.length);
    for (let i = 0; i < nr; i++) job.cols.push(colormap(c.colormap, (i + 0.5) / nr));
  }
  return job;
}

/**
 * 采样若干行，用完预算即返回 "more"。
 * 每行单独调用采样器：行内步长仍按整幅算，所以逐行结果与整幅一次采样逐字节相同。
 */
export function stepRasterJob(s: GeoLabState, job: RasterJob, budgetMs: number): "done" | "more" | "stale" {
  if (rasterKey(s, job.source, s.views[s.mode], job.step, job.w, job.h) !== job.key) return "stale";
  const dark = s.settings.dark;
  const [left, right, bottom, top] = job.world;
  const dy = (bottom - top) / (job.h - 1);
  const t0 = performance.now();
  // 本帧要画的是这张图才继续占池；换图了就放掉上一张还没派完的带
  releaseRaster(job);
  if (job.source === "nn") {
    const m = s.nn.model;
    const cache = job.cache;
    if (!m || !cache) return "stale";
    while (job.rows < job.h) {
      decisionRow(m, left, right, top + dy * job.rows, job.w, dark, job.rgb, job.rows * job.w * 3, cache);
      job.rows++;
      if (performance.now() - t0 >= budgetMs) return "more";
    }
    return "done";
  }
  const c = s.cplx;
  const f = cfn(s.engine, c.f);
  if (!f) return "stale";
  /* 行与行互不依赖，编得出快路径的表达式就整带交给 worker 池：
     主线程这一帧不必采样，等回带填进 job.rgb 即可；池塌了则从当前行起继续串行。 */
  if (
    f.cf &&
    takeRaster(
      {
        src: c.f,
        kind: job.mode === "newton" ? "newton" : "domain",
        globals: numericGlobals(s.engine),
        iter: NEWTON_FULL_ITER,
        plan: job.plan,
        levelStep: c.levelStep,
        saturation: c.mode === "log" ? 0.95 : 0.85,
        selfIter: c.mode === "log" || c.iterative,
        dark,
      },
      job,
    )
  )
    return job.rows >= job.h ? "done" : "more";
  while (job.rows < job.h) {
    const y = top + dy * job.rows;
    const off = job.rows * job.w * 3;
    if (job.mode === "newton") {
      const nz = newtonFractal(f, left, right, y, y + dy, job.w, 1, {
        maxIter: NEWTON_FULL_ITER,
        plan: job.plan ?? undefined,
      });
      shadeNewton(nz.iter, nz.root, job.w, job.cols, dark, job.rgb, off);
    } else {
      const row = domainColor(f, left, right, y, y + dy, job.w, 1, {
        levelStep: c.levelStep,
        dark,
        iterFn: c.mode === "log" || c.iterative ? f : undefined,
        saturation: c.mode === "log" ? 0.95 : 0.85,
        alpha: true,
      });
      job.rgb.set(row, off);
    }
    job.rows++;
    if (performance.now() - t0 >= budgetMs) return "more";
  }
  return "done";
}

/** 续算任务能否直接充当本帧的栅格：采样身份（含视口、档位、色标）必须逐字符相同 */
function readyJob(job: RasterJob | null | undefined, key: string): job is RasterJob {
  return !!job && job.key === key && job.rows === job.h;
}

function drawComplexMode(
  p: P.Paper,
  s: GeoLabState,
  out: SceneOut,
  draft: boolean,
  job?: RasterJob | null,
): void {
  const { vp, dark } = p;
  const c = s.cplx;
  const f = cfn(s.engine, c.f, out, `f(z) = ${c.f}（自变量写作 z）`);
  if (!f) return;
  const step = gridStep(c, draft);
  const w = gridDim(p.w, step);
  const h = gridDim(p.h, step);
  const world: [number, number, number, number] = [vp.left, vp.right, vp.bottom, vp.top];
  const ready = readyJob(job, rasterKey(s, "cplx", vp, step, w, h)) ? job : null;

  if (c.mode === "domain" || c.mode === "log") {
    const note = c.mode === "log" ? "迭代吸引盆 z←f(z)" : "Needham 共形着色：色相=arg f，明度=|f|";
    /* 一帧要采十万像素，交互中不可能现算：把上一次全质量图按当前视口映射贴回，
       位置仍精确，也比降采样那一版看得清 */
    const shot = draft ? shotOf(s) : null;
    if (shot) {
      P.blitCanvas(p, shot.tile, shot.world, 1, s.settings.antialias);
      P.drawFrame(p, { xPi: s.settings.piTicksX, yPi: s.settings.piTicksY, minor: false });
      out.info.push(`${note}：交互中沿用上一次采样`);
      return;
    }
    const rgb = ready
      ? ready.rgb
      : domainColor(f, vp.left, vp.right, vp.top, vp.bottom, w, h, {
          levelStep: c.levelStep,
          dark,
          iterFn: (c.mode === "log" || c.iterative) && !draft ? f : undefined,
          saturation: c.mode === "log" ? 0.95 : 0.85,
          alpha: true,
        });
    P.blitPixels(p, rgb, w, h, world, 1, s.settings.antialias);
    P.drawFrame(p, { xPi: s.settings.piTicksX, yPi: s.settings.piTicksY, minor: false });
    out.info.push(note);
    if (!draft) keepShot(s, rgb, w, h, world, []);
    return;
  }

  if (c.mode === "newton") {
    const showRoots = (list: C[], nr: number) => {
      list.forEach((r, i) =>
        P.drawPoint(p, r.re, r.im, {
          color: rgbCss(colormap(c.colormap, (i + 0.5) / nr)),
          r: 5,
          hollow: true,
          label: `z${i + 1}`,
        }),
      );
    };
    const shot = draft ? shotOf(s) : null;
    /* 每个像素要迭代到收敛并带回溯，降采样也压不进一帧；
       交互中直接把上一次全质量采样按当前视口映射贴回，几何位置仍然正确 */
    if (shot) {
      P.blitCanvas(p, shot.tile, shot.world, 1, s.settings.antialias);
      P.drawFrame(p, { xPi: s.settings.piTicksX, yPi: s.settings.piTicksY, minor: false });
      showRoots(shot.roots, Math.max(1, shot.roots.length));
      out.info.push(`Newton 法 z←z−f/f'：交互中沿用上一次采样`);
      return;
    }
    const iters = draft ? DRAFT_NEWTON_ITER : NEWTON_FULL_ITER;
    let roots: C[];
    let rgb: Uint8ClampedArray;
    if (ready) {
      roots = ready.plan?.roots ?? [];
      rgb = ready.rgb;
    } else {
      const nz = newtonFractal(f, vp.left, vp.right, vp.top, vp.bottom, w, h, { maxIter: iters });
      roots = nz.roots;
      const nr = Math.max(1, nz.roots.length);
      const cols: number[][] = [];
      for (let i = 0; i < nr; i++) cols.push(colormap(c.colormap, (i + 0.5) / nr));
      rgb = new Uint8ClampedArray(w * h * 3);
      shadeNewton(nz.iter, nz.root, w * h, cols, dark, rgb, 0);
    }
    P.blitPixels(p, rgb, w, h, world, 1, s.settings.antialias);
    P.drawFrame(p, { xPi: s.settings.piTicksX, yPi: s.settings.piTicksY, minor: false });
    showRoots(roots, Math.max(1, roots.length));
    if (!draft) keepShot(s, rgb, w, h, world, roots);
    out.info.push(`Newton 法 z←z−f/f'：自动发现 ${roots.length} 个根`);
    return;
  }

  /* 保角映射：把选定的原像曲线族推向像平面，折叠处（f'=0）标记 */
  const span = Math.max(vp.right - vp.left, vp.top - vp.bottom);
  const n = 200;
  const pre: [number, number][][] = [];
  const post: [number, number][][] = [];
  for (const mk of curveFamily(p, c.curve, span)) {
    const pts: [number, number][] = [];
    for (let i = 0; i <= n; i++) pts.push(mk(i / n));
    pre.push(pts);
    for (const ch of mapCurve(f, pts, { splitOnPole: span * 6 })) post.push(ch);
  }
  P.strokeChains(
    p,
    pre.map((c2) => c2.map((q) => P.vpScreen(p, q[0], q[1]))),
    { color: dark ? "rgba(150,160,210,0.3)" : "rgba(70,80,130,0.28)", width: 1 },
  );
  post.forEach((c2, i) => {
    P.strokeChains(p, [c2.map((q) => P.vpScreen(p, q[0], q[1]))], {
      color: rgbCss(colormap(c.colormap, (i % 7) / 6)),
      width: 1.9,
      alpha: 0.95,
    });
  });
  P.drawFrame(p, { minor: false, axes: true, labels: true });
  const cps = criticalPoints(f, vp.left, vp.right, vp.bottom, vp.top, 46);
  cps.forEach((q, i) =>
    P.drawPoint(p, q.re, q.im, { color: "#f472b6", r: 4.5, label: i === 0 ? "f'(z)=0" : "", hollow: true }),
  );
  out.info.push(`原像曲线族（淡）与 f 的像（彩）：保角处正交保持，折叠处 f'=0`);
}

/**
 * 保角映射的原像曲线族：
 * grid 直角栅格 / circle 同心圆 / ray 过原点射线 / polar 圆+射线 / both 全部
 */
function curveFamily(p: P.Paper, kind: string, span: number): ((t: number) => [number, number])[] {
  const { vp } = p;
  const out: ((t: number) => [number, number])[] = [];
  const wantLines = kind === "grid" || kind === "both";
  const wantRings = kind === "circle" || kind === "polar" || kind === "both";
  const wantSpokes = kind === "ray" || kind === "polar" || kind === "both";
  if (wantLines) {
    const xt = niceTicks(vp.left, vp.right, Math.max(4, Math.round(p.w / 120)));
    const yt = niceTicks(vp.bottom, vp.top, Math.max(4, Math.round(p.h / 110)));
    for (const x of xt.values) out.push((t) => [x, vp.bottom + (vp.top - vp.bottom) * t]);
    for (const y of yt.values) out.push((t) => [vp.left + (vp.right - vp.left) * t, y]);
  }
  const R = span * 0.62;
  for (let i = 1; wantRings && i <= 7; i++) {
    const r = (R * i) / 7;
    out.push((t) => [r * Math.cos(TAU * t), r * Math.sin(TAU * t)]);
  }
  for (let i = 0; wantSpokes && i < 12; i++) {
    const a = (TAU * i) / 12;
    out.push((t) => [R * t * Math.cos(a), R * t * Math.sin(a)]);
  }
  return out;
}

/* ============================================================= 向量模式 */

function toXY(v: Val): [number, number] {
  if (v.k === VK.Vec && v.v && v.v.length >= 1) return [v.v[0] ?? 0, v.v[1] ?? 0];
  if (v.k === VK.Num) return [v.re, v.im];
  return [NaN, NaN];
}

function drawVectorMode(p: P.Paper, s: GeoLabState, out: SceneOut): void {
  const { vp } = p;
  const v = s.vec;
  const n = Math.max(4, Math.min(46, Math.round(v.density)));
  const span = Math.max(vp.right - vp.left, vp.top - vp.bottom);

  if (v.fieldMode === "slope") {
    const df = fn(s.engine, v.dfxy, 2, out, "dy/dx");
    if (df) {
      const items = slopeField((x, y) => df.at(x, y), vp.left, vp.right, vp.bottom, vp.top, n, n);
      P.drawSlopeField(p, items, { len: (span / n) * 0.82, color: P.themeOf(p.dark).accent, alpha: 0.8 });
      out.info.push(`方向场 dy/dx = ${v.dfxy}`);
    }
  } else if (v.fieldMode === "quiver") {
    const fu = fn(s.engine, v.fx, 2, out, "u(x,y)");
    const fv = fn(s.engine, v.fy, 2, out, "v(x,y)");
    if (fu && fv) {
      const items = quiverField(
        (x, y) => [fu.at(x, y), fv.at(x, y)],
        vp.left,
        vp.right,
        vp.bottom,
        vp.top,
        n,
        n,
        { logScale: true },
      );
      P.drawQuiver(p, items, {
        scale: span / n / 1.8,
        colormapFn: (t) => rgbCss(colormap(v.colormap, t)),
      });
      out.info.push(`向量场 (${v.fx}, ${v.fy})，箭长按 |F| 对数缩放`);
    }
  } else {
    // 流线 / 相图：流线按弧长推进，相图按真实时间推进
    const fu = fn(s.engine, v.fx, 2, out, "dx/dt");
    const fv = fn(s.engine, v.fy, 2, out, "dy/dt");
    if (fu && fv) {
      const raw = (x: number, y: number): [number, number] => [fu.at(x, y), fv.at(x, y)];
      const speedAt = (x: number, y: number) => {
        const [a, b] = raw(x, y);
        return Math.hypot(a, b);
      };
      let avg = 0;
      let cnt = 0;
      for (let i = 0; i <= 8; i++)
        for (let j = 0; j <= 8; j++) {
          const m = speedAt(vp.left + ((vp.right - vp.left) * i) / 8, vp.bottom + ((vp.top - vp.bottom) * j) / 8);
          if (isNum(m)) {
            avg += m;
            cnt++;
          }
        }
      avg = cnt ? avg / cnt : 1;
      const phase = v.fieldMode === "phase";
      const f = phase
        ? raw
        : (x: number, y: number): [number, number] => {
            const [a, b] = raw(x, y);
            const m = Math.hypot(a, b);
            return m > 1e-12 ? [a / m, b / m] : [0, 0];
          };
      const side = Math.max(2, Math.round(Math.sqrt(Math.max(4, v.streamlineCount))));
      const seeds: number[][] = [];
      for (let i = 0; i < side; i++)
        for (let j = 0; j < side; j++)
          seeds.push([
            vp.left + ((vp.right - vp.left) * (i + 0.5)) / side,
            vp.bottom + ((vp.top - vp.bottom) * (j + 0.5)) / side,
          ]);
      const steps = Math.max(60, Math.round(v.steps));
      const dt = phase ? (12 * avg > 0 ? span / Math.max(1e-9, avg) / steps : 0.01) : span / steps;
      const lines = streamlines(seeds, f, {
        dt,
        steps,
        bounds: [vp.left, vp.right, vp.bottom, vp.top],
      });
      P.drawStreams(p, lines, { color: rgbCss(colormap(v.colormap, 0.62)), width: 1.9, arrows: true });
      if (phase) {
        const vecFn = (y: number[]): number[] => [fu.at(y[0], y[1]), fv.at(y[0], y[1])];
        const eqs = equilibria(vecFn, [
          [vp.left, vp.right],
          [vp.bottom, vp.top],
        ]);
        for (const e of eqs) {
          const cls = classifyEquilibrium(jacobianAt(vecFn, e));
          const col =
            cls.type === "saddle"
              ? "#fbbf24"
              : cls.stable
                ? ["center", "spiral"].includes(cls.type)
                  ? "#667eea"
                  : "#34d399"
                : "#f87171";
          P.drawPoint(p, e[0], e[1], { color: col, r: 5, label: `${cls.type}${cls.stable ? "" : "ⁿ"}` });
        }
        out.info.push(`相图：${eqs.length} 个奇点（鞍点/结点/焦点/中心按本征值着色）`);
      } else {
        out.info.push(`流线：沿方向场的等弧长积分`);
      }
    }
  }

  /* 自由向量与平行四边形法则 */
  const drawn: { tail: [number, number]; head: [number, number]; color: string; label: string }[] = [];
  for (const a of v.arrows) {
    let t: [number, number] = [0, 0];
    let d: [number, number];
    try {
      t = toXY(evalString(s.engine, a.tail));
      d = toXY(evalString(s.engine, a.vec));
    } catch (e) {
      out.errors.push(`${a.label || a.vec}：${msg(e)}`);
      continue;
    }
    if (!isNum(t[0]) || !isNum(t[1]) || !isNum(d[0]) || !isNum(d[1])) continue;
    const head: [number, number] = [t[0] + d[0], t[1] + d[1]];
    drawn.push({ tail: t, head, color: a.color, label: a.label });
    P.drawArrow(p, t[0], t[1], head[0], head[1], { color: a.color, width: 2.6, label: a.label });
    P.drawPoint(p, t[0], t[1], { color: a.color, r: 3, halo: false });
  }
  if (drawn.length === 2) {
    const [a, b] = drawn;
    const da: [number, number] = [a.head[0] - a.tail[0], a.head[1] - a.tail[1]];
    const db: [number, number] = [b.head[0] - b.tail[0], b.head[1] - b.tail[1]];
    const o = a.tail;
    const sum: [number, number] = [o[0] + da[0] + db[0], o[1] + da[1] + db[1]];
    const diff: [number, number] = [o[0] + da[0] - db[0], o[1] + da[1] - db[1]];
    /* 平行四边形法则：平移 b 到 a 的终点，和向量起于 a 的起点 */
    P.drawArrow(p, a.head[0], a.head[1], sum[0], sum[1], { color: "#94a3b8", width: 1.4, dashed: true });
    P.drawArrow(p, o[0] + db[0], o[1] + db[1], sum[0], sum[1], { color: "#94a3b8", width: 1.4, dashed: true });
    P.drawArrow(p, o[0], o[1], sum[0], sum[1], { color: "#f472b6", width: 3, label: "u+v" });
    P.drawArrow(p, o[0], o[1], diff[0], diff[1], { color: "#38bdf8", width: 1.8, dashed: true, label: "u−v" });
  }
}

/* ============================================================ 线性代数模式 */

/** 线性映射各图元的配色：沿用向量场与相图已在用的那组色值，不再引入新色感 */
const LIN_COLORS = {
  image: "#60a5fa",
  ellipse: "#f472b6",
  eigen: "#fbbf24",
  svd: "#34d399",
  flow: "#38bdf8",
  sol: "#f87171",
  col: ["#667eea", "#22d3ee", "#f0abfc"],
};
const LIN_COL_LABEL = ["a₁", "a₂", "a₃"];
/** 轨道种子：两条基向量加两个象限外的点，剪切与反向一眼可辨 */
const FLOW_SEEDS: [number, number][] = [
  [1, 0],
  [0, 1],
  [-1, -1],
  [2, -1],
];

/** 等分采样步长：n 个点之间有 n−1 个间隔，与内核各栅格同一条约定 */
function stepOf(a: number, b: number, n: number): number {
  return n > 1 ? (b - a) / (n - 1) : 0;
}

/** 复特征值的写法与复平面读数板一致 */
function cShow(z: C2): string {
  if (!isNum(z.im) || z.im === 0) return show3(z.re);
  return `${show3(z.re)}${z.im >= 0 ? " + " : " − "}${show3(Math.abs(z.im))}i`;
}

function traceOf(m: Mat): number {
  return m.reduce((sum, row, i) => sum + (row[i] ?? 0), 0);
}

/** 内核的特征值顺序不统一（对称走 Jacobi 降序、2 阶走闭式 tr+√ 在前），显示前自己按模排 */
function sortedEigen(m: Mat): C2[] {
  const mod = (z: C2) => Math.abs(z.re) + Math.abs(z.im);
  return eigen(m)
    .slice()
    .sort((x, y) => mod(y) - mod(x) || y.re - x.re || y.im - x.im);
}

function eigenSummary(m: Mat): string {
  return sortedEigen(m).map(cShow).join("，");
}

/** 整数幂：24 次自乘很容易冲出浮点量程，一出现非有限值就停在最后一个可用幂次 */
function matPow(a: Mat, k: number): { m: Mat; k: number } {
  let m = identity(a.length);
  let done = 0;
  for (let i = 1; i <= k; i++) {
    const next = matMul(m, a);
    if (!next.every((row) => row.every(isNum))) break;
    m = next;
    done = i;
  }
  return { m, k: done };
}

interface LinMap {
  /** 面板里填的 A */
  a: Mat;
  /** 本帧真正作用的变换 M */
  m: Mat;
  /** 实际用到的整数幂次（连续流下无意义） */
  k: number;
  exp: boolean;
  /** 图上/info 里怎么称呼这个 M */
  tag: string;
}

/**
 * 本帧作用的变换：连续流走 e^{tA}（内核仅支持 2×2），否则走离散幂 A^k。
 * k=0 即单位阵，滑块推到 0 时所有像都落回原像 —— 这正是这个滑块的意义。
 * parseMatrix 抛出的中文错误由调用方写进 out.errors。
 */
function linMap(s: GeoLabState): LinMap {
  const { dim, a: cells, t, useExp } = s.lin;
  const a = parseMatrix(s.engine, cells, dim);
  if (useExp && dim === 2) return { a, m: expMat2(a, t), k: 0, exp: true, tag: `e^(${show3(t)}·A)` };
  const want = Math.max(0, Math.min(24, Math.round(t)));
  const pw = matPow(a, want);
  return { a, m: pw.m, k: pw.k, exp: false, tag: `A^${pw.k}` };
}

/**
 * 投到屏幕上的那张平面：3×3 时取 z=0 平面沿 z 轴的正交俯视。
 * 只留前两个分量，原像才与背景网格严格重合，「k=0 落回原像」这条视觉契约才成立。
 */
function topView(m: Mat): Mat {
  if (m.length === 2) return m;
  return [
    [m[0][0], m[0][1]],
    [m[1][0], m[1][1]],
  ];
}

function drawLinMode(p: P.Paper, s: GeoLabState, out: SceneOut): void {
  const { vp, dark } = p;
  const lin = s.lin;
  const th = P.themeOf(dark);
  let lm: LinMap;
  try {
    lm = linMap(s);
  } catch (e) {
    out.errors.push(`矩阵 A：${msg(e)}`);
    return;
  }
  const { a, m, tag } = lm;
  const dim = a.length;
  if (lin.useExp && dim === 3) out.info.push("e^{tA} 内核只支持 2×2，本帧仍按 A^k 绘制");
  const mv = topView(m);
  if (!mv.every((row) => row.every(isNum))) {
    out.errors.push(`${tag}：数值溢出，把 t 调小一点`);
    return;
  }
  const sv = svd(mv);
  const img = (x: number, y: number): [number, number] => {
    const [u, v] = matVec(mv, [x, y]);
    return [u, v];
  };

  /* 整数格线的像：线性映射把直线送成直线，两个端点就够 */
  if (lin.showGrid) {
    const gx = Math.max(1, Math.round(niceTicks(vp.left, vp.right, Math.max(4, Math.round(p.w / 110))).step));
    const gy = Math.max(1, Math.round(niceTicks(vp.bottom, vp.top, Math.max(4, Math.round(p.h / 80))).step));
    const chains: number[][][] = [];
    for (let c = Math.ceil(vp.left / gx) * gx; c <= vp.right && chains.length < 96; c += gx)
      chains.push([P.vpScreen(p, ...img(c, vp.bottom)), P.vpScreen(p, ...img(c, vp.top))]);
    for (let c = Math.ceil(vp.bottom / gy) * gy; c <= vp.top && chains.length < 192; c += gy)
      chains.push([P.vpScreen(p, ...img(vp.left, c)), P.vpScreen(p, ...img(vp.right, c))]);
    P.strokeChains(p, chains, { color: LIN_COLORS.image, width: 1.25, alpha: dark ? 0.7 : 0.55 });
    /* 单位方格的像：光看线网看不出面积缩放，填出这一格才和 |det| 对得上 */
    const scr = ([[0, 0], [1, 0], [1, 1], [0, 1]] as [number, number][]).map(([x, y]) => {
      const [u, v] = img(x, y);
      return P.vpScreen(p, u, v);
    });
    const ctx = p.ctx;
    ctx.save();
    ctx.fillStyle = th.selection;
    ctx.beginPath();
    ctx.moveTo(scr[0][0], scr[0][1]);
    for (let i = 1; i < scr.length; i++) ctx.lineTo(scr[i][0], scr[i][1]);
    ctx.closePath();
    ctx.fill();
    ctx.restore();
    const [mx, my] = img(0.5, 0.5);
    P.annotate(p, mx, my, `${dim === 2 ? "面积" : "俯视面积"} ×${show3(Math.abs(det(mv)))}`, {
      color: LIN_COLORS.image,
      size: 12,
      box: true,
      align: "center",
      dx: 0,
      dy: 0,
    });
  }

  /* 单位圆与它的像：椭圆的扁率即奇异值之比 */
  const circlePts = (n: number, f: (x: number, y: number) => [number, number]): number[][] => {
    const dt = stepOf(0, TAU, n);
    const pts: number[][] = [];
    for (let i = 0; i < n; i++) {
      const [x, y] = f(Math.cos(dt * i), Math.sin(dt * i));
      pts.push(P.vpScreen(p, x, y));
    }
    if (pts.length) pts.push(pts[0]);
    return pts;
  };
  if (lin.showCircle)
    P.strokeChains(p, [circlePts(96, (x, y) => [x, y])], { color: th.muted, width: 1.5, dashed: true });
  if (lin.showEllipse)
    P.strokeChains(p, P.splitChains(circlePts(160, img), p.h * 3), {
      color: LIN_COLORS.ellipse,
      width: 2.4,
      glow: true,
    });

  /* 轨道：p ↦ M p ↦ M² p …，滑 t 时折线随之生长 */
  if (lin.showFlow) {
    const steps = lm.exp ? Math.max(2, Math.min(64, Math.round(Math.abs(lin.t) * 10))) : Math.max(1, lm.k);
    const dt = stepOf(0, lin.t, steps + 1);
    for (const seed of FLOW_SEEDS) {
      const ws: [number, number][] = [[seed[0], seed[1]]];
      for (let i = 1; i <= steps; i++) {
        /* 连续流按定义取 e^{itA}·p；离散幂在上一点像的基础上再乘一次 M */
        const from = lm.exp ? seed : ws[ws.length - 1];
        const M = lm.exp ? topView(expMat2(a, dt * i)) : mv;
        const [nx, ny] = matVec(M, [from[0], from[1]]);
        if (!isNum(nx) || !isNum(ny) || Math.hypot(nx, ny) > 1e7) break;
        ws.push([nx, ny]);
      }
      const pts = ws.map(([x, y]) => P.vpScreen(p, x, y));
      for (let i = 1; i < pts.length; i++)
        P.strokePolyline(p, [pts[i - 1], pts[i]], {
          color: LIN_COLORS.flow,
          width: 2,
          alpha: 0.12 + 0.85 * (i / pts.length),
        });
      const last = ws[ws.length - 1];
      P.drawPoint(p, last[0], last[1], { color: LIN_COLORS.flow, r: 4.5, halo: false });
    }
  }

  if (lin.showSVD) {
    /* 标注必须与画出来的椭圆同解，所以主轴与 σ 都取本帧作图的 2×2；A 真正的奇异值另说 */
    for (let k = 0; k < sv.s.length && k < 2; k++) {
      P.drawArrow(p, 0, 0, sv.v[0][k], sv.v[1][k], {
        color: LIN_COLORS.svd,
        width: 1.6,
        dashed: true,
        alpha: 0.9,
      });
      P.drawArrow(p, 0, 0, sv.u[0][k] * sv.s[k], sv.u[1][k] * sv.s[k], {
        color: LIN_COLORS.svd,
        width: 2.6,
        label: `σ${k + 1} = ${show3(sv.s[k])}`,
      });
    }
    if (dim === 3) out.info.push(`A 的奇异值 σ = ${svd(a).s.map(show3).join("，")}`);
  }

  if (lin.showEigen) {
    let shown = 0;
    for (const z of sortedEigen(m)) {
      if (Math.abs(z.im) > 1e-9) continue;
      const w = eigenVector(m, z.re);
      if (!w) continue;
      const d: [number, number] = [w[0], w[1]];
      const dl = Math.hypot(d[0], d[1]);
      if (dl < 1e-6) {
        out.info.push(`λ = ${show3(z.re)} 的特征方向几乎垂直于屏幕，俯视图里只是一个点`);
        continue;
      }
      shown++;
      const u: [number, number] = [d[0] / dl, d[1] / dl];
      const span = clipToBox([0, 0], u, vp.left, vp.right, vp.bottom, vp.top, -Infinity);
      if (span)
        P.strokeChains(
          p,
          [[P.vpScreen(p, u[0] * span[0], u[1] * span[0]), P.vpScreen(p, u[0] * span[1], u[1] * span[1])]],
          { color: LIN_COLORS.eigen, width: 1.2, dashed: true, alpha: 0.75 },
        );
      /* 箭杆取 λ·d：d 是单位特征向量的俯视图，故箭长即该方向上的伸缩率，负 λ 直接看成反向 */
      P.drawArrow(p, 0, 0, z.re * d[0], z.re * d[1], {
        color: LIN_COLORS.eigen,
        width: 2.6,
        label: `λ = ${show3(z.re)}`,
      });
    }
    /* 复特征值没有实特征方向：给出 |λ|（缩放）与 arg λ（每步转角） */
    if (!shown && dim === 2) {
      const z = sortedEigen(m).find((q) => q.im > 0);
      if (z) {
        const r = Math.hypot(z.re, z.im);
        const ang = Math.atan2(z.im, z.re);
        const rad = Math.min(p.w, p.h) * 0.2;
        P.drawAngleMark(p, 0, 0, 0, ang, rad, LIN_COLORS.eigen);
        P.annotate(p, 0, 0, `λ = ${cShow(z)} 及其共轭：|λ| = ${show3(r)}，每步转 ${show3((ang * 180) / Math.PI)}°`, {
          color: LIN_COLORS.eigen,
          size: 12,
          box: true,
          dx: rad * 0.6,
          dy: -rad * 0.75,
        });
      }
    }
  }

  /* A 的列向量 = 基向量的像，正是 x₁a₁ + x₂a₂ = b 那张线性组合图的边 */
  for (let j = 0; j < dim; j++) {
    const cx = a[0][j];
    const cy = a[1][j];
    if (!isNum(cx) || !isNum(cy)) continue;
    P.drawArrow(p, 0, 0, cx, cy, {
      color: LIN_COLORS.col[j % LIN_COLORS.col.length],
      width: 2.4,
      alpha: 0.8,
      label: LIN_COL_LABEL[j] ?? `a${j + 1}`,
    });
  }

  let b: number[] | null = null;
  try {
    b = parseVector(s.engine, lin.b, dim);
  } catch (e) {
    out.errors.push(`右端向量 b：${msg(e)}`);
  }
  if (b) {
    try {
      const x = solve(a, b);
      const coord = dim === 3 ? `${show3(x[0])}, ${show3(x[1])}, ${show3(x[2])}` : `${show3(x[0])}, ${show3(x[1])}`;
      P.drawArrow(p, 0, 0, x[0], x[1], { color: LIN_COLORS.sol, width: 1.6, dashed: true, alpha: 0.85 });
      P.drawPoint(p, x[0], x[1], { color: LIN_COLORS.sol, r: 5.5, label: `解 x = (${coord})` });
    } catch (e) {
      out.errors.push(`Ax=b：${msg(e)}`);
    }
  }

  out.info.push(`A：det = ${show3(det(a))} · tr = ${show3(traceOf(a))} · 秩 ${rref(a).rank}/${dim}`);
  out.info.push(`${tag}：λ = ${eigenSummary(m)} · σ = ${sv.s.map(show3).join("，")}`);
  if (isSymmetric(a)) out.info.push("A 对称：特征向量互相正交，奇异值 = |特征值|");
  if (dim === 3) out.info.push("3×3：画的是 z=0 平面沿 z 轴的正交俯视，垂直分量只写在数值里");
}

/* ============================================================ 神经网络模式 */

/** 悬停读数每帧都要一个 cache；模型原地训练，对象没换就一直复用 */
let nnProbe: { model: Model; cache: Cache } | null = null;

function nnCacheOf(m: Model): Cache {
  if (!nnProbe || nnProbe.model !== m) nnProbe = { model: m, cache: makeCache(m) };
  return nnProbe.cache;
}

function drawNnMode(
  p: P.Paper,
  s: GeoLabState,
  out: SceneOut,
  draft: boolean,
  job?: RasterJob | null,
): void {
  const { vp, dark } = p;
  const n = s.nn;
  const m = n.model;
  const data = n.data;
  if (!m || !data) {
    P.drawFrame(p, { minor: s.settings.showMinorGrid, axes: true, labels: true });
    P.drawHint(p, "在右侧面板点「训练」，决策边界随 epoch 逐轮变形");
    out.info.push("神经网络：还没有模型，先在面板里选好数据集与结构再点「训练」");
    return;
  }
  if (n.showBoundary) {
    const { step, w, h } = rasterGeom(s, "nn", p.w, p.h, draft);
    const world: [number, number, number, number] = [vp.left, vp.right, vp.bottom, vp.top];
    const ready = readyJob(job, rasterKey(s, "nn", vp, step, w, h)) ? job : null;
    /* 续算任务画满了就贴现成的 rgb；否则整幅一次采样 —— boundaryRes 才几十列，本来就轻 */
    const rgb = ready ? ready.rgb : decisionRaster(m, vp.left, vp.right, vp.top, vp.bottom, w, h, dark);
    P.blitPixels(p, rgb, w, h, world, 1, s.settings.antialias);
  }
  /* 边界是不透明栅格，坐标轴必须后画才不会被盖掉（与复平面模式同一条约定） */
  P.drawFrame(p, { minor: false, axes: true, labels: true });
  for (let i = 0; i < data.n; i++) {
    const label = data.ys[i];
    P.drawPoint(p, data.xs[2 * i], data.xs[2 * i + 1], {
      color: rgbCss(CLASS_COLORS[label % CLASS_COLORS.length]),
      r: 3.8,
      /* 实心=0 类，空心=其余：只靠配色分不出多类样本 */
      hollow: label % 2 === 1,
    });
  }
  out.info.push(
    `决策边界：${data.name} · ${data.k} 类 · ${data.n} 样本 · ${n.epochs} 轮 · 准确率 ${(n.acc * 100).toFixed(1)}%`,
  );
  out.info.push(
    n.running
      ? "训练进行中，边界每轮跟手"
      : n.showBoundary
        ? "色相=预测类别，明度=top1 与 top2 的置信差"
        : "已关闭边界着色，只画样本点",
  );
}

/* ============================================================= 几何模式 */

const GEO_COLORS = {
  point: "#f0abfc",
  guide: "#667eea",
  conic: "#60a5fa",
  locus: "#f472b6",
  poly: "#34d399",
  measure: "#fbbf24",
};

function drawGeoMode(p: P.Paper, s: GeoLabState, out: SceneOut): void {
  const doc = s.geo.doc;
  doc.recompute();
  const th = P.themeOf(p.dark);
  const labels = s.geo.showLabels;
  const ids = doc.ids();
  // 点永远画在曲线/直线之上，避免被后画的边遮住
  const order = [...ids.filter((id) => doc.get(id)?.kind !== "point"), ...ids.filter((id) => doc.get(id)?.kind === "point")];
  for (const id of order) {
    const g = doc.get(id);
    if (!g || g.visible === false) continue;
    const sel = s.geo.selected === id || s.geo.pending.includes(id);
    const w = sel ? 3.2 : 2;
    switch (g.kind) {
      case "polygon": {
        const pts = (g.pts ?? []).map((q) => P.vpScreen(p, q[0], q[1]));
        if (pts.length > 2) {
          const ctx = p.ctx;
          ctx.save();
          ctx.fillStyle = sel ? "rgba(102,126,234,0.2)" : "rgba(52,211,153,0.13)";
          ctx.beginPath();
          ctx.moveTo(pts[0][0], pts[0][1]);
          for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
          ctx.closePath();
          ctx.fill();
          ctx.restore();
          P.strokeChains(p, [[...pts, pts[0]]], { color: GEO_COLORS.poly, width: w });
        }
        break;
      }
      case "locus": {
        if (!s.geo.showTrace) break;
        const pts = (g.pts ?? []).map((q) => P.vpScreen(p, q[0], q[1]));
        P.strokeChains(p, P.splitChains(pts, p.h * 2), { color: GEO_COLORS.locus, width: sel ? 3 : 2.2, glow: true });
        break;
      }
      case "circle":
      case "ellipse":
      case "arc": {
        const pts = doc.samplePoints(id, 160).map((q) => P.vpScreen(p, q[0], q[1]));
        P.strokeChains(p, [pts], {
          color: sel ? th.accent : GEO_COLORS.conic,
          width: w,
          dashed: g.kind === "arc",
        });
        break;
      }
      case "segment":
      case "vector":
      case "line":
      case "ray": {
        const seg = screenSpan(p, doc, id);
        if (!seg) break;
        if (g.kind === "vector") {
          P.drawArrow(p, seg[0][0], seg[0][1], seg[1][0], seg[1][1], {
            color: sel ? th.accent : GEO_COLORS.guide,
            width: sel ? 3.2 : 2.6,
            label: labels ? g.label : undefined,
          });
        } else {
          P.strokeChains(p, [[P.vpScreen(p, seg[0][0], seg[0][1]), P.vpScreen(p, seg[1][0], seg[1][1])]], {
            color: sel ? th.accent : g.kind === "line" ? th.muted : th.text,
            width: w,
            alpha: g.kind === "line" ? 0.85 : 1,
          });
        }
        break;
      }
      case "angle": {
        const b = g.p1;
        const start = g.b;
        const sweep = g.a;
        if (b && isNum(start) && isNum(sweep)) {
          const r = 24;
          P.drawAngleMark(p, b[0], b[1], start, start + sweep, r, GEO_COLORS.measure);
          const mid = start + sweep / 2;
          const wm = 1 / p.vp.scale;
          P.annotate(p, b[0] + Math.cos(mid) * r * wm * 1.5, b[1] - Math.sin(mid) * r * wm * 1.5, g.text ?? "", {
            color: GEO_COLORS.measure,
            size: 12,
            align: "center",
            dx: 0,
            dy: 0,
          });
        }
        break;
      }
      case "text": {
        P.annotate(p, g.x, g.y, g.text ?? g.label, {
          color: GEO_COLORS.measure,
          size: 13,
          box: true,
          dx: 0,
          dy: 0,
        });
        break;
      }
      case "point": {
        P.drawPoint(p, g.x, g.y, {
          color: sel ? th.accent : GEO_COLORS.point,
          r: sel ? 6.5 : 5,
          label: labels ? g.label : undefined,
        });
        break;
      }
    }
  }
  /* 工具提示 */
  const hint = toolHint(s.geo.tool, s.geo.pending.length);
  if (hint) P.drawHint(p, hint);
  const need = NEEDS[s.geo.tool];
  if (need && s.geo.pending.length >= 1) {
    const pts = s.geo.pending.map((id) => doc.get(id)).filter(Boolean) as { x: number; y: number }[];
    if (pts.length >= 2) {
      const chains = pts.map((q) => P.vpScreen(p, q.x, q.y));
      P.strokeChains(p, [chains], { color: th.accent, width: 1.4, dashed: true, alpha: 0.8 });
    }
  }
  const nPt = ids.filter((i) => doc.get(i)?.kind === "point").length;
  out.info.push(`对象 ${ids.length}（点 ${nPt}）· 拖动点即改图形，Cmd/Ctrl+Z 撤销`);
}

/** 直线对象裁剪到视口，线段/向量取端点 */
function screenSpan(p: P.Paper, doc: GeometryDoc, id: string): [[number, number], [number, number]] | null {
  const e = doc.ends(id);
  if (!e) return null;
  const g = doc.get(id)!;
  if (g.kind !== "line" && g.kind !== "ray") return e;
  const o = e[0];
  const d: [number, number] = [e[1][0] - o[0], e[1][1] - o[1]];
  const { vp } = p;
  const t = clipToBox(o, d, vp.left, vp.right, vp.bottom, vp.top, g.kind === "ray" ? 0 : -Infinity);
  if (!t) return null;
  return [
    [o[0] + d[0] * t[0], o[1] + d[1] * t[0]],
    [o[0] + d[0] * t[1], o[1] + d[1] * t[1]],
  ];
}

/** Liang–Barsky 参数区间裁剪 */
function clipToBox(
  o: [number, number],
  d: [number, number],
  x0: number,
  x1: number,
  y0: number,
  y1: number,
  tFloor: number,
): [number, number] | null {
  let tmin = Math.max(-1e9, tFloor);
  let tmax = 1e9;
  const pp = [-d[0], d[0], -d[1], d[1]];
  const qq = [x0 - o[0], o[0] - x1, y0 - o[1], o[1] - y1];
  for (let i = 0; i < 4; i++) {
    if (Math.abs(pp[i]) < 1e-15) {
      if (qq[i] > 0) return null;
      continue;
    }
    const t = qq[i] / pp[i];
    if (pp[i] < 0) tmin = Math.max(tmin, t);
    else tmax = Math.min(tmax, t);
    if (tmin > tmax) return null;
  }
  return [tmin, tmax];
}

/* ================================================================= 入口 */

export function drawScene2D(p: P.Paper, s: GeoLabState, opt: SceneOpt = {}): SceneOut {
  const out: SceneOut = { errors: [], info: [] };
  P.begin(p, s.settings.dark);
  /* 复平面、神经网络画的是不透明栅格，坐标轴必须后画才不会被盖掉；
     线性映射的轴标恒为整数（π 刻度会把轴换成 π/2，与它画的整数格线对不上） */
  const frameFirst = s.mode !== "complex" && s.mode !== "nn" && s.mode !== "lin";
  if (frameFirst) {
    if (
      s.mode === "func" &&
      !p.vp.logX &&
      !p.vp.logY &&
      s.layers.every((l) => !l.visible || l.kind === "polar")
    )
      P.drawPolarFrame(p);
    else if (s.mode !== "geom" || s.geo.grid)
      P.drawFrame(p, {
        xPi: s.settings.piTicksX,
        yPi: s.settings.piTicksY,
        minor: s.settings.showMinorGrid,
      });
  }
  switch (s.mode) {
    case "func":
      drawFuncMode(p, s, out);
      break;
    case "complex":
      drawComplexMode(p, s, out, !!opt.draft, opt.raster);
      break;
    case "vector":
      P.drawFrame(p, { xPi: false, yPi: false, minor: s.settings.showMinorGrid });
      drawVectorMode(p, s, out);
      break;
    case "lin":
      P.drawFrame(p, { xPi: false, yPi: false, minor: s.settings.showMinorGrid });
      drawLinMode(p, s, out);
      break;
    case "nn":
      drawNnMode(p, s, out, !!opt.draft, opt.raster);
      break;
    case "geom":
      drawGeoMode(p, s, out);
      break;
    default:
      P.drawFrame(p, { minor: s.settings.showMinorGrid });
      break;
  }
  P.end(p);
  return out;
}

/** 悬停读数：返回若干行文本与可选的吸附点 */
export function probe(
  s: GeoLabState,
  x: number,
  y: number,
): { lines: string[]; snap: [number, number] | null } {
  const out: SceneOut = { errors: [], info: [] };
  try {
    if (s.mode === "func") {
      for (const l of s.layers) {
        if (!l.visible) continue;
        if (l.kind === "cartesian" || l.kind === "derivative" || l.kind === "integral") {
          const f = fn(s.engine, l.expr, 1, out);
          if (!f) continue;
          const yv = l.kind === "derivative" ? (f.at(x + 1e-7) - f.at(x - 1e-7)) / 2e-7 : f.at(x);
          if (!isNum(yv)) continue;
          return { lines: [`x = ${show3(x)}`, `${l.label || "y"} = ${show3(yv)}`], snap: [x, yv] };
        }
      }
    }
    if (s.mode === "complex") {
      const f = cfn(s.engine, s.cplx.f);
      if (f) {
        const v = f({ re: x, im: y });
        const w = isNum(v.re) && isNum(v.im) ? `${show3(v.re)}${v.im >= 0 ? " + " : " − "}${show3(Math.abs(v.im))}i` : "无定义";
        const mod = isNum(v.re) ? Math.hypot(v.re, v.im) : NaN;
        return {
          lines: [`z = ${show3(x)}${y >= 0 ? " + " : " − "}${show3(Math.abs(y))}i`, `f(z) = ${w}`, isNum(mod) ? `|f| = ${show3(mod)}` : ""].filter(Boolean),
          snap: null,
        };
      }
    }
    if (s.mode === "lin") {
      const lm = linMap(s);
      const dim = lm.a.length;
      /* 与画面同一张俯视图：3×3 时输入按 z=0 起，第三个分量单列出来 */
      const v = dim === 3 ? matVec(lm.m, [x, y, 0]) : matVec(topView(lm.m), [x, y]);
      const lines = [
        `(x, y) = (${show3(x)}, ${show3(y)})`,
        `${lm.tag} 映到 (${show3(v[0])}, ${show3(v[1])})`,
        `det A = ${show3(det(lm.a))} · tr A = ${show3(traceOf(lm.a))} · 秩 ${rref(lm.a).rank}/${dim}`,
        `λ = ${eigenSummary(lm.a)}`,
      ];
      if (dim === 3) lines.push(`俯视未画的 z 分量 = ${show3(v[2])}`);
      return { lines, snap: null };
    }
    if (s.mode === "nn" && s.nn.model) {
      const { label, margin } = predictMargin(s.nn.model, x, y, nnCacheOf(s.nn.model));
      return {
        lines: [`(x, y) = (${show3(x)}, ${show3(y)})`, `预测类别 ${label} · top1−top2 = ${show3(margin)}`],
        snap: null,
      };
    }
    if (s.mode === "vector") {
      const fu = fn(s.engine, s.vec.fx, 2, out);
      const fv = fn(s.engine, s.vec.fy, 2, out);
      if (fu && fv)
        return { lines: [`(x, y) = (${show3(x)}, ${show3(y)})`, `F = (${show3(fu.at(x, y))}, ${show3(fv.at(x, y))})`], snap: null };
      const df = fn(s.engine, s.vec.dfxy, 2, out);
      if (df) return { lines: [`(x, y) = (${show3(x)}, ${show3(y)})`, `y' = ${show3(df.at(x, y))}`], snap: null };
    }
    if (s.mode === "geom" && s.geo.doc) {
      const doc = s.geo.doc;
      const id = doc.hitTest(x, y, (s.geo.snap ? 0.4 : 0.25) / s.views.geom.scale * 12);
      if (id) {
        const g = doc.get(id)!;
        const lines = [`${g.label} · ${kindZh(g.kind)}`];
        if (g.kind === "point") lines.push(`(${show3(g.x)}, ${show3(g.y)})`);
        else {
          const L = doc.length(id);
          if (isNum(L)) lines.push(`长度 = ${show3(L)}`);
          if (g.kind === "polygon") lines.push(`面积 = ${show3(doc.polygonArea(id))}`);
        }
        const near = doc.get(s.geo.selected ?? "");
        if (near && near.kind === "point" && g.kind === "point") lines.push(`距离 = ${show3(doc.dist(near.id, id))}`);
        return { lines, snap: null };
      }
    }
  } catch {
    /* 读数失败不影响主画面 */
  }
  return { lines: [`(${show3(x)}, ${show3(y)})`], snap: null };
}

function show3(v: number): string {
  if (!isNum(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1e6 || (a > 0 && a < 1e-4)) return v.toExponential(3);
  return String(Number(v.toFixed(4)));
}

export function kindZh(k: string): string {
  const m: Record<string, string> = {
    point: "点",
    segment: "线段",
    line: "直线",
    ray: "射线",
    vector: "向量",
    circle: "圆",
    arc: "圆弧",
    ellipse: "椭圆",
    polygon: "多边形",
    locus: "轨迹",
    angle: "角",
    text: "标注",
  };
  return m[k] ?? k;
}

/* ========================================================= 几何工具状态机 */

/** 工具需要的点击次数（点数或对象数），0 = 单次点击完成 */
export const NEEDS: Partial<Record<ToolKind, number>> = {
  point: 1,
  segment: 2,
  line: 2,
  ray: 2,
  vector: 2,
  circle: 2,
  arc: 3,
  ellipse: 3,
  midpoint: 1,
  perpendicular: 2,
  parallel: 2,
  bisector: 3,
  intersection: 2,
  pointOn: 1,
  locus: 2,
  rotate: 1,
  reflect: 2,
  dilate: 1,
  angle: 3,
  area: 1,
  text: 1,
  erase: 1,
};

const HINTS: Partial<Record<ToolKind, string>> = {
  point: "点击放置点",
  segment: "点击两点作线段",
  line: "点击两点作直线",
  ray: "点击两点作射线",
  vector: "点击两点作向量",
  circle: "点击圆心与圆上一点",
  arc: "点击圆心、起点、终点",
  ellipse: "点击两焦点与椭圆上一点",
  midpoint: "点击线段取中点",
  perpendicular: "先点直线（或两点），再点过的那个点",
  parallel: "先点直线（或两点），再点过的那个点",
  bisector: "点击三点作角平分线（顶点在中间）",
  intersection: "点击两个相交对象",
  pointOn: "点击曲线上一点（自动吸附约束）",
  locus: "先点轨迹上的动点（需在曲线上），再点被追踪的对象",
  rotate: "点击绕原点旋转 90° 的点",
  reflect: "点击点与对称轴（两点确定）",
  dilate: "点击以原点为中心放大 2 倍的点",
  angle: "点击三点测量夹角（顶点在中间）",
  area: "点击多边形显示面积",
  text: "点击放置坐标标注",
  erase: "点击对象删除",
  select: "拖动点改变图形，点击选中查看测量",
};

function toolHint(tool: ToolKind, have: number): string {
  const need = NEEDS[tool] ?? 0;
  const base = HINTS[tool] ?? "";
  return need > 1 ? `${base} · 已选 ${have}/${need}` : base;
}

let letterIdx = 0;
function nextLabel(doc: GeometryDoc): string {
  const used = new Set(doc.all().map((g) => g.label));
  for (let i = 0; i < 26 * 4; i++) {
    const n = (letterIdx + i) % (26 * 4);
    const s = n < 26 ? String.fromCharCode(65 + n) : `${String.fromCharCode(65 + (n % 26))}${Math.floor(n / 26)}`;
    if (!used.has(s)) {
      letterIdx = n + 1;
      return s;
    }
  }
  return `P${doc.ids().length + 1}`;
}

export type ApplyResult = { pending: string[]; done: boolean; error?: string };

/**
 * 执行一次工具点击：picked 为命中的已有对象（可能为 null），world 为光标世界坐标。
 * 返回新的待选队列；done 表示该构造已完成。
 */
export function applyGeoTool(
  doc: GeometryDoc,
  tool: ToolKind,
  pending: string[],
  picked: string | null,
  world: [number, number],
): ApplyResult {
  const need = NEEDS[tool] ?? 1;
  /** 取一个点：命中已有点直接用，否则在点击处新建 */
  const pointOf = (id: string | null, at: [number, number]): string => {
    if (id) {
      const g = doc.get(id);
      if (g && g.kind === "point") return id;
      if (g && (g.kind === "segment" || g.kind === "vector")) {
        // 两点型对象直接作为两点使用
        return id;
      }
    }
    return doc.addPoint(nextLabel(doc), at[0], at[1]);
  };

  if (tool === "select" || tool === "text") {
    if (tool === "text") {
      doc.addText(world[0], world[1], `(${world[0].toFixed(2)}, ${world[1].toFixed(2)})`, "");
      return { pending: [], done: true };
    }
    return { pending: [], done: true };
  }

  if (tool === "erase") {
    if (picked) doc.remove(picked);
    return { pending: [], done: true };
  }

  if (need <= 1) {
    if (!picked) {
      if (tool === "point") {
        doc.addPoint(nextLabel(doc), world[0], world[1]);
        return { pending: [], done: true };
      }
      return { pending: [], done: false, error: `${toolZh(tool)}需要先点击一个已有对象` };
    }
    const g = doc.get(picked)!;
    try {
      switch (tool) {
        case "midpoint": {
          const seg = g.kind === "segment" ? picked : doc.addSegment(picked, picked);
          if (g.kind !== "segment") {
            // 两点选中时才走上面的分支；这里退化为在该点上作自由中点
            doc.remove(seg);
            return { pending: [], done: false, error: "中点工具请点击线段" };
          }
          doc.addMidpoint(nextLabel(doc), seg);
          break;
        }
        case "area": {
          if (g.kind !== "polygon" && g.kind !== "ellipse") return { pending: [], done: false, error: "面积工具请点击多边形或椭圆" };
          doc.addMeasure(g.kind === "polygon" ? "area" : "area", [picked], [g.x, g.y], "S");
          break;
        }
        case "pointOn": {
          const t = doc.paramAt(picked, world);
          if (t === null) return { pending: [], done: false, error: "该对象不能作为宿主曲线" };
          doc.addPointOn(nextLabel(doc), picked, t);
          break;
        }
        case "rotate": {
          if (g.kind !== "point") return { pending: [], done: false, error: "旋转工具请点击点" };
          doc.addRotate(`${g.label}′`, picked, [0, 0], Math.PI / 2);
          break;
        }
        case "dilate": {
          if (g.kind !== "point") return { pending: [], done: false, error: "缩放工具请点击点" };
          doc.addDilate(`${g.label}″`, picked, [0, 0], 2);
          break;
        }
        default:
          return { pending: [], done: false };
      }
    } catch (e) {
      return { pending: [], done: false, error: msg(e) };
    }
    return { pending: [], done: true };
  }

  /* 多步构造：累积待选对象 */
  const next = pending.slice();
  if (tool === "intersection") {
    if (!picked) return { pending: next, done: false, error: "请点击要求交的对象" };
    next.push(picked);
    if (next.length === 2) {
      const id = doc.addIntersection(nextLabel(doc), next[0], next[1], 0);
      if (!id) return { pending: [], done: false, error: "两对象在当前视野内不相交" };
      return { pending: [], done: true };
    }
    return { pending: next, done: false };
  }

  if (tool === "locus") {
    if (!picked) return { pending: next, done: false, error: "请点击驱动点或被追踪对象" };
    next.push(picked);
    if (next.length === 2) {
      const d = doc.get(next[0])!;
      const t = doc.get(next[1])!;
      if (d.kind !== "point") return { pending: [], done: false, error: "轨迹的第一个对象必须是（曲线上的）点" };
      if (!doc.canDrive(next[0]))
        return { pending: [], done: false, error: "驱动点是自由的，无法取样轨迹 —— 先用「曲线上的点」把它约束到某条曲线上" };
      const target = t.kind === "point" ? next[1] : (t.p1 && doc.hitTest(t.p1[0], t.p1[1], 1e-6)) || next[1];
      doc.addLocus(`轨迹_${d.label}${t.kind === "point" ? t.label : ""}`, next[0], target);
      return { pending: [], done: true };
    }
    return { pending: next, done: false };
  }

  if (tool === "perpendicular" || tool === "parallel") {
    if (picked && (doc.get(picked)?.kind === "segment" || doc.get(picked)?.kind === "line" || doc.get(picked)?.kind === "ray" || doc.get(picked)?.kind === "vector")) {
      next.push(picked);
    } else {
      next.push(pointOf(picked, world));
    }
    if (next.length < 2) return { pending: next, done: false };
    const [a, b] = next;
    const lineId = isLineLike(doc, a) ? a : isLineLike(doc, b) ? b : null;
    const ptId = lineId === a ? b : a;
    if (!lineId) return { pending: [], done: false, error: "需要一个直线型对象与一个点" };
    const gid = tool === "perpendicular" ? doc.addPerpendicular(ptId, lineId) : doc.addParallel(ptId, lineId);
    doc.setLabel(gid, `${doc.get(ptId)?.label ?? ""}${tool === "perpendicular" ? " 的垂线" : " 的平行线"}`);
    return { pending: [], done: true };
  }

  /* 其余都是「点序列」型工具 */
  if (tool === "polygon") {
    // 闭合条件：再次点到已在队列中的对象
    if (picked && pending.includes(picked) && pending.length >= 3) {
      doc.addPolygon(pending, "多边形");
      return { pending: [], done: true };
    }
    next.push(pointOf(picked, world));
    return { pending: next, done: false };
  }
  next.push(pointOf(picked, world));
  if (next.length < need) return { pending: next, done: false };
  try {
    switch (tool) {
      case "segment":
        doc.addSegment(next[0], next[1]);
        break;
      case "line":
        doc.addLine(next[0], next[1]);
        break;
      case "ray":
        doc.addRay(next[0], next[1]);
        break;
      case "vector":
        doc.addVector(nextLabel(doc), next[0], next[1]);
        break;
      case "circle":
        doc.addCircle(next[0], next[1]);
        break;
      case "arc":
        doc.addArc(nextLabel(doc), next[0], next[1], next[2]);
        break;
      case "ellipse": {
        doc.addEllipseByFoci(nextLabel(doc), next[0], next[1], next[2]);
        break;
      }
      case "bisector":
        doc.addBisector(next[0], next[1], next[2]);
        break;
      case "angle":
        doc.addAngle(next[0], next[1], next[2], `∠${doc.get(next[1])?.label ?? ""}`);
        break;
      case "reflect": {
        const a = next[0];
        const b = next[1];
        const ln = doc.addLine(a, b);
        doc.addReflect(`${doc.get(next[0])?.label}′`, next[0], ln);
        break;
      }
      default:
        break;
    }
  } catch (e) {
    return { pending: [], done: false, error: msg(e) };
  }
  return { pending: [], done: true };
}

function isLineLike(doc: GeometryDoc, id: string): boolean {
  const k = doc.get(id)?.kind;
  return k === "segment" || k === "line" || k === "ray" || k === "vector" || k === "arc" || k === "circle" || k === "ellipse";
}

export function toolZh(t: ToolKind): string {
  const m: Partial<Record<ToolKind, string>> = {
    select: "选择",
    point: "点",
    segment: "线段",
    line: "直线",
    ray: "射线",
    vector: "向量",
    circle: "圆",
    arc: "圆弧",
    ellipse: "椭圆",
    polygon: "多边形",
    midpoint: "中点",
    perpendicular: "垂线",
    parallel: "平行线",
    bisector: "角平分线",
    intersection: "交点",
    pointOn: "曲线上的点",
    locus: "轨迹",
    rotate: "旋转",
    reflect: "反射",
    dilate: "缩放",
    angle: "角",
    area: "面积",
    text: "标注",
    erase: "删除",
  };
  return m[t] ?? t;
}
