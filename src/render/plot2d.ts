/**
 * 2D 画布绘制层：坐标系、网格、曲线、向量场、隐函数、不等式着色、几何图形
 * 所有几何量使用世界坐标，仅在最内层转换为屏幕坐标。
 */

import type { Viewport, Ticks } from "../core/view.ts";
import { niceTicks } from "../core/view.ts";
import type { Seg } from "../core/contour.ts";

export interface Paper {
  ctx: CanvasRenderingContext2D;
  vp: Viewport;
  w: number;
  h: number;
  dark: boolean;
}

export interface Theme {
  bg: string;
  grid: string;
  gridMinor: string;
  axis: string;
  tick: string;
  text: string;
  muted: string;
  accent: string;
  selection: string;
}

export const DARK: Theme = {
  bg: "#0d1020",
  grid: "rgba(148,163,224,0.16)",
  gridMinor: "rgba(148,163,224,0.07)",
  axis: "rgba(203,213,255,0.75)",
  tick: "rgba(190,200,240,0.85)",
  text: "rgba(226,232,255,0.92)",
  muted: "rgba(160,170,210,0.6)",
  accent: "#8b7cf6",
  selection: "rgba(139,124,246,0.35)",
};

export const LIGHT: Theme = {
  bg: "#fbfbfe",
  grid: "rgba(60,70,120,0.16)",
  gridMinor: "rgba(60,70,120,0.07)",
  axis: "rgba(30,40,80,0.8)",
  tick: "rgba(40,50,90,0.85)",
  text: "rgba(20,25,50,0.92)",
  muted: "rgba(70,80,120,0.6)",
  accent: "#6d5ae0",
  selection: "rgba(109,90,224,0.28)",
};

export function themeOf(dark: boolean): Theme {
  return dark ? DARK : LIGHT;
}

export function begin(p: Paper, dark: boolean): void {
  const th = themeOf(dark);
  p.ctx.save();
  p.ctx.fillStyle = th.bg;
  p.ctx.fillRect(0, 0, p.w, p.h);
}
export function end(p: Paper): void {
  p.ctx.restore();
}

/** 网格 + 坐标轴 + 刻度；返回使用的刻度供标注复用 */
export function drawFrame(
  p: Paper,
  opts: { xPi?: boolean; yPi?: boolean; minor?: boolean; axes?: boolean; labels?: boolean } = {},
): { xt: Ticks; yt: Ticks } {
  const th = themeOf(p.dark);
  const { vp, ctx, w, h } = p;
  const xt = niceTicks(vp.left, vp.right, Math.max(4, Math.round(w / 110)), opts.xPi ? "pi" : "auto");
  const yt = niceTicks(vp.bottom, vp.top, Math.max(4, Math.round(h / 80)), opts.yPi ? "pi" : "auto");
  const [ox, oy] = vp.toScreen(0, 0);

  ctx.lineWidth = 1;
  if (opts.minor !== false) {
    ctx.strokeStyle = th.gridMinor;
    ctx.beginPath();
    for (const v of xt.minor) {
      const [sx] = vp.toScreen(v, 0);
      ctx.moveTo(Math.round(sx) + 0.5, 0);
      ctx.lineTo(Math.round(sx) + 0.5, h);
    }
    for (const v of yt.minor) {
      const [, sy] = vp.toScreen(0, v);
      ctx.moveTo(0, Math.round(sy) + 0.5);
      ctx.lineTo(w, Math.round(sy) + 0.5);
    }
    ctx.stroke();
  }
  ctx.strokeStyle = th.grid;
  ctx.beginPath();
  for (const v of xt.values) {
    const [sx] = vp.toScreen(v, 0);
    ctx.moveTo(Math.round(sx) + 0.5, 0);
    ctx.lineTo(Math.round(sx) + 0.5, h);
  }
  for (const v of yt.values) {
    const [, sy] = vp.toScreen(0, v);
    ctx.moveTo(0, Math.round(sy) + 0.5);
    ctx.lineTo(w, Math.round(sy) + 0.5);
  }
  ctx.stroke();

  if (opts.axes !== false) {
    ctx.strokeStyle = th.axis;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    if (oy >= 0 && oy <= h) {
      ctx.moveTo(0, Math.round(oy) + 0.5);
      ctx.lineTo(w, Math.round(oy) + 0.5);
    }
    if (ox >= 0 && ox <= w) {
      ctx.moveTo(Math.round(ox) + 0.5, 0);
      ctx.lineTo(Math.round(ox) + 0.5, h);
    }
    ctx.stroke();
    // 轴箭头
    ctx.fillStyle = th.axis;
    if (oy >= 0 && oy <= h) {
      ctx.beginPath();
      ctx.moveTo(w - 1, oy);
      ctx.lineTo(w - 10, oy - 4.5);
      ctx.lineTo(w - 10, oy + 4.5);
      ctx.fill();
    }
    if (ox >= 0 && ox <= w) {
      ctx.beginPath();
      ctx.moveTo(ox, 1);
      ctx.lineTo(ox - 4.5, 10);
      ctx.lineTo(ox + 4.5, 10);
      ctx.fill();
    }
  }

  if (opts.labels !== false) {
    ctx.fillStyle = th.tick;
    ctx.font = "12px 'SF Mono', Menlo, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    const labelY = Math.min(h - 16, Math.max(2, oy + 5));
    for (let k = 0; k < xt.values.length; k++) {
      const v = xt.values[k];
      if (Math.abs(v) < xt.step * 1e-6) continue;
      const [sx] = vp.toScreen(v, 0);
      if (sx < 12 || sx > w - 12) continue;
      ctx.fillText(xt.labels[k], sx, labelY);
    }
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    const labelX = Math.min(w - 6, Math.max(26, ox - 7));
    for (let k = 0; k < yt.values.length; k++) {
      const v = yt.values[k];
      if (Math.abs(v) < yt.step * 1e-6) continue;
      const [, sy] = vp.toScreen(0, v);
      if (sy < 10 || sy > h - 10) continue;
      ctx.fillText(yt.labels[k], labelX, sy);
    }
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText("O", Math.min(w - 12, Math.max(2, ox + 4)), Math.min(h - 14, Math.max(2, oy + 4)));
  }
  return { xt, yt };
}

/** 极坐标网格：同心圆 + 射线 */
export function drawPolarFrame(p: Paper, rings = 8, spokes = 24): void {
  const th = themeOf(p.dark);
  const { vp, ctx, w, h } = p;
  const [ox, oy] = vp.toScreen(0, 0);
  const rMax = Math.max(w, h) / vp.scale;
  ctx.strokeStyle = th.gridMinor;
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let k = 1; k <= rings; k++) {
    const r = (rMax * k) / rings;
    ctx.moveTo(ox + r * vp.scale, oy);
    ctx.arc(ox, oy, r * vp.scale, 0, Math.PI * 2);
  }
  for (let k = 0; k < spokes; k++) {
    const a = (Math.PI * 2 * k) / spokes;
    ctx.moveTo(ox, oy);
    ctx.lineTo(ox + Math.cos(a) * rMax * vp.scale, oy - Math.sin(a) * rMax * vp.scale);
  }
  ctx.stroke();
}

export interface StrokeStyle {
  color: string;
  width?: number;
  dashed?: boolean;
  alpha?: number;
  glow?: boolean;
}

export function strokePolyline(p: Paper, pts: number[][], st: StrokeStyle): void {
  const { ctx } = p;
  if (pts.length < 2) return;
  ctx.save();
  ctx.globalAlpha = st.alpha ?? 1;
  ctx.strokeStyle = st.color;
  ctx.lineWidth = st.width ?? 2;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.setLineDash(st.dashed ? [7, 5] : []);
  if (st.glow) {
    ctx.shadowColor = st.color;
    ctx.shadowBlur = 8;
  }
  ctx.beginPath();
  for (let i = 0; i < pts.length; i++) {
    const [x, y] = pts[i];
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.restore();
}

/** 分段折线（自动在断点处提笔） */
export function strokeChains(p: Paper, chains: number[][][], st: StrokeStyle): void {
  const { ctx } = p;
  ctx.save();
  ctx.globalAlpha = st.alpha ?? 1;
  ctx.strokeStyle = st.color;
  ctx.lineWidth = st.width ?? 2;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.setLineDash(st.dashed ? [7, 5] : []);
  if (st.glow) {
    ctx.shadowColor = st.color;
    ctx.shadowBlur = 8;
  }
  for (const pts of chains) {
    if (pts.length < 2) continue;
    ctx.beginPath();
    for (let i = 0; i < pts.length; i++) {
      if (i === 0) ctx.moveTo(pts[i][0], pts[i][1]);
      else ctx.lineTo(pts[i][0], pts[i][1]);
    }
    ctx.stroke();
  }
  ctx.restore();
}

/**
 * 直角坐标曲线采样：自适应加密 + 垂直渐近线断笔。
 * 判定规则：屏幕纵坐标跨越整个画布高度两倍以上，且两侧符号相反 → 视为间断。
 */
export function plotCartesian(
  p: Paper,
  f: (x: number) => number,
  st: StrokeStyle & { samples?: number; fillTo?: number; domain?: [number, number]; stepped?: boolean },
): void {
  const { vp, ctx, w, h } = p;
  const [a, b] = st.domain ?? [vp.left, vp.right];
  const n = Math.max(60, Math.round(st.samples ?? Math.max(w * 1.5, 600)));
  const pts: number[][] = [];
  const yLimit = h * 6;
  for (let i = 0; i <= n; i++) {
    const x = a + ((b - a) * i) / n;
    let y = NaN;
    try {
      y = f(x);
    } catch {
      y = NaN;
    }
    if (!Number.isFinite(y)) {
      pts.push([NaN, NaN]);
      continue;
    }
    const [sx, sy] = vp.toScreen(x, y);
    pts.push([sx, Math.max(-yLimit, Math.min(yLimit, sy))]);
  }
  const chains = splitChains(pts, h * 2.2);
  strokeChains(p, chains, st);
  if (st.fillTo !== undefined) {
    const base = vp.toScreen(0, st.fillTo)[1];
    ctx.save();
    ctx.globalAlpha = 0.2;
    ctx.fillStyle = st.color;
    for (const ch of chains) {
      if (ch.length < 2) continue;
      ctx.beginPath();
      ctx.moveTo(ch[0][0], base);
      for (const [x, y] of ch) ctx.lineTo(x, y);
      ctx.lineTo(ch[ch.length - 1][0], base);
      ctx.closePath();
      ctx.fill();
    }
    ctx.restore();
  }
}

/**
 * 把含 NaN 的屏幕点列拆段，并在“跳变穿过渐近线”处断开。
 * x 恒递增，故 dy>dj 只可能来自竖直跳变：tan、1/x 这类极点两侧的连接。
 */
export function splitChains(pts: number[][], dj: number): number[][][] {
  const chains: number[][][] = [];
  let cur: number[][] = [];
  for (let i = 0; i < pts.length; i++) {
    const q = pts[i];
    if (!Number.isFinite(q[0]) || !Number.isFinite(q[1])) {
      if (cur.length > 1) chains.push(cur);
      cur = [];
      continue;
    }
    if (cur.length) {
      const prev = cur[cur.length - 1];
      if (Math.abs(q[1] - prev[1]) > dj) {
        if (cur.length > 1) chains.push(cur);
        // 保留端点让断口留在屏幕外，视觉上不出现悬空
        cur = [prev];
      }
    }
    cur.push(q);
  }
  if (cur.length > 1) chains.push(cur);
  return chains;
}

/** 极坐标 r = f(theta) */
export function plotPolar(
  p: Paper,
  rf: (t: number) => number,
  t0: number,
  t1: number,
  st: StrokeStyle & { samples?: number },
): void {
  const n = Math.max(120, st.samples ?? 1600);
  const pts: number[][] = [];
  for (let i = 0; i <= n; i++) {
    const t = t0 + ((t1 - t0) * i) / n;
    let r = NaN;
    try {
      r = rf(t);
    } catch {
      r = NaN;
    }
    if (!Number.isFinite(r)) {
      pts.push([NaN, NaN]);
      continue;
    }
    const [sx, sy] = vpScreen(p, r * Math.cos(t), r * Math.sin(t));
    pts.push([sx, sy]);
  }
  strokeChains(p, splitChains(pts, p.h * 3), st);
}

/** 参数曲线 (x(t), y(t)) */
export function plotParam(
  p: Paper,
  xf: (t: number) => number,
  yf: (t: number) => number,
  t0: number,
  t1: number,
  st: StrokeStyle & { samples?: number },
): void {
  const n = Math.max(120, st.samples ?? 1600);
  const pts: number[][] = [];
  for (let i = 0; i <= n; i++) {
    const t = t0 + ((t1 - t0) * i) / n;
    let x = NaN, y = NaN;
    try {
      x = xf(t);
      y = yf(t);
    } catch {
      /* 忽略奇异点 */
    }
    if (!Number.isFinite(x) || !Number.isFinite(y)) {
      pts.push([NaN, NaN]);
      continue;
    }
    const [sx, sy] = vpScreen(p, x, y);
    pts.push([sx, sy]);
  }
  strokeChains(p, splitChains(pts, p.h * 3), st);
}

/** 三维空间曲线的二维投影入口由 scene3d 负责；这里绘制离散点序列 */
export function drawSteps(p: Paper, xs: number[], ys: number[], st: StrokeStyle & { barWidth?: number }): void {
  const { ctx } = p;
  ctx.save();
  ctx.strokeStyle = st.color;
  ctx.fillStyle = st.color;
  ctx.lineWidth = 1.5;
  const base = vpScreen(p, 0, 0)[1];
  for (let i = 0; i < xs.length; i++) {
    const [sx, sy] = vpScreen(p, xs[i], ys[i]);
    if (!Number.isFinite(sy)) continue;
    ctx.globalAlpha = 0.45;
    ctx.beginPath();
    ctx.moveTo(sx, base);
    ctx.lineTo(sx, sy);
    ctx.stroke();
    ctx.globalAlpha = 1;
    ctx.beginPath();
    ctx.arc(sx, sy, st.barWidth ?? 3.5, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

export function vpScreen(p: Paper, x: number, y: number): [number, number] {
  return p.vp.toScreen(x, y);
}

/** 隐函数 F(x,y)=c：marching squares 线段集合 */
export function drawSegments(p: Paper, segs: Seg[], st: StrokeStyle): void {
  const { ctx } = p;
  ctx.save();
  ctx.strokeStyle = st.color;
  ctx.lineWidth = st.width ?? 2;
  ctx.lineCap = "round";
  ctx.setLineDash(st.dashed ? [7, 5] : []);
  ctx.beginPath();
  for (const s of segs) {
    const [x1, y1] = vpScreen(p, s[0], s[1]);
    const [x2, y2] = vpScreen(p, s[2], s[3]);
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
  }
  ctx.stroke();
  ctx.restore();
}

/** 焊接后的隐函数折线，线转折更平滑 */
export function drawChains(p: Paper, chains: Seg[][], st: StrokeStyle): void {
  const pts: number[][][] = [];
  for (const ch of chains) {
    const one: number[][] = [];
    for (let i = 0; i < ch.length; i++) {
      const s = ch[i];
      if (i === 0) one.push(vpScreen(p, s[0], s[1]));
      one.push(vpScreen(p, s[2], s[3]));
    }
    pts.push(one);
  }
  strokeChains(p, pts, st);
}

/** 不等式区域着色：低分辨率 mask + 双线性放大 */
export function fillMask(p: Paper, mask: Uint8Array, w: number, h: number, color: string, alpha = 0.18): void {
  const { ctx } = p;
  const off = getScratch(w, h);
  if (!off) return;
  const img = off.ctx.createImageData(w, h);
  const rgb = parseColor(color);
  for (let i = 0; i < w * h; i++) {
    const on = mask[i] === 1;
    img.data[i * 4] = rgb[0];
    img.data[i * 4 + 1] = rgb[1];
    img.data[i * 4 + 2] = rgb[2];
    img.data[i * 4 + 3] = on ? Math.round(alpha * 255) : 0;
  }
  off.ctx.putImageData(img, 0, 0);
  ctx.save();
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(off.canvas, 0, 0, p.w, p.h);
  ctx.restore();
}

const scratchCache = new Map<string, { canvas: HTMLCanvasElement; ctx: CanvasRenderingContext2D }>();
function getScratch(w: number, h: number) {
  if (typeof document === "undefined") return null;
  const key = `${w}x${h}`;
  let s = scratchCache.get(key);
  if (!s) {
    const canvas = document.createElement("canvas");
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    s = { canvas, ctx };
    scratchCache.set(key, s);
  }
  return s;
}

/** 直接以 ImageData 铺设栅格图（复平面着色、标量场热图） */
export function blitPixels(
  p: Paper,
  rgb: Uint8ClampedArray | Uint8Array,
  w: number,
  h: number,
  world: [number, number, number, number],
  alpha = 1,
  smooth = false,
): void {
  const { ctx } = p;
  const off = getScratch(w, h);
  if (!off) return;
  const img = off.ctx.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    img.data[i * 4] = rgb[i * 3];
    img.data[i * 4 + 1] = rgb[i * 3 + 1];
    img.data[i * 4 + 2] = rgb[i * 3 + 2];
    img.data[i * 4 + 3] = 255;
  }
  off.ctx.putImageData(img, 0, 0);
  const [x0, x1, y0, y1] = world;
  const [sx0, sy1] = vpScreen(p, x0, y0);
  const [sx1, sy0] = vpScreen(p, x1, y1);
  ctx.save();
  ctx.globalAlpha = alpha;
  ctx.imageSmoothingEnabled = smooth;
  ctx.drawImage(off.canvas, sx0, sy0, sx1 - sx0, sy1 - sy0);
  ctx.restore();
}

function parseColor(css: string): [number, number, number] {
  const m = /^#([0-9a-f]{6})$/i.exec(css.trim());
  if (m) {
    const v = parseInt(m[1], 16);
    return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
  }
  const rgbm = /rgba?\(([^)]+)\)/i.exec(css);
  if (rgbm) {
    const [r, g, b] = rgbm[1].split(",").map((q) => parseFloat(q));
    return [r | 0, g | 0, b | 0];
  }
  return [128, 128, 200];
}

export function drawPoint(
  p: Paper,
  x: number,
  y: number,
  o: { color?: string; r?: number; label?: string; hollow?: boolean; halo?: boolean; labelDx?: number } = {},
): void {
  const { ctx } = p;
  const th = themeOf(p.dark);
  const [sx, sy] = vpScreen(p, x, y);
  const r = o.r ?? 4.5;
  ctx.save();
  if (o.halo !== false) {
    ctx.fillStyle = p.dark ? "rgba(13,16,32,0.85)" : "rgba(255,255,255,0.9)";
    ctx.beginPath();
    ctx.arc(sx, sy, r + 2.2, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.fillStyle = o.color ?? th.accent;
  ctx.strokeStyle = o.color ?? th.accent;
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  ctx.arc(sx, sy, r, 0, Math.PI * 2);
  if (o.hollow) ctx.stroke();
  else ctx.fill();
  if (o.label) {
    ctx.fillStyle = th.text;
    ctx.font = "600 12.5px 'SF Mono', Menlo, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "bottom";
    ctx.fillText(o.label, sx + (o.labelDx ?? 8), sy - 6);
  }
  ctx.restore();
}

export function drawArrow(
  p: Paper,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  o: Partial<StrokeStyle> & { head?: number; label?: string } = {},
): void {
  const { ctx } = p;
  const th = themeOf(p.dark);
  const [ax, ay] = vpScreen(p, x0, y0);
  const [bx, by] = vpScreen(p, x1, y1);
  const dx = bx - ax, dy = by - ay;
  const len = Math.hypot(dx, dy);
  if (len < 0.5) return;
  const head = Math.min(14, Math.max(7, len * 0.16)) * (o.head ?? 1);
  const ux = dx / len, uy = dy / len;
  ctx.save();
  ctx.strokeStyle = o.color ?? th.accent;
  ctx.fillStyle = o.color ?? th.accent;
  ctx.lineWidth = o.width ?? 2.2;
  ctx.lineCap = "round";
  ctx.setLineDash(o.dashed ? [6, 4] : []);
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(bx - ux * head * 0.85, by - uy * head * 0.85);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.moveTo(bx, by);
  ctx.lineTo(bx - ux * head + uy * head * 0.38, by - uy * head + ux * head * 0.38);
  ctx.lineTo(bx - ux * head - uy * head * 0.38, by - uy * head - ux * head * 0.38);
  ctx.closePath();
  ctx.fill();
  if (o.label) {
    ctx.font = "600 12.5px 'SF Mono', Menlo, system-ui, sans-serif";
    ctx.fillStyle = th.text;
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    ctx.fillText(o.label, (ax + bx) / 2 + uy * 12, (ay + by) / 2 - 4);
  }
  ctx.restore();
}

/** 向量场箭头 */
export function drawQuiver(
  p: Paper,
  items: { x: number; y: number; u: number; v: number; mag?: number }[],
  o: { color?: string; scale: number; width?: number; colormapFn?: (t: number) => string },
): void {
  let max = 1e-9;
  for (const q of items) max = Math.max(max, Math.hypot(q.u, q.v));
  for (const q of items) {
    const m = Math.hypot(q.u, q.v);
    if (!Number.isFinite(m) || m === 0) continue;
    drawArrow(p, q.x, q.y, q.x + q.u * o.scale, q.y + q.v * o.scale, {
      color: o.colormapFn ? o.colormapFn(Math.min(1, m / max)) : o.color ?? "#8b7cf6",
      width: o.width ?? 1.6,
    });
  }
}

/** 方向场（dy/dx = f(x,y)）：短斜率段 */
export function drawSlopeField(
  p: Paper,
  items: { x: number; y: number; dx: number; dy: number }[],
  o: { color?: string; len: number; alpha?: number },
): void {
  const { ctx } = p;
  const th = themeOf(p.dark);
  ctx.save();
  ctx.strokeStyle = o.color ?? th.muted;
  ctx.lineWidth = 1.3;
  ctx.globalAlpha = o.alpha ?? 0.9;
  ctx.beginPath();
  for (const s of items) {
    const m = Math.hypot(s.dx, s.dy);
    if (!Number.isFinite(m) || m === 0) continue;
    const hx = (s.dx / m) * o.len * 0.5;
    const hy = (s.dy / m) * o.len * 0.5;
    const [ax, ay] = vpScreen(p, s.x - hx, s.y - hy);
    const [bx, by] = vpScreen(p, s.x + hx, s.y + hy);
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
  }
  ctx.stroke();
  ctx.restore();
}

/** 相图 / 流线 */
export function drawStreams(
  p: Paper,
  lines: number[][],
  o: StrokeStyle & { arrows?: boolean },
): void {
  const { ctx } = p;
  ctx.save();
  ctx.strokeStyle = o.color;
  ctx.globalAlpha = o.alpha ?? 0.95;
  ctx.lineWidth = o.width ?? 1.8;
  ctx.lineCap = "round";
  for (const ln of lines) {
    if (ln.length < 4) continue;
    ctx.beginPath();
    for (let i = 0; i + 1 < ln.length; i += 2) {
      const [sx, sy] = vpScreen(p, ln[i], ln[i + 1]);
      if (i === 0) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    }
    ctx.stroke();
    if (o.arrows !== false) {
      const mid = Math.floor(ln.length / 4) * 2;
      if (mid + 3 < ln.length) {
        const [ax, ay] = vpScreen(p, ln[mid], ln[mid + 1]);
        const [bx, by] = vpScreen(p, ln[mid + 2], ln[mid + 3]);
        const a = Math.atan2(by - ay, bx - ax);
        ctx.fillStyle = o.color;
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.lineTo(bx - Math.cos(a - 0.42) * 8, by - Math.sin(a - 0.42) * 8);
        ctx.lineTo(bx - Math.cos(a + 0.42) * 8, by - Math.sin(a + 0.42) * 8);
        ctx.closePath();
        ctx.fill();
      }
    }
  }
  ctx.restore();
}

/** 文本标注（世界坐标定位，可带背景框） */
export function annotate(
  p: Paper,
  x: number,
  y: number,
  text: string,
  o: { color?: string; size?: number; box?: boolean; dx?: number; dy?: number; align?: CanvasTextAlign } = {},
): void {
  const { ctx } = p;
  const th = themeOf(p.dark);
  const [sx, sy] = vpScreen(p, x, y);
  ctx.save();
  ctx.font = `${o.size ?? 12.5}px 'SF Mono', Menlo, system-ui, sans-serif`;
  ctx.textAlign = o.align ?? "left";
  ctx.textBaseline = "middle";
  const tx = sx + (o.dx ?? 8);
  const ty = sy + (o.dy ?? -10);
  if (o.box) {
    const m = ctx.measureText(text);
    ctx.fillStyle = p.dark ? "rgba(15,18,38,0.82)" : "rgba(255,255,255,0.86)";
    ctx.strokeStyle = p.dark ? "rgba(140,150,220,0.3)" : "rgba(60,70,120,0.25)";
    ctx.lineWidth = 1;
    const padX = 5;
    const rx = o.align === "right" ? tx - m.width - padX : o.align === "center" ? tx - m.width / 2 - padX : tx - padX;
    roundRect(ctx, rx, ty - 9.5, m.width + padX * 2, 19, 5);
    ctx.fill();
    ctx.stroke();
  }
  ctx.fillStyle = o.color ?? th.text;
  ctx.fillText(text, tx, ty);
  ctx.restore();
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number): void {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/** 角度弧标记（几何画板的“角”显示） */
export function drawAngleMark(p: Paper, vx: number, vy: number, a1: number, a2: number, r = 22, color = "#8b7cf6"): void {
  const { ctx } = p;
  const [sx, sy] = vpScreen(p, vx, vy);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.6;
  ctx.beginPath();
  ctx.arc(sx, sy, r, -a2, -a1, a2 > a1);
  ctx.stroke();
  ctx.restore();
}

/** 十字光标 + 追踪点（鼠标悬停时显示函数值） */
export function drawCrosshair(p: Paper, x: number, y: number, color = "rgba(139,124,246,0.5)"): void {
  const { ctx } = p;
  const [sx, sy] = vpScreen(p, x, y);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.setLineDash([4, 4]);
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(0, Math.round(sy) + 0.5);
  ctx.lineTo(p.w, Math.round(sy) + 0.5);
  ctx.moveTo(Math.round(sx) + 0.5, 0);
  ctx.lineTo(Math.round(sx) + 0.5, p.h);
  ctx.stroke();
  ctx.restore();
}

/** 右下角读数板 */
export function drawReadout(p: Paper, lines: string[]): void {
  const { ctx } = p;
  const th = themeOf(p.dark);
  if (!lines.length) return;
  ctx.save();
  ctx.font = "12px 'SF Mono', Menlo, monospace";
  const wMax = Math.max(...lines.map((l) => ctx.measureText(l).width));
  const hgt = lines.length * 16 + 12;
  const x = p.w - wMax - 22;
  const y = p.h - hgt - 12;
  ctx.fillStyle = p.dark ? "rgba(12,14,30,0.78)" : "rgba(255,255,255,0.86)";
  ctx.strokeStyle = p.dark ? "rgba(140,150,220,0.28)" : "rgba(60,70,120,0.2)";
  roundRect(ctx, x - 8, y - 6, wMax + 20, hgt + 8, 8);
  ctx.fill();
  ctx.stroke();
  ctx.fillStyle = th.text;
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  lines.forEach((l, i) => ctx.fillText(l, x, y + i * 16 + 2));
  ctx.restore();
}

/** 左上角模式水印/提示 */
export function drawHint(p: Paper, text: string): void {
  const { ctx } = p;
  const th = themeOf(p.dark);
  ctx.save();
  ctx.font = "11.5px system-ui, sans-serif";
  ctx.fillStyle = th.muted;
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(text, 12, 10);
  ctx.restore();
}
