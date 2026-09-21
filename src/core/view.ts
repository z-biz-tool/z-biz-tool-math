/**
 * 视口与刻度：世界坐标（数学坐标）↔ 屏幕坐标（像素，y 向下）的唯一映射入口。
 * 视图对象不可变：pan/zoom 都返回新实例，便于 React 状态管理与撤销。
 */

import { fmtPi, fmt } from "./cnum.ts";

export interface ViewportInit {
  cx: number;
  cy: number;
  scale: number;
  width: number;
  height: number;
}

export const MIN_SCALE = 1e-4;
export const MAX_SCALE = 1e8;

export class Viewport {
  cx: number;
  cy: number;
  scale: number;
  width: number;
  height: number;

  constructor(init: ViewportInit) {
    this.cx = init.cx;
    this.cy = init.cy;
    this.scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, init.scale));
    this.width = Math.max(1, init.width);
    this.height = Math.max(1, init.height);
  }

  get left(): number {
    return this.cx - this.width / 2 / this.scale;
  }
  get right(): number {
    return this.cx + this.width / 2 / this.scale;
  }
  get bottom(): number {
    return this.cy - this.height / 2 / this.scale;
  }
  get top(): number {
    return this.cy + this.height / 2 / this.scale;
  }

  toScreen(x: number, y: number): [number, number] {
    return [this.width / 2 + (x - this.cx) * this.scale, this.height / 2 - (y - this.cy) * this.scale];
  }
  toWorld(sx: number, sy: number): [number, number] {
    return [this.cx + (sx - this.width / 2) / this.scale, this.cy - (sy - this.height / 2) / this.scale];
  }
  dx(worldDelta: number): number {
    return worldDelta * this.scale;
  }
  with(patch: Partial<ViewportInit>): Viewport {
    return new Viewport({ ...this.toJSON(), ...patch });
  }
  toJSON(): ViewportInit {
    return { cx: this.cx, cy: this.cy, scale: this.scale, width: this.width, height: this.height };
  }

  panPixels(dx: number, dy: number): Viewport {
    return this.with({ cx: this.cx - dx / this.scale, cy: this.cy + dy / this.scale });
  }
  /** 以屏幕点 (sx, sy) 为锚点缩放，锚点处的世界坐标保持不变 */
  zoomAt(sx: number, sy: number, factor: number): Viewport {
    const [wx, wy] = this.toWorld(sx, sy);
    const scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, this.scale * factor));
    const next = new Viewport({ cx: this.cx, cy: this.cy, scale, width: this.width, height: this.height });
    const [sx2, sy2] = next.toScreen(wx, wy);
    return next.with({ cx: next.cx + (sx2 - sx) / scale, cy: next.cy - (sy2 - sy) / scale });
  }
  /** 让给定世界矩形恰好充满视口（可留边距） */
  fit(x0: number, x1: number, y0: number, y1: number, pad = 0.12): Viewport {
    const w = Math.max(1e-9, x1 - x0);
    const h = Math.max(1e-9, y1 - y0);
    const scale = Math.min(this.width / (w * (1 + pad * 2)), this.height / (h * (1 + pad * 2)));
    return this.with({ cx: (x0 + x1) / 2, cy: (y0 + y1) / 2, scale });
  }
  squareRange(): [number, number, number, number] {
    return [this.left, this.right, this.bottom, this.top];
  }
}

export type TickMode = "auto" | "pi" | "int" | "log" | "scientific";

export interface Ticks {
  values: number[];
  labels: string[];
  step: number;
  minor: number[];
}

const BASES = [1, 2, 2.5, 5, 10];

/** 生成“好看”的刻度：步长归一到 1/2/2.5/5×10^n；pi 模式按 π 的 1/12…2 倍取整 */
export function niceTicks(min: number, max: number, target = 8, mode: TickMode = "auto"): Ticks {
  const span = Math.max(1e-12, max - min);
  if (mode === "pi") {
    const cands = [1 / 12, 1 / 6, 1 / 4, 1 / 3, 1 / 2, 1, 2, 3, 4, 6, 12];
    let step = Math.PI / 2;
    for (const c of cands) {
      if (span / (c * Math.PI) <= target) {
        step = c * Math.PI;
        break;
      }
    }
    const values: number[] = [];
    const start = Math.ceil(min / step - 1e-9) * step;
    for (let v = start; v <= max + 1e-9; v += step) values.push(Math.abs(v) < 1e-12 ? 0 : v);
    return { values, labels: values.map((v) => fmtPi(v, 3)), step, minor: [] };
  }
  const raw = span / Math.max(2, target);
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  let step = mag;
  for (const b of BASES) {
    if (raw / mag <= b) {
      step = b * mag;
      break;
    }
  }
  if (mode === "int") step = Math.max(1, Math.round(step));
  const values: number[] = [];
  const start = Math.ceil(min / step - 1e-9) * step;
  for (let v = start; v <= max + step * 1e-6; v += step) {
    values.push(Math.abs(v) < step * 1e-9 ? 0 : v);
    if (values.length > 2000) break;
  }
  const decimals = Math.max(0, -Math.floor(Math.log10(step)) + (step < 1 ? 0 : 0));
  const labels = values.map((v) =>
    mode === "scientific" || Math.abs(v) >= 1e6 || (Math.abs(v) < 1e-4 && v !== 0)
      ? v.toExponential(1)
      : fmt(v, Math.max(2, decimals + 1)),
  );
  const minor: number[] = [];
  if (step / 5 > 1e-15) {
    const ms = step / 5;
    for (let v = Math.ceil(min / ms) * ms; v <= max; v += ms) {
      if (values.every((q) => Math.abs(q - v) > ms * 0.2)) minor.push(v);
      if (minor.length > 4000) break;
    }
  }
  return { values, labels, step, minor };
}

/** 对数坐标的刻度（1-2-5 序列） */
export function logTicks(min: number, max: number): Ticks {
  const values: number[] = [];
  const e0 = Math.floor(Math.log10(Math.max(1e-300, min)));
  const e1 = Math.ceil(Math.log10(Math.max(1e-300, max)));
  for (let e = e0; e <= e1; e++) {
    for (const m of [1, 2, 5]) {
      const v = m * Math.pow(10, e);
      if (v >= min && v <= max) values.push(v);
    }
  }
  return { values, labels: values.map((v) => (v >= 1e5 || v < 1e-3 ? v.toExponential(0) : String(v))), step: 1, minor: [] };
}

/** 等分区间，供采样使用 */
export function segmentRange(a: number, b: number, n: number): number[] {
  const out = new Array(n + 1);
  for (let i = 0; i <= n; i++) out[i] = a + ((b - a) * i) / n;
  return out;
}
