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
  /** 对数轴：该轴的 cx 与 scale 解释为 log10 空间（十倍频程），支持 semilogx/semilogy/loglog */
  logX?: boolean;
  logY?: boolean;
}

export const MIN_SCALE = 1e-4;
export const MAX_SCALE = 1e8;

/** 对数轴可表示的世界值下界，再小一律按此处理，避免 log10(0) 与负数产生 NaN */
const LOG_FLOOR = 1e-300;
/** 10 的幂合法区间；超出即浮点溢出，钳位后极端缩放仍能返回有限值 */
const LOG_LIMIT = 300;

/** 世界坐标 → 轴坐标：线性轴恒等，对数轴取 log10 */
function axisOf(v: number, log: boolean): number {
  return log ? Math.log10(v > LOG_FLOOR ? v : LOG_FLOOR) : v;
}
/** 轴坐标 → 世界坐标：线性轴恒等，对数轴取 10^ */
function worldOf(u: number, log: boolean): number {
  if (!log) return u;
  return Math.pow(10, Math.min(LOG_LIMIT, Math.max(-LOG_LIMIT, u)));
}

export class Viewport {
  cx: number;
  cy: number;
  scale: number;
  width: number;
  height: number;
  logX: boolean;
  logY: boolean;

  constructor(init: ViewportInit) {
    this.cx = init.cx;
    this.cy = init.cy;
    this.scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, init.scale));
    this.width = Math.max(1, init.width);
    this.height = Math.max(1, init.height);
    this.logX = init.logX === true;
    this.logY = init.logY === true;
  }

  toAxisX(x: number): number {
    return axisOf(x, this.logX);
  }
  toAxisY(y: number): number {
    return axisOf(y, this.logY);
  }
  fromAxisX(u: number): number {
    return worldOf(u, this.logX);
  }
  fromAxisY(v: number): number {
    return worldOf(v, this.logY);
  }
  /** 视口四边在轴空间中的位置：场/等值线按此等距采样，映回屏幕即线性 */
  get axisLeft(): number {
    return this.cx - this.width / 2 / this.scale;
  }
  get axisRight(): number {
    return this.cx + this.width / 2 / this.scale;
  }
  get axisBottom(): number {
    return this.cy - this.height / 2 / this.scale;
  }
  get axisTop(): number {
    return this.cy + this.height / 2 / this.scale;
  }

  get left(): number {
    return this.fromAxisX(this.axisLeft);
  }
  get right(): number {
    return this.fromAxisX(this.axisRight);
  }
  get bottom(): number {
    return this.fromAxisY(this.axisBottom);
  }
  get top(): number {
    return this.fromAxisY(this.axisTop);
  }

  toScreen(x: number, y: number): [number, number] {
    return [
      this.width / 2 + (this.toAxisX(x) - this.cx) * this.scale,
      this.height / 2 - (this.toAxisY(y) - this.cy) * this.scale,
    ];
  }
  toWorld(sx: number, sy: number): [number, number] {
    return [
      this.fromAxisX(this.cx + (sx - this.width / 2) / this.scale),
      this.fromAxisY(this.cy - (sy - this.height / 2) / this.scale),
    ];
  }
  dx(worldDelta: number): number {
    return worldDelta * this.scale;
  }
  with(patch: Partial<ViewportInit>): Viewport {
    return new Viewport({ ...this.toJSON(), ...patch });
  }
  toJSON(): ViewportInit {
    const j: ViewportInit = {
      cx: this.cx,
      cy: this.cy,
      scale: this.scale,
      width: this.width,
      height: this.height,
    };
    // 线性视口不写标记，工程 JSON 与旧版保持一致
    if (this.logX) j.logX = true;
    if (this.logY) j.logY = true;
    return j;
  }

  /**
   * 切换对数轴。被改动的轴中心回到 0 十倍频（即世界值 1）：
   * 线性中心在对数空间里没有对应物，留在原位会让视口落到一片无意义区间。
   */
  withLog(logX: boolean, logY: boolean): Viewport {
    return new Viewport({
      ...this.toJSON(),
      logX,
      logY,
      cx: logX === this.logX ? this.cx : 0,
      cy: logY === this.logY ? this.cy : 0,
    });
  }

  panPixels(dx: number, dy: number): Viewport {
    return this.with({ cx: this.cx - dx / this.scale, cy: this.cy + dy / this.scale });
  }
  /** 以屏幕点 (sx, sy) 为锚点缩放，锚点处的世界坐标保持不变 */
  zoomAt(sx: number, sy: number, factor: number): Viewport {
    const [wx, wy] = this.toWorld(sx, sy);
    const scale = Math.min(MAX_SCALE, Math.max(MIN_SCALE, this.scale * factor));
    const next = new Viewport({ ...this.toJSON(), scale });
    const [sx2, sy2] = next.toScreen(wx, wy);
    return next.with({ cx: next.cx + (sx2 - sx) / scale, cy: next.cy - (sy2 - sy) / scale });
  }
  /** 让给定世界矩形恰好充满视口（可留边距） */
  fit(x0: number, x1: number, y0: number, y1: number, pad = 0.12): Viewport {
    const w = Math.max(1e-9, this.toAxisX(x1) - this.toAxisX(x0));
    const h = Math.max(1e-9, this.toAxisY(y1) - this.toAxisY(y0));
    const scale = Math.min(this.width / (w * (1 + pad * 2)), this.height / (h * (1 + pad * 2)));
    return this.with({
      cx: (this.toAxisX(x0) + this.toAxisX(x1)) / 2,
      cy: (this.toAxisY(y0) + this.toAxisY(y1)) / 2,
      scale,
    });
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
  if (mode === "log") return logTicks(min, max, target);
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

const SUP = "⁰¹²³⁴⁵⁶⁷⁸⁹";

/** 10^n 的刻度标签：1、10、10²、10⁻³ */
function powLabel(n: number): string {
  if (n === 0) return "1";
  const digits = String(Math.abs(n)).replace(/\d/g, (d) => SUP[+d]);
  return `10${n < 0 ? "⁻" : ""}${digits}`;
}

/**
 * 对数坐标的刻度：主刻度落在 10 的整数幂，次级网格按 1-2-5（视野超过两个十倍频）
 * 或 2…9（不足两个十倍频）铺开。step 固定为 0，告诉调用方这里没有统一步长可滤零。
 */
export function logTicks(min: number, max: number, target = 8): Ticks {
  const lo = Math.floor(Math.log10(Math.max(1e-300, min)));
  const hi = Math.ceil(Math.log10(Math.max(1e-300, max)));
  const span = Math.max(0, hi - lo);
  const mult = span <= 2 ? [2, 3, 4, 5, 6, 7, 8, 9] : [2, 5];
  const k = Math.max(1, Math.ceil((span + 1) / Math.max(2, target)));
  const inRange = (v: number) => v >= min * (1 - 1e-9) && v <= max * (1 + 1e-9);
  const values: number[] = [];
  const labels: string[] = [];
  const minor: number[] = [];
  for (let e = lo; e <= hi; e++) {
    // 抽稀时对齐到 10⁰ = 1，否则「1」这个最该出现的刻度会被跳过
    if (((e % k) + k) % k !== 0) continue;
    const base = Math.pow(10, e);
    if (inRange(base)) {
      values.push(base);
      labels.push(powLabel(e));
    }
    for (const m of mult) {
      const v = m * base;
      if (inRange(v)) minor.push(v);
    }
  }
  // 视野不足一个十倍频且两头都不含整十倍频时，只剩次级网格：把倍数值提为主刻度
  if (values.length === 0 && minor.length > 1) {
    return { values: minor, labels: minor.map((v) => fmt(v, 3)), step: 0, minor: [] };
  }
  return { values, labels, step: 0, minor };
}

/** 等分区间，供采样使用 */
export function segmentRange(a: number, b: number, n: number): number[] {
  const out = new Array(n + 1);
  for (let i = 0; i <= n; i++) out[i] = a + ((b - a) * i) / n;
  return out;
}
