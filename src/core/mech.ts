/**
 * 力学内核：重力场里的刚球——抛体、弹性碰撞、空气阻力（世界模型的第一块砖）。
 * 球体活态（位置/速度/轨迹）在模块单例里，store 只存配置与初始布局；
 * 载入工程 / 应用预设 / 面板「重置」时按配置重建。
 */

import { paletteAt } from "./colormap.ts";

/** 弹性墙围成的运动场（世界坐标，y 向上） */
export const MECH_BOUNDS = { left: -12, right: 12, bottom: -9, top: 9 };

export interface MechBallSpec {
  id: string;
  x: number;
  y: number;
  vx: number;
  vy: number;
  r: number;
  m: number;
  color: string;
}

export interface MechBall extends MechBallSpec {
  /** 轨迹历史（世界坐标平铺 [x0,y0,x1,y1,…]） */
  trail: number[];
}

export interface MechParams {
  gravity: number;
  restitution: number;
  /** 空气阻力系数（速度的比例衰减率） */
  drag: number;
}

/** 单次数值积分的最大步长（秒）：碰撞与抛体在这个粒度下足够稳 */
const MAX_H = 0.008;
/** 每球轨迹点数上限（平铺坐标，900 = 450 个点） */
const TRAIL_CAP = 900;

export class MechWorld {
  balls: MechBall[] = [];
  t = 0;
  /** 被抓住的球暂停演化，位置由指针直接驱动 */
  grabbedId: string | null = null;

  resetFrom(specs: MechBallSpec[]): void {
    this.balls = specs.map((s) => ({ ...s, trail: [] }));
    this.t = 0;
    this.grabbedId = null;
  }

  get(id: string): MechBall | null {
    return this.balls.find((b) => b.id === id) ?? null;
  }

  /** 拖拽中的落点直接写位置；轨迹清空避免拉出一条横穿的假线 */
  place(id: string, x: number, y: number): void {
    const b = this.get(id);
    if (!b) return;
    b.x = x;
    b.y = y;
    b.trail = [];
  }

  remove(id: string): void {
    this.balls = this.balls.filter((b) => b.id !== id);
  }

  updateSpec(id: string, patch: Partial<Pick<MechBallSpec, "r" | "m" | "color" | "vx" | "vy">>): void {
    const b = this.get(id);
    if (!b) return;
    Object.assign(b, patch);
  }

  add(spec: MechBallSpec): void {
    this.balls.push({ ...spec, trail: [] });
  }

  step(dt: number, o: MechParams): void {
    const n = Math.max(1, Math.ceil(dt / MAX_H));
    const h = dt / n;
    for (let k = 0; k < n; k++) this.substep(h, o);
    this.t += dt;
    for (const b of this.balls) {
      b.trail.push(b.x, b.y);
      if (b.trail.length > TRAIL_CAP) b.trail.splice(0, b.trail.length - TRAIL_CAP);
    }
  }

  private substep(h: number, o: MechParams): void {
    const { left, right, bottom, top } = MECH_BOUNDS;
    for (const b of this.balls) {
      if (b.id === this.grabbedId) continue;
      b.vy -= o.gravity * h;
      const d = Math.max(0, 1 - o.drag * h);
      b.vx *= d;
      b.vy *= d;
      b.x += b.vx * h;
      b.y += b.vy * h;
      // 墙：法向速度按恢复系数反弹
      if (b.x - b.r < left) {
        b.x = left + b.r;
        b.vx = Math.abs(b.vx) * o.restitution;
      }
      if (b.x + b.r > right) {
        b.x = right - b.r;
        b.vx = -Math.abs(b.vx) * o.restitution;
      }
      if (b.y - b.r < bottom) {
        b.y = bottom + b.r;
        b.vy = Math.abs(b.vy) * o.restitution;
      }
      if (b.y + b.r > top) {
        b.y = top - b.r;
        b.vy = -Math.abs(b.vy) * o.restitution;
      }
    }
    // 球-球：位置修正按质量反比摊开，冲量沿法向、带恢复系数
    for (let i = 0; i < this.balls.length; i++) {
      for (let j = i + 1; j < this.balls.length; j++) {
        const a = this.balls[i];
        const c = this.balls[j];
        const dx = c.x - a.x;
        const dy = c.y - a.y;
        const dist = Math.hypot(dx, dy) || 1e-9;
        const min = a.r + c.r;
        if (dist >= min) continue;
        const nx = dx / dist;
        const ny = dy / dist;
        const overlap = min - dist;
        const total = a.m + c.m;
        if (a.id !== this.grabbedId) {
          a.x -= nx * overlap * (c.m / total);
          a.y -= ny * overlap * (c.m / total);
        }
        if (c.id !== this.grabbedId) {
          c.x += nx * overlap * (a.m / total);
          c.y += ny * overlap * (a.m / total);
        }
        const vn = (c.vx - a.vx) * nx + (c.vy - a.vy) * ny;
        if (vn >= 0) continue;
        const jimp = (-(1 + o.restitution) * vn) / (1 / a.m + 1 / c.m);
        a.vx -= (jimp / a.m) * nx;
        a.vy -= (jimp / a.m) * ny;
        c.vx += (jimp / c.m) * nx;
        c.vy += (jimp / c.m) * ny;
      }
    }
  }

  kinetic(): number {
    return this.balls.reduce((s, b) => s + 0.5 * b.m * (b.vx * b.vx + b.vy * b.vy), 0);
  }
}

export const mechWorld = new MechWorld();

/** 面板「加球」：在场中随机位置给一颗慢速球 */
export function defaultBallSpec(i: number): MechBallSpec {
  return {
    id: `b${Date.now().toString(36).slice(-4)}${i}`,
    x: -6 + Math.random() * 12,
    y: -2 + Math.random() * 8,
    vx: (Math.random() - 0.5) * 8,
    vy: (Math.random() - 0.5) * 8,
    r: 0.6 + Math.random() * 0.6,
    m: 1,
    color: paletteAt(i % 15),
  };
}

/** 拖拽释放时按最后 ~100ms 的指针位移反推出手速度，上限防甩飞 */
export function launchVelocity(samples: { t: number; x: number; y: number }[]): [number, number] {
  const now = samples[samples.length - 1];
  if (!now) return [0, 0];
  let old = samples[0];
  for (const s of samples) {
    if (now.t - s.t <= 120) {
      old = s;
      break;
    }
  }
  const dt = (now.t - old.t) / 1000;
  if (dt <= 0.005) return [0, 0];
  const clamp = (v: number) => Math.max(-40, Math.min(40, v));
  return [clamp((now.x - old.x) / dt), clamp((now.y - old.y) / dt)];
}
