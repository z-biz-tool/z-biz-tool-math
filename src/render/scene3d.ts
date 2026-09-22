/**
 * 3D 场景绘制：曲面 / 网格 / 线框 / 等高线 / surfc / 隐式曲面 / 旋转体 / 空间曲线
 *
 * 软件渲染 + 画家算法：所有图层的面片与线段汇入同一个深度序列，保证跨图层正确遮挡。
 * 网格按「表达式 + 参数值 + 定义域 + 分辨率」签名缓存，
 * 因此轨道拖拽、缩放不会重建网格，只有改表达式或动画参数时才重算。
 */
import { Engine } from "../core/machine.ts";
import {
  buildGraphGrid,
  buildParametricGrid,
  implicitMesh,
  meshBounds,
  meshContours,
  meshEdges,
  orbitCamera,
  projectScene,
  surfaceNormals,
  surfaceOfRevolution,
  type Camera,
  type Mesh,
  type MeshContour,
  type SceneProjection,
} from "../core/surface.ts";
import { buildLUT } from "../core/colormap.ts";
import { niceTicks } from "../core/view.ts";
import { F1 } from "./scene2d.ts";
import { themeOf, type Theme } from "./plot2d.ts";
import type { GeoLabState, SurfLayer } from "../state.ts";

export interface Scene3DOut {
  errors: string[];
  info: string[];
  /** 自动适配视野所需的相机距离 */
  suggestDist: number;
}

const EMPTY = new Uint32Array(0);
const cache = new Map<string, Mesh>();

/**
 * 只依赖网格本身的派生数据。网格按签名缓存了，但法向、参数线、等高线此前每帧
 * 重算一遍——轨道拖拽时网格根本没变，这部分是纯粹的白给。
 */
interface Derived {
  bounds: [number, number, number, number, number, number];
  norms: Float32Array;
  edges: Uint32Array;
  contours: MeshContour[];
}
const derived = new WeakMap<Mesh, Derived>();
const lutCache = new Map<string, Uint8ClampedArray>();

function derive(m: Mesh): Derived {
  let d = derived.get(m);
  if (!d) {
    const structured = m.tris.length > 0;
    const bounds = meshBounds(m);
    const zmax = bounds[5] === bounds[4] ? bounds[4] + 1 : bounds[5];
    d = {
      bounds,
      norms: structured ? surfaceNormals(m) : new Float32Array(0),
      edges: structured ? meshEdges(m) : EMPTY,
      contours: structured ? meshContours(m, levelsOf(bounds[4], zmax, 12)) : [],
    };
    derived.set(m, d);
  }
  return d;
}

function lutOf(name: string): Uint8ClampedArray {
  let l = lutCache.get(name);
  if (!l) {
    l = buildLUT(name);
    lutCache.set(name, l);
  }
  return l;
}

function msg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function fin(v: number): boolean {
  return Number.isFinite(v);
}

function clamp01(t: number): number {
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

/** 世界点 → 屏幕点；相机后方的点返回 null */
function proj(cam: Camera, x: number, y: number, z: number): [number, number, number] | null {
  const dx = x - cam.pos[0];
  const dy = y - cam.pos[1];
  const dz = z - cam.pos[2];
  const depth = dx * cam.fwd[0] + dy * cam.fwd[1] + dz * cam.fwd[2];
  if (!(depth > 1e-9)) return null;
  const k = cam.ortho ? cam.pxPerUnit ?? cam.focal : cam.focal;
  const denom = cam.ortho ? 1 : depth;
  const sx = cam.width / 2 + (k * (dx * cam.right[0] + dy * cam.right[1] + dz * cam.right[2])) / denom;
  const sy = cam.height / 2 - (k * (dx * cam.up[0] + dy * cam.up[1] + dz * cam.up[2])) / denom;
  return fin(sx) && fin(sy) ? [sx, sy, depth] : null;
}

function need(eng: Engine, src: string, take: number, l: SurfLayer, part = ""): F1 {
  const label = `${l.label || l.expr}${part ? ` (${part})` : ""}`;
  if (!src.trim()) throw new Error(`${label} 表达式为空`);
  const f = new F1(eng, src, take);
  const bad = f.badName();
  if (bad) throw new Error(`${part ? `(${part}) ` : ""}${bad}`);
  return f;
}

/** 缓存键：包含动画参数的当前值 */
function sig(l: SurfLayer, s: GeoLabState): string {
  const pv = s.params.map((p) => `${p.name}=${p.value}`).join(",");
  return `${l.id}|${l.kind}|${l.expr}|${l.expr2 ?? ""}|${l.expr3 ?? ""}|${l.res}|${l.range.join(",")}|${s.surf.box.join(",")}|${pv}`;
}

/** 按图层类型构建网格 */
function buildMesh(eng: Engine, l: SurfLayer, s: GeoLabState, errors: string[]): Mesh | null {
  const key = sig(l, s);
  const hit = cache.get(key);
  if (hit) return hit;
  let m: Mesh | null = null;
  try {
    const [a, b, c, d] = l.range;
    const res = Math.max(4, Math.min(220, Math.round(l.res)));
    switch (l.kind) {
      case "graph": {
        const f = need(eng, l.expr, 2, l);
        m = buildGraphGrid((x, y) => f.at(x, y), [a, b, c, d, res, res]);
        break;
      }
      case "param": {
        const fx = need(eng, l.expr, 2, l, "x");
        const fy = need(eng, l.expr2 ?? "v", 2, l, "y");
        const fz = need(eng, l.expr3 ?? "u", 2, l, "z");
        m = buildParametricGrid((u, v, o) => o.set(fx.at(u, v), fy.at(u, v), fz.at(u, v)), [a, b, c, d, res, res]);
        break;
      }
      case "implicit": {
        const f = need(eng, l.expr, 3, l);
        const [x0, x1, y0, y1, z0, z1] = s.surf.box;
        m = implicitMesh((x, y, z) => f.at(x, y, z), [x0, x1, y0, y1, z0, z1], Math.min(res, 72));
        break;
      }
      case "revolve": {
        const f = need(eng, l.expr, 1, l);
        const axis: "x" | "y" = l.expr2 === "y" ? "y" : "x";
        m = surfaceOfRevolution((t) => [t, f.at(t)], [a, b], axis, {
          steps: res,
          segments: Math.max(8, Math.min(180, Math.round(d) || 60)),
        });
        break;
      }
      case "spacecurve": {
        const fx = need(eng, l.expr, 1, l, "x");
        const fy = need(eng, l.expr2 ?? "sin(t)", 1, l, "y");
        const fz = need(eng, l.expr3 ?? "t", 1, l, "z");
        const n = Math.max(2, Math.min(4000, res * 4));
        const pos = new Float32Array(n * 3);
        for (let i = 0; i < n; i++) {
          const t = a + ((b - a) * i) / (n - 1);
          pos[i * 3] = fx.at(t);
          pos[i * 3 + 1] = fy.at(t);
          pos[i * 3 + 2] = fz.at(t);
        }
        m = { positions: pos, tris: EMPTY };
        break;
      }
    }
  } catch (e) {
    errors.push(`${l.label || l.expr}：${msg(e)}`);
    return null;
  }
  if (m) {
    cache.set(key, m);
    if (cache.size > 20) {
      const first = cache.keys().next().value;
      if (first !== undefined) cache.delete(first);
    }
  }
  return m;
}

interface Item {
  layer: SurfLayer;
  mesh: Mesh;
  d: Derived;
  zmin: number;
  zmax: number;
  lut: Uint8ClampedArray;
  proj?: SceneProjection;
}

type Draw = {
  depth: number;
  face: boolean;
  pts: number[];
  color: string;
  alpha: number;
  width: number;
  seam: boolean;
};

/** 世界 z 轴方向的光向（与相机无关，转动时明暗随之变化，符合 MATLAB lighting gouraud 观感） */
const LIGHT: [number, number, number] = [0.42, 0.32, 0.85];

export function drawScene3D(
  ctx: CanvasRenderingContext2D,
  w: number,
  h: number,
  s: GeoLabState,
  dpr: number,
): Scene3DOut {
  const out: Scene3DOut = { errors: [], info: [], suggestDist: 0 };
  const th: Theme = themeOf(s.settings.dark);
  ctx.save();
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.fillStyle = s.settings.dark ? "#0a0c1c" : "#f7f8fd";
  ctx.fillRect(0, 0, w, h);
  ctx.lineJoin = "round";

  let box: number[] = s.surf.autoBox ? [Infinity, -Infinity, Infinity, -Infinity, Infinity, -Infinity] : s.surf.box.slice();
  const items: Item[] = [];
  for (const l of s.surf.layers) {
    if (!l.visible) continue;
    const m = buildMesh(s.engine, l, s, out.errors);
    if (!m || !m.positions.length) continue;
    const d = derive(m);
    const mb = d.bounds;
    if (s.surf.autoBox)
      box = [
        Math.min(box[0], mb[0]),
        Math.max(box[1], mb[1]),
        Math.min(box[2], mb[2]),
        Math.max(box[3], mb[3]),
        Math.min(box[4], mb[4]),
        Math.max(box[5], mb[5]),
      ];
    items.push({
      layer: l,
      mesh: m,
      d,
      zmin: mb[4],
      zmax: mb[5] === mb[4] ? mb[4] + 1 : mb[5],
      lut: lutOf(l.colormap),
    });
  }
  if (!items.length) {
    ctx.fillStyle = th.muted;
    ctx.font = "13px system-ui, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText(out.errors[0] ?? "添加曲面图层开始作图", w / 2, h / 2);
    ctx.restore();
    return out;
  }
  if (!fin(box[0]) || box[0] >= box[1]) box = s.surf.box.slice();

  const target: [number, number, number] = [(box[0] + box[1]) / 2, (box[2] + box[3]) / 2, (box[4] + box[5]) / 2];
  const radius = Math.max(1e-6, Math.hypot(box[1] - box[0], box[3] - box[2], box[5] - box[4]) / 2);
  const fov = 0.9;
  out.suggestDist = ((radius / Math.tan(fov / 2)) * (h > w ? h / w : 1) + radius) * 1.1;
  const cam = orbitCamera({
    azim: s.surf.cam.azim,
    elev: s.surf.cam.elev,
    dist: Math.max(radius * 0.2, s.surf.cam.dist),
    target,
    width: w,
    height: h,
    fov,
  });

  if (s.surf.showAxes) drawBoxAxes(ctx, cam, box as [number, number, number, number, number, number], th, s.settings.dark);

  const all: Draw[] = [];
  let nTri = 0;
  let nVert = 0;
  for (const it of items) {
    const l = it.layer;
    nVert += it.mesh.positions.length / 3;
    const solid = l.style === "surf" || l.style === "surfc";
    const lined = l.style === "mesh" || l.style === "wire" || l.style === "surf" || l.style === "surfc";
    if (solid || lined) it.proj = projectScene(it.mesh.positions, it.mesh.tris, cam, it.d.norms);
    if (solid && it.proj) {
      nTri += it.proj.faces.length;
      for (const f of it.proj.faces) {
        const t = clamp01((f.z - it.zmin) / (it.zmax - it.zmin));
        const o = Math.round(t * 255) * 3;
        const lit = l.lit ? 0.4 + 0.6 * Math.abs(f.n[0] * LIGHT[0] + f.n[1] * LIGHT[1] + f.n[2] * LIGHT[2]) : 1;
        all.push({
          depth: f.depth,
          face: true,
          pts: f.pts,
          color: `rgb(${Math.round(it.lut[o] * lit)},${Math.round(it.lut[o + 1] * lit)},${Math.round(it.lut[o + 2] * lit)})`,
          alpha: l.opacity,
          width: 0,
          seam: l.style === "surf" && l.opacity > 0.94,
        });
      }
    }
    if ((l.style === "surf" || l.style === "mesh" || l.style === "wire") && it.proj) {
      /* 结构化网格取参数线（MATLAB mesh 观感，无对角线）；surf 只描稀疏网格线增强立体感 */
      const e = it.d.edges;
      const v = it.proj.verts;
      const stride = l.style === "wire" ? 2 : l.style === "surf" ? 6 : 2;
      for (let i = 0; i + 1 < e.length; i += stride * 2) {
        const a = e[i] * 3;
        const b = e[i + 1] * 3;
        if (v[a + 2] < 0 || v[b + 2] < 0) continue;
        all.push({
          depth: (v[a + 2] + v[b + 2]) / 2,
          face: false,
          pts: [v[a], v[a + 1], v[b], v[b + 1]],
          color: l.color,
          alpha: l.opacity * (l.style === "surf" ? 0.5 : 1),
          width: l.style === "wire" ? 1.2 : 1,
          seam: false,
        });
      }
    }
    if (l.style === "contour" || l.style === "surfc") {
      const base = l.style === "surfc";
      const zUse = base ? it.zmin : 0;
      for (const c of it.d.contours) {
        const t = clamp01((c.level - it.zmin) / (it.zmax - it.zmin));
        const o = Math.round(t * 255) * 3;
        const col = base
          ? s.settings.dark
            ? "#161a30"
            : "#2c3358"
          : `rgb(${it.lut[o]},${it.lut[o + 1]},${it.lut[o + 2]})`;
        for (const pl of c.polylines) {
          for (let i = 0; i + 1 < pl.length; i++) {
            const z1 = base ? zUse : pl[i][2];
            const z2 = base ? zUse : pl[i + 1][2];
            const p1 = proj(cam, pl[i][0], pl[i][1], z1);
            const p2 = proj(cam, pl[i + 1][0], pl[i + 1][1], z2);
            if (!p1 || !p2) continue;
            all.push({
              depth: (p1[2] + p2[2]) / 2,
              face: false,
              pts: [p1[0], p1[1], p2[0], p2[1]],
              color: col,
              alpha: 1,
              width: base ? 1 : 1.6,
              seam: false,
            });
          }
        }
      }
    }
    if (l.kind === "spacecurve") {
      const pos = it.mesh.positions;
      let prev: [number, number, number] | null = null;
      for (let i = 0; i < pos.length; i += 3) {
        const q = proj(cam, pos[i], pos[i + 1], pos[i + 2]);
        if (prev && q)
          all.push({ depth: q[2], face: false, pts: [prev[0], prev[1], q[0], q[1]], color: l.color, alpha: l.opacity, width: 2.6, seam: false });
        prev = q;
      }
    }
  }
  all.sort((a, b) => b.depth - a.depth);
  for (const d of all) {
    ctx.globalAlpha = d.alpha;
    if (d.face) {
      const q = d.pts;
      ctx.beginPath();
      ctx.moveTo(q[0], q[1]);
      ctx.lineTo(q[2], q[3]);
      ctx.lineTo(q[4], q[5]);
      ctx.closePath();
      ctx.fillStyle = d.color;
      ctx.fill();
      if (d.seam) {
        ctx.strokeStyle = d.color;
        ctx.lineWidth = 0.8;
        ctx.stroke();
      }
    } else {
      ctx.beginPath();
      ctx.moveTo(d.pts[0], d.pts[1]);
      ctx.lineTo(d.pts[2], d.pts[3]);
      ctx.strokeStyle = d.color;
      ctx.lineWidth = d.width;
      ctx.stroke();
    }
  }
  ctx.globalAlpha = 1;
  out.info.push(`顶点 ${nVert | 0} · 面片 ${nTri} · 图元 ${all.length}`);
  ctx.restore();
  return out;
}

function levelsOf(lo: number, hi: number, n: number): number[] {
  return niceTicks(lo, hi, n).values.filter((v) => v > lo && v < hi);
}

/** 立方体边框 + 三轴刻度：每个轴方向取离相机最近的平行边承载刻度 */
function drawBoxAxes(
  ctx: CanvasRenderingContext2D,
  cam: Camera,
  box: [number, number, number, number, number, number],
  th: Theme,
  dark: boolean,
): void {
  const [x0, x1, y0, y1, z0, z1] = box;
  const at = (i: number, j: number, k: number) => proj(cam, i ? x1 : x0, j ? y1 : y0, k ? z1 : z0);
  type Edge = { a: [number, number, number]; b: [number, number, number]; axis: 0 | 1 | 2; depth: number };
  const edges: Edge[] = [];
  const add = (axis: 0 | 1 | 2, f: [number, number, number], t: [number, number, number]): void => {
    const a = at(f[0], f[1], f[2]);
    const b = at(t[0], t[1], t[2]);
    if (a && b) edges.push({ a, b, axis, depth: (a[2] + b[2]) / 2 });
  };
  for (const j of [0, 1]) for (const k of [0, 1]) add(0, [0, j, k], [1, j, k]);
  for (const i of [0, 1]) for (const k of [0, 1]) add(1, [i, 0, k], [i, 1, k]);
  for (const i of [0, 1]) for (const j of [0, 1]) add(2, [i, j, 0], [i, j, 1]);

  ctx.save();
  ctx.lineWidth = 1;
  ctx.strokeStyle = dark ? "rgba(150,160,220,0.3)" : "rgba(70,80,140,0.26)";
  for (const e of edges) {
    ctx.beginPath();
    ctx.moveTo(e.a[0], e.a[1]);
    ctx.lineTo(e.b[0], e.b[1]);
    ctx.stroke();
  }
  ctx.font = "11px 'SF Mono', Menlo, monospace";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  for (const axis of [0, 1, 2] as const) {
    const same = edges.filter((e) => e.axis === axis);
    if (!same.length) continue;
    const e = same.reduce((m, q) => (q.depth < m.depth ? q : m));
    const lo = axis === 0 ? x0 : axis === 1 ? y0 : z0;
    const hi = axis === 0 ? x1 : axis === 1 ? y1 : z1;
    const t = niceTicks(lo, hi, 6);
    ctx.fillStyle = th.muted;
    ctx.strokeStyle = dark ? "rgba(150,160,220,0.55)" : "rgba(70,80,140,0.5)";
    for (let i = 0; i < t.values.length; i++) {
      const f = (t.values[i] - lo) / Math.max(1e-12, hi - lo);
      const px = e.a[0] + (e.b[0] - e.a[0]) * f;
      const py = e.a[1] + (e.b[1] - e.a[1]) * f;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px + 5, py + 5);
      ctx.stroke();
      ctx.fillText(t.labels[i], px + 15, py + 12);
    }
    ctx.fillStyle = th.accent;
    ctx.fillText(axis === 0 ? "x" : axis === 1 ? "y" : "z", e.b[0] + 10, e.b[1] - 10);
  }
  ctx.restore();
}

/** 清空网格缓存（重新定义用户函数后调用） */
export function clearSurfCache(): void {
  cache.clear();
}
