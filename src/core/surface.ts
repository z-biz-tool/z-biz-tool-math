/**
 * 曲面引擎：GeoLab 的软件 3D 底座（无 WebGL，纯 CPU 几何/投影计算）
 *
 * 覆盖 surf / mesh / surfc / contour3 / 线框：显式曲面 z=f(x,y)、参数曲面 r(u,v)、
 * 旋转体（微积分圆盘/壳层法）、隐式曲面 f(x,y,z)=c（surface nets），
 * 以及轨道相机投影 + 画家算法面片排序。
 *
 * 与求值机配合的热路径写法（compileReal 的 slots 为 re/im 交错：第 i 个变量的实部在 s[2i]）：
 *   const fast = compileReal(eng, node, ["x", "y"])!;
 *   const s = new Float64Array(4);
 *   const g = buildGraphGrid((x, y) => (s[0] = x, s[2] = y, fast(s)), [-2, 2, -2, 2, 80, 80]);
 */

import { TAU, clamp } from "./cnum.ts";

/* ------------------------------------------------------------------ */
/* 网格构造                                                            */
/* ------------------------------------------------------------------ */

export interface Mesh {
  /** 顶点扁平 xyz，长度 = nu*nv*3，索引约定 j*nu+i（u 变化最快） */
  positions: Float32Array;
  tris: Uint32Array;
  /** 每顶点的采样参数 (u,v)，供按参数着色使用 */
  uvs?: Float32Array;
  /** 网格拓扑 [nu, nv]；隐式曲面等非结构化网格为 undefined */
  params?: Float32Array;
}

/** buildParametricGrid 的写出目标：曲面上一点的三个直角坐标分量 */
export interface ParamWriter {
  set(x: number, y: number, z: number): void;
}

export type ParamSampler = (u: number, v: number, o: ParamWriter) => void;

export type GridSpec = [u0: number, u1: number, v0: number, v1: number, nu: number, nv: number];

const fin3 = (a: number, b: number, c: number): boolean =>
  Number.isFinite(a) && Number.isFinite(b) && Number.isFinite(c);

/**
 * 参数曲面网格：u = u0 + (u1-u0)*i/(nu-1)，顶点索引 j*nu+i，每格两块三角形。
 * 采样失效（回调未写出 / 分量非有限）的顶点被留在原点并打上无效标记，
 * 引用它的三角形一律不发射 —— 输出几何里不会出现 NaN。
 */
export function buildParametricGrid(f: ParamSampler, spec: GridSpec): Mesh {
  const nu = Math.max(2, Math.round(spec[4]) || 2);
  const nv = Math.max(2, Math.round(spec[5]) || 2);
  const spanU = spec[1] - spec[0];
  const spanV = spec[3] - spec[2];
  const positions = new Float32Array(nu * nv * 3);
  const uvs = new Float32Array(nu * nv * 2);
  const alive = new Uint8Array(nu * nv);
  let sx = 0;
  let sy = 0;
  let sz = 0;
  let touched = 0;
  const o: ParamWriter = {
    set(x, y, z) {
      sx = x;
      sy = y;
      sz = z;
      touched = 1;
    },
  };
  for (let j = 0; j < nv; j++) {
    const v = spec[2] + (spanV * j) / (nv - 1);
    for (let i = 0; i < nu; i++) {
      const u = spec[0] + (spanU * i) / (nu - 1);
      touched = 0;
      f(u, v, o);
      const vi = j * nu + i;
      uvs[vi * 2] = u;
      uvs[vi * 2 + 1] = v;
      if (!touched || !fin3(sx, sy, sz)) continue;
      alive[vi] = 1;
      positions[vi * 3] = sx;
      positions[vi * 3 + 1] = sy;
      positions[vi * 3 + 2] = sz;
    }
  }
  // 四边形 (a,b,c,d) 拆成 (a,b,c) 与 (b,d,c)：z=f(x,y) 时法向 ∝ (−zx,−zy,1) 朝上
  const tris = new Uint32Array(6 * (nu - 1) * (nv - 1));
  let w = 0;
  for (let j = 0; j < nv - 1; j++) {
    for (let i = 0; i < nu - 1; i++) {
      const a = j * nu + i;
      const b = a + 1;
      const c = a + nu;
      const d = c + 1;
      if (!(alive[a] && alive[b] && alive[c] && alive[d])) continue;
      tris[w++] = a;
      tris[w++] = b;
      tris[w++] = c;
      tris[w++] = b;
      tris[w++] = d;
      tris[w++] = c;
    }
  }
  return { positions, tris: tris.subarray(0, w), uvs, params: new Float32Array([nu, nv]) };
}

/** 显式曲面 z=f(x,y)：buildParametricGrid 的便捷包装 */
export function buildGraphGrid(
  f: (x: number, y: number) => number,
  spec: [x0: number, x1: number, y0: number, y1: number, nu: number, nv: number],
): Mesh {
  return buildParametricGrid((u, v, o) => o.set(u, v, f(u, v)), spec);
}

/**
 * 旋转体：平面曲线 profile(t)=[x,y] 绕坐标轴回转（圆盘/壳层法可视化）。
 * axis="x" 时半径取 |y|、轴向取 x；axis="y" 时半径取 |x|、轴向取 y。
 */
export function surfaceOfRevolution(
  profile: (t: number) => [number, number],
  tRange: [number, number],
  axis: "x" | "y",
  opts: { steps: number; segments: number; angle?: [number, number] },
): Mesh {
  const angle: [number, number] = opts.angle ?? [0, TAU];
  const steps = Math.max(2, Math.round(opts.steps) || 2);
  const segments = Math.max(3, Math.round(opts.segments) || 3);
  return buildParametricGrid(
    (u, v, o) => {
      const p = profile(u);
      const axial = axis === "x" ? p[0] : p[1];
      const radius = Math.abs(axis === "x" ? p[1] : p[0]);
      const rx = radius * Math.cos(v);
      const ry = radius * Math.sin(v);
      if (axis === "x") o.set(axial, rx, ry);
      else o.set(rx, axial, ry);
    },
    [tRange[0], tRange[1], angle[0], angle[1], steps, segments + 1],
  );
}

/* ------------------------------------------------------------------ */
/* 法向                                                                */
/* ------------------------------------------------------------------ */

/** 顶点法向：邻接面法向的面积加权平均（未归一叉积本身就是 2×面积，天然带权） */
export function surfaceNormals(m: Mesh): Float32Array {
  const pos = m.positions;
  const tris = m.tris;
  const out = new Float32Array(pos.length);
  for (let t = 0; t + 2 < tris.length; t += 3) {
    const a = tris[t] * 3;
    const b = tris[t + 1] * 3;
    const c = tris[t + 2] * 3;
    const ux = pos[b] - pos[a];
    const uy = pos[b + 1] - pos[a + 1];
    const uz = pos[b + 2] - pos[a + 2];
    const vx = pos[c] - pos[a];
    const vy = pos[c + 1] - pos[a + 1];
    const vz = pos[c + 2] - pos[a + 2];
    const nx = uy * vz - uz * vy;
    const ny = uz * vx - ux * vz;
    const nz = ux * vy - uy * vx;
    out[a] += nx;
    out[a + 1] += ny;
    out[a + 2] += nz;
    out[b] += nx;
    out[b + 1] += ny;
    out[b + 2] += nz;
    out[c] += nx;
    out[c + 1] += ny;
    out[c + 2] += nz;
  }
  for (let i = 0; i < out.length; i += 3) {
    const len = Math.hypot(out[i], out[i + 1], out[i + 2]);
    if (len < 1e-30) {
      // 退化/孤立顶点指向 +z，避免光照出现 NaN
      out[i] = 0;
      out[i + 1] = 0;
      out[i + 2] = 1;
    } else {
      out[i] /= len;
      out[i + 1] /= len;
      out[i + 2] /= len;
    }
  }
  return out;
}

/** 图形曲面 z=g(x,y) 的单位法向 ∝ (−zx, −zy, 1)，偏导用中心差分 */
export function normalAt(
  g: (x: number, y: number) => number,
  x: number,
  y: number,
): [number, number, number] {
  const h = 1e-4 * Math.max(1, Math.abs(x), Math.abs(y));
  const zx = (g(x + h, y) - g(x - h, y)) / (2 * h);
  const zy = (g(x, y + h) - g(x, y - h)) / (2 * h);
  const len = Math.hypot(zx, zy, 1);
  if (!Number.isFinite(len) || len < 1e-30) return [0, 0, 1];
  return [-zx / len, -zy / len, 1 / len];
}

/* ------------------------------------------------------------------ */
/* 相机与投影                                                          */
/* ------------------------------------------------------------------ */

export interface Camera {
  /** 世界→相机 4×4 矩阵，列主序（与 mat4mul 一致） */
  view: Float64Array;
  right: [number, number, number];
  up: [number, number, number];
  /** 视线方向（由相机指向 target） */
  fwd: [number, number, number];
  pos: [number, number, number];
  /** 透视像素焦距：screen = 中心 + focal·x_cam/depth */
  focal: number;
  width: number;
  height: number;
  ortho?: boolean;
  /** 正交像素/世界单位比（取为在 target 深度处与透视等效） */
  pxPerUnit?: number;
  dist?: number;
}

export interface CameraOpts {
  azim: number;
  elev: number;
  dist: number;
  target: [number, number, number];
  width: number;
  height: number;
  /** 水平视野角（弧度），默认 1 */
  fov?: number;
  ortho?: boolean;
}

/**
 * 右手轨道相机（世界 z 轴向上）。相机位于 target + dist·(cosθ·sinφ, cosθ·cosφ, sinθ)，
 * elev 夹离 ±π/2 防万向节翻转；三轴满足 right × up = −fwd。
 */
export function orbitCamera(opts: CameraOpts): Camera {
  const width = Math.max(1, opts.width);
  const height = Math.max(1, opts.height);
  const fov = clamp(opts.fov ?? 1, 0.05, Math.PI - 0.05);
  const dist = Math.max(1e-6, opts.dist);
  const elev = clamp(opts.elev, -Math.PI / 2 + 0.02, Math.PI / 2 - 0.02);
  const ce = Math.cos(elev);
  const se = Math.sin(elev);
  const cp = Math.cos(opts.azim);
  const sp = Math.sin(opts.azim);
  const pos: [number, number, number] = [
    opts.target[0] + dist * ce * sp,
    opts.target[1] + dist * ce * cp,
    opts.target[2] + dist * se,
  ];
  const fwd: [number, number, number] = [-ce * sp, -ce * cp, -se];
  const right: [number, number, number] = [-cp, sp, 0];
  const up: [number, number, number] = [-se * sp, -se * cp, ce];
  const focal = width / 2 / Math.tan(fov / 2);
  const view = new Float64Array(16);
  view[0] = right[0];
  view[1] = up[0];
  view[2] = -fwd[0];
  view[4] = right[1];
  view[5] = up[1];
  view[6] = -fwd[1];
  view[8] = right[2];
  view[9] = up[2];
  view[10] = -fwd[2];
  view[12] = -(right[0] * pos[0] + right[1] * pos[1] + right[2] * pos[2]);
  view[13] = -(up[0] * pos[0] + up[1] * pos[1] + up[2] * pos[2]);
  view[14] = fwd[0] * pos[0] + fwd[1] * pos[1] + fwd[2] * pos[2];
  view[15] = 1;
  return {
    view,
    right,
    up,
    fwd,
    pos,
    focal,
    width,
    height,
    ortho: !!opts.ortho,
    pxPerUnit: focal / dist,
    dist,
  };
}

export interface Face {
  i0: number;
  i1: number;
  i2: number;
  /** 相机空间深度（沿视线为正，越大越远） */
  depth: number;
  /** 屏幕坐标 x0,y0,x1,y1,x2,y2（y 向下，可直接喂给 canvas） */
  pts: number[];
  /** 世界法向：有顶点法向时取其平均（平滑着色），否则为面法向 */
  n: [number, number, number];
  /** 平均高度，供色图取色 */
  z: number;
}

export interface SceneProjection {
  faces: Face[];
  /** 每顶点屏幕坐标 sx,sy,depth；depth<0 表示该点落在相机后方，线框需跳过 */
  verts: number[];
}

export interface ProjectOpts {
  /** 背面剔除，默认 false（数学曲面常需双面观察） */
  cull?: boolean;
  /** 额外的像素/世界缩放（缩放视图用） */
  scale?: number;
  /** 透视近裁剪面（世界单位），默认 1e-9 */
  near?: number;
}

/**
 * 场景投影 + 画家算法排序（结果按 depth 从远到近）。
 * 第 4 参兼容旧签名：传数字视为历史「网格尺寸」，等价于不给顶点法向（改用面法向）；只有 Float32Array 才做平滑法向。
 */
export function projectScene(
  positions: Float32Array,
  tris: Uint32Array,
  cam: Camera,
  vertexNormals?: Float32Array | number,
  opts?: ProjectOpts,
): SceneProjection {
  const k = (cam.ortho ? (cam.pxPerUnit ?? cam.focal) : cam.focal) * (opts?.scale ?? 1);
  const cull = !!opts?.cull;
  const near = opts?.near ?? 1e-9;
  const vc = Math.floor(positions.length / 3);
  const halfW = cam.width / 2;
  const halfH = cam.height / 2;
  const scr = new Float64Array(vc * 3);
  const vis = new Uint8Array(vc);
  const [rx, ry, rz] = cam.right;
  const [ux, uy, uz] = cam.up;
  const [wx, wy, wz] = cam.fwd;
  for (let i = 0; i < vc; i++) {
    const dx = positions[i * 3] - cam.pos[0];
    const dy = positions[i * 3 + 1] - cam.pos[1];
    const dz = positions[i * 3 + 2] - cam.pos[2];
    const depth = dx * wx + dy * wy + dz * wz;
    if (!cam.ortho && !(depth > near)) continue;
    const denom = cam.ortho ? 1 : depth;
    const sx = halfW + (k * (dx * rx + dy * ry + dz * rz)) / denom;
    const sy = halfH - (k * (dx * ux + dy * uy + dz * uz)) / denom;
    if (!fin3(sx, sy, depth)) continue;
    scr[i * 3] = sx;
    scr[i * 3 + 1] = sy;
    scr[i * 3 + 2] = depth;
    vis[i] = 1;
  }
  const verts: number[] = new Array(vc * 3);
  for (let i = 0; i < vc; i++) {
    const o = i * 3;
    verts[o] = vis[i] ? scr[o] : 0;
    verts[o + 1] = vis[i] ? scr[o + 1] : 0;
    verts[o + 2] = vis[i] ? scr[o + 2] : -1;
  }
  // 数字形的第 4 参（旧「网格尺寸」）在此退化为 null
  const smooth =
    typeof vertexNormals === "object" &&
    vertexNormals !== null &&
    vertexNormals.length >= positions.length
      ? vertexNormals
      : null;
  const faces: Face[] = [];
  for (let t = 0; t + 2 < tris.length; t += 3) {
    const i0 = tris[t];
    const i1 = tris[t + 1];
    const i2 = tris[t + 2];
    if (i0 >= vc || i1 >= vc || i2 >= vc) continue;
    if (!(vis[i0] && vis[i1] && vis[i2])) continue;
    const a = i0 * 3;
    const b = i1 * 3;
    const c = i2 * 3;
    const ex = positions[b] - positions[a];
    const ey = positions[b + 1] - positions[a + 1];
    const ez = positions[b + 2] - positions[a + 2];
    const fx = positions[c] - positions[a];
    const fy = positions[c + 1] - positions[a + 1];
    const fz = positions[c + 2] - positions[a + 2];
    let nx = ey * fz - ez * fy;
    let ny = ez * fx - ex * fz;
    let nz = ex * fy - ey * fx;
    const len = Math.hypot(nx, ny, nz);
    if (!(len > 1e-30)) continue; // 退化面片不入场景
    nx /= len;
    ny /= len;
    nz /= len;
    if (smooth) {
      let vx = smooth[a] + smooth[b] + smooth[c];
      let vy = smooth[a + 1] + smooth[b + 1] + smooth[c + 1];
      let vz = smooth[a + 2] + smooth[b + 2] + smooth[c + 2];
      const l = Math.hypot(vx, vy, vz);
      if (l > 1e-30) {
        // 平滑法向须与几何法向同侧，否则绕序不一致的网格明暗会翻转
        const sgn = vx * nx + vy * ny + vz * nz < 0 ? -1 : 1;
        nx = (sgn * vx) / l;
        ny = (sgn * vy) / l;
        nz = (sgn * vz) / l;
      }
    }
    if (cull && nx * wx + ny * wy + nz * wz > 0) continue;
    const depth = (scr[a + 2] + scr[b + 2] + scr[c + 2]) / 3;
    faces.push({
      i0,
      i1,
      i2,
      depth,
      pts: [scr[a], scr[a + 1], scr[b], scr[b + 1], scr[c], scr[c + 1]],
      n: [nx, ny, nz],
      z: (positions[a + 2] + positions[b + 2] + positions[c + 2]) / 3,
    });
  }
  faces.sort((p, q) => q.depth - p.depth);
  return { faces, verts };
}

/* ------------------------------------------------------------------ */
/* 曲面上的等高线（contour3 / surfc）                                  */
/* ------------------------------------------------------------------ */

export interface MeshContour {
  level: number;
  polylines: number[][][];
}

/**
 * 高度等高线：逐三角形求棱上交点，再以「三角形棱」为键焊接成 3D 折线链。
 * 同一条棱被两个三角形共用时得到完全相同的交点，因此不需要浮点容差匹配。
 */
export function meshContours(m: Mesh, levels: number[]): MeshContour[] {
  const pos = m.positions;
  const tris = m.tris;
  const vc = Math.max(1, Math.floor(pos.length / 3));
  const crossOf = new Map<number, number>();
  const pts: number[] = [];
  const segs: number[] = [];
  const adj = new Map<number, number[]>();
  const v3 = [0, 0, 0];
  const d3 = [0, 0, 0];
  const hit: number[] = [];
  const out: MeshContour[] = [];
  for (const level of levels) {
    crossOf.clear();
    pts.length = 0;
    segs.length = 0;
    adj.clear();
    for (let t = 0; t + 2 < tris.length; t += 3) {
      v3[0] = tris[t];
      v3[1] = tris[t + 1];
      v3[2] = tris[t + 2];
      for (let e = 0; e < 3; e++) d3[e] = pos[v3[e] * 3 + 2] - level;
      hit.length = 0;
      for (let e = 0; e < 3; e++) {
        const a = v3[e];
        const b = v3[(e + 1) % 3];
        const da = d3[e];
        const db = d3[(e + 1) % 3];
        if (!(da < 0 !== db < 0)) continue;
        const key = a < b ? a * vc + b : b * vc + a;
        let id = crossOf.get(key);
        if (id === undefined) {
          // 端点恰在水平面上时 da=0 → w=0，交点即该顶点
          const w = da === db ? 0 : da / (da - db);
          id = pts.length / 3;
          pts.push(
            pos[a * 3] + (pos[b * 3] - pos[a * 3]) * w,
            pos[a * 3 + 1] + (pos[b * 3 + 1] - pos[a * 3 + 1]) * w,
            pos[a * 3 + 2] + (pos[b * 3 + 2] - pos[a * 3 + 2]) * w,
          );
          crossOf.set(key, id);
        }
        hit.push(id);
      }
      if (hit.length === 2 && hit[0] !== hit[1]) {
        segs.push(hit[0], hit[1]);
        const s = (segs.length >> 1) - 1;
        for (const e of [hit[0], hit[1]]) {
          const l = adj.get(e);
          if (l) l.push(s);
          else adj.set(e, [s]);
        }
      }
    }
    const used = new Uint8Array(segs.length >> 1);
    const polylines: number[][][] = [];
    const at = (id: number): number[] => [pts[id * 3], pts[id * 3 + 1], pts[id * 3 + 2]];
    // 从种子段两端各自沿未用段延伸；闭曲线会绕回起点
    const walk = (from: number, cap: number): number[] => {
      const chain: number[] = [];
      let cur = from;
      for (;;) {
        const l = adj.get(cur);
        let nx = -1;
        if (l) for (const q of l) if (!used[q]) { nx = q; break; }
        if (nx < 0 || chain.length > cap) break;
        used[nx] = 1;
        cur = segs[nx * 2] === cur ? segs[nx * 2 + 1] : segs[nx * 2];
        chain.push(cur);
      }
      return chain;
    };
    for (let s = 0; s < segs.length; s += 2) {
      if (used[s >> 1]) continue;
      used[s >> 1] = 1;
      const back = walk(segs[s], segs.length);
      const fwd = walk(segs[s + 1], segs.length);
      back.reverse();
      const ids = back.concat(segs[s], segs[s + 1], fwd);
      polylines.push(ids.map(at));
    }
    out.push({ level, polylines });
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* 隐式曲面 f(x,y,z)=c：surface nets（对偶网格，不用 256 例 marching-cubes 表） */
/* ------------------------------------------------------------------ */

const CORNER_D: [number, number, number][] = [
  [0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1],
];
/** 单元体的 12 条棱：前 4 条沿 x，中 4 条沿 y，末 4 条沿 z */
const CORNER_E: [number, number][] = [
  [0, 1], [2, 3], [4, 5], [6, 7],
  [0, 2], [1, 3], [4, 6], [5, 7],
  [0, 4], [1, 5], [2, 6], [3, 7],
];

/**
 * 每个含符号变化的单元格产生一个对偶顶点：12 条棱的线性交点均值 → 沿 ∇f 做带
 * 步长回撤的 Newton 投影（真正落到等值面上，去掉棱角的阶梯感）；再以「有交点的棱」
 * 为中心，把环绕该棱的至多 4 个对偶顶点连成四边形——棱被共用，故网格保持流形。
 */
export function implicitMesh(
  f: (x: number, y: number, z: number) => number,
  box: [number, number, number, number, number, number],
  res: number,
  level = 0,
  iterations = 3,
): Mesh {
  const n = Math.round(clamp(res, 4, 128));
  const N = n + 1;
  const hx = (box[1] - box[0]) / n;
  const hy = (box[3] - box[2]) / n;
  const hz = (box[5] - box[4]) / n;
  const px0 = box[0];
  const py0 = box[2];
  const pz0 = box[4];
  const off = (i: number, j: number, k: number): number => (k * N + j) * N + i;
  const cellAt = (i: number, j: number, k: number): number =>
    i < 0 || j < 0 || k < 0 || i >= n || j >= n || k >= n ? -1 : (k * n + j) * n + i;
  const val = new Float32Array(N * N * N);
  for (let k = 0; k < N; k++) {
    const z = pz0 + hz * k;
    for (let j = 0; j < N; j++) {
      const y = py0 + hy * j;
      for (let i = 0; i < N; i++) val[off(i, j, k)] = f(px0 + hx * i, y, z) - level;
    }
  }
  const hs = 0.5 * Math.min(hx, hy, hz);
  const grad = (x: number, y: number, z: number, o: number[]): void => {
    o[0] = (f(x + hs, y, z) - f(x - hs, y, z)) / (2 * hs);
    o[1] = (f(x, y + hs, z) - f(x, y - hs, z)) / (2 * hs);
    o[2] = (f(x, y, z + hs) - f(x, y, z - hs)) / (2 * hs);
  };
  const cellId = new Int32Array(n * n * n).fill(-1);
  const positions: number[] = [];
  const grads: number[] = [];
  const co = new Int32Array(8);
  const cx = new Float64Array(8);
  const cy = new Float64Array(8);
  const cz = new Float64Array(8);
  const g = [0, 0, 0];
  for (let k = 0; k < n; k++) {
    for (let j = 0; j < n; j++) {
      for (let i = 0; i < n; i++) {
        let lo = Infinity;
        let hi = -Infinity;
        let bad = 0;
        for (let q = 0; q < 8; q++) {
          const d = CORNER_D[q];
          const o = off(i + d[0], j + d[1], k + d[2]);
          co[q] = o;
          const v = val[o];
          if (!Number.isFinite(v)) {
            bad = 1;
            break;
          }
          if (v < lo) lo = v;
          if (v > hi) hi = v;
        }
        const cid = cellAt(i, j, k);
        if (bad || lo > 0 || hi < 0) {
          cellId[cid] = -2; // 不与等值面相交，或采样失效
          continue;
        }
        let sumx = 0;
        let sumy = 0;
        let sumz = 0;
        let cnt = 0;
        for (let q = 0; q < 8; q++) {
          const d = CORNER_D[q];
          cx[q] = px0 + hx * (i + d[0]);
          cy[q] = py0 + hy * (j + d[1]);
          cz[q] = pz0 + hz * (k + d[2]);
        }
        for (const [ea, eb] of CORNER_E) {
          const va = val[co[ea]];
          const vb = val[co[eb]];
          if (!(va < 0 !== vb < 0)) continue;
          const w = va / (va - vb);
          sumx += cx[ea] + (cx[eb] - cx[ea]) * w;
          sumy += cy[ea] + (cy[eb] - cy[ea]) * w;
          sumz += cz[ea] + (cz[eb] - cz[ea]) * w;
          cnt++;
        }
        if (cnt === 0) {
          cellId[cid] = -2;
          continue;
        }
        let vx = sumx / cnt;
        let vy = sumy / cnt;
        let vz = sumz / cnt;
        // 完整 Newton 步 → 逐次减半，只接受让 |f−c| 下降的迭代（球面等情形防 2-循环过冲）
        let best = Math.abs(f(vx, vy, vz) - level);
        for (let it = 0; it < iterations; it++) {
          grad(vx, vy, vz, g);
          const d2 = g[0] * g[0] + g[1] * g[1] + g[2] * g[2];
          if (!(d2 > 1e-30)) break;
          const fv = f(vx, vy, vz) - level;
          let moved = false;
          for (let q = 0; q < 4 && !moved; q++) {
            const a = Math.pow(0.5, q);
            const tx = vx - (g[0] * fv * a) / d2;
            const ty = vy - (g[1] * fv * a) / d2;
            const tz = vz - (g[2] * fv * a) / d2;
            const r = Math.abs(f(tx, ty, tz) - level);
            if (Number.isFinite(r) && r < best) {
              vx = tx;
              vy = ty;
              vz = tz;
              best = r;
              moved = true;
            }
          }
          if (!moved) break;
        }
        grad(vx, vy, vz, g);
        cellId[cid] = positions.length / 3;
        positions.push(vx, vy, vz);
        grads.push(g[0], g[1], g[2]);
      }
    }
  }
  const tris: number[] = [];
  const pushTri = (a: number, b: number, c: number): void => {
    if (a === b || b === c || a === c) return;
    const ox = positions[a * 3] - positions[b * 3];
    const oy = positions[a * 3 + 1] - positions[b * 3 + 1];
    const oz = positions[a * 3 + 2] - positions[b * 3 + 2];
    const px = positions[a * 3] - positions[c * 3];
    const py = positions[a * 3 + 1] - positions[c * 3 + 1];
    const pz = positions[a * 3 + 2] - positions[c * 3 + 2];
    const nx = oy * pz - oz * py;
    const ny = oz * px - ox * pz;
    const nz = ox * py - oy * px;
    // ∇f 指向 f 增大侧（f<c 为“内”），据此把绕序统一成外法向
    const gx = grads[a * 3] + grads[b * 3] + grads[c * 3];
    const gy = grads[a * 3 + 1] + grads[b * 3 + 1] + grads[c * 3 + 1];
    const gz = grads[a * 3 + 2] + grads[b * 3 + 2] + grads[c * 3 + 2];
    if (nx * gx + ny * gy + nz * gz < 0) tris.push(a, c, b);
    else tris.push(a, b, c);
  };
  const fan = (c0: number, c1: number, c2: number, c3: number): void => {
    const ids: number[] = [];
    for (const c of [c0, c1, c2, c3]) {
      if (c >= 0 && cellId[c] >= 0) ids.push(cellId[c]);
    }
    if (ids.length === 4) {
      pushTri(ids[0], ids[1], ids[2]);
      pushTri(ids[0], ids[2], ids[3]);
    } else if (ids.length === 3) {
      pushTri(ids[0], ids[1], ids[2]);
    }
  };
  for (let k = 0; k < N; k++) {
    for (let j = 0; j < N; j++) {
      for (let i = 0; i < n; i++) {
        if (!(val[off(i, j, k)] < 0 !== val[off(i + 1, j, k)] < 0)) continue;
        // 绕 x 棱的环序：(y,z) 平面内逆时针
        fan(cellAt(i, j - 1, k - 1), cellAt(i, j, k - 1), cellAt(i, j, k), cellAt(i, j - 1, k));
      }
    }
  }
  for (let k = 0; k < N; k++) {
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < n; j++) {
        if (!(val[off(i, j, k)] < 0 !== val[off(i, j + 1, k)] < 0)) continue;
        // 绕 y 棱的环序：(z,x) 平面内逆时针
        fan(cellAt(i - 1, j, k - 1), cellAt(i - 1, j, k), cellAt(i, j, k), cellAt(i, j, k - 1));
      }
    }
  }
  for (let j = 0; j < N; j++) {
    for (let i = 0; i < N; i++) {
      for (let k = 0; k < n; k++) {
        if (!(val[off(i, j, k)] < 0 !== val[off(i, j, k + 1)] < 0)) continue;
        // 绕 z 棱的环序：(x,y) 平面内逆时针
        fan(cellAt(i - 1, j - 1, k), cellAt(i, j - 1, k), cellAt(i, j, k), cellAt(i - 1, j, k));
      }
    }
  }
  return { positions: new Float32Array(positions), tris: new Uint32Array(tris) };
}

/* ------------------------------------------------------------------ */
/* 矩阵糖与网格统计/线框                                               */
/* ------------------------------------------------------------------ */

export function mat4id(): Float64Array {
  const m = new Float64Array(16);
  m[0] = 1;
  m[5] = 1;
  m[10] = 1;
  m[15] = 1;
  return m;
}

/** 列主序 a·b（先应用 b，再应用 a） */
export function mat4mul(a: Float64Array, b: Float64Array): Float64Array {
  const o = new Float64Array(16);
  for (let c = 0; c < 4; c++) {
    for (let r = 0; r < 4; r++) {
      o[c * 4 + r] =
        a[r] * b[c * 4] +
        a[4 + r] * b[c * 4 + 1] +
        a[8 + r] * b[c * 4 + 2] +
        a[12 + r] * b[c * 4 + 3];
    }
  }
  return o;
}

export function mat4rotY(angle: number): Float64Array {
  const m = mat4id();
  const c = Math.cos(angle);
  const s = Math.sin(angle);
  m[0] = c;
  m[2] = -s;
  m[8] = s;
  m[10] = c;
  return m;
}

/** 右手透视矩阵（相机前向为 −z，与 orbitCamera 的 view 配套） */
export function mat4perspective(fov: number, aspect: number, near: number, far: number): Float64Array {
  const t = 1 / Math.tan(clamp(fov, 0.01, Math.PI - 0.01) / 2);
  const m = new Float64Array(16);
  m[0] = t / Math.max(1e-9, aspect);
  m[5] = t;
  m[10] = (far + near) / (near - far);
  m[11] = -1;
  m[14] = (2 * far * near) / (near - far);
  return m;
}

/** 施加 4×4 变换（拓扑与参数保持不变） */
export function transformMesh(m: Mesh, m4: Float64Array): Mesh {
  const src = m.positions;
  const out = new Float32Array(src.length);
  for (let i = 0; i < src.length; i += 3) {
    const x = src[i];
    const y = src[i + 1];
    const z = src[i + 2];
    out[i] = m4[0] * x + m4[4] * y + m4[8] * z + m4[12];
    out[i + 1] = m4[1] * x + m4[5] * y + m4[9] * z + m4[13];
    out[i + 2] = m4[2] * x + m4[6] * y + m4[10] * z + m4[14];
  }
  return { positions: out, tris: m.tris, uvs: m.uvs, params: m.params };
}

/** 轴对齐包围盒（跳过无效槽位与 NaN），供视图自适应 */
export function meshBounds(m: Mesh): [number, number, number, number, number, number] {
  const live = usedMask(m);
  const p = m.positions;
  const r = [Infinity, -Infinity, Infinity, -Infinity, Infinity, -Infinity];
  for (let i = 0; i < live.length; i++) {
    if (!live[i]) continue;
    for (let a = 0; a < 3; a++) {
      const v = p[i * 3 + a];
      if (v < r[a * 2]) r[a * 2] = v;
      if (v > r[a * 2 + 1]) r[a * 2 + 1] = v;
    }
  }
  if (!Number.isFinite(r[0])) return [0, 0, 0, 0, 0, 0];
  return [r[0], r[1], r[2], r[3], r[4], r[5]];
}

/** 被三角形引用且有限的顶点掩码：结构化网格里的无效采样点不会进入统计 */
function usedMask(m: Mesh): Uint8Array {
  const vc = Math.floor(m.positions.length / 3);
  const mask = new Uint8Array(vc);
  for (let t = 0; t + 2 < m.tris.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const i = m.tris[t + e];
      if (i < vc) mask[i] = 1;
    }
  }
  return mask;
}

/**
 * 线框边集（扁平顶点对 i0,i1,…）：结构化网格取参数线（MATLAB mesh 观感，无对角线），
 * 其余网格退化为三角形棱去重。
 */
export function meshEdges(m: Mesh): Uint32Array {
  const p = m.params;
  const out: number[] = [];
  if (p && p.length >= 2) {
    const nu = p[0];
    const nv = p[1];
    if (nu * nv === Math.floor(m.positions.length / 3)) {
      const mask = usedMask(m);
      const link = (a: number, b: number): void => {
        if (mask[a] && mask[b]) out.push(a, b);
      };
      for (let j = 0; j < nv; j++) for (let i = 0; i + 1 < nu; i++) link(j * nu + i, j * nu + i + 1);
      for (let j = 0; j + 1 < nv; j++) for (let i = 0; i < nu; i++) link(j * nu + i, (j + 1) * nu + i);
      return new Uint32Array(out);
    }
  }
  const vc = Math.max(1, Math.floor(m.positions.length / 3));
  const seen = new Set<number>();
  for (let t = 0; t + 2 < m.tris.length; t += 3) {
    for (let e = 0; e < 3; e++) {
      const a = m.tris[t + e];
      const b = m.tris[t + ((e + 1) % 3)];
      const key = a < b ? a * vc + b : b * vc + a;
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(a, b);
    }
  }
  return new Uint32Array(out);
}
