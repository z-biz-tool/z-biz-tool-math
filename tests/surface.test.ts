// 曲面引擎 surface.ts 的纯逻辑回归（软件 3D 底座，无 WebGL）。
// 测什么：参数网格构造、法向、4×4 矩阵、轨道相机、画家算法投影、网格等高线、隐式曲面 surface nets。
// 为什么该测：这一层全是"几何量算错不会报错、只会画错"的代码——法向朝里画面就是黑的、
//   包围盒把死顶点算进去会让视图飞出屏幕、投影深度排序错了就是画家算法穿帮。
//   全部靠 CPU 数组计算，没有 DOM/Canvas 依赖，所以可以纯单测。
// 跑法：node --experimental-strip-types --test tests/*.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  buildGraphGrid,
  buildParametricGrid,
  implicitMesh,
  mat4id,
  mat4mul,
  mat4perspective,
  mat4rotY,
  meshBounds,
  meshContours,
  meshEdges,
  normalAt,
  orbitCamera,
  projectScene,
  surfaceNormals,
  surfaceOfRevolution,
  transformMesh,
  type Mesh,
} from "../src/core/surface.ts";

/* ---------------------------------------------------------------- 断言助手 */

function near(actual: number, expect: number, tol = 1e-6, msg?: string): void {
  assert.ok(
    Math.abs(actual - expect) < tol,
    `${msg ?? "值不相等"}：实际 ${actual}，期望 ${expect}（容差 ${tol}）`,
  );
}
function nearVec(actual: ArrayLike<number>, expect: number[], tol = 1e-6): void {
  for (let i = 0; i < expect.length; i++) near(actual[i], expect[i], tol, `分量 ${i}`);
}
const vtx = (m: Mesh, i: number): [number, number, number] => [
  m.positions[i * 3],
  m.positions[i * 3 + 1],
  m.positions[i * 3 + 2],
];

/* ============================================================ 参数网格 */

test("buildGraphGrid：顶点数 nu·nv、每格两三角形、顶点序 j·nu+i", () => {
  const m = buildGraphGrid(() => 0, [0, 1, 0, 1, 2, 2]);
  assert.equal(m.positions.length, 2 * 2 * 3);
  assert.equal(m.tris.length, 6, "1 个格子 = 2 个三角形");
  assert.deepEqual([...m.tris], [0, 1, 2, 1, 3, 2], "拆成 (a,b,c) 与 (b,d,c)");
  assert.deepEqual([...m.params!], [2, 2], "params 记录拓扑");
  assert.deepEqual([...m.uvs!], [0, 0, 1, 0, 0, 1, 1, 1], "uvs = (u,v) 采样参数");
  for (const [nu, nv] of [[3, 3], [5, 2], [2, 7]]) {
    const g = buildGraphGrid(() => 0, [0, 1, 0, 1, nu, nv]);
    assert.equal(g.positions.length / 3, nu * nv, `${nu}×${nv} 顶点数`);
    assert.equal(g.tris.length / 6, (nu - 1) * (nv - 1), `${nu}×${nv} 三角形数`);
  }
});

test("网格分辨率被 Math.max(2, round(n) || 2) 兜底，逆序区间不报错", () => {
  // nu=1 会让 (nu−1)=0 出现除零/空循环，必须被抬到 2
  for (const [nu, nv] of [[1, 1], [0, 0], [NaN, 4], [-3, 2]] as number[][]) {
    const m = buildGraphGrid(() => 0, [0, 1, 0, 1, nu!, nv!]);
    const nu2 = Math.max(2, Math.round(nu!) || 2);
    const nv2 = Math.max(2, Math.round(nv!) || 2);
    assert.equal(m.positions.length / 3, nu2 * nv2, `nu=${nu} nv=${nv} 的顶点数`);
    assert.equal(m.tris.length, 6 * (nu2 - 1) * (nv2 - 1), `nu=${nu} nv=${nv} 的三角形数`);
  }
  // u0 > u1 时参数从 u0 递减到 u1
  const rev = buildGraphGrid((x) => x, [2, 0, 0, 1, 3, 2]);
  near(rev.positions[0], 2, 1e-6, "i=0 → u0=2");
  near(rev.positions[6], 0, 1e-6, "i=2 → u1=0");
  // 采样落点：u = u0 + span·i/(nu−1)
  const lin = buildParametricGrid((u, v, o) => o.set(u, v, u + 2 * v), [0, 2, 0, 3, 3, 4]);
  nearVec(vtx(lin, 1 * 3 + 1), [1, 1, 3], 1e-6, "i=1,j=1");
  nearVec(vtx(lin, 3 * 3 + 2), [2, 3, 8], 1e-6, "i=2,j=3");
});

test("采样失效的顶点不产生三角形：未写出 / NaN / 部分失效", () => {
  // 这是"输出几何里不会出现 NaN"这条承诺的唯一保障
  const noWrite = buildParametricGrid(() => {}, [0, 1, 0, 1, 4, 4]);
  assert.equal(noWrite.tris.length, 0, "回调不写出 → 没有三角形");
  assert.ok([...noWrite.positions].every((v) => v === 0), "无效顶点留在原点");
  const nanMesh = buildParametricGrid((u, v, o) => o.set(u, v, u === v ? NaN : 1), [0, 1, 0, 1, 2, 2]);
  assert.equal(nanMesh.tris.length, 0, "写出 NaN 的顶点连带整格失效");
  // 只有一部分列失效时，有效区域照常成网格
  const partial = buildParametricGrid((u, v, o) => { if (u < 0.75) o.set(u, v, 0); }, [0, 1, 0, 1, 4, 4]);
  assert.ok(partial.tris.length > 0, "存活区域仍出网格");
  assert.ok([...partial.tris].every((i) => i < 16), "不会引用不存在的顶点");
});

/* ============================================================ 旋转体 */

test("surfaceOfRevolution：半径取垂直于轴的分量，绕 y 轴把 x 映成 z", () => {
  const cylY = surfaceOfRevolution((t) => [1, t], [0, 2], "y", { steps: 5, segments: 6 });
  assert.deepEqual([...cylY.params!], [5, 7], "网格是 steps × (segments+1)，多一列以便闭合");
  nearVec(vtx(cylY, 0), [1, 0, 0], 1e-6, "t=0, v=0");
  near(vtx(cylY, 4)[1], 2, 1e-6, "u 取末列（i=nu−1）→ 轴向 y=2");
  // axis="x"：轴向取 x、半径取 |y|
  const cylX = surfaceOfRevolution((t) => [t, 1], [0, 2], "x", { steps: 5, segments: 6 });
  nearVec(vtx(cylX, 0), [0, 1, 0], 1e-6, "轴向 = x = 0，半径 1 落在 (y,z)");
  // 半径取绝对值：负的 profile 也会折到正侧
  const neg = surfaceOfRevolution((t) => [-1, t], [0, 1], "y", { steps: 3, segments: 4 });
  near(neg.positions[0], 1, 1e-6, "半径取 |x|");
  // 角区间可只扫半圈
  const half = surfaceOfRevolution((t) => [t, 1], [0, 1], "y", { steps: 4, segments: 4, angle: [0, Math.PI] });
  assert.deepEqual([...half.params!], [4, 5]);
  // 退化参数被兜底
  assert.deepEqual(
    [...surfaceOfRevolution((t) => [t, 1], [0, 1], "y", { steps: 0, segments: 0 }).params!],
    [2, 4],
    "steps≥2、segments≥3",
  );
});

/* ================================================================ 法向 */

test("surfaceNormals：面积加权后归一，退化/孤立顶点兜底成 +z", () => {
  const flat = surfaceNormals(buildGraphGrid(() => 0, [0, 1, 0, 1, 3, 3]));
  nearVec(flat, [0, 0, 1], 1e-6, "z=const 平面法向朝上");
  for (let i = 0; i < flat.length; i += 3) {
    near(Math.hypot(flat[i], flat[i + 1], flat[i + 2]), 1, 1e-6, `顶点 ${i / 3} 应为单位向量`);
  }
  // z=x 的法向 ∝ (−1,0,1)/√2
  nearVec(surfaceNormals(buildGraphGrid((x) => x, [0, 1, 0, 1, 3, 3])), [-Math.SQRT1_2, 0, Math.SQRT1_2], 1e-6);
  // 锥面顶点不共面，平均后仍是单位向量
  const cone = surfaceNormals(buildGraphGrid((x, y) => Math.hypot(x, y), [0, 1, 0, 1, 5, 5]));
  for (let i = 0; i < cone.length; i += 3) {
    near(Math.hypot(cone[i], cone[i + 1], cone[i + 2]), 1, 1e-5, "全部顶点法向都是单位向量");
  }
  // 无三角形 → 空数组
  assert.deepEqual([...surfaceNormals({ positions: new Float32Array(0), tris: new Uint32Array(0) })], []);
  // 退化三角形（a=a=a）叉积为零 → 兜底 +z，避免光照 NaN
  const degen: Mesh = { positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 0, 0]), tris: new Uint32Array([0, 0, 0]) };
  nearVec(surfaceNormals(degen), [0, 0, 1, 0, 0, 1, 0, 0, 1], 1e-9);
});

test("normalAt：中心差分求偏导，∝(−zx,−zy,1)；非有限退化回 +z", () => {
  nearVec(normalAt(() => 0, 0, 0), [0, 0, 1], 1e-6, "平面");
  nearVec(normalAt((x) => x, 0.3, 0.4), [-Math.SQRT1_2, 0, Math.SQRT1_2], 1e-6, "z=x");
  const c = 1 / Math.sqrt(3);
  nearVec(normalAt((x, y) => x + y, 0, 0), [-c, -c, c], 1e-6, "z=x+y");
  nearVec(normalAt(() => NaN, 0, 0), [0, 0, 1], 1e-12, "NaN 曲面兜底");
  // 陡坡不溢出
  const n = normalAt((x) => 1e18 * x, 0, 0);
  assert.ok(n.every(Number.isFinite), "极陡坡也不该出 NaN/Inf");
  near(n[2], 0, 1e-12, "法向几乎贴着坡面");
});

/* ============================================================== 4×4 矩阵 */

test("mat4id/mat4mul/mat4rotY：列主序，旋转四次回到单位阵", () => {
  nearVec(mat4id(), [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1], 1e-12);
  nearVec(mat4mul(mat4id(), mat4id()), mat4id(), 1e-12, "单位阵是乘法单位元");
  const R = mat4rotY(Math.PI / 2);
  nearVec(R, [0, 0, -1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1], 1e-12);
  let m = mat4id();
  for (let i = 0; i < 4; i++) m = mat4mul(m, R);
  nearVec(m, mat4id(), 1e-12, "R⁴ = I");
  // 顺序敏感：先 R 再 T 与先 T 再 R 结果不同（mat4mul(a,b) 是先 b 后 a）
  const T = new Float64Array(16);
  T[12] = 1; T[13] = 2; T[14] = 3;
  const a = mat4mul(T, R);
  const b = mat4mul(R, T);
  nearVec([a[12], a[13], a[14]], [1, 2, 3], 1e-12, "T·R 的平移分量就是 T 自己的");
  nearVec([b[12], b[13], b[14]], [3, 2, -1], 1e-12, "R·T 的平移分量被旋转过");
});

test("mat4perspective：fov 夹紧、aspect 有下限，w 分量恒为 −z", () => {
  const P = mat4perspective(Math.PI / 2, 1, 1, 100);
  near(P[0], 1, 1e-12, "fov=90°, aspect=1 → m00=1");
  near(P[5], 1, 1e-12);
  near(P[10], -101 / 99, 1e-12, "(far+near)/(near−far)");
  near(P[11], -1, 1e-12, "w = −z");
  near(P[14], -200 / 99, 1e-12);
  // fov 越界被夹到 [0.01, π−0.01]
  near(mat4perspective(0, 1, 1, 100)[0], mat4perspective(0.01, 1, 1, 100)[0], 1e-12, "下夹");
  near(mat4perspective(100, 1, 1, 100)[0], mat4perspective(Math.PI - 0.01, 1, 1, 100)[0], 1e-12, "上夹");
  // aspect=0 不会除零，靠 max(1e-9, aspect) 兜底（结果极大但有限）
  const wide = mat4perspective(1, 0, 1, 100);
  assert.ok(Number.isFinite(wide[0]) && wide[0] > 1e8, "aspect=0 → m00 极大但有限");
});

test("transformMesh：只动 positions，拓扑与参数原样透传", () => {
  const g = buildGraphGrid((x, y) => x + y, [0, 1, 0, 1, 2, 2]);
  const T = new Float64Array(16);
  T[12] = 1; T[13] = 2; T[14] = 3;
  const t = transformMesh(g, T);
  nearVec(t.positions, [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3], 1e-6, "整体平移");
  assert.equal(t.tris, g.tris, "tris 引用不变");
  assert.equal(t.uvs, g.uvs, "uvs 引用不变");
  assert.equal(t.params, g.params, "params 引用不变");
  // 绕 y 转 90°：(x,y,z) → (z, y, −x)。网格第 2 个顶点是 (1,0,1) → (1,0,−1)
  const r = transformMesh(g, mat4rotY(Math.PI / 2));
  nearVec([r.positions[3], r.positions[4], r.positions[5]], [1, 0, -1], 1e-6);
});

/* ============================================================ 网格统计 */

test("meshBounds：只统计被三角形引用且有限的顶点，索引越界忽略", () => {
  assert.deepEqual(meshBounds(buildGraphGrid((x, y) => x + y, [0, 1, 0, 1, 2, 2])), [0, 1, 0, 1, 0, 2]);
  assert.deepEqual(
    meshBounds({ positions: new Float32Array(0), tris: new Uint32Array(0) }),
    [0, 0, 0, 0, 0, 0],
    "空网格给全零，不是 ±Infinity",
  );
  assert.deepEqual(
    meshBounds({ positions: new Float32Array([0, 0, 0, 9, 9, 9]), tris: new Uint32Array([0, 0, 0]) }),
    [0, 0, 0, 0, 0, 0],
    "未被引用的顶点不参与统计（否则视图会被拉飞）",
  );
  assert.deepEqual(
    meshBounds({ positions: new Float32Array([0, 0, 0, 1, 2, 3]), tris: new Uint32Array([0, 1, 99]) }),
    [0, 1, 0, 2, 0, 3],
    "越界索引不越界读",
  );
  // 部分顶点采样失败（NaN）时这些槽位不可用，网格可能整块退化
  const halfDead = buildGraphGrid((x) => (x > 0.5 ? NaN : x), [0, 1, 0, 1, 2, 2]);
  assert.deepEqual(meshBounds(halfDead), [0, 0, 0, 0, 0, 0]);
});

test("meshEdges：结构化网格走参数线（无对角线），非结构化走三角形棱去重", () => {
  const g = buildGraphGrid(() => 0, [0, 1, 0, 1, 2, 2]);
  assert.deepEqual([...meshEdges(g)], [0, 1, 2, 3, 0, 2, 1, 3], "2×2 只有 4 条参数线");
  const g43 = buildGraphGrid(() => 0, [0, 1, 0, 1, 4, 3]);
  assert.equal(meshEdges(g43).length / 2, 4 * 2 + 3 * 3, "nu·(nv−1) + nv·(nu−1)");
  const tri: Mesh = { positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]), tris: new Uint32Array([0, 1, 2]) };
  assert.deepEqual([...meshEdges(tri)], [0, 1, 1, 2, 2, 0]);
  const two: Mesh = {
    positions: new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0]),
    tris: new Uint32Array([0, 1, 2, 1, 3, 2]),
  };
  assert.deepEqual([...meshEdges(two)], [0, 1, 1, 2, 2, 0, 1, 3, 3, 2], "共享棱只出现一次");
  assert.deepEqual([...meshEdges({ positions: new Float32Array(0), tris: new Uint32Array(0) })], []);
  // 采样失败的顶点不连线
  const dead = buildParametricGrid((u, v, o) => { if (u < 0.75) o.set(u, v, 0); }, [0, 1, 0, 1, 4, 2]);
  assert.deepEqual([...meshEdges(dead)], [0, 1, 1, 2, 4, 5, 5, 6, 0, 4, 1, 5, 2, 6], "死列不产生边");
});

/* ============================================================ 相机 */

test("orbitCamera：azim=0/elev=0 时相机在 +y，fwd 指向 −y，right×up = −fwd", () => {
  const cam = orbitCamera({ azim: 0, elev: 0, dist: 5, target: [0, 0, 0], width: 800, height: 600 });
  nearVec(cam.pos, [0, 5, 0], 1e-12, "相机位");
  nearVec(cam.fwd, [0, -1, 0], 1e-12);
  nearVec(cam.right, [-1, 0, 0], 1e-12);
  nearVec(cam.up, [0, 0, 1], 1e-12);
  near(cam.focal, 400 / Math.tan(0.5), 1e-9, "focal = (w/2)/tan(fov/2)");
  near(cam.pxPerUnit!, cam.focal / 5, 1e-12, "正交等效像素比");
  assert.equal(cam.dist, 5);
  assert.equal(cam.ortho, false);
  // 右手系：right × up = −fwd
  const [rx, ry, rz] = cam.right;
  const [ux, uy, uz] = cam.up;
  const cross = [ry * uz - rz * uy, rz * ux - rx * uz, rx * uy - ry * ux];
  nearVec(cross, [0, 1, 0], 1e-12, "right×up 指向 +y = −fwd");
});

test("orbitCamera：dist/width/height/fov/elev 全部夹紧，target 平移相机位", () => {
  const c = orbitCamera({ azim: 10, elev: 10, dist: -3, target: [0, 0, 0], width: 0, height: -5, fov: 100 });
  assert.equal(c.dist, 1e-6, "负距离兜到 1e-6");
  assert.equal(c.width, 1, "宽高各兜到 1");
  assert.equal(c.height, 1);
  near(c.pos[2], 1e-6 * Math.sin(Math.PI / 2 - 0.02), 1e-9, "elev 夹到 ±(π/2 − 0.02) 防万向节翻转");
  // fov 夹到 [0.05, π−0.05]
  const lo = orbitCamera({ azim: 0, elev: 0, dist: 1, target: [0, 0, 0], width: 100, height: 100, fov: 0.0001 });
  const hi = orbitCamera({ azim: 0, elev: 0, dist: 1, target: [0, 0, 0], width: 100, height: 100, fov: 100 });
  near(lo.focal, 50 / Math.tan(0.025), 1e-9, "fov 下夹到 0.05");
  near(hi.focal, 50 / Math.tan((Math.PI - 0.05) / 2), 1e-9, "fov 上夹到 π−0.05");
  const t = orbitCamera({ azim: 0, elev: 0, dist: 2, target: [1, 1, 1], width: 10, height: 10 });
  nearVec(t.pos, [1, 3, 1], 1e-12, "相机位 = target + dist·方向");
  assert.equal(orbitCamera({ azim: 0, elev: 0, dist: 1, target: [0, 0, 0], width: 10, height: 10, ortho: true }).ortho, true);
});

/* ============================================================ 投影 */

test("projectScene：屏幕坐标 = 中心 + k·相机空间分量/深度，面按深度从远到近排序", () => {
  const g = buildGraphGrid((x, y) => x + y, [0, 1, 0, 1, 2, 2]);
  const cam = orbitCamera({ azim: 0, elev: 0, dist: 10, target: [0, 0, 0], width: 100, height: 100 });
  const p = projectScene(g.positions, g.tris, cam);
  assert.equal(p.verts.length, g.positions.length, "每顶点输出 sx,sy,depth");
  assert.equal(p.faces.length, 2);
  assert.ok(p.faces[0].depth >= p.faces[1].depth, "画家算法：远的先画");
  assert.equal(p.faces[0].pts.length, 6);
  assert.ok(p.faces.every((f) => f.n.every(Number.isFinite)), "面法向是单位向量且有限");
  // y 轴向上翻转：屏幕 y 向下
  const centre = projectScene(new Float32Array([0, 0, 0]), new Uint32Array(), cam);
  assert.deepEqual([centre.verts[0], centre.verts[1]], [50, 50], "世界原点落在画面正中");
  const above = projectScene(new Float32Array([0, 0, 1]), new Uint32Array(), cam);
  assert.ok(above.verts[1] < 50, "世界 +z（画面上方）对应更小的屏幕 y");
  // 正交投影：depth 不做透视除法
  const oc = orbitCamera({ azim: 0, elev: 0, dist: 10, target: [0, 0, 0], width: 100, height: 100, ortho: true });
  const op = projectScene(g.positions, g.tris, oc);
  assert.ok(Number.isFinite(op.verts[0]) && Number.isFinite(op.verts[1]), "正交投影不发散");
});

test("projectScene：背面剔除、相机后方顶点置 depth=−1、退化面与越界索引被丢", () => {
  const g = buildGraphGrid((x, y) => x + y, [0, 1, 0, 1, 2, 2]);
  const cam = orbitCamera({ azim: 0, elev: 0, dist: 10, target: [0, 0, 0], width: 100, height: 100 });
  // 平面法向背对相机（n·fwd > 0）
  assert.equal(projectScene(g.positions, g.tris, cam, undefined, { cull: true }).faces.length, 0, "剔掉背面");
  assert.equal(projectScene(g.positions, g.tris, cam, undefined, { cull: false }).faces.length, 2);
  // 近裁剪：depth ≤ near 的顶点标记为不可见
  const near1 = orbitCamera({ azim: 0, elev: 0, dist: 1, target: [0, 0, 0], width: 10, height: 10 });
  const np = projectScene(g.positions, g.tris, near1);
  const depths = np.verts.filter((_, i) => i % 3 === 2);
  assert.ok(depths.some((d) => d === -1), "落在相机平面上的顶点被标记 depth=−1");
  assert.ok(np.verts.every(Number.isFinite), "不可见顶点的 sx/sy 退化成 0 而不是 NaN/Infinity");
  // 共线三点 → 叉积为零 → 面不入场景
  const collinear = new Float32Array([0, 0, 0, 1, 0, 0, 2, 0, 0]);
  assert.equal(projectScene(collinear, new Uint32Array([0, 1, 2]), cam).faces.length, 0, "退化面片被丢弃");
  // 越界索引
  assert.equal(
    projectScene(new Float32Array([0, 0, 0, 0, 0, 5, 0, 0, 9]), new Uint32Array([0, 1, 99]), cam).faces.length,
    0,
  );
  // 顶点法向长度不足时退化为面法向（旧签名兼容）
  const flat = projectScene(g.positions, g.tris, cam, new Float32Array(3));
  const legacy = projectScene(g.positions, g.tris, cam, 1);
  nearVec(flat.faces[0].n, legacy.faces[0].n, 1e-12, "短数组/数字都不触发平滑");
  // scale 只是额外的像素缩放
  const s2 = projectScene(g.positions, g.tris, cam, undefined, { scale: 2 });
  near(s2.verts[3] - 50, 2 * (projectScene(g.positions, g.tris, cam).verts[3] - 50), 1e-6, "偏移翻倍");
});

/* ============================================================ 等高线 */

test("meshContours：交点严格落在等值面上，鞍面断成两条分支，levels 顺序保留", () => {
  const hills = buildGraphGrid((x, y) => x + y, [-1, 1, -1, 1, 4, 4]);
  const c = meshContours(hills, [0]);
  assert.equal(c.length, 1);
  assert.equal(c[0].level, 0);
  assert.equal(c[0].polylines.length, 1, "z=x+y 的 z=0 是一条直线");
  for (const line of c[0].polylines) {
    for (const p of line) near(p[2], 0, 1e-6, "每个交点的 z 必须等于 level");
  }
  // 锥面一圈
  const cone = meshContours(buildGraphGrid((x, y) => Math.hypot(x, y), [-1, 1, -1, 1, 8, 8]), [0.5]);
  assert.equal(cone[0].polylines.length, 1, "圆锥等高线是一圈");
  for (const p of cone[0].polylines[0]) {
    near(p[2], 0.5, 1e-6);
    near(Math.hypot(p[0], p[1]), 0.5, 0.02, "半径应接近 0.5（8×8 网格的弦交点只会低估）");
  }
  // 鞍面 z=x²−y² 在 z=0 上是两条对角线
  const saddle = meshContours(buildGraphGrid((x, y) => x * x - y * y, [-1, 1, -1, 1, 6, 6]), [0]);
  assert.equal(saddle[0].polylines.length, 2, "鞍面等高线分成两条分支");
  assert.deepEqual(meshContours(hills, [0.5, -0.5, 0]).map((x) => x.level), [0.5, -0.5, 0], "按传入顺序返回");
  assert.deepEqual(meshContours(hills, [99]).map((x) => x.polylines.length), [0], "超出量程 → 空");
  assert.deepEqual(meshContours({ positions: new Float32Array(0), tris: new Uint32Array(0) }, [0]).map((x) => x.polylines.length), [0]);
  assert.equal(meshContours(hills, []).length, 0, "空 levels");
});

/* ============================================================ 隐式曲面 */

test("implicitMesh 单位球：顶点在球面上、法向朝外、索引与三角形合法", () => {
  const m = implicitMesh((x, y, z) => Math.hypot(x, y, z) - 1, [-1.5, 1.5, -1.5, 1.5, -1.5, 1.5], 16);
  const vc = m.positions.length / 3;
  assert.ok(vc > 100, `球面应产生足够顶点，实际 ${vc}`);
  assert.equal(m.tris.length % 3, 0, "索引数是 3 的倍数");
  assert.equal(m.params, undefined, "隐式曲面不是结构化网格");
  assert.equal(m.uvs, undefined);
  assert.ok([...m.positions].every(Number.isFinite), "输出里不出现 NaN");
  assert.ok([...m.tris].every((i) => i < vc), "索引不越界");
  for (let t = 0; t + 2 < m.tris.length; t += 3) {
    const [a, b, c] = [m.tris[t], m.tris[t + 1], m.tris[t + 2]];
    assert.ok(a !== b && b !== c && a !== c, "不发射退化三角形");
  }
  // 顶点被 Newton 投影真正拉到等值面上
  let worst = 0;
  for (let i = 0; i < m.positions.length; i += 3) {
    worst = Math.max(worst, Math.abs(Math.hypot(m.positions[i], m.positions[i + 1], m.positions[i + 2]) - 1));
  }
  assert.ok(worst < 1e-6, `顶点应落在球面上，最大偏差 ${worst}`);
  // 绕序按 ∇f 统一成外法向
  const n = surfaceNormals(m);
  for (let i = 0; i < n.length; i += 3) {
    const L = Math.hypot(m.positions[i], m.positions[i + 1], m.positions[i + 2]);
    const d = (n[i] * m.positions[i] + n[i + 1] * m.positions[i + 1] + n[i + 2] * m.positions[i + 2]) / L;
    assert.ok(d > 0, `顶点 ${i / 3} 的法向应朝外（与径向同侧），实际 ${d}`);
  }
});

test("implicitMesh：环面残差收敛；无变号/采样失效时给空网格；分辨率夹到 [4,128]", () => {
  const torus = implicitMesh(
    (x, y, z) => (Math.hypot(x, y) - 1) ** 2 + z * z - 0.01,
    [-2, 2, -2, 2, -1, 1],
    20,
  );
  assert.ok(torus.positions.length / 3 > 50, "环面应有顶点");
  let worst = 0;
  for (let i = 0; i < torus.positions.length; i += 3) {
    worst = Math.max(
      worst,
      Math.abs((Math.hypot(torus.positions[i], torus.positions[i + 1]) - 1) ** 2 + torus.positions[i + 2] ** 2 - 0.01),
    );
  }
  assert.ok(worst < 1e-4, `环面顶点残差应收敛，最大 ${worst}`);
  // 常数场没有变号单元
  for (const f of [() => 5, () => -1]) {
    const m = implicitMesh(f, [-1, 1, -1, 1, -1, 1], 8, 0);
    assert.equal(m.positions.length, 0);
    assert.equal(m.tris.length, 0);
  }
  // 采样出现 NaN 的单元被跳过，输出仍全是有限值
  const bad = implicitMesh((x) => (x > 0 ? NaN : -1), [-1, 1, -1, 1, -1, 1], 8, 0);
  assert.equal(bad.positions.length, 0);
  assert.ok([...bad.positions].every(Number.isFinite));
  // res 夹紧：0 → 4
  const r4 = implicitMesh((x) => x, [-1, 1, -1, 1, -1, 1], 4, 0);
  const r0 = implicitMesh((x) => x, [-1, 1, -1, 1, -1, 1], 0, 0);
  const r5 = implicitMesh((x) => x, [-1, 1, -1, 1, -1, 1], 5, 0);
  assert.equal(r0.positions.length, r4.positions.length, "res<4 抬到 4");
  assert.equal(r5.positions.length / 3, 25, "res=5 → 5×5 个对偶顶点（x=0 平面穿心）");
  // level 偏移把面平移到 x=0.5
  const shifted = implicitMesh((x) => x, [-1, 1, [-1, 1] as never, 1, 1] as never, 8, 0.5);
  for (let i = 0; i < shifted.positions.length; i += 3) {
    near(shifted.positions[i], 0.5, 1e-9, "等值面应平移到 x=0.5");
  }
});
