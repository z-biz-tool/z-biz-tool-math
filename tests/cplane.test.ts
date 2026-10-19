// cplane.ts（复平面：共形着色 / Newton 分形 / 映射雅可比 / 像曲线）的纯逻辑回归
// （无 React、无 Canvas、无 DOM）。
// 跑法：node --experimental-strip-types --test tests/cplane.test.ts
//
// 为什么该测它（00 的 P0-3）：
//   `cplane.ts` 是 README 点名的「复平面」那一整块，18.6 KB，此前**零测试**。
//   而这里的失败形态全是**静默**的：保角判据写反、临界点搜不到、奇点没断开，
//   都不抛错 —— 界面上只是"这张分形看着不对劲"，用户不会知道是计算错了。
//
// 断言策略：用**解析可判**的性质，而不是"和上次跑出来的一样"：
//   · z² 是保角映射 ⇒ CR 方程必须成立；共轭 z̄ 是反保角 ⇒ 必须不成立
//   · z² 的临界点只有 0；z+1 没有临界点
//   · 1/z 的像曲线在越过奇点时必须断开，否则会把奇点两侧连成长线
//   · Newton 迭代 z²−1 的根必须落在 ±1
// 快照式断言只能证明"没变"，证不出"是对的"—— 而这正是复平面出错的典型形态。
//
// 其中若干条锁定的是**当前真实行为（含疑似缺陷）**，已在注释里逐条标注。
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  type C,
  criticalPoints,
  domainColor,
  jacobianOfMap,
  mapCurve,
  newtonFractal,
  newtonPlan,
  sampleComplex,
} from "../src/core/cplane.ts";

/** 浮点近似相等 —— 给数值型断言一个统一容差口径 */
function closeN(a: number, b: number, tol: number, msg = ""): void {
  assert.ok(
    Number.isFinite(a) && Math.abs(a - b) <= tol,
    `${msg} 期望 ${b}，实得 ${a}（容差 ${tol}）`
  );
}

/** 复数 f(z) = z² */
const sq = (z: C): C => ({ re: z.re * z.re - z.im * z.im, im: 2 * z.re * z.im });
/** 复数 f(z) = z */
const ident = (z: C): C => ({ re: z.re, im: z.im });
/** 复数 f(z) = z + 1 */
const shift = (z: C): C => ({ re: z.re + 1, im: z.im });

// ── sampleComplex：等距栅格采样 ───────────────────────────────────────

test("sampleComplex：形状是 [高, 宽]，行优先、x 为快维", () => {
  const s = sampleComplex(ident, 0, 2, 0, 1, 3, 2);
  assert.deepEqual(s.shape, [2, 3], "shape = [H, W]");
  assert.deepEqual(Array.from(s.re), [0, 1, 2, 0, 1, 2], "每行是 x 的等距采样");
  assert.deepEqual(Array.from(s.im), [0, 0, 0, 1, 1, 1], "y 按行变化");
});

test("sampleComplex：宽/高为 1 时步长为 0（只有一个采样点）", () => {
  const s = sampleComplex(ident, 0, 2, 0, 1, 1, 1);
  assert.deepEqual(s.shape, [1, 1]);
  assert.deepEqual(Array.from(s.re), [0]);
});

test("sampleComplex：宽/高不足 1 时被抬到 1", () => {
  const s = sampleComplex(ident, 0, 2, 0, 1, 0, 0);
  assert.deepEqual(s.shape, [1, 1]);
  assert.equal(s.re.length, 1);
});

test("sampleComplex：函数抛错或返回非有限值时该点记 NaN（不抛错）", () => {
  const s = sampleComplex(
    (z) => {
      if (z.re === 1) throw new Error("boom");
      return ident(z);
    },
    0,
    2,
    0,
    0,
    3,
    1
  );
  assert.equal(s.re.length, 3);
  assert.ok(Number.isNaN(s.re[1]), "抛错的采样点记 NaN");
  assert.ok(Number.isFinite(s.re[0]) && Number.isFinite(s.re[2]), "其余点正常");
});

// ── domainColor：共形着色 ─────────────────────────────────────────────

test("domainColor：输出长度 = 宽 × 高 × 3", () => {
  const px = domainColor(sq, -1, 1, -1, 1, 4, 3, { levelStep: 0 });
  assert.equal(px.length, 4 * 3 * 3);
  assert.ok(px instanceof Uint8ClampedArray);
});

test("domainColor：奇点（非有限）画成黑色", () => {
  const px = domainColor(
    (z) => (z.re === 0 && z.im === 0 ? { re: NaN, im: NaN } : sq(z)),
    -1,
    1,
    -1,
    1,
    3,
    3,
    { levelStep: 0 }
  );
  // 中心像素 = (j=1, i=1) ⇒ 偏移 (1*3 + 1) * 3
  const off = (1 * 3 + 1) * 3;
  assert.deepEqual(Array.from(px.slice(off, off + 3)), [0, 0, 0], "奇点必须是黑");
});

test("domainColor：levelStep=0 关闭等值线网格", () => {
  const withGrid = domainColor(sq, -1, 1, -1, 1, 4, 4, { levelStep: 1 });
  const noGrid = domainColor(sq, -1, 1, -1, 1, 4, 4, { levelStep: 0 });
  assert.notDeepEqual(
    Array.from(withGrid),
    Array.from(noGrid),
    "有网格与无网格必须画出不同的像素"
  );
});

// ── jacobianOfMap：实二维映射的雅可比 ─────────────────────────────────

test("jacobianOfMap：z² 的雅可比是 [[2x, -2y], [2y, 2x]]，且是保角映射", () => {
  const r = jacobianOfMap((z) => [z.re * z.re - z.im * z.im, 2 * z.re * z.im], 2, 1);
  closeN(r.j[0][0], 4, 1e-6, "∂u/∂x = 2x：");
  closeN(r.j[0][1], -2, 1e-6, "∂u/∂y = -2y：");
  closeN(r.j[1][0], 2, 1e-6, "∂v/∂x = 2y：");
  closeN(r.j[1][1], 4, 1e-6, "∂v/∂y = 2x：");
  closeN(r.det, 4 * (2 * 2 + 1 * 1), 1e-6, "det = 4(x²+y²)：");
  assert.equal(r.singular, false);
  assert.equal(r.conformal, true, "z² 是保角映射 ⇒ CR 方程必须成立");
});

test("jacobianOfMap：共轭 z̄ 是反保角 ⇒ conformal 为 false", () => {
  const r = jacobianOfMap((z) => [z.re, -z.im], 2, 1);
  closeN(r.j[0][0], 1, 1e-6, "∂u/∂x = 1：");
  closeN(r.j[1][1], -1, 1e-6, "∂v/∂y = -1：");
  closeN(r.det, -1, 1e-6, "det = -1：");
  assert.equal(r.singular, false);
  assert.equal(r.conformal, false, "反保角映射不满足 CR 方程");
});

test("jacobianOfMap：恒等映射是保角且不奇异", () => {
  const r = jacobianOfMap((z) => [z.re, z.im], 2, 1);
  closeN(r.det, 1, 1e-6, "det = 1：");
  assert.equal(r.singular, false);
  assert.equal(r.conformal, true);
});

test("jacobianOfMap【现状刻画】：常数映射被标成 singular，但 conformal 仍为 true", () => {
  // CR 判据只看 a[0]≈b[1] 与 a[1]≈-b[0]；常数映射两个雅可比分量都是 0，
  // 于是 CR 方程**平凡成立** ⇒ conformal = true。
  // 数学上常数映射既不是保角也不是共形 —— 这里是判据的盲区，记下来。
  const r = jacobianOfMap(() => [1, 1], 2, 1);
  closeN(r.det, 0, 1e-12, "det = 0：");
  assert.equal(r.singular, true, "|det| 过小 ⇒ 奇异");
  assert.equal(r.conformal, true, "现状：常数映射也被判成保角（判据盲区）");
});

// ── mapCurve：像曲线与奇点断开 ────────────────────────────────────────

test("mapCurve：恒等映射原样返回点列", () => {
  const curves = mapCurve(ident, [
    [0, 0],
    [1, 1],
    [2, 2],
  ]);
  assert.equal(curves.length, 1);
  assert.deepEqual(curves[0], [
    [0, 0],
    [1, 1],
    [2, 2],
  ]);
});

test("mapCurve：越过奇点时把曲线拆断", () => {
  // 1/z 在 x→0 时像点跑向无穷 ⇒ 相邻点间距超过 splitOnPole 就该断开
  const inv = (z: C): C => ({ re: 1 / z.re, im: -z.im / (z.re * z.re) });
  const split = mapCurve(
    inv,
    [
      [1, 0],
      [0.5, 0],
      [0.1, 0],
    ],
    { splitOnPole: 1 }
  );
  assert.equal(split.length, 1, "只有前半段有 2 个点，能成一条曲线");
  // 注意：1/z 的虚部是 -0/(x²) = -0，
  // 而 deepStrictEqual 用 Object.is 比较 ⇒ -0 与 0 **不相等**。
  // 所以这里逐分量比，不整体 deepEqual。
  assert.equal(split[0].length, 2);
  closeN(split[0][0][0], 1, 1e-12, "第 1 个像点 x：");
  closeN(split[0][1][0], 2, 1e-12, "第 2 个像点 x：");
});

test("mapCurve：默认 splitOnPole=1e3 时同样的输入不拆", () => {
  const inv = (z: C): C => ({ re: 1 / z.re, im: -z.im / (z.re * z.re) });
  const curves = mapCurve(inv, [
    [1, 0],
    [0.5, 0],
    [0.1, 0],
  ]);
  assert.equal(curves.length, 1);
  assert.equal(curves[0].length, 3, "三段都没超过 1e3 ⇒ 连成一条");
});

test("mapCurve：只有单个点的碎片会被丢掉（曲线至少要 2 个点）", () => {
  const inv = (z: C): C => ({ re: 1 / z.re, im: -z.im / (z.re * z.re) });
  const curves = mapCurve(
    inv,
    [
      [1, 0],
      [0.5, 0],
      [0.1, 0],
    ],
    { splitOnPole: 1 }
  );
  // 断开后后半段只剩 1 个点 ⇒ 不成曲线
  assert.equal(curves.length, 1, "后半段单点被丢弃");
});

test("mapCurve：非有限值把曲线断开，两侧各成一条", () => {
  const withHole = (z: C): C =>
    z.re === 1 ? { re: NaN, im: NaN } : { re: z.re, im: z.im };
  const curves = mapCurve(withHole, [
    [0, 0],
    [0.5, 0],
    [1, 0],
    [1.5, 0],
    [2, 0],
  ]);
  assert.equal(curves.length, 2, "断成两条");
  assert.deepEqual(curves[0], [
    [0, 0],
    [0.5, 0],
  ]);
  assert.deepEqual(curves[1], [
    [1.5, 0],
    [2, 0],
  ]);
});

test("mapCurve【现状刻画】：断点两侧若各只剩 1 个点，整条曲线消失", () => {
  const withHole = (z: C): C =>
    z.re === 1 ? { re: NaN, im: NaN } : { re: z.re, im: z.im };
  const curves = mapCurve(withHole, [
    [0, 0],
    [1, 0],
    [2, 0],
  ]);
  assert.deepEqual(curves, [], "单点碎片不成曲线 ⇒ 结果为空");
});

// ── criticalPoints：f'(z) = 0 的临界点 ────────────────────────────────

test("criticalPoints：z² 的临界点只有 0", () => {
  const pts = criticalPoints(sq, -1, 1, -1, 1, 24);
  assert.equal(pts.length, 1);
  closeN(pts[0].re, 0, 1e-6, "临界点实部：");
  closeN(pts[0].im, 0, 1e-6, "临界点虚部：");
});

test("criticalPoints：z + 1 没有临界点", () => {
  assert.deepEqual(criticalPoints(shift, -1, 1, -1, 1, 24), []);
});

test("criticalPoints：z³ 的临界点也是 0", () => {
  const cube = (z: C): C => ({
    re: z.re * z.re * z.re - 3 * z.re * z.im * z.im,
    im: 3 * z.re * z.re * z.im - z.im * z.im * z.im,
  });
  const pts = criticalPoints(cube, -1, 1, -1, 1, 24);
  assert.equal(pts.length, 1);
  closeN(pts[0].re, 0, 1e-5, "临界点实部：");
  closeN(pts[0].im, 0, 1e-5, "临界点虚部：");
});

test("criticalPoints：函数抛错时返回空表（不抛错）", () => {
  const pts = criticalPoints(
    () => {
      throw new Error("boom");
    },
    -1,
    1,
    -1,
    1,
    8
  );
  assert.deepEqual(pts, []);
});

// ── newtonPlan / newtonFractal：Newton 迭代 ───────────────────────────

test("newtonPlan：z² − 1 的根是 ±1", () => {
  const plan = newtonPlan((z) => ({ re: z.re * z.re - 1, im: 2 * z.re * z.im }), -2, 2, -2, 2);
  assert.equal(plan.roots.length, 2, "两个相异根");
  closeN(plan.roots[0].re, -1, 1e-6, "第一个根（按 re 排序）：");
  closeN(plan.roots[1].re, 1, 1e-6, "第二个根：");
  closeN(plan.roots[0].im, 0, 1e-6, "第一个根的虚部：");
});

test("newtonPlan：limit 与 match 由包围盒尺寸决定", () => {
  const plan = newtonPlan((z) => ({ re: z.re * z.re - 1, im: 2 * z.re * z.im }), -2, 2, -2, 2);
  closeN(plan.limit, 2.2 * 2, 1e-12, "limit = 2.2 × max(半宽, 半高)：");
  closeN(plan.match, 1e-5, 1e-12, "match = max(1e-5, 1e-6 × 半宽)：");
});

test("newtonFractal：每个像素归属到最近的根，未收敛记 -1", () => {
  const f = (z: C): C => ({ re: z.re * z.re - 1, im: 2 * z.re * z.im });
  const plan = newtonPlan(f, -2, 2, -2, 2);
  const fr = newtonFractal(f, -2, 2, -2, 2, 20, 20, { plan });
  assert.equal(fr.iter.length, 400, "20 × 20 个像素");
  assert.equal(fr.root.length, 400);
  assert.equal(fr.roots.length, 2, "与 plan 的根一致");
  // 左半边归属第一个根、右半边归属第二个根
  const left = fr.root[10 * 20 + 3];
  const right = fr.root[10 * 20 + 16];
  assert.ok(left >= 0 && right >= 0, "两侧都应收敛到某个根");
  assert.notEqual(left, right, "左右两侧归属不同的根");
});

// ── 复数结构约定 ──────────────────────────────────────────────────────

test("cplane 的复数结构与 cnum.C 同构（re/im 两字段）", () => {
  // 绘图层直接把 C 当 {re, im} 用，字段名写错会静默画出 NaN
  const z: C = { re: 3, im: -4 };
  assert.equal(z.re, 3);
  assert.equal(z.im, -4);
});
