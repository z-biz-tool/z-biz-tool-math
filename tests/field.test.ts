// field.ts（向量场 / ODE）的纯逻辑回归（无 React、无 Canvas、无 DOM）。
// 跑法：node --experimental-strip-types --test tests/field.test.ts
//
// 为什么该测它（00 的 P0-3）：
//   `field.ts` 是 README 点名的「向量场 / 微分方程方向场 / 相平面」那一整块的地基，
//   此前**零测试**。而这里的失败形态全是**静默**的：
//   积分器少走一步、雅可比差分步长写错、相平面分类把鞍点判成节点 ——
//   都不抛错，界面上只是"看着有点怪"，用户不会知道是计算错了。
//
// 断言策略：绝大多数用例用**数学恒等式 / 解析解**，不是"和上次跑出来的一样"：
//   · y' = y  ⇒ 解是 e^t，末值必须落在 e 的容差内
//   · y'' = -y ⇒ 解是 (cos t, sin t)，走满 π 后必须回到 (-1, 0)
//   · 雅可比：对线性映射是精确的（差分必须精确还原）
//   · 相平面分类：节点/鞍点/中心/螺旋各自的**特征值形态**是解析可判的
// 这样某处把符号写反、把差分步长改小、把容差放松，都会被逮到；
// 快照式断言只能证明"没变"，证不出"是对的"。
//
// 其中若干条锁定的是**当前真实行为（含疑似缺陷）**，已在注释里逐条标注。
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  type RHS,
  type VecField,
  classifyEquilibrium,
  equilibria,
  integrateSystem,
  jacobianAt,
  quiverField,
  slopeField,
  streamlines,
} from "../src/core/field.ts";

/** 浮点近似相等 —— 给数值型断言一个统一容差口径 */
function closeN(a: number, b: number, tol: number, msg = ""): void {
  assert.ok(
    Number.isFinite(a) && Math.abs(a - b) <= tol,
    `${msg} 期望 ${b}，实得 ${a}（容差 ${tol}）`
  );
}

// ── integrateSystem：定步长经典 RK4 ───────────────────────────────────

test("integrateSystem：y' = y 的解是 e^t（步长越小越准）", () => {
  // rhs 收到的 y 是**状态**（不含 t）—— 约定本身要钉住，见下一条
  const sol = integrateSystem((_t, y) => [y[0]], [0, 1], [1, 0.1]);
  assert.equal(sol.length, 11, "t 从 0 到 1、步长 0.1 ⇒ 10 步 + 初值行");
  const last = sol[sol.length - 1];
  closeN(last[1], Math.E, 1e-5, "dt=0.1 的 RK4 末值：");
  closeN(last[0], 1, 1e-12, "末行的 t 必须正好落在 tEnd：");
});

test("integrateSystem：步长减半，误差按 RK4 的**四次方**掉下去", () => {
  // 实测：dt=0.1 误差 2.08e-6、dt=0.05 误差 1.36e-7 ⇒ 比值 ≈ 15.3 ≈ 2⁴。
  // 比值落在 [8, 32] 说明确实是四阶；若退化成二阶（比值 4）或一阶（比值 2），
  // 这条会红。
  const coarse = integrateSystem((_t, y) => [y[0]], [0, 1], [1, 0.1]);
  const fine = integrateSystem((_t, y) => [y[0]], [0, 1], [1, 0.05]);
  const eCoarse = Math.abs(coarse[coarse.length - 1][1] - Math.E);
  const eFine = Math.abs(fine[fine.length - 1][1] - Math.E);
  assert.ok(eFine < eCoarse, `步长减半后误差应下降：${eCoarse} → ${eFine}`);
  const ratio = eCoarse / eFine;
  assert.ok(
    ratio >= 8 && ratio <= 32,
    `误差比应在 2⁴=16 附近（八到三十二之间），实测 ${ratio}`
  );
  closeN(fine[fine.length - 1][1], Math.E, 1e-6, "dt=0.05 的末值：");
});

test("integrateSystem：非自治方程也保持四次方收敛（rhs 用到 t 时）", () => {
  // 为什么必须用**非自治**场再测一遍收敛阶：
  //   上一条用的 y' = y 是自治的 —— rhs 完全不看 t，
  //   于是「k2 的时间点写成 ta+step 还是 ta+0.5*step」对它**没有任何影响**，
  //   那种变异打过来是绿的（实测：F2 变异不咬住）。
  //   换成 y' = t + y（解析解 e^t − t − 1）后，k2/k3 的时间点写错就会掉阶。
  // 实测比值：dt 0.1→0.05 为 15.35、0.05→0.025 为 15.67（≈2⁴=16）。
  // 判据区间取 [12, 20]：三阶方法比值 ≈8、二阶 ≈4、一阶 ≈2，都会被挡下。
  const truth = Math.E - 1 - 1;
  const err = (dt: number) => {
    const sol = integrateSystem((t, y) => [t + y[0]], [0, 0], [1, dt]);
    return Math.abs(sol[sol.length - 1][1] - truth);
  };
  const e1 = err(0.1);
  const e2 = err(0.05);
  const e3 = err(0.025);
  assert.ok(e3 < e2 && e2 < e1, `误差应逐级下降：${e1} → ${e2} → ${e3}`);
  const r1 = e1 / e2;
  const r2 = e2 / e3;
  assert.ok(r1 >= 12 && r1 <= 20, `0.1/0.05 的误差比应 ≈16，实测 ${r1}`);
  assert.ok(r2 >= 12 && r2 <= 20, `0.05/0.025 的误差比应 ≈16，实测 ${r2}`);
});

test("integrateSystem：rhs 收到的是**状态**，不含时刻（约定钉住）", () => {
  // 初值行是 [t0, y1, y2, ...]；rhs 的第二个参数是**去掉 t0 之后的状态**。
  // 写错这条会得到 y[1] === undefined ⇒ 积分一步就停。
  const seen: number[][] = [];
  const rhs: RHS = (_t, y) => {
    seen.push([...y]);
    return [y[0], y[1]];
  };
  const sol = integrateSystem(rhs, [0, 2, 3], [1, 0.5]);
  assert.ok(seen.length > 0, "rhs 至少被调用一次");
  assert.deepEqual(seen[0], [2, 3], "rhs 的 y 必须是去掉 t0 的状态 [y1, y2]：");
  assert.equal(sol.length, 3, "t 0→1、步长 0.5 ⇒ 2 步 + 初值行");
});

test("integrateSystem：y'' = -y 走满 π 回到 (-1, 0)", () => {
  const sol = integrateSystem((_t, y) => [y[1], -y[0]], [0, 1, 0], [Math.PI, 0.05]);
  const last = sol[sol.length - 1];
  closeN(last[0], Math.PI, 1e-12, "末行 t = π：");
  closeN(last[1], Math.cos(Math.PI), 1e-6, "x(π) = cos π = -1：");
  closeN(last[2], Math.sin(Math.PI), 1e-6, "y(π) = sin π = 0：");
});

test("integrateSystem：可以往**负方向**积分", () => {
  const sol = integrateSystem((_t, y) => [y[0]], [0, 1], [-1, 0.1]);
  const last = sol[sol.length - 1];
  closeN(last[0], -1, 1e-12, "末行 t = -1：");
  closeN(last[1], Math.exp(-1), 1e-5, "y(-1) = e^-1：");
});

test("integrateSystem：步长过小时由 maxSteps 兜底，不会把机器跑死", () => {
  const sol = integrateSystem((_t, y) => [y[0]], [0, 1], [1, 1e-9]);
  assert.equal(sol.length, 20001, "默认 maxSteps=20000 ⇒ 20000 步 + 初值行");
  const capped = integrateSystem((_t, y) => [y[0]], [0, 1], [1, 1e-9, 50]);
  assert.equal(capped.length, 51, "maxSteps=50 ⇒ 50 步 + 初值行");
});

test("integrateSystem：退化输入只返回初值行（不抛错）", () => {
  // tEnd === t0：区间长度为 0，没有步可走
  assert.deepEqual(integrateSystem((_t, y) => [y[0]], [0, 1], [0, 0.1]), [[0, 1]]);
  // 空初值：没有状态维度
  assert.deepEqual(integrateSystem(() => [1], [], [1, 0.1]), [[]]);
  // rhs 返回的维度对不上
  assert.deepEqual(integrateSystem(() => [1, 2], [0, 1], [1, 0.1]), [[0, 1]]);
  // rhs 抛出
  assert.deepEqual(
    integrateSystem(() => {
      throw new Error("boom");
    }, [0, 1], [1, 0.1]),
    [[0, 1]]
  );
  // rhs 返回 NaN（数值爆炸）
  assert.deepEqual(integrateSystem(() => [NaN], [0, 1], [1, 0.1]), [[0, 1]]);
});

test("integrateSystem：tEnd 非有限时只返回初值行", () => {
  assert.deepEqual(
    integrateSystem((_t, y) => [y[0]], [0, 1], [Infinity, 0.1]),
    [[0, 1]]
  );
  assert.deepEqual(
    integrateSystem((_t, y) => [y[0]], [0, 1], [NaN, 0.1]),
    [[0, 1]]
  );
});

// ── jacobianAt：中心差分雅可比 ────────────────────────────────────────

test("jacobianAt：对线性映射给出精确雅可比", () => {
  const J = jacobianAt((v) => [v[0] * 2, v[1] * 3], [1, 1]);
  assert.equal(J.length, 2, "输出行数 = 被测函数的输出维数");
  closeN(J[0][0], 2, 1e-6, "∂f0/∂x0：");
  closeN(J[0][1], 0, 1e-6, "∂f0/∂x1：");
  closeN(J[1][0], 0, 1e-6, "∂f1/∂x0：");
  closeN(J[1][1], 3, 1e-6, "∂f1/∂x1：");
});

test("jacobianAt：混合偏导能对上（雅可比不是只算对角）", () => {
  // f(x,y) = (x²y, xy) ⇒ J = [[2xy, x²],[y, x]
  const J = jacobianAt((v) => [v[0] * v[0] * v[1], v[0] * v[1]], [2, 3]);
  closeN(J[0][0], 2 * 2 * 3, 1e-4, "∂f0/∂x = 2xy：");
  closeN(J[0][1], 4, 1e-4, "∂f0/∂y = x²：");
  closeN(J[1][0], 3, 1e-4, "∂f1/∂x = y：");
  closeN(J[1][1], 2, 1e-4, "∂f1/∂y = x：");
});

test("jacobianAt：函数抛错时返回**空矩阵**（不是零矩阵）", () => {
  // 现状刻画：探针在 p 与 p+1e-3 两处都失败 ⇒ probe 为空 ⇒ 行数 0。
  // 零矩阵与空矩阵在下游的语义不同（前者是"导数为 0"），所以要钉住。
  const J = jacobianAt(
    () => {
      throw new Error("boom");
    },
    [1, 1]
  );
  assert.equal(J.length, 0);
});

// ── classifyEquilibrium：相平面分类 ──────────────────────────────────

test("classifyEquilibrium：节点（实特征值同号）", () => {
  const stable = classifyEquilibrium([
    [-1, 0],
    [0, -1],
  ]);
  assert.equal(stable.type, "node");
  assert.equal(stable.stable, true, "两个负实特征值 ⇒ 稳定");
  assert.equal(stable.eigs.length, 2);

  const unstable = classifyEquilibrium([
    [1, 0],
    [0, 1],
  ]);
  assert.equal(unstable.type, "node");
  assert.equal(unstable.stable, false, "两个正实特征值 ⇒ 不稳定");
});

test("classifyEquilibrium：鞍点（特征值一正一负）", () => {
  const r = classifyEquilibrium([
    [1, 0],
    [0, -1],
  ]);
  assert.equal(r.type, "saddle");
  assert.equal(r.stable, false);
});

test("classifyEquilibrium：中心（纯虚特征值，实部为零）", () => {
  const r = classifyEquilibrium([
    [0, -1],
    [1, 0],
  ]);
  assert.equal(r.type, "center");
  assert.equal(r.stable, false, "中心是稳定的，但 stable 判据只看实部 ⇒ false");
  assert.equal(r.eigs.length, 2);
  closeN(Math.abs(r.eigs[0].im), 1, 1e-12, "特征值虚部绝对值：");
});

test("classifyEquilibrium：螺旋（复特征值且实部非零）", () => {
  const r = classifyEquilibrium([
    [1, -1],
    [1, 1],
  ]);
  assert.equal(r.type, "spiral");
  assert.equal(r.stable, false, "实部为正 ⇒ 不稳定");
  closeN(r.eigs[0].re, 1, 1e-12, "特征值实部：");
});

test("classifyEquilibrium：退化（全零矩阵）", () => {
  const r = classifyEquilibrium([
    [0, 0],
    [0, 0],
  ]);
  assert.equal(r.type, "degenerate");
  assert.equal(r.stable, false);
});

test("classifyEquilibrium：1×1 按一维处理，3×3 不支持 ⇒ 退化", () => {
  const one = classifyEquilibrium([[-1]]);
  assert.equal(one.type, "node");
  assert.equal(one.stable, true);
  assert.equal(one.eigs.length, 1);

  const three = classifyEquilibrium([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ]);
  assert.equal(three.type, "degenerate");
  assert.deepEqual(three.eigs, [], "3×3 不解析特征值 ⇒ 本征列表为空");
});

// ── equilibria：平衡点搜索 ────────────────────────────────────────────

test("equilibria：f(x,y) = (x, y) 的唯一平衡点是原点", () => {
  const pts = equilibria((v) => [v[0], v[1]], [
    [-2, 2],
    [-2, 2],
  ], 20);
  assert.equal(pts.length, 1);
  closeN(pts[0][0], 0, 1e-6, "平衡点 x：");
  closeN(pts[0][1], 0, 1e-6, "平衡点 y：");
});

test("equilibria：没有平衡点时返回空表", () => {
  const pts = equilibria((v) => [v[0] + 5, v[1] + 5], [
    [-2, 2],
    [-2, 2],
  ], 20);
  assert.deepEqual(pts, []);
});

test("equilibria：ranges 为空时返回空表", () => {
  assert.deepEqual(equilibria(() => [0, 0], []), []);
});

// ── quiverField：箭头场采样 ───────────────────────────────────────────

test("quiverField：采样点数 = nx × ny", () => {
  const f: VecField = () => [1, 0];
  const pts = quiverField(f, -1, 1, -1, 1, 3, 2);
  assert.equal(pts.length, 6);
});

test("quiverField：normalize 时 mag 恒为 1，方向不变", () => {
  const pts = quiverField(() => [3, 4], 0, 0, 0, 0, 1, 1, { normalize: true });
  assert.equal(pts.length, 1);
  closeN(pts[0].mag, 1, 1e-12, "归一后长度：");
  closeN(pts[0].u, 0.6, 1e-12, "归一后 x 分量：");
  closeN(pts[0].v, 0.8, 1e-12, "归一后 y 分量：");
});

test("quiverField：logScale 时 mag = log1p(原长)", () => {
  const pts = quiverField(() => [0, 10], 0, 0, 0, 0, 1, 1, { logScale: true });
  closeN(pts[0].mag, Math.log1p(10), 1e-12, "log 压缩后的长度：");
});

test("quiverField：默认（不归一）时 mag = 原长", () => {
  const pts = quiverField(() => [3, 4], 0, 0, 0, 0, 1, 1);
  closeN(pts[0].mag, 5, 1e-12, "原长：");
});

test("quiverField：非有限或抛错的采样点被跳过", () => {
  const pts = quiverField(
    (x) => {
      if (x === 0) throw new Error("boom");
      return [x, x];
    },
    -1,
    1,
    0,
    0,
    3,
    1
  );
  assert.equal(pts.length, 2, "x=0 那一格被跳过");
});

test("quiverField【现状刻画】：零向量场在 normalize 下 mag 仍是 1，但方向是零向量", () => {
  // 实测：normalize 分支把长度**无条件**设成 1（`len = opts.normalize ? 1 : raw`），
  // 缩放因子另算（`s = raw > 0 ? len / raw : 0`）⇒ 零场时 u=v=0、mag=1。
  // 画出来仍是一条零长度的箭头，所以界面上看不出差别；
  // 但「mag 表示实际画出的线段长度」这个注释在零场上是不成立的 —— 记在这里。
  const pts = quiverField(() => [0, 0], 0, 0, 0, 0, 1, 1, { normalize: true });
  assert.equal(pts.length, 1);
  assert.ok(Number.isFinite(pts[0].u) && Number.isFinite(pts[0].v), "不能出 NaN");
  closeN(pts[0].u, 0, 1e-12, "零场的方向 x 分量：");
  closeN(pts[0].v, 0, 1e-12, "零场的方向 y 分量：");
  closeN(pts[0].mag, 1, 1e-12, "零场在 normalize 下的 mag（现状）：");
});

// ── streamlines：流线 ────────────────────────────────────────────────

test("streamlines：常向量场给出直线流线（种子在正中，前后各积分 steps 步）", () => {
  const lines = streamlines([[0, 0]], () => [1, 0], {
    dt: 0.5,
    steps: 4,
    bounds: [-10, 10, -10, 10],
  });
  assert.equal(lines.length, 1);
  const pts = lines[0];
  assert.equal(pts.length % 2, 0, "流线是展平的 [x0,y0,x1,y1,...]");
  // 种子在正中，前后各 4 步 ⇒ 9 个点
  assert.equal(pts.length / 2, 9);
  // 折线的第一个点是**向后**积分的最远端（不是种子）
  closeN(pts[0], -2, 1e-9, "折线起点 x（向后 4 步 × 0.5）：");
  closeN(pts[1], 0, 1e-9, "折线起点 y：");
  // 种子在正中间
  closeN(pts[8], 0, 1e-9, "种子 x：");
  closeN(pts[9], 0, 1e-9, "种子 y：");
  // 末端是向前积分的最远端
  closeN(pts[pts.length - 2], 2, 1e-9, "折线末端 x（向前 4 步 × 0.5）：");
});

test("streamlines：驻点上的种子被舍弃（画不出方向）", () => {
  const lines = streamlines([[0, 0]], () => [0, 0], {
    dt: 0.5,
    steps: 4,
    bounds: [-10, 10, -10, 10],
  });
  assert.deepEqual(lines, [], "零速种子不产生流线");
});

test("streamlines：越界的种子被跳过", () => {
  const lines = streamlines([[100, 100]], () => [1, 0], {
    dt: 0.5,
    steps: 4,
    bounds: [-10, 10, -10, 10],
  });
  assert.deepEqual(lines, []);
});

test("streamlines：走出边界就截断（不会无限延伸）", () => {
  const lines = streamlines([[0, 0]], () => [1, 0], {
    dt: 1,
    steps: 100,
    bounds: [-2, 2, -2, 2],
  });
  assert.equal(lines.length, 1);
  const xs = [];
  for (let i = 0; i < lines[0].length; i += 2) xs.push(lines[0][i]);
  assert.ok(Math.max(...xs) <= 2 + 1e-9, "不能越过上界");
  assert.ok(Math.min(...xs) >= -2 - 1e-9, "不能越过下界");
});

// ── slopeField：方向场 ───────────────────────────────────────────────

test("slopeField：给出单位长度的方向向量", () => {
  const pts = slopeField(() => 1, -1, 1, -1, 1, 3, 3);
  assert.equal(pts.length, 9, "3 × 3 个采样点");
  for (const p of pts) {
    assert.ok(Number.isFinite(p.dx) && Number.isFinite(p.dy), "不能出 NaN");
    closeN(Math.hypot(p.dx, p.dy), 1, 1e-12, "方向向量必须是单位长度：");
  }
  // dy/dx = 1 ⇒ 方向角 45°
  closeN(pts[0].dx, Math.SQRT1_2, 1e-12, "dx：");
  closeN(pts[0].dy, Math.SQRT1_2, 1e-12, "dy：");
});

test("slopeField：竖直方向场（f = ±∞）收敛为 (0, ±1)，不产生 NaN", () => {
  const up = slopeField(() => Infinity, -1, 1, -1, 1, 2, 2);
  assert.equal(up.length, 4);
  closeN(up[0].dx, 0, 1e-12, "+∞ ⇒ dx = 0：");
  closeN(up[0].dy, 1, 1e-12, "+∞ ⇒ dy = 1：");

  const down = slopeField(() => -Infinity, -1, 1, -1, 1, 2, 2);
  closeN(down[0].dx, 0, 1e-12, "-∞ ⇒ dx = 0：");
  closeN(down[0].dy, -1, 1e-12, "-∞ ⇒ dy = -1：");
});

test("slopeField：NaN 的采样点被跳过，抛错的也跳过", () => {
  const pts = slopeField(
    (x) => {
      if (x === 0) throw new Error("boom");
      return NaN;
    },
    -1,
    1,
    0,
    0,
    3,
    1
  );
  assert.equal(pts.length, 0, "NaN 与抛错的采样点都不产出");
});
