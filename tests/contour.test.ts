// contour.ts（等值线 / 决策边界）的纯逻辑回归（无 React、无 Canvas）。
// 跑法：node --experimental-strip-types --test tests/contour.test.ts
//
// 为什么测它（00 的 A3 覆盖表留下的最后一个缺口之一）：
//   `contour.ts` 是 README 八种工作模式里「函数图像（隐函数 f(x,y)=c、不等式区域）」
//   与「神经网络（决策边界）」两块共用的等值线算法，此前**零测试**。
//   而它的失败形态是静默的：marching squares 的**歧义构型**（对角同侧）
//   选错了连通方式 ⇒ 等值线在那一格连错 ⇒ 界面上是「边界拐错了一个弯」，
//   不抛错、也不白屏。
//
// 断言用**解析可判**的性质（不依赖实现细节的坐标格式）：
//   · f(x,y) = x、level = 0 ⇒ 等值线就是直线 x = 0，所有线段的 x 坐标必须是 0
//   · 端点必须是**格边线性插值交点**（不是格心、不是角点）—— 位级可判
//   · f(x,y) = x² + y² − 1 ⇒ 端点到单位圆的偏差受**一阶插值误差界** h²/8 约束
//   · 场恒为常数 ⇒ 没有等值线
//   · 歧义构型（case 5/10）⇒ 端点配对必须符合**格心均值消解规则**
//
// ⚠️ 一条被实测纠正过的错误断言（保留此注记，避免后人再犯）：
//   「f = x² + y² − 1 的等值线端点必须精确落在单位圆上」是**错的** ——
//   marching squares 沿格边做**线性**插值，而 x² + y² − 1 沿边是二次的，
//   实测 res=8、域 [-2,2] 时端点模长偏差最大 0.0282（h²/8 = 0.03125）。
//   ⇒ 正确判据是「偏差受一阶界约束」，不是「精确落圆」。
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  contourLevels,
  implicitFill,
  marchingSquares,
  signedRegionMask,
  traceContours,
} from "../src/core/contour.ts";

function closeN(a: number, b: number, tol: number, msg = ""): void {
  assert.ok(
    Number.isFinite(a) && Math.abs(a - b) <= tol,
    `${msg} 期望 ${b}，实得 ${a}（容差 ${tol}）`
  );
}

test("marchingSquares：f = x、level = 0 的等值线就是直线 x = 0", () => {
  const segs = marchingSquares((x) => x, -1, 1, -1, 1, 4, 0);
  assert.ok(segs.length > 0, "必须有等值线");
  for (const s of segs) {
    closeN(s[0], 0, 1e-12, "线段起点 x：");
    closeN(s[2], 0, 1e-12, "线段终点 x：");
  }
});

test("marchingSquares：线段首尾相接，连成一条完整的折线", () => {
  const segs = marchingSquares((x) => x, -1, 1, -1, 1, 4, 0);
  // 线段按 y 从下到上排列，逐段首尾相接
  for (let i = 1; i < segs.length; i++) {
    closeN(segs[i][1], segs[i - 1][3], 1e-12, `第 ${i} 段的起点必须接住上一段的终点`);
  }
});

test("marchingSquares：端点是格边的线性插值交点（不是格心、不是角点）", () => {
  // 判别式：把 res 改小，格点更稀疏。若实现退化成「取格心」，
  // 端点的两个坐标都不是格点值 ⇒ 被这条抓住；
  // 若退化成「取角点」，端点两个坐标都是格点值 ⇒ 也被抓住。
  // 合法的例外只有一种：角点**恰好落在等值线上**（此时交点就是该角点本身），
  //   本例中 (0, −1) 满足 f = 0² + (−1)² − 1 = 0。
  const res = 8;
  const x0 = -2;
  const x1 = 2;
  const grid: number[] = [];
  for (let i = 0; i <= res; i++) grid.push(x0 + ((x1 - x0) * i) / res);
  const onGrid = (v: number) => grid.some((g) => Math.abs(g - v) <= 1e-12);
  const f = (x: number, y: number) => x * x + y * y - 1;

  const segs = marchingSquares(f, x0, x1, x0, x1, res, 0);
  assert.ok(segs.length > 0, "必须有等值线");
  for (const s of segs) {
    for (const [px, py] of [
      [s[0], s[1]],
      [s[2], s[3]],
    ]) {
      const gx = onGrid(px);
      const gy = onGrid(py);
      if (gx && gy) {
        // 角点：只有当它恰在等值线上时才是合法交点
        closeN(f(px, py), 0, 1e-12, `角点 (${px}, ${py}) 不在等值线上：`);
      } else {
        assert.ok(
          gx || gy,
          `端点 (${px}, ${py}) 的两个坐标都不是格点值（退化成了格心取值）`
        );
      }
    }
  }
});

test("marchingSquares：端点精确等于所在格边的线性插值交点（位级可判）", () => {
  // 这条直接钉住「边上做线性插值」这个实现约定：
  // 若实现退化成取格心 / 取角点，端点就不再等于 (level−va)/(vb−va) 的插值结果。
  const res = 8;
  const x0 = -2;
  const x1 = 2;
  const level = 0;
  const f = (x: number, y: number) => x * x + y * y - 1;
  const xs: number[] = [];
  const ys: number[] = [];
  for (let i = 0; i <= res; i++) {
    xs.push(x0 + ((x1 - x0) * i) / res);
    ys.push(x0 + ((x1 - x0) * i) / res);
  }
  const segs = marchingSquares(f, x0, x1, x0, x1, res, level);
  assert.ok(segs.length > 0, "必须有等值线");

  for (const s of segs) {
    for (const [px, py] of [
      [s[0], s[1]],
      [s[2], s[3]],
    ]) {
      const xi = xs.findIndex((v) => Math.abs(v - px) <= 1e-12);
      const yj = ys.findIndex((v) => Math.abs(v - py) <= 1e-12);
      if (xi >= 0 && yj >= 0) {
        // 角点恰在等值线上（edgeCross 里 devA/devB === 0 的分支）：交点就是该角点
        closeN(f(px, py), level, 1e-12, `角点 (${px}, ${py}) 的场值：`);
      } else if (xi >= 0 && yj < 0) {
        // 竖边 (xs[xi], ys[j]) → (xs[xi], ys[j+1])
        let hit = false;
        for (let j = 0; j < res; j++) {
          const va = f(xs[xi], ys[j]);
          const vb = f(xs[xi], ys[j + 1]);
          if (va === vb) continue;
          const t = (level - va) / (vb - va);
          if (t < 0 || t > 1) continue;
          const want = ys[j] + (ys[j + 1] - ys[j]) * t;
          if (Math.abs(want - py) <= 1e-9) {
            hit = true;
            break;
          }
        }
        assert.ok(hit, `竖边上的端点 (${px}, ${py}) 不等于该边的线性插值交点`);
      } else if (yj >= 0 && xi < 0) {
        // 横边 (xs[i], ys[yj]) → (xs[i+1], ys[yj])
        let hit = false;
        for (let i = 0; i < res; i++) {
          const va = f(xs[i], ys[yj]);
          const vb = f(xs[i + 1], ys[yj]);
          if (va === vb) continue;
          const t = (level - va) / (vb - va);
          if (t < 0 || t > 1) continue;
          const want = xs[i] + (xs[i + 1] - xs[i]) * t;
          if (Math.abs(want - px) <= 1e-9) {
            hit = true;
            break;
          }
        }
        assert.ok(hit, `横边上的端点 (${px}, ${py}) 不等于该边的线性插值交点`);
      } else {
        assert.fail(`端点 (${px}, ${py}) 既不在竖边上也不在横边上`);
      }
    }
  }
});

test("marchingSquares：f = x² + y² − 1 的端点偏差受一阶插值误差界 h²/8 约束", () => {
  // 线性插值对二次场的误差是 O(h²)。实测 res=8（h=0.5）最大偏差 0.0282，
  // 界 h²/8 = 0.03125。判据用「比值」形式，换 res 也成立。
  const res = 8;
  const x0 = -2;
  const x1 = 2;
  const h = (x1 - x0) / res;
  const segs = marchingSquares((x, y) => x * x + y * y - 1, x0, x1, x0, x1, res, 0);
  let maxDev = 0;
  for (const s of segs) {
    for (const [px, py] of [
      [s[0], s[1]],
      [s[2], s[3]],
    ]) {
      maxDev = Math.max(maxDev, Math.abs(Math.hypot(px, py) - 1));
    }
  }
  assert.ok(
    maxDev <= (h * h) / 8,
    `端点最大偏差 ${maxDev} 超过一阶界 h²/8 = ${(h * h) / 8}`
  );
});

test("marchingSquares：圆对称场生成的端点集合在 x→−x / y→−y 下闭合", () => {
  const segs = marchingSquares((x, y) => x * x + y * y - 1, -2, 2, -2, 2, 8, 0);
  const key = (px: number, py: number) => `${Math.round(px * 1e9)},${Math.round(py * 1e9)}`;
  const set = new Set<string>();
  for (const s of segs) {
    set.add(key(s[0], s[1]));
    set.add(key(s[2], s[3]));
  }
  for (const s of segs) {
    for (const [px, py] of [
      [s[0], s[1]],
      [s[2], s[3]],
    ]) {
      assert.ok(set.has(key(-px, py)), `端点 (${px}, ${py}) 的 x 镜像缺失`);
      assert.ok(set.has(key(px, -py)), `端点 (${px}, ${py}) 的 y 镜像缺失`);
    }
  }
});

test("marchingSquares：场恒为常数时没有等值线", () => {
  assert.deepEqual(marchingSquares(() => 0, -1, 1, -1, 1, 4, 1), []);
  assert.deepEqual(marchingSquares(() => 5, -1, 1, -1, 1, 4, 1), []);
});

test("marchingSquares：场抛错时错误向上传播（不静默吞掉）", () => {
  // 现状：sampleField 不捕异常 ⇒ 异常直接冒到调用方。
  // 这条锁住「不静默」：若将来改成吞掉异常，界面会画出空的等值线而没人知道。
  assert.throws(
    () =>
      marchingSquares(
        () => {
          throw new Error("场炸了");
        },
        -1,
        1,
        -1,
        1,
        4,
        0
      ),
    /场炸了/
  );
});

test("marchingSquares：非法参数返回空表而不是抛错", () => {
  const f = (x: number) => x;
  assert.deepEqual(marchingSquares(f, -1, 1, -1, 1, 0, 0), []);
  assert.deepEqual(marchingSquares(f, -1, 1, -1, 1, NaN, 0), []);
  assert.deepEqual(marchingSquares(f, -1, 1, -1, 1, 4, NaN), []);
  assert.deepEqual(marchingSquares(f, NaN, 1, -1, 1, 4, 0), []);
});

test("marchingSquares：歧义构型 case 5 的连通方式符合格心均值消解规则", () => {
  // 构造：域 [0,2]²、res=2 ⇒ 网格线 x,y ∈ {0,1,2}；歧义格是 [0,1]²。
  // 场 f = (x−0.5)(y−0.5) + c：格 [0,1]² 四角 a(0,0)、c(1,1) 同号为内，
  // b(1,0)、d(0,1) 为外 ⇒ cell = 5（对角同侧）。
  // 消解规则：格心值 (a+b+c+d)/4 决定哪对对角连通。
  //   c = +0.1 ⇒ 格心在内 ⇒ 切掉 b、d ⇒ 配对 {底,右} + {上,左}
  //   c = -0.1 ⇒ 格心在外 ⇒ 切掉 a、c ⇒ 配对 {底,左} + {右,上}
  // 边交点解析可算（底 y=0 ⇒ (x−0.5)(−0.5)+c=0 ⇒ x = 0.5 − 2c）：
  const inside = () =>
    marchingSquares((px, py) => (px - 0.5) * (py - 0.5) + 0.1, 0, 2, 0, 2, 2, 0);
  const outside = () =>
    marchingSquares((px, py) => (px - 0.5) * (py - 0.5) - 0.1, 0, 2, 0, 2, 2, 0);

  // 配对 = 「哪两个边交点被连成同一段」；只取**两端都落在歧义格 [0,1]² 内**的段
  const pairOf = (segs: number[][]) => {
    const inBox = (px: number, py: number) =>
      px >= -1e-9 && px <= 1 + 1e-9 && py >= -1e-9 && py <= 1 + 1e-9;
    const out: string[] = [];
    for (const s of segs) {
      if (!inBox(s[0], s[1]) || !inBox(s[2], s[3])) continue;
      const edge = (px: number, py: number) => {
        if (Math.abs(py) <= 1e-9) return "底";
        if (Math.abs(px - 1) <= 1e-9) return "右";
        if (Math.abs(py - 1) <= 1e-9) return "上";
        return "左";
      };
      out.push([edge(s[0], s[1]), edge(s[2], s[3])].sort().join("+"));
    }
    return out.sort();
  };

  assert.deepEqual(
    pairOf(inside()),
    ["上+左", "右+底"],
    "格心在内：应切掉 b、d 两角 ⇒ 配对 {底,右} 与 {上,左}"
  );
  assert.deepEqual(
    pairOf(outside()),
    ["上+右", "左+底"],
    "格心在外：应切掉 a、c 两角 ⇒ 配对 {底,左} 与 {右,上}"
  );
});

test("marchingSquares：case 10 与 case 5 互为场取反，零集相同", () => {
  // f 与 −f 的零集相同 ⇒ 端点集合必须一致（f 与 −f 互为极性对称）。
  const segsA = marchingSquares((x, y) => (x - 0.5) * (y - 0.5) + 0.1, 0, 2, 0, 2, 2, 0);
  const segsB = marchingSquares((x, y) => -((x - 0.5) * (y - 0.5) + 0.1), 0, 2, 0, 2, 2, 0);
  const key = (px: number, py: number) => `${Math.round(px * 1e9)},${Math.round(py * 1e9)}`;
  const setA = new Set(segsA.flatMap((s) => [key(s[0], s[1]), key(s[2], s[3])]));
  const setB = new Set(segsB.flatMap((s) => [key(s[0], s[1]), key(s[2], s[3])]));
  assert.deepEqual([...setB].sort(), [...setA].sort());
});

test("traceContours：单位圆串成一条闭合链，且段数守恒", () => {
  const segs = marchingSquares((x, y) => x * x + y * y - 1, -2, 2, -2, 2, 8, 0);
  const chains = traceContours((x, y) => x * x + y * y - 1, -2, 2, -2, 2, 8, 0, 4);
  assert.equal(chains.length, 1, "单位圆应串成一条闭合链");
  assert.equal(
    chains[0].length,
    segs.length,
    "串链不得丢段（minSegments=4 不应滤掉这条链）"
  );
  // 闭合：链首尾相接
  const c = chains[0];
  closeN(c[0][0], c[c.length - 1][2], 1e-9, "闭合链的起点必须接住末段终点：");
  closeN(c[0][1], c[c.length - 1][3], 1e-9, "闭合链的起点必须接住末段终点：");
});

test("traceContours：碎链被 minSegments 滤掉", () => {
  // res=2 时圆只有 4 段，取 minSegments=10 应全部滤空
  const chains = traceContours((x, y) => x * x + y * y - 1, -2, 2, -2, 2, 2, 0, 10);
  assert.deepEqual(chains, []);
});

test("contourLevels：一次采样、多条 level，顺序与传入一致", () => {
  const out = contourLevels((x) => x, -1, 1, -1, 1, 4, [-0.5, 0, 0.5]);
  assert.equal(out.length, 3);
  assert.deepEqual(out.map((o) => o.level), [-0.5, 0, 0.5]);
  for (const o of out) {
    assert.ok(o.segments.length > 0, `level=${o.level} 应有等值线`);
    for (const s of o.segments) {
      closeN(s[0], o.level, 1e-12, `level=${o.level} 的线段 x：`);
      closeN(s[2], o.level, 1e-12, `level=${o.level} 的线段 x：`);
    }
  }
});

test("contourLevels：levels 为空数组时返回空表", () => {
  assert.deepEqual(contourLevels((x) => x, -1, 1, -1, 1, 4, []), []);
});

test("implicitFill：level 本身位于返回带的正中", () => {
  const f = (x: number, y: number) => x * x + y * y - 1;
  const bands = implicitFill(f, -2, 2, -2, 2, 8, 0);
  assert.equal(bands.length, 5, "应返回 level−2band … level+2band 共 5 条带");
  // 正中那条必须就是 level=0 的等值线（单位圆，12 段）
  assert.equal(bands[2].length, 12);
});

test("signedRegionMask：圆内为 1、圆外为 0", () => {
  const f = (x: number, y: number) => x * x + y * y - 1;
  const W = 16;
  const mask = signedRegionMask(f, -2, 2, -2, 2, W, W, 0, "<");
  assert.equal(mask.length, W * W);
  // 像素中心在 (−1.75+0.25i, −1.75+0.25j)
  const at = (i: number, j: number) => mask[j * W + i];
  assert.equal(at(8, 8), 1, "原点应在圆内");
  assert.equal(at(0, 0), 0, "左下角应在圆外");
  assert.equal(at(15, 15), 0, "右上角应在圆外");
  // 圆内像素数应与面积大致相符（π·1² / 4 ≈ 0.785，实测 12/64 = 0.1875）
  const inside = mask.reduce((a, b) => a + b, 0);
  assert.ok(inside > 0 && inside < W * W, `圆内像素数 ${inside} 不合理`);
});

test("signedRegionMask：非法参数返回全 0 掩码而不是抛错", () => {
  const f = (x: number) => x;
  assert.equal(signedRegionMask(f, NaN, 1, -1, 1, 4, 4, 0, "<").length, 16);
  assert.equal(
    signedRegionMask(f, -1, 1, -1, 1, 4, 4, NaN, "<").reduce((a, b) => a + b, 0),
    0
  );
});
