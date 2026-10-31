/** 分形内核：逃逸迭代、对称性、平滑值域、逐行着色 */
import test from "node:test";
import assert from "node:assert/strict";
import { escapeAt, escapeText, fracRow } from "../src/core/fractal.ts";

test("Mandelbrot：c=0 在集合内，c=2 越过逃逸半径，标准心脏线内点未逃逸", () => {
  assert.equal(escapeAt("mandelbrot", 0, 0, 0, 0, 100).inside, true);
  const far = escapeAt("mandelbrot", 2, 0, 0, 0, 100);
  assert.equal(far.inside, false);
  // 逃逸半径取 |z|>4：第一步 z=2 还不够，第二步 z=5 才算逃逸
  assert.equal(far.n, 2);
  // c = 1/4 恰在尖点上，属集合；c = 1/4 + ε 逃逸
  assert.equal(escapeAt("mandelbrot", 0.25, 0, 0, 0, 200).inside, true);
  assert.equal(escapeAt("mandelbrot", 0.26, 0, 0, 0, 200).inside, false);
});

test("Julia：c=-1 时 z=0 与 z=±1 行为符合 z²-1 动力系统", () => {
  // c=-1：0 ↦ -1 ↦ 0 循环，不逃逸
  assert.equal(escapeAt("julia", 0, 0, -1, 0, 100).inside, true);
  // z=2 ↦ 3 ↦ 8 ↦ … 逃逸
  assert.equal(escapeAt("julia", 2, 0, -1, 0, 100).inside, false);
});

test("Burning Ship：与 Mandelbrot 的差别体现在远离实轴的船身火焰区", () => {
  // 抽样核实过的「仅 Burning Ship 在集合内」点与「仅 Mandelbrot 在集合内」点各一
  assert.equal(escapeAt("burning", -0.804, -0.894, 0, 0, 120).inside, true);
  assert.equal(escapeAt("mandelbrot", -0.804, -0.894, 0, 0, 120).inside, false);
  assert.equal(escapeAt("mandelbrot", 0.298, -0.412, 0, 0, 120).inside, true);
  assert.equal(escapeAt("burning", 0.298, -0.412, 0, 0, 120).inside, false);
});

test("逃逸点满足 Mandelbrot 对称性：共轭点迭代数相同", () => {
  const a = escapeAt("mandelbrot", -0.72, 0.24, 0, 0, 300);
  const b = escapeAt("mandelbrot", -0.72, -0.24, 0, 0, 300);
  assert.equal(a.inside, b.inside);
  assert.equal(a.n, b.n);
});

test("平滑迭代数落在 [n, n+1] 区间且 fracRow 输出合法 RGB", () => {
  const e = escapeAt("mandelbrot", -0.72, 0.24, 0, 0, 300);
  assert.ok(e.mu >= e.n && e.mu <= e.n + 1);
  const rgb = new Uint8ClampedArray(8 * 3);
  fracRow(
    { kind: "mandelbrot", cr: 0, ci: 0, maxIter: 120, colormap: "turbo", smooth: true, dark: true, left: -2.2, right: 1, y: 0, w: 8 },
    rgb,
    0,
  );
  for (const v of rgb) assert.ok(Number.isFinite(v) && v >= 0 && v <= 255);
  // 集合内部点位是近黑
  const insideRow = new Uint8ClampedArray(8 * 3);
  fracRow(
    { kind: "mandelbrot", cr: 0, ci: 0, maxIter: 120, colormap: "turbo", smooth: true, dark: true, left: -0.1, right: 0.1, y: 0, w: 8 },
    insideRow,
    0,
  );
  assert.deepEqual([...insideRow.slice(0, 3)], [5, 6, 12]);
});

test("escapeText：文案区分未逃逸与逃逸步数", () => {
  assert.match(escapeText("mandelbrot", 0, 0, 0, 0, 100), /未逃逸/);
  assert.match(escapeText("mandelbrot", 2, 0, 0, 0, 100), /第 2 步逃逸/);
});
