// view.ts（刻度 / 视口换算）、colormap.ts（色标）、parsemat.ts（矩阵输入格）的纯逻辑回归。
// 跑法：node --experimental-strip-types --test tests/viewColormap.test.ts
//
// 为什么测这三个（00 的 A3 覆盖表留下的缺口）：
//   它们都不在 2026-10-04 建档时被测的 4 个模块里，但都是**纯函数**、零 DOM：
//
//   · `view.ts` 的 niceTicks —— 坐标轴刻度。步长归一、刻度落在区间内、
//     主刻度与次刻度不重合，这三条一旦错了，界面上就是"刻度挤在一起"或
//     "轴上出现两个 0.6"，不抛错。
//   · `colormap.ts` —— 热力图的取色。t 越界要夹住；名字写错要回退；
//     查找表首尾必须与 colormap(0)/colormap(1) 一致，否则热图两端颜色错位。
//   · `parsemat.ts` —— 界面矩阵输入格。**错误提示必须指名道姓**
//     （"第 3 格为空"），因为用户是在网格里一个个填的，不知道是哪一格错了。
import { test } from "node:test";
import assert from "node:assert/strict";
import { Engine } from "../src/core/machine.ts";
import { logTicks, niceTicks, segmentRange } from "../src/core/view.ts";
import { buildLUT, colormap, paletteAt, rgbCss } from "../src/core/colormap.ts";
import { parseCell, parseMatrix, parseVector } from "../src/core/parsemat.ts";

function closeN(a: number, b: number, tol: number, msg = ""): void {
  assert.ok(
    Number.isFinite(a) && Math.abs(a - b) <= tol,
    `${msg} 期望 ${b}，实得 ${a}（容差 ${tol}）`
  );
}

// ── niceTicks：坐标轴刻度 ────────────────────────────────────────────

test("niceTicks：步长归一到 1/2/2.5/5×10^n", () => {
  closeN(niceTicks(0, 10).step, 2, 1e-12, "0..10 的步长：");
  closeN(niceTicks(0, 1).step, 0.2, 1e-12, "0..1 的步长：");
  closeN(niceTicks(0, 100).step, 20, 1e-12, "0..100 的步长：");
  closeN(niceTicks(0, 3).step, 0.5, 1e-12, "0..3 的步长：");
});

test("niceTicks：主刻度全部落在 [min, max] 内，且等距", () => {
  for (const [a, b] of [
    [0, 10],
    [-3, 7],
    [0.25, 0.75],
    [-100, 100],
  ] as [number, number][]) {
    const t = niceTicks(a, b);
    assert.ok(t.values.length > 0, `区间 ${a}..${b} 必须有刻度`);
    for (const v of t.values) {
      assert.ok(v >= a - 1e-9 && v <= b + 1e-9, `刻度 ${v} 跑出了区间 ${a}..${b}`);
    }
    for (let i = 1; i < t.values.length; i++) {
      closeN(t.values[i] - t.values[i - 1], t.step, 1e-9, "相邻刻度间距必须等于步长：");
    }
  }
});

test("niceTicks：标签与刻度一一对应", () => {
  const t = niceTicks(0, 10);
  assert.equal(t.labels.length, t.values.length);
  assert.deepEqual(t.labels, ["0", "2", "4", "6", "8", "10"]);
});

test("niceTicks：int 模式的步长至少为 1", () => {
  const t = niceTicks(0, 10, 8, "int");
  assert.equal(t.step, 2);
  const small = niceTicks(0, 0.5, 8, "int");
  assert.ok(small.step >= 1, `int 模式的步长必须 ≥ 1，实得 ${small.step}`);
});

test("niceTicks：pi 模式按 π 的分数取刻度，标签是 π 的倍数", () => {
  const t = niceTicks(0, Math.PI * 2, 8, "pi");
  assert.equal(t.values.length, 9);
  assert.deepEqual(t.labels, ["0", "π/4", "π/2", "3π/4", "π", "5π/4", "3π/2", "7π/4", "2π"]);
  closeN(t.step, Math.PI / 4, 1e-12, "pi 模式的步长：");
});

test("niceTicks：退化区间（min == max）不会死循环", () => {
  const t = niceTicks(5, 5);
  assert.equal(t.values.length, 1);
  closeN(t.values[0], 5, 1e-12, "退化区间仍给一个刻度：");
});

test("niceTicks：次刻度不与主刻度画在同一个点上", () => {
  // 实现里的排除带宽是"次刻度步长（step/5）的 0.2 倍"，
  // 所以次刻度可以离主刻度很近（step=2 时 0.4 挨着 0），
  // 但**不能落在同一个点上** —— 否则轴上同一位置画两条线。
  const t = niceTicks(0, 10);
  for (const m of t.minor) {
    for (const v of t.values) {
      assert.ok(
        Math.abs(v - m) > t.step * 1e-6,
        `次刻度 ${m} 与主刻度 ${v} 落在同一个点上`
      );
    }
  }
});

test("niceTicks：scientific 模式用科学计数法标标签", () => {
  const t = niceTicks(0, 1e7, 8, "scientific");
  assert.equal(t.labels[0], "0.0e+0");
  assert.ok(t.labels.every((l) => /e[+-]/.test(l)), "所有标签都应是科学计数法");
});

test("logTicks：主刻度落在 10 的整数幂上，标签是 10^n", () => {
  const t = logTicks(1, 1000);
  assert.deepEqual(t.values, [1, 10, 100, 1000]);
  assert.deepEqual(t.labels, ["1", "10¹", "10²", "10³"]);
  assert.equal(t.step, 0, "对数轴没有统一步长，step 固定为 0");
});

test("logTicks：不足一个十倍频时把倍数值提为主刻度", () => {
  const t = logTicks(1, 10);
  assert.deepEqual(t.values, [1, 10]);
  assert.deepEqual(t.minor, [2, 3, 4, 5, 6, 7, 8, 9]);
});

test("segmentRange：等分且含两端，共 n+1 个点", () => {
  const r = segmentRange(0, 1, 4);
  assert.equal(r.length, 5);
  assert.deepEqual(r, [0, 0.25, 0.5, 0.75, 1]);
});

// ── colormap：色标 ───────────────────────────────────────────────────

test("colormap：gray 的两端是纯黑与纯白", () => {
  assert.deepEqual(colormap("gray", 0), [0, 0, 0]);
  assert.deepEqual(colormap("gray", 1), [255, 255, 255]);
  assert.deepEqual(colormap("gray", 0.5), [128, 128, 128]);
});

test("colormap：t 越界被夹到 [0, 1]，不会溢出色域", () => {
  assert.deepEqual(colormap("gray", -1), [0, 0, 0]);
  assert.deepEqual(colormap("gray", 2), [255, 255, 255]);
});

test("colormap：未知色标名回退到默认（parula），不抛错", () => {
  assert.deepEqual(colormap("nope", 0.5), colormap("parula", 0.5));
});

test("colormap：hsv 的六个扇区起点颜色正确", () => {
  assert.deepEqual(colormap("hsv", 0), [255, 0, 0], "0° 必须是红");
  assert.deepEqual(colormap("hsv", 1 / 3), [0, 0, 255], "120° 必须是蓝");
});

test("buildLUT：长度 = n×3，首尾与 colormap(0)/colormap(1) 一致", () => {
  const lut = buildLUT("parula", 8);
  assert.equal(lut.length, 24);
  assert.deepEqual(Array.from(lut.slice(0, 3)), colormap("parula", 0));
  assert.deepEqual(Array.from(lut.slice(21, 24)), colormap("parula", 1));
});

test("rgbCss：不透明给 rgb、带透明度给 rgba", () => {
  assert.equal(rgbCss([1, 2, 3]), "rgb(1,2,3)");
  assert.equal(rgbCss([1, 2, 3], 0.5), "rgba(1,2,3,0.5)");
});

test("paletteAt：按图层序号循环取色，负数也安全", () => {
  const n = 15;
  assert.equal(paletteAt(0), "#5b6bd6");
  assert.equal(paletteAt(n), "#5b6bd6", "越界要绕回第一个");
  assert.equal(paletteAt(-1), "#0284c7", "负数要绕回最后一个");
});

// ── parsemat：界面矩阵输入格 ─────────────────────────────────────────

test("parseCell：能算的表达式求值", () => {
  assert.equal(parseCell(new Engine(), "1+2", 0), 3);
  assert.equal(parseCell(new Engine(), "  4*5 ", 0), 20);
});

test("parseCell：错误提示必须指名是第几格", () => {
  assert.throws(() => parseCell(new Engine(), "", 2), /第 3 格为空/);
  assert.throws(() => parseCell(new Engine(), "1i", 0), /第 1 格/);
  assert.throws(() => parseCell(new Engine(), "zzz", 1), /第 2 格/);
});

test("parseMatrix：按行优先展开，维度不符时明确报错", () => {
  const e = new Engine();
  assert.deepEqual(parseMatrix(e, ["1", "2", "3", "4"], 2), [
    [1, 2],
    [3, 4],
  ]);
  assert.throws(() => parseMatrix(e, ["1"], 2), /需要 4 个元素，实际 1/);
});

test("parseVector：维度不符时明确报错", () => {
  const e = new Engine();
  assert.deepEqual(parseVector(e, ["1", "2"], 2), [1, 2]);
  assert.throws(() => parseVector(e, ["1"], 2), /需要 2 个元素，实际 1/);
});
