/**
 * 力学内核测试：自由落体、弹性对撞守恒、非弹性恢复系数、墙反弹、拖拽出手速度。
 * 世界是模块单例，每个用例先 resetFrom 铺自己的初始条件。
 */
import assert from "node:assert/strict";
import { test } from "node:test";

import { launchVelocity, mechWorld } from "../src/core/mech.ts";
import { applyPreset } from "../src/presets.ts";
import { useStore } from "../src/state.ts";

const P0 = { gravity: 0, restitution: 1, drag: 0 };

test("自由落体：1 秒后 v≈−9.8、y≈−4.9（半隐式欧拉的容差内）", () => {
  mechWorld.resetFrom([{ id: "a", x: 0, y: 0, vx: 0, vy: 0, r: 0.4, m: 1, color: "#000" }]);
  mechWorld.step(1, { gravity: 9.8, restitution: 0.8, drag: 0 });
  const b = mechWorld.get("a")!;
  assert.ok(Math.abs(b.vy + 9.8) < 1e-9, `vy=${b.vy}`);
  assert.ok(Math.abs(b.y + 4.9) < 0.1, `y=${b.y}`);
});

test("等质量弹性对撞：速度交换，动量与动能守恒", () => {
  mechWorld.resetFrom([
    { id: "a", x: -2, y: 0, vx: 3, vy: 0, r: 0.3, m: 1, color: "#000" },
    { id: "b", x: 2, y: 0, vx: -3, vy: 0, r: 0.3, m: 1, color: "#111" },
  ]);
  const ke0 = mechWorld.kinetic();
  mechWorld.step(1, P0);
  const a = mechWorld.get("a")!;
  const b = mechWorld.get("b")!;
  assert.ok(Math.abs(a.vx + 3) < 1e-6, `a.vx=${a.vx}`);
  assert.ok(Math.abs(b.vx - 3) < 1e-6, `b.vx=${b.vx}`);
  assert.ok(Math.abs(mechWorld.kinetic() - ke0) < 1e-6);
});

test("不等质量 e=1：动量守恒，重球撞轻球把速度让出去", () => {
  /* m=2、v0=2 撞静止 m=1：v1'=(m1−m2)/(m1+m2)·v0=2/3, v2'=2m1/(m1+m2)·v0=8/3 */
  mechWorld.resetFrom([
    { id: "a", x: -3, y: 0, vx: 2, vy: 0, r: 0.3, m: 2, color: "#000" },
    { id: "b", x: 3, y: 0, vx: 0, vy: 0, r: 0.3, m: 1, color: "#111" },
  ]);
  const p0 = 2 * 2;
  const ke0 = mechWorld.kinetic();
  mechWorld.step(3, P0);
  const a = mechWorld.get("a")!;
  const b = mechWorld.get("b")!;
  assert.ok(Math.abs(a.vx * 2 + b.vx - p0) < 1e-6, `p=${a.vx * 2 + b.vx}`);
  assert.ok(Math.abs(a.vx - 2 / 3) < 1e-6, `a.vx=${a.vx}`);
  assert.ok(Math.abs(b.vx - 8 / 3) < 1e-6, `b.vx=${b.vx}`);
  assert.ok(Math.abs(mechWorld.kinetic() - ke0) < 1e-6);
});

test("e=0 完全非弹性：法向相对速度归零，动量守恒、动能损失", () => {
  mechWorld.resetFrom([
    { id: "a", x: -2, y: 0, vx: 4, vy: 0, r: 0.3, m: 1, color: "#000" },
    { id: "b", x: 2, y: 0, vx: 0, vy: 0, r: 0.3, m: 3, color: "#111" },
  ]);
  const p0 = 4;
  const ke0 = mechWorld.kinetic();
  mechWorld.step(1, { gravity: 0, restitution: 0, drag: 0 });
  const a = mechWorld.get("a")!;
  const b = mechWorld.get("b")!;
  assert.ok(Math.abs(a.vx - b.vx) < 1e-6, `相对速度=${a.vx - b.vx}`);
  assert.ok(Math.abs(a.vx * 1 + b.vx * 3 - p0) < 1e-6);
  assert.ok(mechWorld.kinetic() < ke0);
});

test("墙反弹：vx 按 e 折返", () => {
  mechWorld.resetFrom([{ id: "a", x: -11.9, y: 0, vx: -5, vy: 0, r: 0.5, m: 1, color: "#000" }]);
  mechWorld.step(0.016, { gravity: 0, restitution: 0.5, drag: 0 });
  const a = mechWorld.get("a")!;
  assert.ok(Math.abs(a.vx - 2.5) < 1e-9, `vx=${a.vx}`);
  assert.ok(a.x - a.r >= -12, `x=${a.x}`);
});

test("launchVelocity：只用最近 120ms 窗口，越界钳到 ±40", () => {
  const v = launchVelocity([
    { t: 0, x: 0, y: 0 },
    { t: 100, x: 4, y: 1 },
    { t: 200, x: 7, y: 3 },
  ]);
  assert.deepEqual(v.map((q) => Math.round(q)), [30, 20]);
  const clamped = launchVelocity([
    { t: 0, x: 0, y: 0 },
    { t: 100, x: -1000, y: 1000 },
  ]);
  assert.deepEqual(clamped, [-40, 40]);
  assert.deepEqual(launchVelocity([]), [0, 0]);
});

test("力学预设：applyPreset 把球表铺进世界单例并停住运行态", () => {
  applyPreset("mech-projectile");
  assert.equal(mechWorld.balls.length, 3);
  assert.equal(mechWorld.balls[0].trail.length, 0);
  applyPreset("mech-newton");
  assert.equal(mechWorld.balls.length, 4);
  const st = useStore.getState();
  assert.equal(st.mech.gravity, 0);
  assert.equal(st.mech.restitution, 1);
  assert.equal(st.mech.running, false);
  assert.equal(st.mode, "mech");
});

test("轨迹封顶 450 点、resetFrom 清空全部活态", () => {
  mechWorld.resetFrom([{ id: "a", x: 0, y: 0, vx: 1, vy: 0, r: 0.4, m: 1, color: "#000" }]);
  for (let k = 0; k < 600; k++) mechWorld.step(0.016, P0);
  assert.equal(mechWorld.balls[0].trail.length, 900);
  mechWorld.grabbedId = "a";
  mechWorld.resetFrom([{ id: "b", x: 0, y: 0, vx: 0, vy: 0, r: 0.4, m: 1, color: "#000" }]);
  assert.equal(mechWorld.t, 0);
  assert.equal(mechWorld.grabbedId, null);
  assert.equal(mechWorld.balls.length, 1);
  assert.equal(mechWorld.balls[0].trail.length, 0);
});
