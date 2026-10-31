/** 生命游戏内核：规则解析、振荡子、滑翔机平移、环形边界 */
import test from "node:test";
import assert from "node:assert/strict";
import { LifeWorld, parseRule, ruleOf, applySeed, world } from "../src/core/life.ts";
import { applyPreset } from "../src/presets.ts";
import { useStore } from "../src/state.ts";

const B3S23 = parseRule("B3/S23")!;

test("parseRule：合法记法落表，非法返回 null", () => {
  const r = parseRule("B36/S23")!;
  assert.deepEqual(
    [3, 6].map((n) => r.birth[n]),
    [true, true],
  );
  assert.equal(r.birth[2], false);
  assert.equal(r.survive[2], true);
  assert.equal(parseRule("B3/S9"), null);
  assert.equal(parseRule("hello"), null);
});

test("ruleOf：同一字符串不重复解析，字符串变化后重解析", () => {
  const a = ruleOf("B3/S23");
  assert.equal(ruleOf("B3/S23"), a);
  assert.notEqual(ruleOf("B2/S"), a);
});

test("振荡子：横放闪灯一步变竖放，两步复原", () => {
  const w = new LifeWorld(8, 8);
  w.set(3, 4, 1);
  w.set(4, 4, 1);
  w.set(5, 4, 1);
  w.step(B3S23);
  assert.equal(w.get(4, 3), 1);
  assert.equal(w.get(4, 5), 1);
  assert.equal(w.get(3, 4), 0);
  w.step(B3S23);
  assert.equal(w.get(3, 4), 1);
  assert.equal(w.get(5, 4), 1);
  assert.equal(w.population, 3);
});

test("滑翔机：四步沿对角线平移一格（向右上）", () => {
  const w = new LifeWorld(16, 16);
  w.set(1, 1, 1);
  w.set(2, 1, 1);
  w.set(3, 1, 1);
  w.set(3, 2, 1);
  w.set(2, 3, 1);
  const pop = w.population;
  for (let i = 0; i < 4; i++) w.step(B3S23);
  assert.equal(w.population, pop);
  // 整体平移 (+1, -1)：与 t0 的五个格子一一对照
  assert.equal(w.get(2, 0), 1);
  assert.equal(w.get(3, 0), 1);
  assert.equal(w.get(4, 0), 1);
  assert.equal(w.get(4, 1), 1);
  assert.equal(w.get(3, 2), 1);
  assert.equal(w.get(1, 1), 0);
});

test("环形边界：贴边的活格能数到对面的邻居", () => {
  const w = new LifeWorld(6, 6);
  // (0,0) 的右、下、右下三邻居全活：B3 出生
  w.set(1, 0, 1);
  w.set(0, 1, 1);
  w.set(1, 1, 1);
  w.step(B3S23);
  assert.equal(w.get(0, 0), 1);
});

test("set/toggle/population 与清空", () => {
  const w = new LifeWorld(4, 4);
  w.set(1, 1, 1);
  w.set(2, 2, 1);
  assert.equal(w.population, 2);
  w.toggle(1, 1);
  assert.equal(w.population, 1);
  w.clear();
  assert.equal(w.population, 0);
  assert.equal(w.gen, 0);
  // 世界外写入被忽略
  w.set(-1, 0, 1);
  assert.equal(w.population, 0);
});

test("applyPreset：life-gun 切模式、撒下 36 枚机枪、带回规则配置", () => {
  applyPreset("life-gun");
  const st = useStore.getState();
  assert.equal(st.mode, "life");
  assert.equal(world.population, 36);
  assert.equal(world.gen, 0);
  assert.equal(st.life.rule, "B3/S23");
  assert.equal(st.life.running, false);
  applySeed({ kind: "random", density: 0.3 });
  assert.ok(world.population > 0);
  // 收尾清干净，别把单例世界留给后面可能的用例
  world.clear();
});
