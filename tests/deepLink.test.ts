import { test } from "node:test";
import assert from "node:assert/strict";
import { resolveMathDeepLink } from "../src/_shared/deepLink.ts";
import type { Handoff } from "z-biz-tool-shared/protocol";

function item(params: Record<string, string>, action = "open"): Handoff {
  return { product: "math", action, params };
}

test("正常路径：open + preset 应用真实预设", () => {
  const out = resolveMathDeepLink(item({ preset: "func-basic" }));
  assert.equal(out.problem, null);
  assert.equal(out.preset, "func-basic");
  assert.equal(out.mode, null);
});

test("正常路径：open + mode 直接切模式", () => {
  const out = resolveMathDeepLink(item({ mode: "geom" }));
  assert.equal(out.problem, null);
  assert.equal(out.mode, "geom");
  assert.equal(out.preset, null);
});

test("正常路径：preset + mode 同时给，互不冲突", () => {
  const out = resolveMathDeepLink(item({ preset: "cpx-newton", mode: "surf" }));
  assert.equal(out.problem, null);
  assert.equal(out.preset, "cpx-newton");
  assert.equal(out.mode, "surf");
});

test("缺参数：open 不带任何参数 → no-op，不报错", () => {
  const out = resolveMathDeepLink(item({}));
  assert.equal(out.problem, null);
  assert.equal(out.preset, null);
  assert.equal(out.mode, null);
});

test("不认识的动作：fly → problem", () => {
  const out = resolveMathDeepLink(item({ preset: "func-basic" }, "fly"));
  assert.equal(out.problem, "不认识的动作 fly");
  assert.equal(out.preset, null);
});

test("对象 id 不存在：preset 不是真实预设 → problem，且不部分设置", () => {
  const out = resolveMathDeepLink(item({ preset: "ghost-preset" }));
  assert.equal(out.problem, "没有这个预设 ghost-preset");
  assert.equal(out.preset, null);
});

test("不认识的 mode：mode=weather → problem", () => {
  const out = resolveMathDeepLink(item({ mode: "weather" }));
  assert.equal(out.problem, "不认识的 mode weather");
  assert.equal(out.mode, null);
});

test("参数前后空格：preset=' cpx-newton ' 仍能命中", () => {
  const out = resolveMathDeepLink(item({ preset: "  cpx-newton  " }));
  assert.equal(out.problem, null);
  assert.equal(out.preset, "cpx-newton");
});

test("空字符串参数：preset='' → 视为未提供，不报错", () => {
  const out = resolveMathDeepLink(item({ preset: "" }));
  assert.equal(out.problem, null);
  assert.equal(out.preset, null);
});

test("各类预设 key 都被识别（nn-xor / lin-rot60 / surf-graph / vec-quiver）", () => {
  for (const k of ["nn-xor", "lin-rot60", "surf-graph", "vec-quiver"]) {
    const out = resolveMathDeepLink(item({ preset: k }));
    assert.equal(out.problem, null, `预设 ${k} 应被识别`);
    assert.equal(out.preset, k);
  }
});

test("from 透传：仅用于日志", () => {
  const out = resolveMathDeepLink(item({ preset: "func-basic", from: "box" }));
  assert.equal(out.problem, null);
  assert.equal(out.from, "box");
});
