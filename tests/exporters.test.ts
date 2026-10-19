// ui/exporters.ts 里**不碰浏览器与 store** 的那部分：工程文件的结构校验。
// 跑法：node --experimental-strip-types --test tests/exporters.test.ts
//
// 为什么测它（00 的 P1「render/ 与 ui/ 的可测性评估」的结论之一）：
//   `exporters.ts` 205 行里，真正值得测的是 `applyProject` **开头那几道校验**——
//   它们在触碰 store 之前就抛错，所以**不需要 mock zustand 单例、不需要 DOM**。
//   而这几道校验恰好是「静默失败」的反面：工程文件损坏时必须**报错**，
//   不能悄悄把用户的图换成一堆默认值。
//
// 本文件**不测**的部分（写进 05 的未落地项，不假装测过）：
//   · 完整往返（serializeProject → applyProject）会改全局 store 与视口，需要 mock；
//   · saveProject / exportPng 依赖 Blob / document / canvas，属浏览器侧；
//   · applyProject 成功路径同样会 patch store。
import { test } from "node:test";
import assert from "node:assert/strict";
import { applyProject } from "../src/ui/exporters.ts";

/** 造一份**结构合法**的工程 JSON（各校验分支用它当底） */
function validProject(extra: Record<string, unknown> = {}): string {
  return JSON.stringify({
    v: 1,
    layers: [],
    params: [],
    ...extra,
  });
}

test("applyProject：不是合法 JSON 时报中文错", () => {
  assert.throws(() => applyProject("not json"), /文件不是合法的 JSON/);
  assert.throws(() => applyProject(""), /文件不是合法的 JSON/);
  assert.throws(() => applyProject("{"), /文件不是合法的 JSON/);
});

test("applyProject：缺版本标记时拒绝（不接受将来的格式）", () => {
  assert.throws(() => applyProject("{}"), /缺少工程版本标记 v:1/);
  assert.throws(() => applyProject(JSON.stringify({ v: 2 })), /缺少工程版本标记 v:1/);
});

test("applyProject：图层或参数列表缺失时拒绝", () => {
  assert.throws(() => applyProject(JSON.stringify({ v: 1 })), /图层或参数列表缺失/);
  assert.throws(
    () => applyProject(JSON.stringify({ v: 1, layers: [] })),
    /图层或参数列表缺失/
  );
  assert.throws(
    () => applyProject(JSON.stringify({ v: 1, layers: [], params: "x" })),
    /图层或参数列表缺失/
  );
});

test("applyProject：几何文档损坏时报错，而不是静默丢图", () => {
  // 几何文档是**可选**段落：缺失时保留当前文档，损坏时报错。
  // 这两种情形的区别很容易被后来者改坏，所以钉住。
  assert.throws(
    () => applyProject(validProject({ geo: "{这不是合法 JSON" })),
    /几何文档损坏/
  );
});

test("applyProject：顶层不是对象时拒绝", () => {
  // 注意："[1,2,3]" 是**合法 JSON**，只是不是合法工程 ⇒ 报的是版本标记那条错，
  // 不是「不是合法 JSON」。这两条错别混。
  assert.throws(() => applyProject("null"), /缺少工程版本标记 v:1/);
  assert.throws(() => applyProject("42"), /缺少工程版本标记 v:1/);
  assert.throws(() => applyProject("[1,2,3]"), /缺少工程版本标记 v:1/);
});
