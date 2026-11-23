// 工程结构校验及真实 store / engine 的导入回归；不 mock 内核、不依赖 DOM。
// 跑法：node --experimental-strip-types --test tests/exporters.test.ts
import { afterEach, test } from "node:test";
import assert from "node:assert/strict";
import { applyProject, serializeProject } from "../src/ui/exporters.ts";
import { useStore, type Param } from "../src/state.ts";
import { evalString } from "../src/core/machine.ts";

const original = useStore.getState();
const originalGlobals = new Map(original.engine.globals);
const originalFns = new Map(original.engine.fns);
afterEach(() => {
  original.engine.globals.clear();
  for (const [name, value] of originalGlobals) original.engine.globals.set(name, value);
  original.engine.fns.clear();
  for (const [name, fn] of originalFns) original.engine.fns.set(name, fn);
  useStore.setState(original, true);
});

function param(name: string, value: number): Param {
  return { name, value, min: -50, max: 50, step: 0.01, animate: false, speed: 1 };
}

function seedSliders(): void {
  const s = useStore.getState();
  s.engine.globals.clear();
  const params = [param("a", 2), param("c", 37)];
  for (const p of params) s.engine.setNum(p.name, p.value);
  s.engine.define("consoleOnly = 42");
  s.patch({ params });
}

/** 造一份**结构合法**的工程 JSON（各校验分支用它当底） */
function validProject(extra: Record<string, unknown> = {}): string {
  return JSON.stringify({
    v: 1,
    layers: [],
    params: [],
    ...extra,
  });
}

test("applyProject：实际导入移除旧 c、更新同名 a 并同步新增参数，保留控制台变量", () => {
  seedSliders();
  const engine = useStore.getState().engine;
  assert.equal(evalString(engine, "c").re, 37);
  const project = JSON.parse(serializeProject());
  project.params = [param("a", 9), param("b", -4)];
  applyProject(JSON.stringify(project));

  const s = useStore.getState();
  assert.equal(s.engine, engine, "仍使用共享引擎，而不是换一个新单例");
  assert.equal(s.geo.doc.engine, engine);
  assert.deepEqual(s.params, project.params);
  assert.deepEqual([...engine.globals.keys()].sort(), [...s.params.map((p) => p.name), "consoleOnly"].sort());
  for (const p of s.params) assert.equal(evalString(engine, p.name).re, p.value);
  assert.equal(evalString(engine, "a").re, 9);
  assert.throws(() => evalString(engine, "c"), /未定义/);
  assert.equal(evalString(engine, "consoleOnly").re, 42);
});

test("applyProject：空参数表仅清除旧滑块变量，不清除控制台变量和同名函数", () => {
  seedSliders();
  const engine = useStore.getState().engine;
  engine.define("c(x) = x + 1");
  applyProject(validProject());
  assert.deepEqual(useStore.getState().params, []);
  assert.deepEqual([...engine.globals.keys()], ["consoleOnly"]);
  assert.equal(evalString(engine, "consoleOnly").re, 42);
  assert.equal(evalString(engine, "c(2)").re, 3);
});

test("applyProject：所有参数先校验，后部畸形不能留下前部新值或删除旧 c", () => {
  seedSliders();
  const before = useStore.getState();
  const globals = new Map(before.engine.globals);
  const malformed = [
    null, [], "bad", {}, { name: 123, value: 1 },
    param("", 1), param("a b", 1), param("1a", 1), param("pi", 1),
    { ...param("b", 1), value: "3" },
    { ...param("b", 1), value: null },
    { name: "b" }, param("a", 99),
  ];
  const texts = malformed.map((bad) => validProject({ params: [param("a", 9), bad] }));
  // JSON 可以把 1e400 解析成 Infinity，不能只校验 typeof number。
  texts.push('{"v":1,"layers":[],"params":[{"name":"a","value":9},{"name":"b","value":1e400}]}');
  for (const text of texts) {
    assert.throws(() => applyProject(text), /参数/);
    assert.equal(useStore.getState(), before, "失败不得 patch store");
    assert.deepEqual(before.engine.globals, globals);
    assert.equal(evalString(before.engine, "a").re, 2);
    assert.equal(evalString(before.engine, "c").re, 37);
  }
});

test("applyProject：几何解析或重建失败保留原参数、引擎和几何文档", () => {
  seedSliders();
  const before = useStore.getState();
  const globals = new Map(before.engine.globals);
  for (const geo of ["{坏 JSON", "{}", JSON.stringify({ n: 1, order: ["missing"], items: [] })]) {
    assert.throws(() => applyProject(validProject({ params: [param("a", 9)], geo })), /几何文档损坏/);
    assert.equal(useStore.getState(), before);
    assert.equal(useStore.getState().geo.doc, before.geo.doc);
    assert.deepEqual(before.engine.globals, globals);
    assert.equal(evalString(before.engine, "a").re, 2);
    assert.equal(evalString(before.engine, "c").re, 37);
  }
});

test("applyProject：后续力学重置所需列表损坏时也不能先写入参数", () => {
  seedSliders();
  const before = useStore.getState();
  const globals = new Map(before.engine.globals);
  assert.throws(() => applyProject(validProject({ params: [param("a", 9)], mech: { balls: null } })), /力学球体列表无效/);
  assert.equal(useStore.getState(), before);
  assert.deepEqual(before.engine.globals, globals);
});

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
