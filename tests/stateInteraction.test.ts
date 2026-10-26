// state.ts 里与交互结果直接相关的动作：删参数必须连引擎里的变量一起删。
// 跑法：node --experimental-strip-types --test tests/stateInteraction.test.ts
//
// 为什么要测：参数滑块显示的是一份 store 列表，表达式求值用的却是引擎里的 globals。
// 只删滑块不删变量时，画布上引用该名字的曲线仍然照常画出来——界面在骗人。
import { test } from "node:test";
import assert from "node:assert/strict";
import { useStore, type Param } from "../src/state.ts";

function param(name: string, value = 1): Param {
  return { name, value, min: -5, max: 5, step: 0.01, animate: false, speed: 1 };
}

test("removeParam：滑块和引擎变量一起删掉", () => {
  const st = useStore.getState();
  st.engine.setNum("zz_test", 3);
  st.addParam(param("zz_test", 3));
  assert.equal(st.engine.globals.has("zz_test"), true);

  useStore.getState().removeParam("zz_test");

  assert.equal(useStore.getState().params.some((p) => p.name === "zz_test"), false);
  assert.equal(useStore.getState().engine.globals.has("zz_test"), false);
});

test("removeParam：不误删同名之外的其它参数", () => {
  const st = useStore.getState();
  st.engine.setNum("zz_keep", 2);
  st.addParam(param("zz_keep", 2));
  st.engine.setNum("zz_drop", 7);
  st.addParam(param("zz_drop", 7));

  useStore.getState().removeParam("zz_drop");

  assert.equal(useStore.getState().engine.globals.has("zz_drop"), false);
  assert.equal(useStore.getState().engine.globals.has("zz_keep"), true);
  assert.equal(useStore.getState().params.some((p) => p.name === "zz_keep"), true);
});
