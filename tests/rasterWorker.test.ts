// render/rasterWorker.ts（栅格采样 worker）的纯逻辑回归。
// 跑法：node --experimental-strip-types --test tests/rasterWorker.test.ts
//
// 为什么测它（00 的 P1「render/ 与 ui/ 的可测性评估」的结论之二）：
//   `rasterWorker.ts` 的文件头把「与主线程逐行续算同解」写成**三条约束**，
//   其中第 2 条是「行坐标用同一式子 y = yTop + dy * row，采样区间传 (y, y+dy)、
//   高度传 1」——**这正是最容易写错、且写错后完全静默**的一处：
//   行坐标差一个 dy，画面就是整幅错位，不抛错、不白屏。
//
// 怎么测：worker 入口在 Node 里可以直接 import（顶层只设 globalThis.onmessage），
//   把 globalThis.postMessage 换成收集器，喂一条带，拿回采样结果，
//   与**主线程用同样公式逐行采样**的结果逐字节比对。
import { test } from "node:test";
import assert from "node:assert/strict";
import { domainColor, newtonFractal, type CFn } from "../src/core/cplane.ts";
import { compileCplx, Engine, num } from "../src/core/machine.ts";
import { parseExpr } from "../src/core/parser.ts";
import type { RasterReq, RasterRes } from "../src/render/rasterPool.ts";

/** 收集 worker 回交的结果 */
const posted: RasterRes[] = [];
(globalThis as unknown as { postMessage: (m: RasterRes) => void }).postMessage = (m) => {
  posted.push(m);
};

// 顶层副作用：设 globalThis.onmessage
await import("../src/render/rasterWorker.ts");

/** 与 worker 内部**同一份**的快路径闭包构造（否则着色结果不可比） */
function fastFn(src: string, globals: [string, number, number][] = []): CFn {
  const eng = new Engine();
  for (const [nm, re, im] of globals) eng.globals.set(nm, num(re, im));
  const cf = compileCplx(eng, parseExpr(src), "z");
  assert.ok(cf, `表达式 ${src} 必须能编出快路径`);
  const slot = { re: 0, im: 0 };
  const g = ((z: { re: number; im: number }) => {
    cf(z.re, z.im, slot);
    return { re: slot.re, im: slot.im };
  }) as unknown as CFn;
  g.cf = cf;
  return g;
}

function baseReq(over: Partial<RasterReq> = {}): RasterReq {
  return {
    id: 1,
    src: "z^2",
    kind: "domain",
    globals: [],
    iter: 32,
    plan: null,
    levelStep: 1,
    saturation: 1,
    selfIter: false,
    dark: true,
    x0: -1,
    x1: 1,
    yTop: 1,
    dy: 0.25,
    w: 4,
    start: 1,
    count: 2,
    ...over,
  };
}

function fire(req: RasterReq): RasterRes {
  const before = posted.length;
  const onmessage = (globalThis as unknown as {
    onmessage: ((e: { data: RasterReq }) => void) | null;
  }).onmessage;
  assert.ok(onmessage, "worker 入口必须已挂上 onmessage");
  onmessage({ data: req });
  assert.equal(posted.length, before + 1, "worker 必须回交一条结果");
  return posted[posted.length - 1];
}

test("rasterWorker：按带采样的 domain 结果与主线程逐行采样逐字节相同", () => {
  const req = baseReq();
  const res = fire(req);
  assert.ok(res.rgb, "domain 模式必须回交 rgb");

  // 主线程逐行：行 k 的纵坐标 y = yTop + dy*(start+k)，采样区间 (y, y+dy)、高度 1
  const f = fastFn(req.src, req.globals);
  const expected = new Uint8ClampedArray(req.w * req.count * 3);
  for (let k = 0; k < req.count; k++) {
    const y = req.yTop + req.dy * (req.start + k);
    const row = domainColor(f, req.x0, req.x1, y, y + req.dy, req.w, 1, {
      levelStep: req.levelStep,
      dark: req.dark,
      iterFn: req.selfIter ? f : undefined,
      saturation: req.saturation,
      alpha: true,
    });
    expected.set(row, k * req.w * 3);
  }
  assert.deepEqual(
    Array.from(new Uint8ClampedArray(res.rgb)),
    Array.from(expected),
    "带采样必须与逐行采样逐字节相同"
  );
});

test("rasterWorker：按带采样的 newton 结果与主线程逐行采样逐字节相同", () => {
  const req = baseReq({ kind: "newton", iter: 24 });
  const res = fire(req);
  assert.ok(res.iterBuf && res.rootBuf, "newton 模式必须回交 iter/root");

  const f = fastFn(req.src, req.globals);
  const n = req.w * req.count;
  const it = new Uint16Array(n);
  const rt = new Int16Array(n);
  for (let k = 0; k < req.count; k++) {
    const y = req.yTop + req.dy * (req.start + k);
    const r = newtonFractal(f, req.x0, req.x1, y, y + req.dy, req.w, 1, {
      maxIter: req.iter,
      plan: req.plan ?? undefined,
    });
    it.set(r.iter, k * req.w);
    rt.set(r.root, k * req.w);
  }
  assert.deepEqual(
    Array.from(new Uint16Array(res.iterBuf)),
    Array.from(it),
    "迭代步数必须逐字节相同"
  );
  assert.deepEqual(
    Array.from(new Int16Array(res.rootBuf)),
    Array.from(rt),
    "归属根必须逐字节相同"
  );
});

test("rasterWorker：回交的行号与请求一致（start/count 原样带出）", () => {
  const req = baseReq({ start: 3, count: 1 });
  const res = fire(req);
  assert.equal(res.id, req.id);
  assert.equal(res.start, req.start);
  assert.equal(res.count, req.count);
});
