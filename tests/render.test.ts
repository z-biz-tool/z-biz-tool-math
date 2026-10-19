// render/ 与 core/ 交界处的纯逻辑回归（无 React、无 Canvas、无 DOM）。
// 跑法：node --experimental-strip-types --test tests/render.test.ts
//
// 为什么测这两处（00 的 P1「render/ 与 ui/ 的可测性评估」的结论）：
//   `render/` 五个文件里绝大多数是 Canvas 绘制（测不了，也不该测）；
//   但有两处是**纯逻辑**、且失败形态全是静默的：
//
//   1. `globalsSig` —— 采样身份里的**参数指纹**。
//      同一表达式配不同参数值是两张图；指纹算错 ⇒ 栅格缓存复用到上一张图，
//      界面上就是"改了参数但画面没变"。不抛错、不白屏。
//
//   2. `rasterPool` —— 复平面全质量栅格的 worker 池。
//      派发的行带必须**恰好铺满** [0, h)：多算会重复写、少算会留白带；
//      环境不支持 worker 时必须**优雅退回**主线程串行续算，而不是崩掉。
//
// 这两处都不是"为凑覆盖率而测"——它们各自对应一类"界面上看着像坏了、
// 但没有任何报错"的故障。
import { test } from "node:test";
import assert from "node:assert/strict";
import { Engine } from "../src/core/machine.ts";
import { globalsSig } from "../src/render/scene2d.ts";
// RasterJob 声明在 scene2d.ts（栅格池只是 import 它），
// RasterSpec 才是栅格池自己导出的。
import type { RasterJob } from "../src/render/scene2d.ts";
import type { RasterSpec } from "../src/render/rasterPool.ts";

/** 造一个最小可用的 RasterSpec（字段与栅格池关心的那些一致） */
function spec(over: Partial<RasterSpec> = {}): RasterSpec {
  return {
    src: "z^2",
    kind: "newton",
    globals: [],
    iter: 64,
    plan: null,
    levelStep: 1,
    saturation: 1,
    selfIter: false,
    dark: false,
    ...over,
  };
}

/** 造一个最小可用的 RasterJob */
function job(h: number, w = 8): RasterJob {
  return {
    world: [-1, 1, -1, 1],
    w,
    h,
    rows: 0,
    rgb: new Uint8ClampedArray(w * h * 3),
    cols: [],
  } as unknown as RasterJob;
}

/**
 * 等回信排空队列：每回一条就再派一条，直到队列空。
 * 不用固定 sleep —— 改成「连续 20ms 没有新带就认为排空了」，快且不 flaky。
 */
async function drain(posted: { start: number; count: number }[]): Promise<void> {
  let last = -1;
  let stable = 0;
  for (let i = 0; i < 200 && stable < 3; i++) {
    await new Promise((r) => setTimeout(r, 10));
    if (posted.length === last) stable++;
    else {
      stable = 0;
      last = posted.length;
    }
  }
}

// ── globalsSig：参数指纹 ──────────────────────────────────────────────

test("globalsSig：空引擎的指纹是空串", () => {
  assert.equal(globalsSig(new Engine()), "");
});

test("globalsSig：参数变了指纹必须变（否则会复用上一张图）", () => {
  const e = new Engine();
  e.define("a = 1 + 2");
  const before = globalsSig(e);
  e.define("a = 10");
  const after = globalsSig(e);
  assert.notEqual(before, after, "参数改了，指纹必须变");
  assert.equal(after, "a=10,0");
});

test("globalsSig：新增一个参数，指纹也必须变", () => {
  const e = new Engine();
  e.define("a = 1");
  const before = globalsSig(e);
  e.define("b = 2");
  const after = globalsSig(e);
  assert.notEqual(before, after);
  assert.equal(after, "a=1,0,b=2,0");
});

test("globalsSig：指纹按参数名排序 ⇒ 与定义顺序无关", () => {
  const e1 = new Engine();
  e1.define("a = 1");
  e1.define("b = 2");
  const e2 = new Engine();
  e2.define("b = 2");
  e2.define("a = 1");
  assert.equal(globalsSig(e1), globalsSig(e2), "同样两个参数 ⇒ 同一张图");
});

// ── rasterPool：环境不支持 worker 时的降级 ────────────────────────────

test("rasterPool：没有 Worker 环境时退回主线程串行续算（不抛错）", async () => {
  // 本仓的 worker 池是**可选加速**：环境不支持时必须优雅退回，
  // 而不是让整幅栅格算不出来。契约是 takeRaster 返回 false。
  const saved = (globalThis as { Worker?: unknown }).Worker;
  delete (globalThis as { Worker?: unknown }).Worker;
  try {
    const pool = await import("../src/render/rasterPool.ts");
    assert.equal(pool.takeRaster(spec(), job(4)), false, "必须退回主线程");
    assert.equal(pool.rasterPoolSize(), 0, "池规模为 0");
    // 释放一张不是当前任务的图，不该出事
    pool.releaseRaster(job(4));
  } finally {
    if (saved === undefined) delete (globalThis as { Worker?: unknown }).Worker;
    else (globalThis as { Worker?: unknown }).Worker = saved;
  }
});

// ── rasterPool：行带必须恰好铺满 [0, h) ──────────────────────────────

test("rasterPool：派发的行带恰好铺满 [0, h)，不重不漏", async () => {
  // 用假 Worker 接住 postMessage，记录每一条带。
  // 关键不变量：所有带的 [start, start+count) **恰好**拼成 [0, h)。
  // 多算 ⇒ 重复写同一行；少算 ⇒ 留一条没画的带（画面缺一条）。
  //
  // 假 worker 必须**回话**：池里只有 4 个 worker，一次只能派 4 条带，
  // 其余排在队列里等回信。不回话就只看得见前几条，铺不满的断言就废了。
  // 回信是异步的（setTimeout）—— 真 worker 也是异步；
  // 同步回信会让池在同一轮里把队列排空，与真实行为不同。
  const posted: { start: number; count: number }[] = [];
  class FakeWorker {
    onmessage: ((ev: { data: unknown }) => void) | null = null;
    constructor() {}
    postMessage(m: { id: number; start: number; count: number }) {
      posted.push({ start: m.start, count: m.count });
      setTimeout(() => {
        this.onmessage?.({ data: { id: m.id, start: m.start, count: m.count } });
      }, 0);
    }
    terminate() {}
  }

  const saved = (globalThis as { Worker?: unknown }).Worker;
  (globalThis as { Worker?: unknown }).Worker = FakeWorker;
  try {
    // 用 query 破模块缓存，拿到一份**全新的**池状态
    const url = new URL("../src/render/rasterPool.ts?fake=1", import.meta.url).href;
    const pool = (await import(url)) as typeof import("../src/render/rasterPool.ts");

    for (const h of [1, 7, 40, 97]) {
      const before = posted.length;
      pool.takeRaster(spec(), job(h));
      await drain(posted);
      const bands = posted.slice(before).sort((a, b) => a.start - b.start);
      assert.ok(bands.length > 0, `h=${h}：至少派发一条带`);

      let cursor = 0;
      for (const b of bands) {
        assert.equal(b.start, cursor, `h=${h}：带必须首尾相接（期望 start=${cursor}）`);
        assert.ok(b.count > 0, `h=${h}：带的行数必须为正`);
        cursor += b.count;
      }
      assert.equal(cursor, h, `h=${h}：所有带必须恰好铺满 ${h} 行`);
    }
  } finally {
    if (saved === undefined) delete (globalThis as { Worker?: unknown }).Worker;
    else (globalThis as { Worker?: unknown }).Worker = saved;
  }
});

test("rasterPool：行带的大小由 WORKERS×BANDS_EACH 决定（不是任意切分）", async () => {
  // 为什么要钉「带多大」而不只钉「铺满」：
  //   带长直接决定「换图后多久能把线程腾出来」——实现注释写明
  //   「worker 收不到取消，带不能长」。带算错 ⇒ 交互卡顿，但界面不报错。
  //   只断言「铺满 [0,h)」的话，把带大小改成一半仍然铺满 ⇒ 判据不敏感。
  const posted: { start: number; count: number }[] = [];
  class FakeWorker {
    onmessage: ((ev: { data: unknown }) => void) | null = null;
    constructor() {}
    postMessage(m: { id: number; start: number; count: number }) {
      posted.push({ start: m.start, count: m.count });
      setTimeout(() => {
        this.onmessage?.({ data: { id: m.id, start: m.start, count: m.count } });
      }, 0);
    }
    terminate() {}
  }

  // 复算实现里的两个常量（它们是模块私有的）：
  //   WORKERS = max(2, min(4, cores>0 ? floor(cores/2) : 2))，BANDS_EACH = 10
  const cores =
    typeof navigator === "undefined" ? 0 : navigator.hardwareConcurrency || 0;
  const workers = Math.max(2, Math.min(4, cores > 0 ? Math.floor(cores / 2) : 2));
  const bandsEach = 10;

  const saved = (globalThis as { Worker?: unknown }).Worker;
  (globalThis as { Worker?: unknown }).Worker = FakeWorker;
  try {
    const url = new URL("../src/render/rasterPool.ts?fake=3", import.meta.url).href;
    const pool = (await import(url)) as typeof import("../src/render/rasterPool.ts");

    for (const h of [7, 40, 97]) {
      const before = posted.length;
      pool.takeRaster(spec(), job(h));
      await drain(posted);
      const bands = posted.slice(before).sort((a, b) => a.start - b.start);
      assert.ok(bands.length > 0, `h=${h}：至少派发一条带`);
      const expected = Math.max(1, Math.ceil(h / (workers * bandsEach)));
      // 末带可以是余数（h=97、band=3 ⇒ 末带只有 1 行），其余带必须等长
      for (let i = 0; i < bands.length - 1; i++) {
        assert.equal(
          bands[i].count,
          expected,
          `h=${h}：第 ${i} 条带必须是 ${expected} 行（WORKERS=${workers}×BANDS_EACH=${bandsEach}）`
        );
      }
      const last = bands[bands.length - 1];
      assert.ok(
        last.count >= 1 && last.count <= expected,
        `h=${h}：末带长度 ${last.count} 应在 [1, ${expected}]`
      );
      assert.equal(
        (bands.length - 1) * expected + last.count,
        h,
        `h=${h}：等长带 + 余数带必须正好铺满`
      );
    }
  } finally {
    if (saved === undefined) delete (globalThis as { Worker?: unknown }).Worker;
    else (globalThis as { Worker?: unknown }).Worker = saved;
  }
});

test("rasterPool：同一张图重复 takeRaster 不会重复派发", async () => {
  const posted: { start: number; count: number }[] = [];
  class FakeWorker {
    onmessage: ((ev: { data: unknown }) => void) | null = null;
    constructor() {}
    postMessage(m: { id: number; start: number; count: number }) {
      posted.push({ start: m.start, count: m.count });
      // 故意**不回话**：模拟 worker 还没回来
    }
    terminate() {}
  }

  const saved = (globalThis as { Worker?: unknown }).Worker;
  (globalThis as { Worker?: unknown }).Worker = FakeWorker;
  try {
    const url = new URL("../src/render/rasterPool.ts?fake=2", import.meta.url).href;
    const pool = (await import(url)) as typeof import("../src/render/rasterPool.ts");
    const j = job(40);
    assert.equal(pool.takeRaster(spec(), j), true);
    const first = posted.length;
    assert.ok(first > 0, "第一次应当派发若干条带");
    assert.equal(pool.takeRaster(spec(), j), true, "同一张图再次调用仍归池管");
    assert.equal(posted.length, first, "不允许重复派发（会重复写同一行）");
  } finally {
    if (saved === undefined) delete (globalThis as { Worker?: unknown }).Worker;
    else (globalThis as { Worker?: unknown }).Worker = saved;
  }
});
