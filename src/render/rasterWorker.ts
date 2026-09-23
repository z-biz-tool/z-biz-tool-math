/**
 * 栅格采样 worker：收一行带（start..start+count），交回该带采样结果。
 *
 * 与主线程逐行续算同解的三条约束：
 *  1. 表达式与数值全局量取同一份，编译走同一个 compileCplx；
 *  2. 行坐标用同一式子 y = yTop + dy * row，采样区间传 (y, y+dy)、高度传 1，
 *     行内步长仍是整幅的 dy，于是不论按行还是按带都逐字节相同；
 *  3. 只走 f.cf 快路径，编不出快路径的表达式主线程不会派过来。
 */
import { Engine, compileCplx, num } from "../core/machine.ts";
import { parseExpr } from "../core/parser.ts";
import { domainColor, newtonFractal, type CFn } from "../core/cplane.ts";
import type { RasterReq, RasterRes } from "./rasterPool.ts";

const ctx = globalThis as unknown as {
  postMessage: (m: RasterRes, t: Transferable[]) => void;
  onmessage: ((e: { data: RasterReq }) => void) | null;
};

let lastSig = "";
let lastFn: CFn | null = null;

/** 表达式 + 参数值 → 无分配快路径闭包；参数没动就复用上一次编译 */
function fnFor(req: RasterReq): CFn | null {
  const sig = req.src + "\u0000" + req.globals.map(([k, re, im]) => `${k}=${re},${im}`).join(",");
  if (sig === lastSig && lastFn) return lastFn;
  const eng = new Engine();
  for (const [nm, re, im] of req.globals) eng.globals.set(nm, num(re, im));
  let cf = null;
  try {
    cf = compileCplx(eng, parseExpr(req.src), "z");
  } catch {
    cf = null;
  }
  if (!cf) {
    lastSig = "";
    lastFn = null;
    return null;
  }
  const slot = { re: 0, im: 0 };
  const g = ((z: { re: number; im: number }) => {
    cf!(z.re, z.im, slot);
    return { re: slot.re, im: slot.im };
  }) as unknown as CFn;
  g.cf = cf;
  lastSig = sig;
  lastFn = g;
  return g;
}

ctx.onmessage = (e) => {
  const req = e.data;
  const f = fnFor(req);
  if (!f) return;
  const w = req.w;
  const out: RasterRes = { id: req.id, start: req.start, count: req.count };
  if (req.kind === "newton") {
    const n = w * req.count;
    const it = new Uint16Array(n);
    const rt = new Int16Array(n);
    for (let k = 0; k < req.count; k++) {
      const y = req.yTop + req.dy * (req.start + k);
      const r = newtonFractal(f, req.x0, req.x1, y, y + req.dy, w, 1, { maxIter: req.iter, plan: req.plan ?? undefined });
      it.set(r.iter, k * w);
      rt.set(r.root, k * w);
    }
    out.iterBuf = it.buffer;
    out.rootBuf = rt.buffer;
    ctx.postMessage(out, [it.buffer, rt.buffer]);
    return;
  }
  const buf = new Uint8ClampedArray(w * req.count * 3);
  for (let k = 0; k < req.count; k++) {
    const y = req.yTop + req.dy * (req.start + k);
    const row = domainColor(f, req.x0, req.x1, y, y + req.dy, w, 1, {
      levelStep: req.levelStep,
      dark: req.dark,
      iterFn: req.selfIter ? f : undefined,
      saturation: req.saturation,
      alpha: true,
    });
    buf.set(row, k * w * 3);
  }
  out.rgb = buf.buffer;
  ctx.postMessage(out, [buf.buffer]);
};
