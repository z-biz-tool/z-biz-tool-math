/**
 * 复平面全质量栅格的 worker 池。
 *
 * 整幅 Newton / 共形着色是 1~3 秒的纯算术，行与行互不依赖，按行带分给多个 worker 即可；
 * worker 只负责采样（Newton 交回 iter/root），着色仍由主线程的 shadeNewton 完成，
 * 于是分带结果与主线程逐行续算出自同一份代码，逐字节一致。
 *
 * 只在表达式编得出无分配快路径（f.cf）时使用：通用解释路径依赖引擎里的用户函数，
 * 那部分状态不过线程，编不出快路径就留在主线程串行采样。
 */
import { shadeNewton, type NewtonPlan } from "../core/cplane.ts";
import type { RasterJob } from "./scene2d.ts";

export interface RasterSpec {
  /** 表达式文本，worker 侧自己 parse + compile */
  src: string;
  kind: "newton" | "domain";
  /** 参数指纹的全部来源：名字与数值，worker 按同一份编闭包 */
  globals: [string, number, number][];
  iter: number;
  plan: NewtonPlan | null;
  levelStep: number;
  saturation: number;
  /** 共形着色的不动点迭代（z←f(z)），与主线程同判据 */
  selfIter: boolean;
  dark: boolean;
}

export interface RasterReq extends RasterSpec {
  id: number;
  x0: number;
  x1: number;
  yTop: number;
  dy: number;
  w: number;
  start: number;
  count: number;
}

export interface RasterRes {
  id: number;
  start: number;
  count: number;
  /** domain：整带 RGB；newton：迭代步数与归属根 */
  rgb?: ArrayBuffer;
  iterBuf?: ArrayBuffer;
  rootBuf?: ArrayBuffer;
}

interface Task {
  job: RasterJob;
  req: RasterReq;
  queue: [number, number][];
  /** 已派发未回的带数 */
  out: number;
  /** 池接不动了（出错 / 剩尾数），交回主线程串行续算 */
  dead: boolean;
}

const cores = typeof navigator === "undefined" ? 0 : navigator.hardwareConcurrency || 0;
/** 主线程还得留一核画布与合成，worker 数按核数折半封顶 4 */
const WORKERS = Math.max(2, Math.min(4, cores > 0 ? Math.floor(cores / 2) : 2));
/**
 * 每个 worker 排 10 带。带长决定「换图后多久能把线程腾出来」：
 * worker 在一带里是同步算到底的，收不到取消，所以带不能长——
 * 全 4 带时一次拖拽会把 4 个 worker 各锁 200ms，新图只能排在旧图后面。
 */
const BANDS_EACH = 10;

let broken = false;
const pool: Worker[] = [];
const idle: Worker[] = [];
const busy = new Map<Worker, Task>();
const owner = new Map<RasterJob, Task>();
let current: Task | null = null;
let seq = 1;

/** 起池：一个 worker 都起不来就算不可用 */
function ensurePool(): boolean {
  if (broken || typeof Worker === "undefined") return false;
  if (pool.length) return true;
  for (let i = 0; i < WORKERS; i++) {
    let wk: Worker;
    try {
      // 必须写成行内的 new URL：Vite 靠这个字面式识别 worker 入口并单独出包
      wk = new Worker(new URL("./rasterWorker.ts", import.meta.url), { type: "module" });
    } catch {
      // CSP 或环境不支持 worker：整幅栅格退回主线程串行续算
      break;
    }
    wk.onmessage = (ev: MessageEvent<RasterRes>) => settle(wk, ev.data);
    wk.onerror = () => drop(wk);
    pool.push(wk);
    idle.push(wk);
  }
  if (!pool.length) {
    broken = true;
    return false;
  }
  return true;
}

/** 出错的 worker 摘掉：它手上那一带永远不回来，任务转主线程兜底 */
function drop(w: Worker): void {
  const t = busy.get(w);
  busy.delete(w);
  const i = idle.indexOf(w);
  if (i >= 0) idle.splice(i, 1);
  const k = pool.indexOf(w);
  if (k >= 0) pool.splice(k, 1);
  w.terminate();
  if (t) {
    t.out--;
    t.dead = true;
  }
  if (!pool.length) broken = true;
}

function settle(w: Worker, res: RasterRes): void {
  const t = busy.get(w);
  busy.delete(w);
  if (pool.includes(w)) idle.push(w);
  if (!t || t.req.id !== res.id) return;
  t.out--;
  if (!t.dead) write(t, res);
  pump();
}

/** 行偏移由 start 定位，写序无关；job.rows 只是完成计数，凑满 h 即整幅就绪 */
function write(t: Task, res: RasterRes): void {
  const job = t.job;
  const n = res.count * job.w;
  const off = res.start * job.w * 3;
  if (t.req.kind === "domain") {
    if (!res.rgb) return;
    job.rgb.set(new Uint8ClampedArray(res.rgb), off);
  } else {
    if (!res.iterBuf || !res.rootBuf) return;
    shadeNewton(new Uint16Array(res.iterBuf), new Int16Array(res.rootBuf), n, job.cols, t.req.dark, job.rgb, off);
  }
  job.rows += res.count;
}

/** 有空闲 worker 就把队列里的行带发出去 */
function pump(): void {
  const t = current;
  if (!t || t.dead) return;
  while (idle.length && t.queue.length) {
    const w = idle.shift()!;
    const band = t.queue.shift()!;
    t.out++;
    busy.set(w, t);
    w.postMessage({ ...t.req, start: band[0], count: band[1] });
  }
  // 带派发完、也全回来了，却没满幅：worker 侧算不动（例如表达式无定义），交回主线程
  if (!t.out && !t.queue.length) t.dead = true;
}

/** 换图了：上一张还占着池就放掉，它剩下的带不再外派 */
export function releaseRaster(job: RasterJob): void {
  if (current && current.job !== job) {
    current.dead = true;
    current = null;
  }
}

/**
 * 这张图的采样归 worker 池管了吗？
 * true 表示主线程这一帧不必动手（还没填满，等回带）；
 * false 表示池接不了或中途塌了，主线程要自己按行续算。
 */
export function takeRaster(spec: RasterSpec, job: RasterJob): boolean {
  let t = owner.get(job);
  if (!t) {
    if (!ensurePool()) return false;
    const [left, right, bottom, top] = job.world;
    const req: RasterReq = {
      ...spec,
      id: seq++,
      x0: left,
      x1: right,
      yTop: top,
      dy: (bottom - top) / (job.h - 1),
      w: job.w,
      start: 0,
      count: 0,
    };
    t = { job, req, queue: [], out: 0, dead: false };
    const band = Math.max(1, Math.ceil(job.h / (WORKERS * BANDS_EACH)));
    for (let s = 0; s < job.h; s += band) t.queue.push([s, Math.min(band, job.h - s)]);
    owner.set(job, t);
    current = t;
    pump();
    return true;
  }
  if (job.rows >= job.h) {
    owner.delete(job);
    if (current === t) current = null;
    return !t.dead;
  }
  if (t.dead) {
    owner.delete(job);
    if (current === t) current = null;
    return false;
  }
  pump();
  return true;
}

/** 池现在的规模：自检用 */
export function rasterPoolSize(): number {
  return pool.length;
}
