/**
 * 工程与图片导出：几何画板类的工具必须能把画出来的东西带走。
 *  - PNG：把舞台上的若干层 canvas（场景层 + 叠加层）按原分辨率合成；
 *  - 工程 JSON：完整状态（图层、参数、各模式子状态、视口、几何文档），
 *    载入时按文件内容重建，表达式内核仍用全局唯一 engine。
 */

import { GeometryDoc } from "../core/geometry.ts";
import { Viewport, type ViewportInit } from "../core/view.ts";
import {
  useStore,
  type ComplexState,
  type GeoLabState,
  type Layer,
  type LinState,
  type Mode,
  type NnState,
  type Param,
  type Settings,
  type SurfState,
  type VectorState,
} from "../state.ts";

const MODES: Mode[] = ["func", "geom", "complex", "vector", "lin", "nn", "surf", "console"];

/**
 * 神经网络落盘的部分：只有超参数与显示设置。
 * model/data 是活的 TypedArray —— JSON.stringify 把 Float64Array 写成 {"0":…,"1":…}
 * 这样的普通对象，读回来仍是个真值，但 .length 变 undefined、.fill 直接没有，
 * 面板会以为「已有模型」而拿着一堆索引对象去前向/清零。所以导出时剥掉，
 * 载入后置空，由面板按这些超参数重建；训练进度（epochs/loss/curve）属于被丢掉
 * 的那个模型，跟着一起丢，免得报出一张对不上号的准确率。
 */
type NnPersist = Omit<NnState, "model" | "data" | "running" | "epochs" | "loss" | "acc" | "curve">;

/** 逐字段挑出可序列化的那部分：NnState 将来添字段时靠类型报出来，而不是被静默丢件或整份带出 */
function nnToProject(n: NnState): NnPersist {
  return {
    dataset: n.dataset,
    samples: n.samples,
    hidden: n.hidden,
    depth: n.depth,
    act: n.act,
    lr: n.lr,
    momentum: n.momentum,
    batch: n.batch,
    seed: n.seed,
    showBoundary: n.showBoundary,
    boundaryRes: n.boundaryRes,
  };
}

export interface Project {
  v: 1;
  mode: Mode;
  layers: Layer[];
  params: Param[];
  cplx: ComplexState;
  vec: VectorState;
  lin: LinState;
  nn: NnPersist;
  surf: SurfState;
  settings: Settings;
  views: Partial<Record<Mode, ViewportInit>>;
  /** 几何文档快照（GeometryDoc.toJSON 的原文） */
  geo: string;
  geoUi: { tool: string; showTrace: boolean; showLabels: boolean; snap: boolean; grid: boolean };
}

function stamp(): string {
  const d = new Date();
  const p = (n: number) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${p(d.getMonth() + 1)}${p(d.getDate())}-${p(d.getHours())}${p(d.getMinutes())}`;
}

function saveBlob(name: string, mime: string, data: BlobPart): void {
  const url = URL.createObjectURL(new Blob([data], { type: mime }));
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  // 交给浏览器发起下载后再回收，立即 revoke 会让部分浏览器拿到空文件
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

export function serializeProject(): string {
  const s = useStore.getState();
  const views: Partial<Record<Mode, ViewportInit>> = {};
  for (const m of MODES) views[m] = s.views[m].toJSON();
  const p: Project = {
    v: 1,
    mode: s.mode,
    layers: s.layers,
    params: s.params,
    cplx: s.cplx,
    vec: s.vec,
    lin: s.lin,
    nn: nnToProject(s.nn),
    surf: s.surf,
    settings: s.settings,
    views,
    geo: s.geo.doc.toJSON(),
    geoUi: {
      tool: s.geo.tool,
      showTrace: s.geo.showTrace,
      showLabels: s.geo.showLabels,
      snap: s.geo.snap,
      grid: s.geo.grid,
    },
  };
  return JSON.stringify(p, null, 1);
}

export function saveProject(): string {
  const text = serializeProject();
  const name = `geolab-${stamp()}.json`;
  saveBlob(name, "application/json", text);
  return name;
}

/** 几何文档是可选段落：缺失时保留当前文档，损坏时报错而不是静默丢图 */
function loadGeo(raw: string | undefined, s: GeoLabState): GeometryDoc {
  if (typeof raw !== "string" || !raw.trim()) return s.geo.doc;
  try {
    return GeometryDoc.fromJSON(raw, s.engine);
  } catch (e) {
    throw new Error(`几何文档损坏：${e instanceof Error ? e.message : String(e)}`);
  }
}

/** 载入工程；文件不合结构时抛中文错误，供界面提示 */
export function applyProject(text: string): void {
  let data: Project;
  try {
    data = JSON.parse(text) as Project;
  } catch {
    throw new Error("文件不是合法的 JSON");
  }
  if (!data || typeof data !== "object" || data.v !== 1) throw new Error("缺少工程版本标记 v:1");
  if (!Array.isArray(data.layers) || !Array.isArray(data.params)) throw new Error("图层或参数列表缺失");
  const s = useStore.getState();
  const views = { ...s.views };
  for (const m of MODES) {
    const init = data.views?.[m];
    if (init && Number.isFinite(init.scale) && init.scale > 0) views[m] = new Viewport(init);
  }
  const doc = loadGeo(data.geo, s);
  for (const p of data.params) s.engine.setNum(p.name, p.value);
  s.patch({
    mode: MODES.includes(data.mode) ? data.mode : s.mode,
    layers: data.layers,
    params: data.params,
    activeLayer: data.layers[0]?.id ?? null,
    cplx: { ...s.cplx, ...data.cplx },
    vec: { ...s.vec, ...data.vec },
    lin: { ...s.lin, ...data.lin },
    /* 老工程没有 nn 段落时整个跳过，别把当前那个训练中的模型顺手清掉；
       有的话只恢复超参数，model/data 交回 null 让面板按新形状重建 */
    nn: data.nn
      ? { ...s.nn, ...data.nn, model: null, data: null, running: false, epochs: 0, loss: 0, acc: 0, curve: [] }
      : s.nn,
    surf: { ...s.surf, ...data.surf },
    settings: { ...s.settings, ...data.settings },
    views,
  });
  s.setGeo({
    doc,
    pending: [],
    selected: null,
    tool: (data.geoUi?.tool ?? "select") as typeof s.geo.tool,
    showTrace: data.geoUi?.showTrace ?? true,
    showLabels: data.geoUi?.showLabels ?? true,
    snap: data.geoUi?.snap ?? false,
    grid: data.geoUi?.grid ?? true,
  });
}

/** 合成舞台上所有画布并下载为 PNG；返回 false 表示当前没有可导出的可见画布 */
export function exportPng(): boolean {
  const layers = Array.from(document.querySelectorAll<HTMLCanvasElement>(".gl-stage canvas"));
  if (!layers.length) return false;
  // 折叠/隐藏时画布只有 2×2 的保底尺寸，导出一张噪点图不如直说不可见
  if (!layers.some((c) => c.getBoundingClientRect().width > 4 && c.getBoundingClientRect().height > 4)) return false;
  const w = Math.max(...layers.map((c) => c.width));
  const h = Math.max(...layers.map((c) => c.height));
  if (!(w > 4 && h > 4)) return false;
  const out = document.createElement("canvas");
  out.width = w;
  out.height = h;
  const ctx = out.getContext("2d");
  if (!ctx) return false;
  for (const c of layers) ctx.drawImage(c, 0, 0);
  const name = `geolab-${stamp()}.png`;
  const a = document.createElement("a");
  a.href = out.toDataURL("image/png");
  a.download = name;
  a.click();
  return true;
}
