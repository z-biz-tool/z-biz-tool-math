/** GeoLab 应用状态：工作模式、图层、参数滑块、各模式独立视口 */

import { create } from "zustand";
import { Viewport } from "./core/view.ts";
import { paletteAt } from "./core/colormap.ts";
import { Engine } from "./core/machine.ts";
import { GeometryDoc } from "./core/geometry.ts";
import type { Act, Dataset, DatasetName, Model } from "./core/nn.ts";

export type Mode = "func" | "geom" | "complex" | "vector" | "lin" | "nn" | "surf" | "console";

export type LayerKind =
  | "cartesian"
  | "polar"
  | "param"
  | "implicit"
  | "inequality"
  | "derivative"
  | "integral"
  | "sequence";

export interface Layer {
  id: string;
  kind: LayerKind;
  expr: string;
  /** 参数方程的 y 分量 / 积分上限 */
  expr2?: string;
  label: string;
  color: string;
  width: number;
  visible: boolean;
  dashed: boolean;
  fill: boolean;
  /** x 或 t 的取值范围；未提供则用视口范围 */
  domain?: [number, number];
  /** 隐函数 / 不等式的比较方向 */
  rel?: "<" | ">" | "<=" | ">=";
  level?: number;
  samples: number;
  /** 动画参数（几何画板式的 a/b/c 滑块） */
  param?: string;
}

export interface Param {
  name: string;
  value: number;
  min: number;
  max: number;
  step: number;
  animate: boolean;
  speed: number;
}

export type ToolKind =
  | "select"
  | "point"
  | "segment"
  | "line"
  | "ray"
  | "vector"
  | "circle"
  | "arc"
  | "ellipse"
  | "polygon"
  | "midpoint"
  | "perpendicular"
  | "parallel"
  | "bisector"
  | "intersection"
  | "pointOn"
  | "locus"
  | "rotate"
  | "reflect"
  | "dilate"
  | "angle"
  | "area"
  | "text"
  | "erase";

export interface ComplexState {
  /** domain：共形着色；map：保角映射网格；newton：Newton 分形；vector：复向量场 */
  mode: "domain" | "map" | "newton" | "log";
  f: string;
  /** map 模式下被映射的曲线族 */
  curve: string;
  iterative: boolean;
  levelStep: number;
  colormap: string;
  resolution: number;
}

export interface VectorState {
  /** 自由向量列表（表达式） */
  arrows: { id: string; tail: string; vec: string; color: string; label: string }[];
  fieldMode: "quiver" | "slope" | "stream" | "phase";
  fx: string;
  fy: string;
  dfxy: string;
  density: number;
  streamlineCount: number;
  steps: number;
  colormap: string;
}

/** 线性代数工作台：矩阵 acts on the plane */
export interface LinState {
  dim: 2 | 3;
  /** 行优先展开的 dim² 个表达式串，走引擎求值（可引用参数滑块 a/b） */
  a: string[];
  /** Ax=b 的 b，长度 dim */
  b: string[];
  /** 整数网格线的像 */
  showGrid: boolean;
  /** 单位圆 */
  showCircle: boolean;
  /** 单位圆的像（椭圆/退化线段） */
  showEllipse: boolean;
  /** 实特征向量与特征值标注 */
  showEigen: boolean;
  /** SVD 奇异向量与像椭圆主轴 */
  showSVD: boolean;
  /** 轨道点 p ↦ M·p */
  showFlow: boolean;
  /** 幂次 k 或 e^{tA} 的时间 t */
  t: number;
  /** true：连续流 e^{tA}；false：离散迭代 A^k */
  useExp: boolean;
}

/** 神经网络工作台：小型 MLP 的决策边界与训练过程 */
export interface NnState {
  dataset: DatasetName;
  samples: number;
  /** 隐藏层宽度 */
  hidden: number;
  /** 隐藏层数（≥1） */
  depth: number;
  act: Act;
  lr: number;
  momentum: number;
  batch: number;
  seed: number;
  /** 训练循环开关（面板里的 rAF 驱动） */
  running: boolean;
  showBoundary: boolean;
  /** 决策边界栅格的水平采样数 */
  boundaryRes: number;
  epochs: number;
  loss: number;
  acc: number;
  /** 最近若干轮的 loss，供面板画迷你曲线 */
  curve: number[];
  /** 可变对象：原地训练，不随 revision 重建 */
  model: Model | null;
  data: Dataset | null;
}

export type SurfKind = "graph" | "param" | "implicit" | "revolve" | "spacecurve";
export type SurfStyle = "surf" | "mesh" | "wire" | "contour" | "surfc";

export interface SurfLayer {
  id: string;
  kind: SurfKind;
  expr: string;
  /** 参数曲面的 v 分量 / 旋转轴等 */
  expr2?: string;
  expr3?: string;
  style: SurfStyle;
  colormap: string;
  res: number;
  range: [number, number, number, number];
  zRange?: [number, number];
  visible: boolean;
  color: string;
  opacity: number;
  lit: boolean;
  label: string;
}

export interface ConsoleLine {
  src: string;
  out: string;
  ok: boolean;
}

export interface GeoState {
  tool: ToolKind;
  /** 构造队列：工具需要的前置选中对象 */
  pending: string[];
  doc: GeometryDoc;
  /** 当前选中对象（用于联动测量与高亮） */
  selected: string | null;
  showTrace: boolean;
  showLabels: boolean;
  snap: boolean;
  grid: boolean;
}

export interface Settings {
  dark: boolean;
  showMinorGrid: boolean;
  piTicksX: boolean;
  piTicksY: boolean;
  showCrosshair: boolean;
  antialias: boolean;
}

export interface SurfCam {
  azim: number;
  elev: number;
  dist: number;
  target: [number, number, number];
}

export interface SurfState {
  layers: SurfLayer[];
  cam: SurfCam;
  /** 世界坐标包围盒 [x0,x1,y0,y1,z0,z1] */
  box: [number, number, number, number, number, number];
  /** 是否按数据自动缩放包围盒 */
  autoBox: boolean;
  showAxes: boolean;
  lightAngle: number;
}

export interface GeoLabState {
  mode: Mode;
  layers: Layer[];
  activeLayer: string | null;
  params: Param[];
  engine: Engine;
  views: Record<Mode, Viewport>;
  geo: GeoState;
  cplx: ComplexState;
  vec: VectorState;
  lin: LinState;
  nn: NnState;
  surf: SurfState;
  console: { lines: ConsoleLine[]; input: string };
  settings: Settings;
  /** 递增以强制重绘（拖动、参数变化等） */
  revision: number;

  setMode(m: Mode): void;
  setView(m: Mode, v: Viewport): void;
  patch(p: Partial<GeoLabState>): void;
  bump(): void;
  addLayer(kind?: LayerKind, expr?: string): string;
  updateLayer(id: string, patch: Partial<Layer>): void;
  removeLayer(id: string): void;
  reorder(id: string, dir: -1 | 1): void;
  setParam(name: string, value: number): void;
  addParam(p: Param): void;
  removeParam(name: string): void;
  toggleAnimate(name: string): void;
  setSettings(s: Partial<Settings>): void;
  setGeo(p: Partial<GeoState>): void;
  setCplx(p: Partial<ComplexState>): void;
  setVec(p: Partial<VectorState>): void;
  setLin(p: Partial<LinState>): void;
  setNn(p: Partial<NnState>): void;
  setSurf(p: Partial<SurfState>): void;
}

let seq = 0;
export const uid = (p = "o"): string => `${p}${++seq}${Date.now().toString(36).slice(-3)}`;

const defaultView = (mode: Mode): Viewport =>
  new Viewport({
    cx: 0,
    cy: 0,
    scale:
      mode === "geom"
        ? 48
        : mode === "complex"
          ? 70
          : mode === "lin"
            ? 88
            : mode === "nn"
              ? 112
              : 60,
    width: 900,
    height: 640,
  });

/** 引擎与几何文档共享同一个表达式内核 */
const engine = new Engine();

/** 初始几何示例：三角形 + 两条中线 + 重心 + 边长测量 */
function seedGeo(): GeometryDoc {
  const doc = new GeometryDoc();
  doc.engine = engine;
  const A = doc.addPoint("A", -2.2, 0);
  const B = doc.addPoint("B", 2.6, 0);
  const C = doc.addPoint("C", 0.4, 3.1);
  const ab = doc.addSegment(A, B);
  doc.addSegment(B, C);
  doc.addSegment(C, A);
  const M = doc.addMidpoint("M", ab);
  const N = doc.addMidpoint("N", doc.addSegment(B, C));
  const cm = doc.addSegment(C, M);
  const an = doc.addSegment(A, N);
  doc.addIntersection("G", cm, an);
  doc.addMeasure("dist", [A, B], [0.2, -1.1], "|AB|");
  doc.recompute();
  return doc;
}

const initialLayers: Layer[] = [
  {
    id: uid("L"),
    kind: "cartesian",
    expr: "sin(x)*exp(-x/6)",
    label: "f(x) = sin(x)·e^(-x/6)",
    color: paletteAt(0),
    width: 2.4,
    visible: true,
    dashed: false,
    fill: false,
    samples: 1400,
  },
  {
    id: uid("L"),
    kind: "cartesian",
    expr: "x^2/6 - 1",
    label: "g(x) = x²/6 − 1",
    color: paletteAt(1),
    width: 2.2,
    visible: true,
    dashed: true,
    fill: false,
    samples: 1400,
  },
];

export const useStore = create<GeoLabState>((set, get) => ({
  mode: "func",
  layers: initialLayers,
  activeLayer: initialLayers[0].id,
  params: [
    { name: "a", value: 1, min: -3, max: 3, step: 0.01, animate: false, speed: 1 },
    { name: "b", value: 2, min: -5, max: 5, step: 0.01, animate: false, speed: 0.6 },
  ],
  engine,
  views: {
    func: defaultView("func"),
    geom: defaultView("geom"),
    complex: defaultView("complex"),
    vector: defaultView("vector"),
    lin: defaultView("lin"),
    nn: defaultView("nn"),
    surf: defaultView("surf"),
    console: defaultView("console"),
  },
  geo: {
    tool: "select",
    pending: [],
    doc: seedGeo(),
    selected: null,
    showTrace: true,
    showLabels: true,
    snap: false,
    grid: true,
  },
  cplx: {
    mode: "domain",
    f: "(z^3 - 1)/(z^2 + 2i)",
    curve: "circle",
    iterative: true,
    levelStep: 1,
    colormap: "hsv",
    resolution: 2,
  },
  vec: {
    arrows: [
      { id: uid("V"), tail: "[0,0]", vec: "[3,1]", color: paletteAt(2), label: "u" },
      { id: uid("V"), tail: "[0,0]", vec: "[-1,2.5]", color: paletteAt(3), label: "v" },
    ],
    fieldMode: "quiver",
    fx: "y - sin(x)",
    fy: "1 - x/3",
    dfxy: "(x^2 - y)/(x + y + 1)",
    density: 18,
    streamlineCount: 16,
    steps: 600,
    colormap: "parula",
  },
  lin: {
    dim: 2,
    a: ["2", "1", "1", "-1"],
    b: ["3", "1"],
    showGrid: true,
    showCircle: true,
    showEllipse: true,
    showEigen: true,
    showSVD: false,
    showFlow: true,
    t: 1,
    useExp: false,
  },
  nn: {
    dataset: "moons",
    samples: 160,
    hidden: 8,
    depth: 2,
    act: "tanh",
    lr: 0.08,
    momentum: 0.9,
    batch: 16,
    seed: 5,
    running: false,
    showBoundary: true,
    boundaryRes: 96,
    epochs: 0,
    loss: 0,
    acc: 0,
    curve: [],
    model: null,
    data: null,
  },
  surf: {
    layers: [
      {
        id: uid("S"),
        kind: "graph",
        expr: "sin(sqrt(x^2 + y^2)*3)/2",
        style: "surf",
        colormap: "parula",
        res: 90,
        range: [-3, 3, -3, 3],
        visible: true,
        color: paletteAt(0),
        opacity: 1,
        lit: true,
        label: "z = sin(3r)/2",
      },
    ],
    cam: { azim: 0.7, elev: 0.5, dist: 4.2, target: [0, 0, 0] },
    box: [-3, 3, -3, 3, -1.4, 1.4],
    autoBox: true,
    showAxes: true,
    lightAngle: 0.6,
  },
  console: { lines: [], input: "" },
  settings: {
    dark: true,
    showMinorGrid: true,
    piTicksX: false,
    piTicksY: false,
    showCrosshair: true,
    antialias: true,
  },
  revision: 0,

  setMode: (m) => set({ mode: m }),
  setView: (m, v) => set((s) => ({ views: { ...s.views, [m]: v } })),
  patch: (p) => set(p as Partial<GeoLabState>),
  bump: () => set((s) => ({ revision: s.revision + 1 })),
  addLayer: (kind = "cartesian", expr = "") => {
    const id = uid("L");
    const n = get().layers.length;
    set((s) => ({
      layers: [
        ...s.layers,
        {
          id,
          kind,
          expr,
          label: "",
          color: paletteAt(n),
          width: 2.2,
          visible: true,
          dashed: false,
          fill: false,
          samples: 1400,
          level: 0,
        },
      ],
      activeLayer: id,
    }));
    return id;
  },
  updateLayer: (id, patch) =>
    set((s) => ({ layers: s.layers.map((l) => (l.id === id ? { ...l, ...patch } : l)), revision: s.revision + 1 })),
  removeLayer: (id) =>
    set((s) => ({
      layers: s.layers.filter((l) => l.id !== id),
      activeLayer: s.layers.find((l) => l.id !== id)?.id ?? null,
      revision: s.revision + 1,
    })),
  reorder: (id, dir) =>
    set((s) => {
      const i = s.layers.findIndex((l) => l.id === id);
      const j = Math.max(0, Math.min(s.layers.length - 1, i + dir));
      if (i === j) return {};
      const arr = s.layers.slice();
      const [it] = arr.splice(i, 1);
      arr.splice(j, 0, it);
      return { layers: arr, revision: s.revision + 1 };
    }),
  setParam: (name, value) => {
    set((s) => ({ params: s.params.map((p) => (p.name === name ? { ...p, value } : p)), revision: s.revision + 1 }));
    get().engine.setNum(name, value);
  },
  addParam: (p) =>
    set((s) => ({ params: [...s.params.filter((q) => q.name !== p.name), p], revision: s.revision + 1 })),
  removeParam: (name) =>
    set((s) => ({ params: s.params.filter((p) => p.name !== name), revision: s.revision + 1 })),
  toggleAnimate: (name) => {
    set((s) => ({ params: s.params.map((p) => (p.name === name ? { ...p, animate: !p.animate } : p)) }));
  },
  setSettings: (p) => set((s) => ({ settings: { ...s.settings, ...p }, revision: s.revision + 1 })),
  setGeo: (p) => set((s) => ({ geo: { ...s.geo, ...p }, revision: s.revision + 1 })),
  setCplx: (p) => set((s) => ({ cplx: { ...s.cplx, ...p }, revision: s.revision + 1 })),
  setVec: (p) => set((s) => ({ vec: { ...s.vec, ...p }, revision: s.revision + 1 })),
  setSurf: (p) => set((s) => ({ surf: { ...s.surf, ...p }, revision: s.revision + 1 })),
  setLin: (p) => set((s) => ({ lin: { ...s.lin, ...p }, revision: s.revision + 1 })),
  setNn: (p) => set((s) => ({ nn: { ...s.nn, ...p }, revision: s.revision + 1 })),
}));
