/**
 * 示例库：每个模式一组「拿来就能看」的预设，覆盖各类函数图像、复平面、
 * 向量/场、动态几何构造与 MATLAB 风格的 3D 可视化。
 */
import { GeometryDoc } from "./core/geometry.ts";
import type { Engine } from "./core/machine.ts";
import { paletteAt } from "./core/colormap.ts";
import {
  useStore,
  uid,
  type ComplexState,
  type GeoState,
  type Layer,
  type Mode,
  type Param,
  type SurfLayer,
  type SurfState,
  type VectorState,
} from "./state.ts";

export interface Preset {
  key: string;
  title: string;
  mode: Mode;
  /** 一句话说明，显示在下拉项里 */
  hint: string;
  layers?: () => Layer[];
  params?: () => Param[];
  defs?: string[];
  cplx?: Partial<ComplexState>;
  vec?: Partial<VectorState>;
  surf?: Partial<SurfState>;
  /** 几何：给出一个全新的构造文档 */
  geo?: (doc: GeometryDoc, eng: Engine) => void;
  /** [cx, cy, scale] */
  view?: [number, number, number];
}

const L = (p: Partial<Layer> & { expr: string }): Layer => ({
  id: uid("L"),
  kind: "cartesian",
  label: p.expr,
  color: paletteAt(0),
  width: 2.3,
  visible: true,
  dashed: false,
  fill: false,
  samples: 1400,
  ...p,
});

const S = (p: Partial<SurfLayer> & { expr: string }): SurfLayer => ({
  id: uid("S"),
  kind: "graph",
  style: "surf",
  colormap: "parula",
  res: 80,
  range: [-3, 3, -3, 3],
  visible: true,
  color: paletteAt(0),
  opacity: 1,
  lit: true,
  label: p.expr,
  ...p,
});

export const PRESETS: Preset[] = [
  /* ---------------------------------------------------------- 函数 */
  {
    key: "func-basic",
    title: "基本初等函数",
    mode: "func",
    hint: "多项式 / 三角 / 指数 / 对数 / 有理函数同屏",
    view: [0, 0, 70],
    layers: () => [
      L({ expr: "sin(x)", label: "sin x", color: paletteAt(0) }),
      L({ expr: "x^3/8 - x", label: "x³/8 − x", color: paletteAt(1) }),
      L({ expr: "exp(x/3)", label: "e^(x/3)", color: paletteAt(2), dashed: true }),
      L({ expr: "ln(abs(x)+0.05)", label: "ln|x|", color: paletteAt(3), samples: 2600 }),
      L({ expr: "1/(x-1.6)", label: "1/(x−1.6)", color: paletteAt(6), samples: 3000 }),
    ],
  },
  {
    key: "func-damped",
    title: "衰减振荡与包络",
    mode: "func",
    hint: "e^(-ax) 包络 + 参数动画",
    view: [3, 0, 95],
    params: () => [
      { name: "a", value: 0.28, min: 0.02, max: 1.2, step: 0.01, animate: true, speed: 0.5 },
      { name: "w", value: 3, min: 0.5, max: 9, step: 0.05, animate: false, speed: 0.7 },
    ],
    layers: () => [
      L({ expr: "exp(-a*x)*sin(w*x)", label: "e^(−ax)·sin(ωx)", color: paletteAt(0), width: 2.6 }),
      L({ expr: "exp(-a*x)", label: "上包络", color: paletteAt(2), dashed: true, width: 1.5 }),
      L({ expr: "-exp(-a*x)", label: "下包络", color: paletteAt(2), dashed: true, width: 1.5 }),
    ],
  },
  {
    key: "func-polar",
    title: "极坐标曲线",
    mode: "func",
    hint: "玫瑰线 / 阿基米德螺线 / 心形线",
    view: [0, 0, 70],
    layers: () => [
      L({ kind: "polar", expr: "sin(4*theta)", label: "r = sin 4θ", color: paletteAt(4), samples: 2200 }),
      L({ kind: "polar", expr: "0.35*theta", label: "r = 0.35θ", color: paletteAt(1), samples: 2200 }),
      L({ kind: "polar", expr: "1 + cos(theta)", label: "r = 1 + cos θ", color: paletteAt(3), width: 1.8 }),
    ],
  },
  {
    key: "func-param",
    title: "参数方程：摆线与椭圆",
    mode: "func",
    hint: "滚轮上一点的轨迹（摆线）与参数椭圆",
    view: [3.2, 0, 62],
    params: () => [{ name: "k", value: 1, min: 0.2, max: 2.4, step: 0.01, animate: false, speed: 0.5 }],
    layers: () => [
      L({ kind: "param", expr: "k*(t - sin(t))", expr2: "k*(1 - cos(t))", label: "摆线 (k)", color: paletteAt(0), samples: 2000 }),
      L({ kind: "param", expr: "2.4*cos(t)", expr2: "1.3*sin(t)", label: "椭圆", color: paletteAt(3), dashed: true, samples: 900 }),
      L({ kind: "cartesian", expr: "0", label: "地面", color: paletteAt(5), width: 1 }),
    ],
  },
  {
    key: "func-implicit",
    title: "隐函数与不等式",
    mode: "func",
    hint: "x²+y²=1、心形线、|x|+|y|<1 区域",
    view: [0, 0, 105],
    layers: () => [
      L({ kind: "implicit", expr: "x^2 + y^2 - 1", label: "单位圆", color: paletteAt(0) }),
      L({ kind: "implicit", expr: "(x^2+y^2-1)^3 - x^2*y^3", label: "心形线", color: paletteAt(4), samples: 200 }),
      L({ kind: "inequality", expr: "abs(x) + abs(y) - 0.9", rel: "<", label: "|x|+|y|<0.9", color: paletteAt(2), fill: true }),
    ],
  },
  {
    key: "func-calc",
    title: "微积分：导数与积分",
    mode: "func",
    hint: "f、f' 与 ∫f 面积着色（上限可用参数 b 动画）",
    view: [1.6, 0.6, 78],
    params: () => [{ name: "b", value: 3.2, min: -4, max: 6.2, step: 0.02, animate: true, speed: 0.35 }],
    defs: ["f(x) = sin(x)*exp(-x/8) + 0.6"],
    layers: () => [
      L({ expr: "f(x)", label: "f(x)", color: paletteAt(0), width: 2.6 }),
      L({ kind: "derivative", expr: "f(x)", label: "f '(x)", color: paletteAt(1), dashed: true }),
      L({ kind: "integral", expr: "f(x)", expr2: "b", label: "∫₀ᵇ f", color: paletteAt(4), fill: true }),
    ],
  },
  {
    key: "func-seq",
    title: "数列与离散采样",
    mode: "func",
    hint: "收敛数列点列与取整阶梯",
    view: [3, 1.4, 78],
    layers: () => [
      L({ kind: "sequence", expr: "1 + 1/n", label: "1 + 1/n", color: paletteAt(0) }),
      L({ kind: "sequence", expr: "sin(n/2)*n/6", label: "sin(n/2)·n/6", color: paletteAt(2) }),
      L({ kind: "cartesian", expr: "floor(x)", label: "⌊x⌋", color: paletteAt(3), dashed: true }),
    ],
  },

  /* ---------------------------------------------------------- 复平面 */
  {
    key: "cpx-domain",
    title: "共形着色：有理函数",
    mode: "complex",
    hint: "Needham 相位着色，零点/极点一目了然",
    view: [0, 0, 110],
    cplx: { mode: "domain", f: "(z^3 - 1)/(z^2 + 2i)", colormap: "hsv", levelStep: 1, resolution: 2 },
  },
  {
    key: "cpx-map",
    title: "保角映射：Joukowsky",
    mode: "complex",
    hint: "w = z + 1/z 把圆栅格变成机翼型",
    view: [0, 0, 92],
    cplx: { mode: "map", f: "z + 1/z", curve: "grid", colormap: "hsv", resolution: 2 },
  },
  {
    key: "cpx-newton",
    title: "Newton 分形：z³ − 1",
    mode: "complex",
    hint: "三个根的吸引盆与分形边界",
    view: [0, 0, 150],
    cplx: { mode: "newton", f: "z^3 - 1", iterative: true, resolution: 2 },
  },
  {
    key: "cpx-log",
    title: "多值函数：log 与 sqrt",
    mode: "complex",
    hint: "分支切割处的颜色跳变",
    view: [0, 0, 120],
    cplx: { mode: "log", f: "sqrt(z)*log(z)", colormap: "hsv", levelStep: 0.7, resolution: 2 },
  },

  /* ---------------------------------------------------------- 向量与场 */
  {
    key: "vec-parallelogram",
    title: "向量的平行四边形法则",
    mode: "vector",
    hint: "u、v 与 u+v、u−v",
    view: [0.6, 0.8, 90],
    vec: {
      fieldMode: "quiver",
      arrows: [
        { id: uid("V"), tail: "[0,0]", vec: "[3,1]", color: paletteAt(2), label: "u" },
        { id: uid("V"), tail: "[0,0]", vec: "[-1,2.5]", color: paletteAt(3), label: "v" },
      ],
      fx: "0",
      fy: "0",
    },
  },
  {
    key: "vec-quiver",
    title: "向量场（quiver）",
    mode: "vector",
    hint: "F = (y − sin x, 1 − x/3)，按模长着色",
    view: [0, 0, 78],
    vec: { fieldMode: "quiver", fx: "y - sin(x)", fy: "1 - x/3", density: 22, colormap: "parula" },
  },
  {
    key: "vec-slope",
    title: "斜率场与解曲线",
    mode: "vector",
    hint: "y' = (x² − y)/(x + y + 1)",
    view: [0, 0, 78],
    vec: { fieldMode: "slope", dfxy: "(x^2 - y)/(x + y + 1)", density: 26 },
  },
  {
    key: "vec-stream",
    title: "流线图（streamslice）",
    mode: "vector",
    hint: "绕涡流的流线",
    view: [0, 0, 82],
    vec: {
      fieldMode: "stream",
      fx: "-y + x*(1 - x^2 - y^2)/3",
      fy: "x + y*(1 - x^2 - y^2)/3",
      density: 20,
      streamlineCount: 26,
      steps: 900,
    },
  },
  {
    key: "vec-phase",
    title: "相图：单摆",
    mode: "vector",
    hint: "自动求奇点并按本征值分类",
    view: [0, 0, 70],
    vec: { fieldMode: "phase", fx: "y", fy: "-sin(x) - 0.25*y", density: 18, streamlineCount: 22 },
  },

  /* ---------------------------------------------------------- 曲面 */
  {
    key: "surf-graph",
    title: "曲面 surf：波纹",
    mode: "surf",
    hint: "z = sin(3r)/2，MATLAB surf 观感",
    surf: {
      layers: [S({ expr: "sin(sqrt(x^2 + y^2)*3)/2", label: "z = sin(3r)/2", res: 110, colormap: "parula" })],
      cam: { azim: 0.7, elev: 0.45, dist: 8.5, target: [0, 0, 0] },
      autoBox: true,
    },
  },
  {
    key: "surf-mesh",
    title: "网格 mesh：马鞍面",
    mode: "surf",
    hint: "z = x²/2 − y²/2（mesh 参数线）",
    surf: {
      layers: [
        S({
          expr: "x^2/2 - y^2/2",
          label: "z = x²/2 − y²/2",
          style: "mesh",
          res: 46,
          colormap: "jet",
          color: "#5eead4",
        }),
      ],
      cam: { azim: 0.9, elev: 0.5, dist: 9, target: [0, 0, 0] },
      autoBox: true,
    },
  },
  {
    key: "surf-torus",
    title: "参数曲面：环面",
    mode: "surf",
    hint: "x = (R + r cos v) cos u 等",
    surf: {
      layers: [
        S({
          kind: "param",
          expr: "(2 + 0.8*cos(v))*cos(u)",
          expr2: "(2 + 0.8*cos(v))*sin(u)",
          expr3: "0.8*sin(v)",
          label: "环面 R=2, r=0.8",
          res: 90,
          range: [0, 6.2831853, 0, 6.2831853],
          colormap: "hsv",
        }),
      ],
      cam: { azim: 0.7, elev: 0.42, dist: 8, target: [0, 0, 0] },
      autoBox: true,
    },
  },
  {
    key: "surf-implicit",
    title: "隐式曲面：球与陀螺面",
    mode: "surf",
    hint: "isosurface：球面 + (x²+y²+z²)²=3(x²+y²)",
    surf: {
      layers: [
        S({ kind: "implicit", expr: "x^2 + y^2 + z^2 - 1", label: "球面", res: 60, colormap: "cool", opacity: 0.85 }),
        S({
          kind: "implicit",
          expr: "(x^2+y^2+z^2)^2 - 3*(x^2 + y^2)",
          label: "陀螺面",
          res: 48,
          style: "wire",
          color: "#f0abfc",
        }),
      ],
      box: [-1.9, 1.9, -1.9, 1.9, -1.5, 1.5],
      autoBox: false,
      cam: { azim: 0.8, elev: 0.4, dist: 8.5, target: [0, 0, 0] },
    },
  },
  {
    key: "surf-revolve",
    title: "旋转体：y = 1/x 绕 x 轴",
    mode: "surf",
    hint: "壳层法立体（surface of revolution）",
    surf: {
      layers: [
        S({
          kind: "revolve",
          expr: "0.5 + 1.2/x",
          expr2: "x",
          label: "y = 0.5 + 1.2/x 绕 x 轴",
          res: 60,
          range: [0.6, 4, 0, 90],
          colormap: "turbo",
        }),
      ],
      cam: { azim: 0.5, elev: 0.3, dist: 10, target: [0, 0, 0] },
      autoBox: true,
    },
  },
  {
    key: "surf-contour",
    title: "等高线 contour3 / surfc",
    mode: "surf",
    hint: "双峰曲面 + 底面等高投影",
    surf: {
      layers: [
        S({
          expr: "3*exp(-x^2-y^2) - 2*exp(-((x-1.4)^2+(y+1)^2))",
          label: "双峰",
          style: "contour",
          res: 90,
          colormap: "viridis",
        }),
        S({
          expr: "sin(x)*cos(y)",
          label: "surfc",
          style: "surfc",
          res: 60,
          colormap: "gray",
          opacity: 0.55,
          range: [-3.2, 3.2, -3.2, 3.2],
        }),
      ],
      cam: { azim: 0.7, elev: 0.6, dist: 9.5, target: [0, 0, 0] },
      autoBox: true,
    },
  },
  {
    key: "surf-curve",
    title: "空间曲线：螺旋线与扭结",
    mode: "surf",
    hint: "r(t) = (cos t, sin t, t/3) 与环面扭结",
    surf: {
      layers: [
        S({
          kind: "spacecurve",
          expr: "cos(3*t)*(1 + 0.35*cos(7*t))",
          expr2: "sin(3*t)*(1 + 0.35*cos(7*t))",
          expr3: "0.5*sin(7*t)",
          label: "扭结",
          res: 300,
          range: [0, 6.2831853, 0, 0],
          colormap: "hsv",
          color: "#f472b6",
        }),
        S({
          kind: "spacecurve",
          expr: "cos(t)",
          expr2: "sin(t)",
          expr3: "t/3",
          label: "螺旋线",
          res: 200,
          range: [0, 18.85, 0, 0],
          color: "#22d3ee",
        }),
      ],
      cam: { azim: 0.7, elev: 0.35, dist: 9, target: [0, 0, 0] },
      autoBox: true,
    },
  },

  /* ---------------------------------------------------------- 几何 */
  {
    key: "geo-triangle",
    title: "三角形四心与九点圆",
    mode: "geom",
    hint: "重心/垂心/外心共线（欧拉线），拖动顶点看九点圆",
    view: [0.4, 1.1, 62],
    geo: (doc) => {
      const A = doc.addPoint("A", -2.4, 0.2);
      const B = doc.addPoint("B", 3.1, 0);
      const C = doc.addPoint("C", 0.6, 3.4);
      const ab = doc.addSegment(A, B);
      const bc = doc.addSegment(B, C);
      const ca = doc.addSegment(C, A);
      const M = doc.addMidpoint("M", ab);
      const N = doc.addMidpoint("N", bc);
      const L3 = doc.addMidpoint("L", ca);
      const centroid = doc.addIntersection("G", doc.addSegment(C, M), doc.addSegment(A, N));
      doc.addSegment(B, L3, "中线 b");
      const ortho = doc.addIntersection("H", doc.addPerpendicular(C, ab, "高 h_C"), doc.addPerpendicular(A, bc, "高 h_A"));
      const circum = doc.addIntersection(
        "O",
        doc.addPerpendicular(M, ab, "中垂线 c₁"),
        doc.addPerpendicular(N, bc, "中垂线 c₂"),
      );
      if (centroid && circum) doc.addLine(centroid, circum, "欧拉线");
      if (circum) doc.addCircle(circum, A, "外接圆");
      if (ortho && circum) {
        /* 九点圆：外心与垂心中点为圆心，半径为外接圆一半（过三边中点） */
        const n9 = doc.addMidpoint("N₉", doc.addSegment(ortho, circum, "H–O"));
        doc.addCircle(n9, M, "九点圆");
      }
      doc.addMeasure("dist", [A, B], [-2.9, -0.8], "|AB|");
      doc.addMeasure("angle", [A, B, C], [3.4, 3.7], "∠ABC");
    },
  },
  {
    key: "geo-locus",
    title: "轨迹：中点扫出的圆",
    mode: "geom",
    hint: "动点在圆上，追踪它与定点连线的中点",
    view: [0, 0.8, 74],
    geo: (doc) => {
      const O = doc.addPoint("O", 0, 0);
      const R = doc.addPoint("R", 3, 0);
      const circ = doc.addCircle(O, R);
      const D = doc.addPointOn("D", circ, 0.12);
      const C = doc.addPoint("C", -2.4, 2.2);
      const seg = doc.addSegment(D, C);
      const N = doc.addMidpoint("N", seg);
      doc.addLocus("轨迹 N", D, N);
      doc.addSegment(O, D, "半径");
      doc.addMeasure("length", [seg], [1.4, -2.4], "|DC|");
    },
  },
  {
    key: "geo-conic",
    title: "圆锥曲线：焦点定义",
    mode: "geom",
    hint: "椭圆上点到两焦点距离之和为常数",
    view: [0, 0, 70],
    geo: (doc) => {
      const f1 = doc.addPoint("F₁", -2.2, 0);
      const f2 = doc.addPoint("F₂", 2.2, 0);
      const P = doc.addPoint("P", 0, 2.4);
      const e = doc.addEllipseByFoci("E", f1, f2, P);
      const Q = doc.addPointOn("Q", e, 0.18);
      const r1 = doc.addSegment(f1, Q, "r₁");
      doc.addSegment(f2, Q, "r₂");
      doc.addMeasure("length", [r1], [-3.3, 2.6], "r₁");
      doc.addMeasure("dist", [f1, f2], [-0.6, -3.1], "|F₁F₂|");
    },
  },
  {
    key: "geo-rotate",
    title: "变换：旋转与反射对称",
    mode: "geom",
    hint: "正六边形绕中心逐次旋转，再关于轴对称复制",
    view: [0, 0, 66],
    geo: (doc) => {
      const O = doc.addPoint("O", 0, 0);
      const A = doc.addPoint("A", 2.6, 0);
      const pts = [A];
      let prev = A;
      for (let i = 1; i < 6; i++) {
        prev = doc.addRotate(`P${i}`, prev, O, (Math.PI * 2) / 6);
        pts.push(prev);
      }
      doc.addPolygon(pts, "正六边形");
      const axis = doc.addLine(A, O, "对称轴");
      for (let i = 0; i < 3; i++) doc.addReflect(`S${i}`, pts[i], axis);
      doc.addCircle(O, A, "外接圆");
      doc.addMeasure("angle", [O, A, pts[1]], [2.9, 1.1], "中心角");
    },
  },
  {
    key: "geo-driven",
    title: "表达式驱动的点（几何画板式）",
    mode: "geom",
    hint: "点 P 由 (t, sin t) 生成，选中后可用参数动画扫过曲线",
    view: [0, 0, 52],
    geo: (doc, eng) => {
      const O = doc.addPoint("O", -6.4, 0);
      const X = doc.addPoint("X", 6.4, 0);
      const axis = doc.addSegment(O, X, "x 轴");
      doc.addCurveExpr("y = 2 sin x", "t", "2*sin(t)", eng, -6.4, 6.4, 360);
      /* 驱动点的参数 p∈[0,1] 即表达式的 t，这里做一次区间映射 */
      const P = doc.addPointExpr("P", "-6.4 + 12.8*t", "2*sin(-6.4 + 12.8*t)", eng);
      doc.setParam(P, 0.42);
      const drop = doc.addPerpendicular(P, axis, "投影线");
      const F = doc.addIntersection("F", axis, drop);
      if (F) {
        doc.addSegment(P, F, "y(P)");
        doc.addMeasure("dist", [P, F], [-7.6, 2.4], "y(P)");
      }
      doc.addMeasure("dist", [O, P], [-7.6, -2.4], "|OP|");
    },
  },
];

/** 应用预设：把示例写回 store，并按需重建几何文档 */
export function applyPreset(key: string): void {
  const p = PRESETS.find((q) => q.key === key);
  if (!p) return;
  const st = useStore.getState();
  const { engine } = st;
  if (p.defs)
    for (const d of p.defs) {
      try {
        engine.define(d);
      } catch {
        /* 单个定义失败不影响其余部分 */
      }
    }
  const layers = p.layers?.();
  const params = p.params?.();
  if (params) for (const q of params) engine.setNum(q.name, q.value);

  let geo: GeoState | undefined;
  if (p.geo) {
    const doc = new GeometryDoc();
    doc.engine = engine;
    p.geo(doc, engine);
    doc.recompute();
    geo = { ...st.geo, doc, pending: [], selected: null };
  }

  st.patch({
    mode: p.mode,
    layers: layers ?? st.layers,
    activeLayer: layers?.[0]?.id ?? st.activeLayer,
    params: params ?? st.params,
    cplx: p.cplx ? { ...st.cplx, ...p.cplx } : st.cplx,
    vec: p.vec ? { ...st.vec, ...p.vec } : st.vec,
    surf: p.surf ? { ...st.surf, ...p.surf } : st.surf,
    geo: geo ?? st.geo,
    revision: st.revision + 1,
  });

  if (p.view) {
    const v = st.views[p.mode];
    st.setView(p.mode, v.with({ cx: p.view[0], cy: p.view[1], scale: p.view[2] }));
  }
}
