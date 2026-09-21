/**
 * 动态几何内核（几何画板式）
 *
 * 数据结构：对象表 + 创建序拓扑序。每个对象携带一个"构造配方"（Ctor），
 * 依赖只指向更早创建的对象，因此 recompute 按创建顺序单遍扫描即可完成整张图。
 * 渲染、测量、轨迹全部从 recompute 之后的几何位置派生，拖动时只需重跑一遍。
 *
 * 退化策略：任一步得到非有限值时保留上一次结果（NaN 不上屏），绝不在
 * recompute 中抛错，保证共线、重合点、零半径等极端拖动下交互不断。
 */

import { Engine, compileReal } from "./machine.ts";
import { parseExpr } from "./parser.ts";

type P = [number, number];

export type GeoKind =
  | "point"
  | "segment"
  | "line"
  | "ray"
  | "vector"
  | "circle"
  | "arc"
  | "ellipse"
  | "polygon"
  | "locus"
  | "angle"
  | "text";

/** 构造配方：可序列化、可重放 */
export type Ctor =
  | { t: "free"; x: number; y: number }
  | { t: "on"; host: string; p: number }
  | { t: "expr"; x: string; y: string; v: string; p: number }
  | { t: "exprCurve"; x: string; y: string; t0: number; t1: number; n: number }
  | { t: "mid"; seg: string }
  | { t: "int"; a: string; b: string; br: number }
  | { t: "seg" | "line" | "ray" | "vector"; a: string; b: string }
  | { t: "poly"; pts: string[] }
  | { t: "circ"; c: string; p: string }
  | { t: "arc"; c: string; s: string; e: string }
  | { t: "ell"; c: P; a: number; b: number; rot: number; f1?: string; f2?: string; p?: string }
  | { t: "perp"; p: string; l: string }
  | { t: "par"; p: string; l: string }
  | { t: "bis"; a: string; b: string; c: string }
  | { t: "rot"; s: string; c: P | string; ang: number; ox: number; oy: number }
  | { t: "refl"; s: string; l: string }
  | { t: "dil"; s: string; c: P | string; k: number }
  | { t: "trans"; s: string; v: P; a: string; b: string }
  | { t: "locus"; d: string; target: string }
  | { t: "angle"; a: string; b: string; c: string }
  | { t: "measure"; what: MeasureWhat; ids: string[]; ax: number; ay: number }
  | { t: "text"; x: number; y: number; s: string };

export type MeasureWhat = "area" | "length" | "dist" | "perimeter" | "angle" | "slope" | "radius";

export interface Geo {
  id: string;
  kind: GeoKind;
  label: string;
  /** 点的位置；非点对象为标签锚点 */
  x: number;
  y: number;
  /** 长半轴 / 半径 / 测量值 */
  a: number;
  /** 短半轴 / 起始角 */
  b: number;
  rot: number;
  center?: P;
  deps?: string[];
  frozen?: boolean;
  visible?: boolean;
  host?: string;
  param?: number;
  branch?: number;
  ctor: Ctor;
  /** 以下为 recompute 派生字段，不参与序列化 */
  p1?: P;
  p2?: P;
  dir?: P;
  pts?: P[];
  text?: string;
}

function fin(v: number): boolean {
  return Number.isFinite(v);
}

function ok(p: P | null | undefined): p is P {
  return !!p && fin(p[0]) && fin(p[1]);
}

function sub(a: P, b: P): P {
  return [a[0] - b[0], a[1] - b[1]];
}

function add(a: P, b: P): P {
  return [a[0] + b[0], a[1] + b[1]];
}

function mul(a: P, k: number): P {
  return [a[0] * k, a[1] * k];
}

function len(a: P): number {
  return Math.hypot(a[0], a[1]);
}

function norm(a: P): P | null {
  const m = len(a);
  return m > 1e-12 ? [a[0] / m, a[1] / m] : null;
}

function perpOf(a: P): P {
  return [-a[1], a[0]];
}

function rot(a: P, ang: number): P {
  const c = Math.cos(ang),
    s = Math.sin(ang);
  return [a[0] * c - a[1] * s, a[0] * s + a[1] * c];
}

function dist(a: P, b: P): number {
  return Math.hypot(a[0] - b[0], a[1] - b[1]);
}

/** 点到线段的距离（用于 hitTest） */
function segDist(p: P, a: P, b: P): number {
  const dx = b[0] - a[0],
    dy = b[1] - a[1];
  const dd = dx * dx + dy * dy;
  if (dd <= 1e-30) return dist(p, a);
  let t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / dd;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return dist(p, [a[0] + t * dx, a[1] + t * dy]);
}

/** 到折线的最小距离 */
function polylineDist(p: P, pts: P[], closed: boolean): number {
  let best = Infinity;
  for (let i = 0; i + 1 < pts.length; i++) best = Math.min(best, segDist(p, pts[i], pts[i + 1]));
  if (closed && pts.length > 2) best = Math.min(best, segDist(p, pts[pts.length - 1], pts[0]));
  return best;
}

const TWO_PI = Math.PI * 2;

/** 角度归一到 [0, 2π) */
function wrap(ang: number): number {
  const a = ang % TWO_PI;
  return a < 0 ? a + TWO_PI : a;
}

function fracIn(a: number, start: number, sweep: number): number {
  return sweep === 0 ? 0 : (((a - start) % TWO_PI) + TWO_PI) % TWO_PI / Math.abs(sweep);
}

export class GeometryDoc {
  private map = new Map<string, Geo>();
  private order: string[] = [];
  private n = 0;
  private rev = 0;
  private undoStack: string[] = [];
  private redoStack: string[] = [];
  private gesture = false;
  /** 轨迹采样中标记，防止嵌套轨迹递归重入 */
  private sampling = false;
  private traces = new Map<string, { rev: number; pts: P[] }>();
  /** 表达式驱动对象需要的求值环境 */
  engine: Engine | null = null;
  private compiled = new Map<string, { x: (s: Float64Array) => number; y: (s: Float64Array) => number }>();

  /* ----------------------------------------------------------- 基础访问 */

  ids(): string[] {
    return this.order.slice();
  }

  get(id: string): Geo | null {
    return this.map.get(id) ?? null;
  }

  all(): Geo[] {
    return this.order.map((i) => this.map.get(i)!).filter((g) => !!g);
  }

  /** 构造实际依赖（表达式与测量值也依赖其取值对象） */
  private depsOf(g: Geo): string[] {
    const d: string[] = [];
    const c = g.ctor;
    switch (c.t) {
      case "seg":
      case "line":
      case "ray":
      case "vector":
        d.push(c.a, c.b);
        break;
      case "poly":
        d.push(...c.pts);
        break;
      case "on":
        d.push(c.host);
        break;
      case "mid":
        d.push(c.seg);
        break;
      case "int":
        d.push(c.a, c.b);
        break;
      case "circ":
        d.push(c.c, c.p);
        break;
      case "arc":
        d.push(c.c, c.s, c.e);
        break;
      case "ell":
        if (c.f1 && c.f2 && c.p) d.push(c.f1, c.f2, c.p);
        break;
      case "perp":
      case "par":
        d.push(c.p, c.l);
        break;
      case "bis":
        d.push(c.a, c.b, c.c);
        break;
      case "rot":
      case "dil":
        d.push(c.s);
        if (typeof c.c === "string") d.push(c.c);
        break;
      case "refl":
        d.push(c.s, c.l);
        break;
      case "trans":
        d.push(c.s, c.a, c.b);
        break;
      case "locus":
        d.push(c.d, c.target);
        break;
      case "angle":
      case "measure":
        if (c.t === "angle") d.push(c.a, c.b, c.c);
        else d.push(...c.ids);
        break;
      default:
        break;
    }
    return d.filter((q) => this.map.has(q));
  }

  /** 环检测由创建顺序保证：依赖总是指向更早的对象 */

  /* ------------------------------------------------------------ 构造操作 */

  private mk(kind: GeoKind, label: string, ctor: Ctor, x = 0, y = 0): Geo {
    const id = `${kind[0]}${++this.n}`;
    const g: Geo = {
      id,
      kind,
      label: label || autoLabel(kind, this.n),
      x,
      y,
      a: NaN,
      b: NaN,
      rot: 0,
      ctor,
      visible: true,
    };
    this.map.set(id, g);
    this.order.push(id);
    return g;
  }

  addPoint(label: string, x: number, y: number): string {
    this.commit();
    const g = this.mk("point", label, { t: "free", x, y }, x, y);
    this.mark();
    return g.id;
  }

  addSegment(a: string, b: string, label = ""): string {
    this.commit();
    const g = this.mk("segment", label, { t: "seg", a, b });
    this.mark();
    return g.id;
  }

  addLine(a: string, b: string, label = ""): string {
    this.commit();
    const g = this.mk("line", label, { t: "line", a, b });
    this.mark();
    return g.id;
  }

  addRay(a: string, b: string, label = ""): string {
    this.commit();
    const g = this.mk("ray", label, { t: "ray", a, b });
    this.mark();
    return g.id;
  }

  addVector(label: string, tail: string, head: string): string {
    this.commit();
    const g = this.mk("vector", label, { t: "vector", a: tail, b: head });
    this.mark();
    return g.id;
  }

  addPolygon(pts: string[], label = ""): string {
    this.commit();
    const g = this.mk("polygon", label, { t: "poly", pts });
    this.mark();
    return g.id;
  }

  addCircle(center: string, pointOn: string, label = ""): string {
    this.commit();
    const g = this.mk("circle", label, { t: "circ", c: center, p: pointOn });
    this.mark();
    return g.id;
  }

  addArc(label: string, center: string, start: string, end: string): string {
    this.commit();
    const g = this.mk("arc", label, { t: "arc", c: center, s: start, e: end });
    this.mark();
    return g.id;
  }

  /** 椭圆：中心、长半轴、短半轴、长轴倾角 */
  addEllipse(label: string, center: P, a: number, b: number, rotAng = 0): string {
    this.commit();
    const g = this.mk("ellipse", label, { t: "ell", c: center, a, b, rot: rotAng });
    g.center = [center[0], center[1]];
    g.a = a;
    g.b = b;
    g.rot = rotAng;
    g.x = center[0];
    g.y = center[1];
    this.mark();
    return g.id;
  }

  /** 椭圆定义：到两焦点距离之和 = 过 p 点的 2a（几何画板式圆锥曲线） */
  addEllipseByFoci(label: string, f1: string, f2: string, p: string): string {
    this.commit();
    const g = this.mk("ellipse", label, { t: "ell", c: [0, 0], a: 1, b: 1, rot: 0, f1, f2, p });
    this.mark();
    return g.id;
  }

  addMidpoint(label: string, segment: string): string {
    this.commit();
    const g = this.mk("point", label, { t: "mid", seg: segment });
    this.mark();
    return g.id;
  }

  /** 过 point 作 segment/line 的垂线（返回 line 的 id） */
  addPerpendicular(point: string, segmentOrLine: string, label = ""): string {
    this.commit();
    const g = this.mk("line", label, { t: "perp", p: point, l: segmentOrLine });
    this.mark();
    return g.id;
  }

  addParallel(point: string, line: string, label = ""): string {
    this.commit();
    const g = this.mk("line", label, { t: "par", p: point, l: line });
    this.mark();
    return g.id;
  }

  /** ∠abc 在顶点 b 处的角平分线 */
  addBisector(a: string, b: string, c: string, label = ""): string {
    this.commit();
    const g = this.mk("line", label, { t: "bis", a, b, c });
    this.mark();
    return g.id;
  }

  /** 约束在曲线上的点，param∈[0,1] */
  addPointOn(label: string, host: string, param: number): string {
    this.commit();
    const g = this.mk("point", label, { t: "on", host, p: norm01(param) });
    g.host = host;
    g.param = norm01(param);
    this.mark();
    return g.id;
  }

  addIntersection(label: string, a: string, b: string, branch = 0): string | null {
    this.commit();
    const g = this.mk("point", label, { t: "int", a, b, br: branch });
    g.branch = branch;
    this.recompute();
    const cands = this.intCandidates(g);
    if (!cands.length) {
      this.map.delete(g.id);
      this.order.pop();
      this.mark();
      return null;
    }
    this.mark();
    return g.id;
  }

  /** 绕 center 旋转 pointId；center 可为点 id、坐标或纯数字（数字视作 [n,n]） */
  addRotate(label: string, pointId: string, center: P | string | number, angle: number): string {
    this.commit();
    const c = typeof center === "number" ? ([center, center] as P) : center;
    const g = this.mk("point", label, { t: "rot", s: pointId, c, ang: angle, ox: 0, oy: 0 });
    this.mark();
    return g.id;
  }

  addReflect(label: string, pointId: string, lineId: string): string {
    this.commit();
    const g = this.mk("point", label, { t: "refl", s: pointId, l: lineId });
    this.mark();
    return g.id;
  }

  addDilate(label: string, pointId: string, center: P | string | number, ratio: number): string {
    this.commit();
    const c = typeof center === "number" ? ([center, center] as P) : center;
    const g = this.mk("point", label, { t: "dil", s: pointId, c, k: ratio });
    this.mark();
    return g.id;
  }

  /** 按向量平移；向量取自 a→b 两点 */
  addTranslate(label: string, pointId: string, from: string, to: string): string {
    this.commit();
    const g = this.mk("point", label, { t: "trans", s: pointId, v: [0, 0], a: from, b: to });
    this.mark();
    return g.id;
  }

  addTransform(
    kind: "translate" | "rotate" | "reflect" | "dilate",
    pointIds: string[],
    params: {
      vector?: P;
      center?: P | string;
      angle?: number;
      ratio?: number;
      line?: string;
      labels?: string[];
    } = {},
  ): string[] {
    this.commit();
    const out: string[] = [];
    pointIds.forEach((pid, i) => {
      const src = this.map.get(pid);
      const label = params.labels?.[i] ?? (src ? `${src.label}′` : "");
      if (kind === "rotate") {
        const c = params.center ?? ([0, 0] as P);
        out.push(this.addRotate(label, pid, c, params.angle ?? 0));
      } else if (kind === "reflect") {
        out.push(this.addReflect(label, pid, params.line ?? ""));
      } else if (kind === "dilate") {
        out.push(this.addDilate(label, pid, params.center ?? ([0, 0] as P), params.ratio ?? 1));
      } else {
        const v = params.vector ?? [0, 0];
        const g = this.mk("point", label, { t: "trans", s: pid, v, a: pid, b: pid });
        (g.ctor as { t: "trans"; v: P }).v = [v[0], v[1]];
        out.push(g.id);
      }
    });
    this.mark();
    return out;
  }

  /** 轨迹：driver 沿其宿主曲线走完一周时 target 描出的曲线 */
  addLocus(label: string, driverId: string, targetId: string): string {
    this.commit();
    const g = this.mk("locus", label, { t: "locus", d: driverId, target: targetId });
    this.mark();
    return g.id;
  }

  /** 角度标记（三点，顶点在中间） */
  addAngle(a: string, b: string, c: string, label = ""): string {
    this.commit();
    const g = this.mk("angle", label, { t: "angle", a, b, c });
    this.mark();
    return g.id;
  }

  /** 动态测量文本：随拖动刷新数值 */
  addMeasure(what: MeasureWhat, ids: string[], at?: P, label = ""): string {
    this.commit();
    const a = at ?? [0, 0];
    const g = this.mk("text", label, { t: "measure", what, ids, ax: a[0], ay: a[1] });
    g.x = a[0];
    g.y = a[1];
    this.mark();
    return g.id;
  }

  addText(x: number, y: number, s: string, label = ""): string {
    this.commit();
    const g = this.mk("text", label, { t: "text", x, y, s });
    g.x = x;
    g.y = y;
    this.mark();
    return g.id;
  }

  /** 表达式驱动的点：x=f(t), y=g(t)，t 为参数（可用 setParam 动画） */
  addPointExpr(label: string, xExpr: string, yExpr: string, engine?: Engine, varName = "t"): string {
    if (engine) this.engine = engine;
    this.commit();
    const g = this.mk("point", label, { t: "expr", x: xExpr, y: yExpr, v: varName, p: 0 });
    g.param = 0;
    this.mark();
    return g.id;
  }

  /** 表达式驱动的参数曲线（t 从 t0 到 t1 采样 n 点） */
  addCurveExpr(label: string, xExpr: string, yExpr: string, engine: Engine, t0 = 0, t1 = 1, n = 400): string {
    this.engine = engine;
    this.commit();
    const g = this.mk("locus", label, { t: "exprCurve", x: xExpr, y: yExpr, t0, t1, n: Math.max(2, n) });
    this.mark();
    return g.id;
  }

  remove(id: string): void {
    const g = this.map.get(id);
    if (!g) return;
    this.commit();
    const doomed = new Set<string>([id]);
    // 级联：删除所有依赖链经过它的对象
    for (;;) {
      let grew = false;
      for (const q of this.order) {
        if (doomed.has(q)) continue;
        const item = this.map.get(q)!;
        if (this.depsOf(item).some((d) => doomed.has(d))) {
          doomed.add(q);
          grew = true;
        }
      }
      if (!grew) break;
    }
    this.order = this.order.filter((q) => !doomed.has(q));
    for (const q of doomed) {
      this.map.delete(q);
      this.compiled.delete(q);
    }
    this.mark();
  }

  clear(): void {
    this.commit();
    this.map.clear();
    this.order = [];
    this.compiled.clear();
    this.mark();
  }

  setVisible(id: string, v: boolean): void {
    const g = this.map.get(id);
    if (g) g.visible = v;
  }

  setLabel(id: string, label: string): void {
    const g = this.map.get(id);
    if (g) g.label = label;
    this.mark();
  }

  /* ------------------------------------------------------------ 参数与拖动 */

  move(id: string, x: number, y: number): void {
    const g = this.map.get(id);
    if (!g || g.kind !== "point") return;
    if (!this.gesture) this.commit();
    const c = g.ctor;
    if (c.t === "free") {
      c.x = x;
      c.y = y;
      g.x = x;
      g.y = y;
    } else if (c.t === "on") {
      const t = this.paramAt(c.host, [x, y]);
      if (t !== null) c.p = t;
    } else if (c.t === "expr") {
      const t = this.exprParam(g, [x, y]);
      if (t !== null) c.p = t;
    } else {
      return; // 派生点不可直接拖动
    }
    this.recompute();
    this.mark();
  }

  /** 平移一个自由点（拖动增量） */
  nudge(id: string, dx: number, dy: number): void {
    const g = this.map.get(id);
    if (!g) return;
    this.move(id, g.x + dx, g.y + dy);
  }

  setParam(id: string, t: number): void {
    const g = this.map.get(id);
    if (!g) return;
    const c = g.ctor;
    if (c.t !== "on" && c.t !== "expr") return;
    if (!this.gesture) this.commit();
    c.p = norm01(t);
    this.recompute();
    this.mark();
  }

  getParam(id: string): number {
    const g = this.map.get(id);
    const c = g?.ctor;
    return c && (c.t === "on" || c.t === "expr") ? c.p : 0;
  }

  /** 可拖动的对象：自由点、曲线上的点、参数点 */
  dragTarget(id: string): boolean {
    const g = this.map.get(id);
    if (!g) return false;
    return g.kind === "point" && ["free", "on", "expr"].includes(g.ctor.t);
  }

  /* ------------------------------------------------------------ 重算 */

  recompute(): void {
    for (const id of this.order) {
      const g = this.map.get(id)!;
      const save = { x: g.x, y: g.y };
      try {
        this.build(g);
      } catch {
        /* 保持上一次结果，拖动不断线 */
      }
      if (!fin(g.x) || !fin(g.y)) {
        g.x = save.x;
        g.y = save.y;
      }
    }
  }

  private pt(id: string): P | null {
    const g = this.map.get(id);
    if (!g) return null;
    return ok([g.x, g.y]) ? [g.x, g.y] : null;
  }

  /** 线状对象的锚点与方向（单位向量） */
  private lineOf(id: string): { o: P; d: P } | null {
    const g = this.map.get(id);
    if (!g || !g.dir || !g.p1) return null;
    return ok(g.p1) && ok(g.dir) ? { o: g.p1, d: g.dir } : null;
  }

  private circleOf(id: string): { c: P; r: number; ref: number } | null {
    const g = this.map.get(id);
    if (!g) return null;
    if (g.kind === "circle" && g.center && ok(g.center) && fin(g.a)) {
      const on = this.pt((g.ctor as { t: "circ"; p: string }).p);
      return { c: g.center, r: g.a, ref: on ? Math.atan2(on[1] - g.center[1], on[0] - g.center[0]) : 0 };
    }
    if (g.kind === "arc" && g.center && ok(g.center) && fin(g.a)) {
      return { c: g.center, r: g.a, ref: g.b ?? 0 };
    }
    return null;
  }

  private build(g: Geo): void {
    const c = g.ctor;
    switch (c.t) {
      case "free": {
        g.x = c.x;
        g.y = c.y;
        break;
      }
      case "text": {
        g.x = c.x;
        g.y = c.y;
        g.text = c.s;
        break;
      }
      case "on": {
        const p = this.pointAtT(c.host, norm01(c.p));
        if (p) {
          g.x = p[0];
          g.y = p[1];
        }
        g.param = norm01(c.p);
        break;
      }
      case "expr": {
        this.buildExpr(g, c);
        break;
      }
      case "exprCurve": {
        this.buildExprCurve(g, c);
        break;
      }
      case "mid": {
        const s = this.map.get(c.seg);
        if (s?.p1 && s.p2 && ok(s.p1) && ok(s.p2)) {
          g.x = (s.p1[0] + s.p2[0]) / 2;
          g.y = (s.p1[1] + s.p2[1]) / 2;
        }
        break;
      }
      case "seg":
      case "line":
      case "ray":
      case "vector": {
        const a = this.pt(c.a);
        const b = this.pt(c.b);
        if (a && b) {
          g.p1 = a;
          g.p2 = b;
          g.dir = norm(sub(b, a)) ?? [1, 0];
          g.x = (a[0] + b[0]) / 2;
          g.y = (a[1] + b[1]) / 2;
          g.a = dist(a, b);
        }
        break;
      }
      case "poly": {
        const pts = c.pts.map((q) => this.pt(q)).filter(Boolean) as P[];
        g.pts = pts;
        if (pts.length) {
          g.x = pts.reduce((s, q) => s + q[0], 0) / pts.length;
          g.y = pts.reduce((s, q) => s + q[1], 0) / pts.length;
        }
        g.a = polygonSignedArea(pts);
        break;
      }
      case "circ": {
        const cc = this.pt(c.c);
        const on = this.pt(c.p);
        if (cc && on) {
          g.center = cc;
          g.x = cc[0];
          g.y = cc[1];
          g.a = dist(cc, on);
          g.b = Math.atan2(on[1] - cc[1], on[0] - cc[0]);
        }
        break;
      }
      case "arc": {
        const cc = this.pt(c.c);
        const s = this.pt(c.s);
        const e = this.pt(c.e);
        if (cc && s && e) {
          const a0 = Math.atan2(s[1] - cc[1], s[0] - cc[0]);
          let a1 = Math.atan2(e[1] - cc[1], e[0] - cc[0]);
          if (a1 <= a0) a1 += TWO_PI;
          g.center = cc;
          g.x = cc[0];
          g.y = cc[1];
          g.a = dist(cc, s);
          g.b = a0;
          g.rot = a1;
        }
        break;
      }
      case "ell": {
        let a = Math.abs(c.a);
        let b = Math.abs(c.b);
        let cen: P = [c.c[0], c.c[1]];
        let rr = c.rot;
        if (c.f1 && c.f2 && c.p) {
          // 2a = |PF1|+|PF2|，c = |F1F2|/2，长轴沿两焦点连线
          const f1 = this.pt(c.f1);
          const f2 = this.pt(c.f2);
          const p = this.pt(c.p);
          if (f1 && f2 && p) {
            cen = [(f1[0] + f2[0]) / 2, (f1[1] + f2[1]) / 2];
            a = (dist(p, f1) + dist(p, f2)) / 2;
            const half = dist(f1, f2) / 2;
            b = Math.sqrt(Math.max(0, a * a - half * half));
            rr = Math.atan2(f2[1] - f1[1], f2[0] - f1[0]);
          }
        }
        g.center = cen;
        g.x = cen[0];
        g.y = cen[1];
        g.a = a;
        g.b = b;
        g.rot = rr;
        g.dir = [Math.cos(rr), Math.sin(rr)];
        break;
      }
      case "perp":
      case "par": {
        const p = this.pt(c.p);
        const l = this.lineOf(c.l);
        if (p && l) {
          const d = c.t === "perp" ? perpOf(l.d) : l.d;
          g.p1 = p;
          g.dir = d;
          g.p2 = add(p, d);
          g.x = p[0];
          g.y = p[1];
        }
        break;
      }
      case "bis": {
        const a = this.pt(c.a);
        const b = this.pt(c.b);
        const cc = this.pt(c.c);
        if (a && b && cc) {
          const u = norm(sub(a, b));
          const v = norm(sub(cc, b));
          if (u && v) {
            const d = norm(add(u, v)) ?? perpOf(u);
            g.p1 = b;
            g.dir = d;
            g.p2 = add(b, d);
            g.x = b[0];
            g.y = b[1];
          }
        }
        break;
      }
      case "int": {
        const cands = this.intCandidates(g);
        if (cands.length) {
          let br = g.branch;
          if (br === undefined || br < 0 || br >= cands.length) {
            // 所选分支消失时贴合上一个位置，避免拖动中在两个解之间跳变
            let bi = 0;
            let bd = Infinity;
            for (let i = 0; i < cands.length; i++) {
              const d = dist(cands[i], [g.x, g.y]);
              if (d < bd) {
                bd = d;
                bi = i;
              }
            }
            br = bi;
            g.branch = bi;
          }
          g.x = cands[br][0];
          g.y = cands[br][1];
        }
        break;
      }
      case "rot": {
        const s = this.pt(c.s);
        const cc: P | null = typeof c.c === "string" ? this.pt(c.c) : [c.c[0], c.c[1]];
        if (s && cc) {
          const r = rot(sub(s, cc), c.ang);
          g.x = cc[0] + r[0];
          g.y = cc[1] + r[1];
        }
        break;
      }
      case "refl": {
        const s = this.pt(c.s);
        const l = this.lineOf(c.l);
        if (s && l) {
          const v = sub(s, l.o);
          const proj = mul(l.d, v[0] * l.d[0] + v[1] * l.d[1]);
          const foot = add(l.o, proj);
          g.x = 2 * foot[0] - s[0];
          g.y = 2 * foot[1] - s[1];
        }
        break;
      }
      case "dil": {
        const s = this.pt(c.s);
        const cc: P | null = typeof c.c === "string" ? this.pt(c.c) : [c.c[0], c.c[1]];
        if (s && cc) {
          g.x = cc[0] + (s[0] - cc[0]) * c.k;
          g.y = cc[1] + (s[1] - cc[1]) * c.k;
        }
        break;
      }
      case "trans": {
        const s = this.pt(c.s);
        let v: P = c.v;
        if (c.a !== c.b) {
          const a = this.pt(c.a);
          const bq = this.pt(c.b);
          if (a && bq) v = sub(bq, a);
        }
        if (s) {
          g.x = s[0] + v[0];
          g.y = s[1] + v[1];
        }
        break;
      }
      case "locus": {
        const tr = this.trace(g.id);
        g.pts = tr;
        break;
      }
      case "angle": {
        const a = this.pt(c.a);
        const b = this.pt(c.b);
        const cc = this.pt(c.c);
        if (a && b && cc) {
          const a0 = Math.atan2(a[1] - b[1], a[0] - b[0]);
          const a1 = Math.atan2(cc[1] - b[1], cc[0] - b[0]);
          // 取劣角：把起始角调整为逆时针扫到另一条边的那个
          let start = a1;
          let sweep = wrap(a0 - a1);
          if (sweep > Math.PI) {
            start = a0;
            sweep = TWO_PI - sweep;
          }
          g.x = b[0];
          g.y = b[1];
          g.p1 = b;
          g.p2 = a;
          g.dir = [Math.cos(a1), Math.sin(a1)];
          g.a = sweep;
          g.b = start;
          g.text = `${((180 * sweep) / Math.PI).toFixed(1)}°`;
        }
        break;
      }
      case "measure": {
        const v = this.measureValue(c.what, c.ids);
        g.x = c.ax;
        g.y = c.ay;
        g.a = v;
        g.text = `${c.what} = ${fin(v) ? v.toFixed(3) : "?"}`;
        break;
      }
    }
  }

  /* -------------------------------------------------------- 表达式驱动 */

  private buildExpr(g: Geo, c: { t: "expr"; x: string; y: string; v: string; p: number }): void {
    const cf = this.exprFn(g.id, c.x, c.y);
    if (!cf) return;
    const s = new Float64Array(2);
    s[0] = c.p;
    const x = cf.x(s);
    const y = cf.y(s);
    if (fin(x) && fin(y)) {
      g.x = x;
      g.y = y;
    }
    g.param = c.p;
  }

  /** 参数曲线：把 t∈[t0,t1] 采样成上屏折线 */
  private buildExprCurve(g: Geo, c: { t: "exprCurve"; x: string; y: string; t0: number; t1: number; n: number }): void {
    const cf = this.exprFn(g.id, c.x, c.y);
    if (!cf) {
      g.pts = [];
      return;
    }
    const s = new Float64Array(2);
    const pts: P[] = [];
    for (let i = 0; i < c.n; i++) {
      s[0] = c.t0 + ((c.t1 - c.t0) * i) / (c.n - 1);
      const x = cf.x(s);
      const y = cf.y(s);
      pts.push(fin(x) && fin(y) ? [x, y] : [NaN, NaN]);
    }
    g.pts = pts;
  }

  private exprFn(
    id: string,
    xExpr: string,
    yExpr: string,
  ): { x: (s: Float64Array) => number; y: (s: Float64Array) => number } | null {
    let cf = this.compiled.get(id);
    if (!cf) {
      if (!this.engine) return null;
      cf = compilePair(this.engine, xExpr, yExpr) ?? undefined;
      if (cf) this.compiled.set(id, cf);
    }
    return cf ?? null;
  }

  private exprParam(g: Geo, at: P): number | null {
    const c = g.ctor;
    if (c.t !== "expr") return null;
    const cf = this.exprFn(g.id, c.x, c.y);
    if (!cf) return null;
    let best: number | null = null;
    let bd = Infinity;
    const s = new Float64Array(2);
    for (let i = 0; i <= 200; i++) {
      const t = i / 200;
      s[0] = t;
      const d = dist([cf.x(s), cf.y(s)], at);
      if (d < bd) {
        bd = d;
        best = t;
      }
    }
    return best;
  }

  /* -------------------------------------------------------- 曲线参数化 */

  /** 曲线上的参数点：t∈[0,1] */
  pointAtT(hostId: string, t: number): P | null {
    const g = this.map.get(hostId);
    if (!g) return null;
    const q = norm01(t);
    switch (g.kind) {
      case "segment":
      case "vector": {
        if (!g.p1 || !g.p2 || !ok(g.p1) || !ok(g.p2)) return null;
        return add(g.p1, mul(sub(g.p2, g.p1), q));
      }
      case "line":
      case "ray": {
        if (!g.p1 || !g.dir || !ok(g.p1) || !ok(g.dir)) return null;
        const span = g.a && fin(g.a) && g.a > 1e-9 ? g.a : 1;
        const k = g.kind === "ray" ? q * span : (q - 0.5) * 2 * span;
        return add(g.p1, mul(g.dir, k));
      }
      case "circle": {
        const ci = this.circleOf(hostId);
        if (!ci) return null;
        const a = ci.ref + q * TWO_PI;
        return [ci.c[0] + ci.r * Math.cos(a), ci.c[1] + ci.r * Math.sin(a)];
      }
      case "arc": {
        if (!g.center || !ok(g.center) || !fin(g.a) || !fin(g.b) || !fin(g.rot)) return null;
        const a = g.b + q * (g.rot - g.b);
        return [g.center[0] + g.a * Math.cos(a), g.center[1] + g.a * Math.sin(a)];
      }
      case "ellipse": {
        if (!g.center || !ok(g.center)) return null;
        const a = q * TWO_PI;
        const p = rot([g.a * Math.cos(a), g.b * Math.sin(a)], g.rot ?? 0);
        return add(g.center, p);
      }
      case "polygon":
      case "locus": {
        const pts = g.pts;
        if (!pts || pts.length < 2) return null;
        const closed = g.kind === "polygon";
        const seg = closed ? pts.length : pts.length - 1;
        const f = q * seg;
        const i = Math.min(seg - 1, Math.floor(f));
        const u = f - i;
        const a = pts[i];
        const b = pts[(i + 1) % pts.length];
        return add(a, mul(sub(b, a), u));
      }
      default:
        return null;
    }
  }

  /** 驱动点的参数域：宿主曲线决定闭合与否，闭合曲线取 720 份 */
  private driverDomain(drv: Geo): { closed: boolean; samples: number } | null {
    const c = drv.ctor;
    if (c.t === "on") {
      const host = this.map.get(c.host);
      if (!host) return null;
      const closed =
        host.kind === "circle" || host.kind === "ellipse" || host.kind === "polygon" || host.kind === "locus";
      return { closed, samples: closed ? 720 : 480 };
    }
    if (c.t === "expr") return { closed: false, samples: 480 };
    return null;
  }

  /** 把世界坐标投影到宿主曲线上，返回参数 */
  paramAt(hostId: string, at: P): number | null {
    const g = this.map.get(hostId);
    if (!g) return null;
    switch (g.kind) {
      case "segment":
      case "vector": {
        if (!g.p1 || !g.p2 || !ok(g.p1) || !ok(g.p2)) return null;
        const ab = sub(g.p2, g.p1);
        const dd = ab[0] * ab[0] + ab[1] * ab[1];
        if (dd < 1e-24) return 0;
        return clamp01(((at[0] - g.p1[0]) * ab[0] + (at[1] - g.p1[1]) * ab[1]) / dd);
      }
      case "line":
      case "ray": {
        if (!g.p1 || !g.dir || !ok(g.p1) || !ok(g.dir)) return null;
        const v = sub(at, g.p1);
        const span = g.a && g.a > 1e-9 ? g.a : 1;
        const k = v[0] * g.dir[0] + v[1] * g.dir[1];
        return g.kind === "ray" ? clamp01(k / span) : clamp01(k / (2 * span) + 0.5);
      }
      case "circle": {
        const ci = this.circleOf(hostId);
        if (!ci) return null;
        return norm01(fracIn(Math.atan2(at[1] - ci.c[1], at[0] - ci.c[0]), ci.ref, TWO_PI));
      }
      case "arc": {
        if (!g.center || !ok(g.center) || !fin(g.b) || !fin(g.rot)) return null;
        const sweep = (g.rot as number) - (g.b as number);
        return clamp01(fracIn(Math.atan2(at[1] - g.center[1], at[0] - g.center[0]), g.b, sweep));
      }
      case "ellipse": {
        if (!g.center || !ok(g.center) || !g.a || !g.b) return null;
        // 离心角：把点换算到椭圆主轴坐标系后按 atan2(y/b, x/a) 取参数
        const v = rot(sub(at, g.center), -(g.rot ?? 0));
        return norm01(Math.atan2(v[1] / g.b, v[0] / g.a) / TWO_PI);
      }
      case "polygon":
      case "locus": {
        const pts = g.pts;
        if (!pts || pts.length < 2) return null;
        let best = 0;
        let bd = Infinity;
        const closed = g.kind === "polygon";
        const seg = closed ? pts.length : pts.length - 1;
        for (let i = 0; i < seg; i++) {
          const a = pts[i];
          const b = pts[(i + 1) % pts.length];
          const ab = sub(b, a);
          const dd = ab[0] * ab[0] + ab[1] * ab[1];
          const u = dd < 1e-24 ? 0 : clamp01(((at[0] - a[0]) * ab[0] + (at[1] - a[1]) * ab[1]) / dd);
          const d = dist(add(a, mul(ab, u)), at);
          if (d < bd) {
            bd = d;
            best = (i + u) / seg;
          }
        }
        return best;
      }
      default:
        return null;
    }
  }

  /* ------------------------------------------------------------ 求交 */

  /** 解析求交，候选按极角、半径排序 */
  private intCandidates(g: Geo): P[] {
    const c = g.ctor as { t: "int"; a: string; b: string };
    const A = this.map.get(c.a);
    const B = this.map.get(c.b);
    if (!A || !B) return [];
    const out = intersect(this.shapeOf(A), this.shapeOf(B));
    out.sort((p, q) => {
      const ap = Math.atan2(p[1], p[0]),
        aq = Math.atan2(q[1], q[0]);
      if (Math.abs(ap - aq) > 1e-12) return ap - aq;
      return len(p) - len(q);
    });
    return out;
  }

  /** 求交用的隐式形状 */
  private shapeOf(
    g: Geo,
  ): { k: "line"; o: P; d: P } | { k: "seg"; a: P; b: P } | { k: "circ"; c: P; r: number } | { k: "ell"; c: P; a: number; b: number; rot: number } | null {
    switch (g.kind) {
      case "line":
      case "ray": {
        if (g.p1 && g.dir && ok(g.p1) && ok(g.dir)) return { k: "line", o: g.p1, d: g.dir };
        return null;
      }
      case "segment":
      case "vector": {
        if (g.p1 && g.p2 && ok(g.p1) && ok(g.p2)) return { k: "seg", a: g.p1, b: g.p2 };
        return null;
      }
      case "circle": {
        if (g.center && ok(g.center) && fin(g.a)) return { k: "circ", c: g.center, r: g.a };
        return null;
      }
      case "arc": {
        if (g.center && ok(g.center) && fin(g.a)) return { k: "circ", c: g.center, r: g.a };
        return null;
      }
      case "ellipse": {
        if (g.center && ok(g.center) && fin(g.a) && fin(g.b))
          return { k: "ell", c: g.center, a: g.a, b: g.b, rot: g.rot ?? 0 };
        return null;
      }
      case "polygon":
      case "locus": {
        return null; // 折线求交由采样近似，UI 层不必
      }
      default:
        return null;
    }
  }

  /* ------------------------------------------------------------ 轨迹 */

  /** 该点能否充当轨迹驱动：必须被约束在曲线或参数路径上，自由点没有参数域 */
  canDrive(id: string): boolean {
    const g = this.map.get(id);
    return !!g && g.kind === "point" && !!this.driverDomain(g);
  }

  /** 采样驱动的整个参数域，逐点解析重算目标位置（结果按修订号缓存） */
  trace(locusId: string): P[] {
    const g = this.map.get(locusId);
    if (!g) return [];
    const c = g.ctor;
    if (c.t !== "locus") return g.pts ?? [];
    if (this.sampling) return g.pts ?? [];
    const hit = this.traces.get(locusId);
    if (hit && hit.rev === this.rev) return hit.pts;
    const drv = this.map.get(c.d);
    const dom = drv ? this.driverDomain(drv) : null;
    if (!drv || !dom) return [];
    const dc = drv.ctor as { p: number };
    const save = dc.p;
    this.sampling = true;
    const pts: P[] = [];
    const n = dom.samples;
    for (let i = 0; i < n; i++) {
      dc.p = dom.closed ? i / n : i / (n - 1);
      this.recompute();
      const tgt = this.map.get(c.target);
      if (tgt && fin(tgt.x) && fin(tgt.y)) pts.push([tgt.x, tgt.y]);
    }
    dc.p = save;
    /* 收尾这次重算仍要留在 sampling 里：否则嵌套的轨迹会拿"最后一个采样位置"当作 save 复原，
       驱动点就永远停在轨迹末端 */
    this.recompute();
    this.sampling = false;
    this.traces.set(locusId, { rev: this.rev, pts });
    return pts;
  }

  /* ------------------------------------------------------------ 测量 */

  dist(a: string, b: string): number {
    const p = this.pt(a),
      q = this.pt(b);
    return p && q ? dist(p, q) : NaN;
  }

  /** ∠abc，顶点在 b，返回 [0,π] 弧度 */
  angleOf(a: string, b: string, c: string): number {
    const p = this.pt(a),
      q = this.pt(b),
      r = this.pt(c);
    if (!p || !q || !r) return NaN;
    const u = norm(sub(p, q));
    const v = norm(sub(r, q));
    if (!u || !v) return NaN;
    const d = clamp1(u[0] * v[0] + u[1] * v[1]);
    return Math.acos(d);
  }

  slope(id: string): number {
    const g = this.map.get(id);
    if (!g?.dir || !ok(g.dir)) return NaN;
    return Math.abs(g.dir[0]) < 1e-12 ? Infinity : g.dir[1] / g.dir[0];
  }

  /** 长度：线段/向量=长，圆=周长，圆弧=弧长，椭圆=Ramanujan 近似，多边形/轨迹=周长 */
  length(id: string): number {
    const g = this.map.get(id);
    if (!g) return NaN;
    switch (g.kind) {
      case "segment":
      case "vector":
        return fin(g.a) ? g.a : NaN;
      case "circle":
        return fin(g.a) ? TWO_PI * g.a : NaN;
      case "arc":
        return fin(g.a) && fin(g.b) && fin(g.rot) ? g.a * Math.abs((g.rot as number) - (g.b as number)) : NaN;
      case "ellipse": {
        if (!fin(g.a) || !fin(g.b)) return NaN;
        const a = g.a,
          b = g.b;
        return Math.PI * (3 * (a + b) - Math.sqrt((3 * a + b) * (a + 3 * b)));
      }
      case "polygon":
      case "locus":
        return polylineLength(g.pts ?? [], g.kind === "polygon");
      default:
        return NaN;
    }
  }

  polygonArea(id: string): number {
    const g = this.map.get(id);
    if (!g) return 0;
    if (g.kind === "polygon") return Math.abs(polygonSignedArea(g.pts ?? []));
    if (g.kind === "ellipse") return Math.PI * (g.a ?? 0) * (g.b ?? 0);
    return 0;
  }

  private measureValue(what: MeasureWhat, ids: string[]): number {
    switch (what) {
      case "area":
        return ids.length ? this.polygonArea(ids[0]) : NaN;
      case "length":
      case "perimeter":
        return ids.length ? this.length(ids[0]) : NaN;
      case "dist":
        return ids.length >= 2 ? this.dist(ids[0], ids[1]) : NaN;
      case "angle":
        return ids.length >= 3 ? this.angleOf(ids[0], ids[1], ids[2]) : NaN;
      case "slope":
        return ids.length ? this.slope(ids[0]) : NaN;
      case "radius": {
        const g = ids.length ? this.map.get(ids[0]) : null;
        return g && fin(g.a) ? g.a : NaN;
      }
      default:
        return NaN;
    }
  }

  bbox(): [number, number, number, number] {
    let x0 = Infinity,
      y0 = Infinity,
      x1 = -Infinity,
      y1 = -Infinity;
    for (const g of this.all()) {
      const push = (p: P) => {
        if (!ok(p)) return;
        x0 = Math.min(x0, p[0]);
        y0 = Math.min(y0, p[1]);
        x1 = Math.max(x1, p[0]);
        y1 = Math.max(y1, p[1]);
      };
      if (g.kind === "point") push([g.x, g.y]);
      else if (g.pts) for (const q of g.pts) push(q);
      else if (g.center && fin(g.a)) {
        push([g.center[0] - g.a, g.center[1] - g.a]);
        push([g.center[0] + g.a, g.center[1] + g.a]);
      }
    }
    if (!fin(x0)) return [-5, 5, -4, 4];
    return [x0, x1, y0, y1];
  }

  /** 拾取：点优先，其次按容差内的曲线距离 */
  hitTest(x: number, y: number, tolWorld: number): string | null {
    const at: P = [x, y];
    let bestPt: { id: string; d: number } | null = null;
    let bestLine: { id: string; d: number } | null = null;
    for (const g of this.all()) {
      if (g.visible === false) continue;
      const tol = tolWorld * (g.kind === "point" ? 1.6 : 1);
      if (g.kind === "point") {
        const d = dist(at, [g.x, g.y]);
        if (d <= tol && (!bestPt || d < bestPt.d)) bestPt = { id: g.id, d };
        continue;
      }
      const d = this.distTo(g, at);
      if (d <= tol && (!bestLine || d < bestLine.d)) bestLine = { id: g.id, d };
    }
    return bestPt?.id ?? bestLine?.id ?? null;
  }

  private distTo(g: Geo, at: P): number {
    switch (g.kind) {
      case "segment":
      case "vector":
        return g.p1 && g.p2 && ok(g.p1) && ok(g.p2) ? segDist(at, g.p1, g.p2) : Infinity;
      case "line":
      case "ray": {
        if (!g.p1 || !g.dir || !ok(g.p1) || !ok(g.dir)) return Infinity;
        const v = sub(at, g.p1);
        const k = v[0] * g.dir[0] + v[1] * g.dir[1];
        if (g.kind === "ray" && k < 0) return dist(at, g.p1);
        return dist(at, add(g.p1, mul(g.dir, k)));
      }
      case "circle": {
        if (!g.center || !ok(g.center) || !fin(g.a)) return Infinity;
        return Math.abs(dist(at, g.center) - g.a);
      }
      case "arc": {
        if (!g.center || !ok(g.center) || !fin(g.a) || !fin(g.b) || !fin(g.rot)) return Infinity;
        const a = Math.atan2(at[1] - g.center[1], at[0] - g.center[0]);
        const sweep = (g.rot as number) - (g.b as number);
        if (fracIn(a, g.b as number, sweep) > 1) return dist(at, this.pointAtT(g.id, 0)!);
        return Math.abs(dist(at, g.center) - g.a);
      }
      case "ellipse": {
        if (!g.center || !ok(g.center) || !g.a || !g.b) return Infinity;
        const v = rot(sub(at, g.center), -(g.rot ?? 0));
        const r = Math.hypot(v[0] / g.a, v[1] / g.b);
        return Math.abs(r - 1) * Math.min(g.a, g.b);
      }
      case "polygon":
      case "locus":
        return g.pts && g.pts.length > 1 ? polylineDist(at, g.pts, g.kind === "polygon") : Infinity;
      default:
        return Infinity;
    }
  }

  /** 渲染采样：统一给出折线点列（圆/椭圆/弧/多边形/轨迹/线状） */
  samplePoints(id: string, n = 128): P[] {
    const g = this.map.get(id);
    if (!g) return [];
    switch (g.kind) {
      case "circle":
      case "ellipse": {
        const out: P[] = [];
        if (!g.center || !ok(g.center) || !fin(g.a)) return [];
        const b = g.kind === "circle" ? g.a : g.b ?? 0;
        for (let i = 0; i <= n; i++) {
          const t = (i / n) * TWO_PI;
          const p = rot([g.a * Math.cos(t), b * Math.sin(t)], g.rot ?? 0);
          out.push(add(g.center, p));
        }
        return out;
      }
      case "arc": {
        if (!g.center || !ok(g.center) || !fin(g.a) || !fin(g.b) || !fin(g.rot)) return [];
        const out: P[] = [];
        for (let i = 0; i <= n; i++) {
          const t = (g.b as number) + ((g.rot as number) - (g.b as number)) * (i / n);
          out.push([g.center[0] + g.a * Math.cos(t), g.center[1] + g.a * Math.sin(t)]);
        }
        return out;
      }
      case "polygon":
      case "locus":
        return g.pts ?? [];
      case "segment":
      case "vector":
        return g.p1 && g.p2 ? [g.p1, g.p2] : [];
      case "line":
      case "ray":
        return g.p1 && g.p2 ? [g.p1, g.p2] : [];
      case "point":
        return ok([g.x, g.y]) ? [[g.x, g.y]] : [];
      default:
        return [];
    }
  }

  /** 端点（供箭头、角度标记、标签定位使用） */
  ends(id: string): [P, P] | null {
    const g = this.map.get(id);
    if (!g || !g.p1 || !g.p2 || !ok(g.p1) || !ok(g.p2)) return null;
    return [g.p1, g.p2];
  }

  /* ------------------------------------------------------------ 历史/序列化 */

  beginGesture(): void {
    if (!this.gesture) {
      this.gesture = true;
      this.commit();
    }
  }

  endGesture(): void {
    this.gesture = false;
  }

  /** 快照入栈：保存变更前的状态 */
  private commit(): void {
    if (this.gesture && this.undoStack.length && this.undoStack[this.undoStack.length - 1] === this.snapshot()) {
      return;
    }
    this.undoStack.push(this.snapshot());
    if (this.undoStack.length > 100) this.undoStack.shift();
    this.redoStack.length = 0;
  }

  private snapshot(): string {
    return JSON.stringify({ n: this.n, order: this.order, items: this.order.map((i) => ser(this.map.get(i)!)) });
  }

  private restore(snap: string): void {
    const data = JSON.parse(snap) as { n: number; order: string[]; items: unknown[] };
    this.map = new Map();
    this.order = data.order;
    this.n = data.n;
    this.compiled.clear();
    for (const it of data.items) this.map.set((it as Geo).id, deser(it as Geo, this.map));
    this.recompute();
  }

  undo(): boolean {
    const prev = this.undoStack.pop();
    if (prev === undefined) return false;
    this.redoStack.push(this.snapshot());
    this.restore(prev);
    this.mark();
    return true;
  }

  redo(): boolean {
    const next = this.redoStack.pop();
    if (next === undefined) return false;
    this.undoStack.push(this.snapshot());
    this.restore(next);
    this.mark();
    return true;
  }

  private mark(): void {
    this.rev++;
    this.recompute();
  }

  toJSON(): string {
    return JSON.stringify({ v: 1, ...JSON.parse(this.snapshot()) });
  }

  static fromJSON(s: string, engine?: Engine): GeometryDoc {
    const doc = new GeometryDoc();
    const data = JSON.parse(s) as { n: number; order: string[]; items: unknown[] };
    doc.order = data.order.slice();
    doc.n = data.n;
    for (const it of data.items) {
      const g = deser(it as Geo, doc.map);
      doc.map.set(g.id, g);
    }
    doc.engine = engine ?? null;
    doc.recompute();
    return doc;
  }

  /** 载入/重建后调用，重新编译表达式对象 */
  bindEngine(engine: Engine): void {
    this.engine = engine;
    this.compiled.clear();
    this.recompute();
  }
}

/* ------------------------------------------------------------ 自由函数 */

function autoLabel(kind: GeoKind, n: number): string {
  const prefix: Partial<Record<GeoKind, string>> = {
    point: "P",
    segment: "s",
    line: "l",
    ray: "r",
    vector: "v",
    circle: "c",
    arc: "a",
    ellipse: "e",
    polygon: "多边形",
    locus: "轨迹",
    angle: "∠",
    text: "文本",
  };
  return `${prefix[kind] ?? "o"}${n}`;
}

function norm01(t: number): number {
  if (!fin(t)) return 0;
  return t - Math.floor(t);
}

function clamp01(t: number): number {
  return t < 0 ? 0 : t > 1 ? 1 : t;
}

function clamp1(t: number): number {
  return t < -1 ? -1 : t > 1 ? 1 : t;
}

function polygonSignedArea(pts: P[]): number {
  let s = 0;
  for (let i = 0; i + 1 < pts.length; i++) s += pts[i][0] * pts[i + 1][1] - pts[i + 1][0] * pts[i][1];
  if (pts.length > 2) s += pts[pts.length - 1][0] * pts[0][1] - pts[0][0] * pts[pts.length - 1][1];
  return s / 2;
}

function polylineLength(pts: P[], closed: boolean): number {
  let s = 0;
  for (let i = 0; i + 1 < pts.length; i++) s += dist(pts[i], pts[i + 1]);
  if (closed && pts.length > 2) s += dist(pts[pts.length - 1], pts[0]);
  return s;
}

type Shape =
  | { k: "line"; o: P; d: P }
  | { k: "seg"; a: P; b: P }
  | { k: "circ"; c: P; r: number }
  | { k: "ell"; c: P; a: number; b: number; rot: number };

/** 两形状的解析求交；线段只在自身范围内取解 */
function intersect(A: Shape | null, B: Shape | null): P[] {
  if (!A || !B) return [];
  const cut = (s: Shape, pts: P[]): P[] =>
    s.k === "seg" ? pts.filter((p) => segDist(p, s.a, s.b) <= 1e-7 * Math.max(1, len(sub(s.b, s.a)))) : pts;

  if (A.k === "line" || A.k === "seg") {
    const la = A.k === "seg" ? { o: A.a, d: norm(sub(A.b, A.a)) ?? ([1, 0] as P) } : A;
    if (B.k === "line" || B.k === "seg") {
      const lb = B.k === "seg" ? { o: B.a, d: norm(sub(B.b, B.a)) ?? ([1, 0] as P) } : B;
      const den = la.d[0] * lb.d[1] - lb.d[0] * la.d[1];
      if (Math.abs(den) < 1e-12) return [];
      const w = sub(lb.o, la.o);
      const t = (w[0] * lb.d[1] - w[1] * lb.d[0]) / den;
      const p = add(la.o, mul(la.d, t));
      return cut(A, cut(B, ok(p) ? [p] : []));
    }
    if (B.k === "circ") return cut(A, cut(B, lineCircle(la, B)));
    return cut(A, cut(B, lineEllipse(la, B)));
  }
  if (A.k === "circ") {
    if (B.k === "circ") return circCirc(A, B);
    if (B.k === "line" || B.k === "seg") {
      const lb = B.k === "seg" ? { o: B.a, d: norm(sub(B.b, B.a)) ?? ([1, 0] as P) } : B;
      return cut(B, lineCircle(lb, A));
    }
  }
  // 椭圆与圆/椭圆：牛顿迭代求交（初值取椭圆参数栅格）
  const ell = A.k === "ell" ? A : B.k === "ell" ? B : null;
  const other = A.k === "ell" ? B : A;
  if (ell && other && ell !== other) {
    const pts: P[] = [];
    for (let i = 0; i < 240; i++) {
      const t = (i / 240) * TWO_PI;
      const q = add(ell.c, rot([ell.a * Math.cos(t), ell.b * Math.sin(t)], ell.rot));
      const f = () => resid(other, q);
      let cur: P = q;
      for (let k = 0; k < 24; k++) {
        const r0 = resid(other, cur);
        if (Math.abs(r0) < 1e-12) break;
        const hx = 1e-6;
        const dr = (resid(other, [cur[0] + hx, cur[1]]) - r0) / hx;
        if (Math.abs(dr) < 1e-14) break;
        cur = [cur[0] - r0 / dr, cur[1]];
      }
      if (Math.abs(f()) < 1e-7 && ok(cur) && !pts.some((p) => dist(p, cur) < 1e-6)) pts.push(cur);
    }
    return pts;
  }
  return [];
}

function resid(s: Shape, p: P): number {
  if (s.k === "circ") return dist(p, s.c) - s.r;
  if (s.k === "ell") {
    const v = rot(sub(p, s.c), -s.rot);
    return Math.hypot(v[0] / s.a, v[1] / s.b) - 1;
  }
  const o = s.k === "seg" ? s.a : s.o;
  const d = s.k === "seg" ? norm(sub(s.b, s.a)) ?? ([1, 0] as P) : s.d;
  const n = perpOf(d);
  return (p[0] - o[0]) * n[0] + (p[1] - o[1]) * n[1];
}

function lineCircle(l: { o: P; d: P }, c: { c: P; r: number }): P[] {
  const w = sub(l.o, c.c);
  const b = w[0] * l.d[0] + w[1] * l.d[1];
  const cc = w[0] * w[0] + w[1] * w[1] - c.r * c.r;
  const disc = b * b - cc;
  if (disc < 0 || !fin(disc)) return [];
  const q = Math.sqrt(disc);
  return [add(l.o, mul(l.d, -b + q)), add(l.o, mul(l.d, -b - q))].filter(ok);
}

function lineEllipse(l: { o: P; d: P }, e: { c: P; a: number; b: number; rot: number }): P[] {
  const o = rot(sub(l.o, e.c), -e.rot);
  const d = rot(l.d, -e.rot);
  const A = (d[0] / e.a) ** 2 + (d[1] / e.b) ** 2;
  const B = 2 * ((o[0] * d[0]) / e.a ** 2 + (o[1] * d[1]) / e.b ** 2);
  const C = (o[0] / e.a) ** 2 + (o[1] / e.b) ** 2 - 1;
  if (A < 1e-18) return [];
  const disc = B * B - 4 * A * C;
  if (disc < 0) return [];
  const q = Math.sqrt(disc);
  return [(-B + q) / (2 * A), (-B - q) / (2 * A)].map((t) => add(l.o, mul(l.d, t))).filter(ok);
}

function circCirc(a: { c: P; r: number }, b: { c: P; r: number }): P[] {
  const d0 = dist(a.c, b.c);
  if (d0 < 1e-12 || d0 > a.r + b.r || d0 < Math.abs(a.r - b.r)) return [];
  const u = norm(sub(b.c, a.c))!;
  const x = (d0 * d0 + a.r * a.r - b.r * b.r) / (2 * d0);
  const h2 = a.r * a.r - x * x;
  if (h2 < 0) return [];
  const h = Math.sqrt(h2);
  const m = add(a.c, mul(u, x));
  const v = perpOf(u);
  return [add(m, mul(v, h)), add(m, mul(v, -h))];
}

/** 表达式驱动的 (x(t), y(t)) 编译；任一失败即整体不可用 */
function compilePair(
  eng: Engine,
  xExpr: string,
  yExpr: string,
): { x: (s: Float64Array) => number; y: (s: Float64Array) => number } | null {
  try {
    const xs = compileReal(eng, parseExpr(xExpr), ["t"]);
    const ys = compileReal(eng, parseExpr(yExpr), ["t"]);
    return xs && ys ? { x: xs, y: ys } : null;
  } catch {
    return null;
  }
}

/** 序列化：只保留构造配方与元数据 */
function ser(g: Geo): Record<string, unknown> {
  return {
    id: g.id,
    kind: g.kind,
    label: g.label,
    ctor: g.ctor,
    visible: g.visible,
    branch: g.branch,
    param: g.param,
    host: g.host,
    a: g.a,
    b: g.b,
    rot: g.rot,
    center: g.center,
    frozen: g.frozen,
  };
}

function deser(raw: Geo, into: Map<string, Geo>): Geo {
  const g: Geo = {
    id: raw.id,
    kind: raw.kind,
    label: raw.label,
    x: 0,
    y: 0,
    ctor: raw.ctor,
    visible: raw.visible !== false,
    branch: raw.branch,
    param: raw.param,
    host: raw.host,
    a: typeof raw.a === "number" ? raw.a : NaN,
    b: typeof raw.b === "number" ? raw.b : NaN,
    rot: typeof raw.rot === "number" ? raw.rot : 0,
    center: raw.center ? ([raw.center[0], raw.center[1]] as P) : undefined,
    frozen: raw.frozen,
  };
  into.set(g.id, g);
  return g;
}
