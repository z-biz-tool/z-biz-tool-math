// 动态几何内核 GeometryDoc 的纯逻辑回归。
// 测什么：几何量的数学定义 + 全部退化路径（缺 id / NaN / 重合点 / 零半径 / 共线 / 越界参数）。
// 为什么该测：文件头声明的契约是"任一步得到非有限值时保留上一次结果，绝不在 recompute 中抛错，
//   保证共线、重合点、零半径等极端拖动下交互不断"——这条契约全靠这些分支撑着，却没有任何测试钉住。
//   交互不断 = 不抛错；NaN 不上屏 = 结果要么是旧值要么是有限值。两者都要断言。
// 注意：本文件只测经 public API（GeometryDoc 的方法）能观察到的行为，geometry.ts 里的自由函数
//   （intersect / polygonSignedArea / norm01 …）均未导出，故不做直测。
// 跑法：node --experimental-strip-types --test tests/*.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import { GeometryDoc } from "../src/core/geometry.ts";
import { Engine } from "../src/core/machine.ts";

/* ---------------------------------------------------------------- 断言助手 */

/** 浮点比较：几何断言一律走容差，不做严格相等 */
function near(actual: number, expect: number, tol = 1e-9, msg?: string): void {
  assert.ok(
    Math.abs(actual - expect) < tol,
    `${msg ?? "值不相等"}：实际 ${actual}，期望 ${expect}（容差 ${tol}）`,
  );
}
function nearPt(actual: [number, number], expect: [number, number], tol = 1e-9): void {
  near(actual[0], expect[0], tol, "x");
  near(actual[1], expect[1], tol, "y");
}
/** 方向向量容差比较：必须用这个而不是 deepEqual——perpOf 会算出 -0，deepEqual 判 -0 ≠ 0 */
function nearVec(actual: [number, number] | undefined, expect: [number, number], tol = 1e-12): void {
  assert.ok(actual, "方向向量为 undefined");
  near(actual![0], expect[0], tol, "dx");
  near(actual![1], expect[1], tol, "dy");
}
/** 取点的位置；构造出错时给出可读信息 */
function at(d: GeometryDoc, id: string): [number, number] {
  const g = d.get(id);
  assert.ok(g, `对象 ${id} 不存在`);
  return [g!.x, g!.y];
}

/* ================================================================ 基础度量 */

test("两点距离、夹角、斜率：3-4-5 直角三角形", () => {
  const d = new GeometryDoc();
  const a = d.addPoint("", 0, 0);
  const b = d.addPoint("", 3, 4);
  const c = d.addPoint("", 0, 4);
  near(d.dist(a, b), 5);
  // 顶点 b 处：u=(-3,-4)/5、v=(-1,0) → cos=0.6
  near(d.angleOf(a, b, c), Math.acos(0.6), 1e-12, "∠abc");
  near(d.angleOf(a, c, b), Math.PI / 2, 1e-12, "直角的另一侧");
  // a=(0,0) b=(3,4)：把 (3,4) 挪到 b 的另一侧得到直角 π
  near(d.angleOf(a, b, d.addPoint("", 6, 8)), Math.PI, 1e-12, "共线反向 = π");
});

test("夹角 180° 与 0° 落在 acos 定义域内，不因浮点越界变 NaN", () => {
  // 防的是 clamp1 失效：点积因舍入超过 1 时 acos 返回 NaN，角度标记会整个消失
  const d = new GeometryDoc();
  const o = d.addPoint("", 0, 0);
  const a = d.addPoint("", 1, 0);
  const nearlySame = d.addPoint("", 1 + 1e-7, 1e-7);
  assert.ok(Number.isFinite(d.angleOf(a, o, nearlySame)), "接近零的角必须是有限值");
  assert.ok(d.angleOf(a, o, nearlySame) < 1e-5, "应接近 0");
  near(d.angleOf(a, o, a), 0, 1e-12, "完全同向");
  near(d.angleOf(a, o, d.addPoint("", -1, 0)), Math.PI, 1e-12, "完全反向");
});

test("斜率：竖直线给 Infinity，水平线给 0，无方向的点给 NaN", () => {
  const d = new GeometryDoc();
  const o = d.addPoint("", 0, 0);
  const up = d.addPoint("", 0, 5);
  const right = d.addPoint("", 4, 2);
  assert.equal(d.slope(d.addLine(o, up)), Infinity, "竖直方向 x 分量为 0");
  near(d.slope(d.addLine(o, right)), 0.5);
  assert.ok(Number.isNaN(d.slope(o)), "点没有方向向量");
  assert.ok(Number.isNaN(d.slope("不存在")), "不存在的 id");
});

test("长度：线段/向量取长，圆取周长，圆弧取弧长，椭圆走 Ramanujan 近似", () => {
  const d = new GeometryDoc();
  const o = d.addPoint("", 0, 0);
  near(d.length(d.addSegment(o, d.addPoint("", 3, 4))), 5);
  near(d.length(d.addVector("", o, d.addPoint("", 0, 3))), 3);
  near(d.length(d.addCircle(o, d.addPoint("", 1, 0))), 2 * Math.PI, 1e-12, "周长 2πr");
  // 四分之一圆弧：r=1，扫角 π/2
  const arc = d.addArc("A", o, d.addPoint("", 1, 0), d.addPoint("", 0, 1));
  near(d.length(arc), Math.PI / 2, 1e-12, "弧长 = r·θ");
  // 椭圆周长：π(3(a+b) − √((3a+b)(a+3b)))，a=3 b=2 → π(15−√99)
  const ell = d.addEllipse("E", [0, 0], 3, 2, 0);
  near(d.length(ell), Math.PI * (15 - Math.sqrt(99)), 1e-9, "Ramanujan");
});

test("折线/多边形周长与面积：闭合补一条边，两点与单点的退化", () => {
  const d = new GeometryDoc();
  const a = d.addPoint("", 0, 0);
  const b = d.addPoint("", 4, 0);
  const c = d.addPoint("", 4, 3);
  const tri = d.addPolygon([a, b, c]);
  near(d.polygonArea(tri), 6);
  near(d.length(tri), 12, 1e-12, "3+5+4");
  near(d.length(d.addPolygon([a, b])), 4, 1e-12, "两点：不算闭合边");
  near(d.length(d.addPolygon([a])), 0);
  near(d.polygonArea(d.addPolygon([a, b])), 0, 1e-12, "两点无面积");
  near(d.polygonArea(d.addPolygon([])), 0);
  // 顺/逆时针：有向面积符号相反，取绝对值后一致
  near(d.get(tri)!.a!, 6, 1e-12, "有向面积（逆时针为正）");
  near(d.polygonArea(d.addPolygon([a, c, b])), 6, 1e-12, "反向绕序面积仍为正");
  assert.ok(d.get(d.addPolygon([a, c, b]))!.a! < 0, "反向绕序有向面积为负");
});

test("area/length/perimeter 之外的 measure 类型都落到各自的值上", () => {
  const d = new GeometryDoc();
  const a = d.addPoint("", 0, 0);
  const b = d.addPoint("", 3, 0);
  const c = d.addPoint("", 3, 4);
  const tri = d.addPolygon([a, b, c]);
  const ell = d.addEllipse("E", [0, 0], 2, 1);
  const circ = d.addCircle(a, c);
  const line = d.addLine(a, b);
  const txt = (what: string, ids: string[]) => d.get(d.addMeasure(what as never, ids))!.text;
  assert.equal(txt("area", [tri]), "area = 6.000");
  assert.equal(txt("perimeter", [tri]), "perimeter = 12.000");
  assert.equal(txt("dist", [a, c]), "dist = 5.000");
  assert.equal(txt("angle", [a, b, c]), "angle = 1.571");
  assert.equal(txt("radius", [circ]), "radius = 5.000");
  assert.equal(txt("slope", [line]), "slope = 0.000");
  assert.equal(txt("area", [ell]), "area = 6.283", "椭圆面积 πab");
  // 非多边形/椭圆的对象，area 退化为 0（不是 NaN）
  assert.equal(txt("area", [a]), "area = 0.000");
});

test("measure 值非有限时显示 ? 而不是 NaN/∞ 文本", () => {
  // 防的是测量值在屏幕上显示出 "dist = NaN"
  const d = new GeometryDoc();
  const a = d.addPoint("", 0, 0);
  assert.equal(d.get(d.addMeasure("dist", [a]))!.text, "dist = ?", "id 不足");
  assert.ok(Number.isNaN(d.get(d.addMeasure("dist", [a]))!.a!));
  assert.equal(d.get(d.addMeasure("radius", ["不存在"]))!.text, "radius = ?", "目标不存在");
  // 直线没有有限长度、竖直线斜率是 Infinity —— 两者都必须显示 ?
  const vert = d.addLine(a, d.addPoint("", 0, 1));
  assert.equal(d.get(d.addMeasure("slope", [vert]))!.text, "slope = ?", "Infinity 不是有限值");
  assert.equal(d.get(d.addMeasure("length", [vert]))!.text, "length = ?", "直线长度是 NaN");
});

/* =========================================================== 退化与不存在 */

test("引用不存在的 id：一律给 NaN/null，绝不抛错", () => {
  // 交互不断的前提：recompute 里任何一步拿到坏引用都不能冒泡
  const d = new GeometryDoc();
  const a = d.addPoint("", 0, 0);
  assert.equal(d.get("不存在"), null);
  assert.ok(Number.isNaN(d.dist(a, "不存在")));
  assert.ok(Number.isNaN(d.angleOf("不存在", a, a)));
  assert.ok(Number.isNaN(d.length("不存在")));
  near(d.polygonArea("不存在"), 0, 1e-12, "polygonArea 对缺失 id 返回 0");
  assert.deepEqual(d.pointAtT("不存在", 0.5), null);
  assert.equal(d.paramAt("不存在", [0, 0]), null);
  assert.deepEqual(d.samplePoints("不存在"), []);
  assert.equal(d.ends("不存在"), null);
  assert.equal(d.hitTest(100, 100, 1), null, "空旷处没有可拾取对象");
  assert.deepEqual(d.trace("不存在"), []);
  assert.equal(d.dragTarget("不存在"), false);
  assert.equal(d.canDrive("不存在"), false);
  d.remove("不存在"); // 不能抛
  d.setLabel("不存在", "X");
  d.setVisible("不存在", false);
});

test("重合点做线段：长度为 0，方向兜底成 +x，不产生 NaN", () => {
  // 防的是 norm() 返回 null 后 dir 变 undefined，几何线整条消失
  const d = new GeometryDoc();
  const a = d.addPoint("", 1, 1);
  const b = d.addPoint("", 1, 1);
  const s = d.addSegment(a, b);
  near(d.length(s), 0, 1e-12);
  nearVec(d.get(s)!.dir, [1, 0]);
  nearPt(d.pointAtT(s, 0.5)!, [1, 1]);
  // 退化线段仍可当反射镜：等价于过该点的水平线
  const mirror = d.addSegment(d.addPoint("", 0, 0), d.addPoint("", 0, 0));
  nearPt(at(d, d.addReflect("", a, mirror)), [1, -1]);
  nearVec(d.get(d.addPerpendicular(d.addPoint("", 5, 5), mirror))!.dir, [0, 1]);
});

test("零半径圆：半径 0、周长 0、包围盒退化，但不崩", () => {
  const d = new GeometryDoc();
  const c = d.addPoint("", 2, 3);
  const circ = d.addCircle(c, d.addPoint("", 2, 3));
  near(d.get(circ)!.a!, 0, 1e-12);
  near(d.length(circ), 0, 1e-12);
  nearPt(d.pointAtT(circ, 0.3)!, [2, 3], 1e-12);
  assert.deepEqual(d.bbox(), [2, 2, 3, 3], "单点包围盒");
  // 零半径圆仍能与穿过圆心的直线求交
  const line = d.addLine(d.addPoint("", -1, 3), d.addPoint("", 5, 3));
  const i = d.addIntersection("", circ, line);
  assert.ok(i, "零半径圆与过心直线应有交点");
  nearPt(at(d, i!), [2, 3]);
});

test("NaN 坐标：recompute 保留上一次结果，派生量不被污染成 NaN", () => {
  // 文件头承诺"任一步得到非有限值时保留上一次结果"
  const d = new GeometryDoc();
  const a = d.addPoint("", 1, 2);
  const seg = d.addSegment(a, d.addPoint("", 3, 4));
  const mid = d.addMidpoint("M", seg);
  const perp = d.addPerpendicular(d.addPoint("", 5, 5), seg);
  // 改动前的正确值
  const segDir = d.get(seg)!.dir!.slice() as [number, number];
  const segLen = d.get(seg)!.a!;
  const midBefore = at(d, mid);
  const perpBefore = at(d, perp);
  // 把端点写成 NaN：seg/mid/perp 全部过不了 ok() 守卫，整步跳过 → 保留上一次结果
  d.move(a, NaN, NaN);
  assert.ok(Number.isNaN(d.get(a)!.x), "自由点自身被写成了 NaN（见报告）");
  nearVec(d.get(seg)!.dir, segDir, 1e-12);
  near(d.get(seg)!.a!, segLen, 1e-12, "seg 长度保持旧值");
  assert.deepEqual(at(d, mid), midBefore, "中点保持旧值");
  assert.deepEqual(at(d, perp), perpBefore, "垂线保持旧值");
  assert.ok(
    [...d.get(seg)!.p1!, ...d.get(seg)!.p2!].every(Number.isFinite),
    "派生坐标里不应冒出 NaN",
  );
  // NaN 点不参与包围盒统计
  const d2 = new GeometryDoc();
  d2.addPoint("", 0, 0);
  const bad = d2.addPoint("", 10, 10);
  d2.move(bad, NaN, NaN);
  assert.deepEqual(d2.bbox(), [0, 0, 0, 0], "只剩一个有效点");
});

test("move 写入 NaN 会永久毒化该点：recompute 的守卫救不回来", () => {
  // 实测记录（未修 src）：move() 先把 NaN 写进 g.x/g.y，recompute 的 save 快照因此也已是 NaN，
  // "保留上一次结果"的守卫形同虚设。钉住现状，防止将来无意中被当成正常行为。
  const d = new GeometryDoc();
  const p = d.addPoint("", 3, 4);
  d.move(p, NaN, NaN);
  d.addPoint("", 1, 1); // 触发一次全量 recompute
  assert.ok(Number.isNaN(d.get(p)!.x), "重算后仍是 NaN");
  assert.ok(Number.isNaN(d.dist(p, d.ids()[1])), "距离随之变 NaN");
  d.move(p, Infinity, 1);
  assert.equal(d.get(p)!.x, Infinity, "Infinity 同样不被拦截");
});

/* ================================================================ 包围盒 */

test("bbox：空文档给默认视野，有限值才算数，顺序是 [x0,x1,y0,y1]", () => {
  assert.deepEqual(new GeometryDoc().bbox(), [-5, 5, -4, 4], "空文档默认视野");
  const d = new GeometryDoc();
  d.addPoint("", 1, 2);
  d.addPoint("", -3, 8);
  assert.deepEqual(d.bbox(), [-3, 1, 2, 8], "x0,x1,y0,y1");
  // 只有一个有效点 → 退化成单点
  const d2 = new GeometryDoc();
  d2.addPoint("", 7, 8);
  assert.deepEqual(d2.bbox(), [7, 7, 8, 8]);
  // 圆按半径外扩
  const d3 = new GeometryDoc();
  d3.addCircle(d3.addPoint("", 10, 10), d3.addPoint("", 10, 11));
  assert.deepEqual(d3.bbox(), [9, 11, 9, 11], "圆心 ±r");
  // 中心为 NaN 的椭圆：两次 push 都过不了 ok()，退到默认视野
  const d4 = new GeometryDoc();
  d4.addEllipse("E", [NaN, 0], 3, 2, 0);
  assert.deepEqual(d4.bbox(), [-5, 5, -4, 4]);
});

test("bbox 把椭圆的短半轴 b 当成了 a：视野被高估（实测记录）", () => {
  // 源码 bbox() 对 center 类对象统一按 g.a 外扩，椭圆 b 被忽略。
  // 实测：a=3,b=2 的椭圆给出 [-3,3,-3,3] 而非 [-3,3,-2,2]，视图会多留一圈空白。
  const d = new GeometryDoc();
  const e = d.addEllipse("E", [0, 0], 3, 2, 0);
  assert.deepEqual([d.get(e)!.a, d.get(e)!.b], [3, 2], "半轴本身是对的");
  assert.deepEqual(d.bbox(), [-3, 3, -3, 3], "y 方向用了 a=3 而非 b=2");
});

/* ============================================================ 曲线参数化 */

test("线段/直线/射线上取参数点：t∈[0,1] 落在曲线定义域内", () => {
  const d = new GeometryDoc();
  const a = d.addPoint("", 0, 0);
  const b = d.addPoint("", 10, 0);
  const seg = d.addSegment(a, b);
  nearPt(d.pointAtT(seg, 0)!, [0, 0]);
  nearPt(d.pointAtT(seg, 0.25)!, [2.5, 0]);
  const line = d.addLine(a, b);
  // 直线的参数域是 [p1 − |ab|, p1 + |ab|]，p1 在正中
  nearPt(d.pointAtT(line, 0.5)!, [0, 0]);
  nearPt(d.pointAtT(line, 0)!, [-10, 0]);
  const ray = d.addRay(a, b);
  nearPt(d.pointAtT(ray, 0.5)!, [5, 0], 1e-12, "射线只走正向半段");
  nearPt(d.pointAtT(ray, 0)!, [0, 0]);
  // 非 point 类对象没有参数域
  assert.equal(d.pointAtT(a, 0.5), null);
  assert.equal(d.pointAtT(d.addText(0, 0, "x"), 0.5), null);
});

test("参数归一化：t=1 被折回 0（norm01 = t − floor(t)）", () => {
  // 实测记录：norm01(1)=0，所以开放曲线（线段/直线/射线/多边形/轨迹）的"末端"参数点
  // 会跳回起点。圆/椭圆是闭合曲线，t=1 与 t=0 本就重合，看不出问题。
  const d = new GeometryDoc();
  const seg = d.addSegment(d.addPoint("", 0, 0), d.addPoint("", 10, 0));
  nearPt(d.pointAtT(seg, 0.999)!, [9.99, 0], 1e-9, "0.999 是正常的");
  nearPt(d.pointAtT(seg, 1)!, [0, 0], 1e-12, "t=1 落到起点而不是 (10,0)");
  const q = d.addPointOn("Q", seg, 1);
  assert.equal(d.get(q)!.param, 0, "addPointOn(host, 1) 把参数存成 0");
  nearPt(at(d, q), [0, 0]);
  // 越界参数是取小数部分而不是夹紧：1.5 → 0.5，−0.25 → 0.75
  nearPt(d.pointAtT(seg, 1.5)!, [5, 0], 1e-12);
  nearPt(d.pointAtT(seg, -0.25)!, [7.5, 0], 1e-12);
  nearPt(d.pointAtT(seg, NaN)!, [0, 0], 1e-12, "非有限 t 归一到 0");
  // 闭合曲线不受影响
  const circ = d.addCircle(d.addPoint("", 0, 0), d.addPoint("", 2, 0));
  nearPt(d.pointAtT(circ, 0)!, [2, 0]);
  nearPt(d.pointAtT(circ, 0.25)!, [0, 2], 1e-12);
});

test("paramAt：投影回参数并夹紧到 [0,1]；退化线段返回 0", () => {
  const d = new GeometryDoc();
  const a = d.addPoint("", 0, 0);
  const seg = d.addSegment(a, d.addPoint("", 10, 0));
  near(d.paramAt(seg, [2, 0])!, 0.2, 1e-12);
  near(d.paramAt(seg, [20, 0])!, 1, 1e-12, "越界夹紧");
  near(d.paramAt(seg, [-5, 0])!, 0, 1e-12, "越界夹紧");
  // 竖直方向的偏移不该改变参数
  near(d.paramAt(seg, [2, 99])!, 0.2, 1e-12);
  const circ = d.addCircle(a, d.addPoint("", 2, 0));
  near(d.paramAt(circ, [2, 0])!, 0, 1e-12);
  near(d.paramAt(circ, [0, 2])!, 0.25, 1e-12);
  const deg = d.addSegment(a, d.addPoint("", 0, 0));
  near(d.paramAt(deg, [5, 5])!, 0, 1e-12, "零长线段没有参数方向");
});

test("pointAtT ↔ paramAt 在圆/弧/椭圆/多边形上往返一致", () => {
  // 这是"把点拖到曲线上"与"沿线滑动"的共同基础，往返必须闭合
  const d = new GeometryDoc();
  const o = d.addPoint("", 0, 0);
  const circ = d.addCircle(o, d.addPoint("", 2, 0));
  for (const t of [0, 0.25, 0.5, 0.9]) {
    const q = d.pointAtT(circ, t)!;
    near(d.paramAt(circ, q)!, t, 1e-9, `圆 t=${t}`);
  }
  const arc = d.addArc("A", o, d.addPoint("", 1, 0), d.addPoint("", 0, 1));
  for (const t of [0, 0.25, 0.5, 0.75]) {
    const q = d.pointAtT(arc, t)!;
    near(d.paramAt(arc, q)!, t, 1e-9, `弧 t=${t}`);
    // 弧上取点必须在弧扫角内
    const ang = Math.atan2(q[1] - 0, q[0] - 0);
    assert.ok(ang >= -1e-9 && ang <= Math.PI / 2 + 1e-9, "点落在弧的扫角之外");
  }
  const ell = d.addEllipse("E", [0, 0], 3, 2, 0.4);
  for (const t of [0, 0.13, 0.37, 0.5, 0.81]) {
    near(d.paramAt(ell, d.pointAtT(ell, t)!)!, t, 1e-9, `椭圆 t=${t}`);
  }
  const poly = d.addPolygon([d.addPoint("", 0, 0), d.addPoint("", 4, 0), d.addPoint("", 4, 3)]);
  near(d.paramAt(poly, d.pointAtT(poly, 0.5)!)!, 0.5, 1e-12, "多边形");
  near(d.paramAt(poly, [4, 1.5])!, 0.5, 1e-12, "闭多边形中点参数是 1/2");
});

test("椭圆取参数点：主轴端点、旋转后仍按 rot 定向", () => {
  const d = new GeometryDoc();
  const e = d.addEllipse("E", [0, 0], 3, 2, 0);
  nearPt(d.pointAtT(e, 0)!, [3, 0], 1e-12);
  nearPt(d.pointAtT(e, 0.25)!, [0, 2], 1e-9);
  near(d.polygonArea(e), 6 * Math.PI, 1e-12);
  const d2 = new GeometryDoc();
  const e2 = d2.addEllipse("E", [0, 0], 3, 2, Math.PI / 2);
  nearPt(d2.pointAtT(e2, 0)!, [0, 3], 1e-9, "旋转 90° 后长轴指向 +y");
  near(d2.polygonArea(e2), 6 * Math.PI, 1e-12, "旋转不改变面积");
  // 负半轴取绝对值
  const d3 = new GeometryDoc();
  const e3 = d3.addEllipse("E", [0, 0], -3, -2, 0);
  assert.deepEqual([d3.get(e3)!.a, d3.get(e3)!.b], [3, 2]);
});

test("两焦点定义椭圆：2a = |PF1|+|PF2|，短轴由 c²=a²−b² 反推", () => {
  const d = new GeometryDoc();
  const f1 = d.addPoint("", -2, 0);
  const f2 = d.addPoint("", 2, 0);
  const p = d.addPoint("", 0, 3);
  const e = d.addEllipseByFoci("E", f1, f2, p);
  const g = d.get(e)!;
  // a = (√13+√13)/2 = √13, c = 2, b = √(13−4) = 3
  near(g.a!, Math.sqrt(13), 1e-12, "长半轴");
  near(g.b!, 3, 1e-12, "短半轴");
  near(g.rot!, 0, 1e-12, "长轴沿焦点连线");
  assert.deepEqual(g.center, [0, 0]);
  // 焦点互换不改变结果
  const e2 = d.addEllipseByFoci("E2", f2, f1, p);
  near(d.get(e2)!.b!, 3, 1e-12);
});

test("圆弧：逆时针扫角，起止同点得到整圆", () => {
  const d = new GeometryDoc();
  const o = d.addPoint("", 0, 0);
  const quarter = d.addArc("Q", o, d.addPoint("", 1, 0), d.addPoint("", 0, 1));
  near(d.get(quarter)!.b!, 0, 1e-12);
  near(d.get(quarter)!.rot!, Math.PI / 2, 1e-12);
  const full = d.addArc("F", o, d.addPoint("", 1, 0), d.addPoint("", 1, 0));
  near(d.get(full)!.rot!, 2 * Math.PI, 1e-12, "起止重合 → 整圆");
  near(d.length(full), 2 * Math.PI, 1e-12);
  // 终点角不大于起点角时加一整圈，保证逆时针
  const half = d.addArc("H", o, d.addPoint("", -1, 0), d.addPoint("", 1, 0));
  near(d.get(half)!.rot! - d.get(half)!.b!, Math.PI, 1e-12);
  // samplePoints 沿扫角取 n+1 点，两端恰是起止点
  const sp = d.samplePoints(quarter, 2);
  assert.equal(sp.length, 3);
  nearPt(sp[0], [1, 0], 1e-12, "起点");
  nearPt(sp[1], [Math.SQRT1_2, Math.SQRT1_2], 1e-12, "中点");
  nearPt(sp[2], [0, 1], 1e-12, "终点");
});

/* ================================================================= 求交 */

test("直线×线段：只有落在段内的交点才算数", () => {
  const d = new GeometryDoc();
  const a = d.addPoint("", 0, 0);
  const line = d.addLine(a, d.addPoint("", 10, 0));
  const crossing = d.addSegment(d.addPoint("", 5, -5), d.addPoint("", 5, 5));
  const i = d.addIntersection("", line, crossing);
  assert.ok(i, "交点在段内");
  nearPt(at(d, i!), [5, 0]);
  // 同样的线，交点落在段外 → 判定为不相交，返回 null 且不留下对象
  // 竖段整体在 y>0 一侧，与直线 y=0 不相交（别用穿过 y=0 的段，那会真的相交）
  const far = d.addSegment(d.addPoint("", 50, 5), d.addPoint("", 50, 10));
  const before = d.ids().length;
  assert.equal(d.addIntersection("", line, far), null);
  assert.equal(d.ids().length, before, "失败的求交不留残骸");
});

test("求交失败的四种退化：平行、重合线、点、折线", () => {
  const d = new GeometryDoc();
  const a = d.addPoint("", 0, 0);
  const b = d.addPoint("", 1, 0);
  const c = d.addPoint("", 0, 1);
  const e = d.addPoint("", 1, 1);
  const l1 = d.addLine(a, b);
  const l2 = d.addLine(c, e);
  assert.equal(d.addIntersection("", l1, l2), null, "平行线");
  // 重合的两点定义的两条线：方向叉积为 0
  const same = d.addLine(a, d.addPoint("", 2, 0));
  assert.equal(d.addIntersection("", l1, same), null, "重合直线");
  assert.equal(d.addIntersection("", a, l1), null, "点不是可求交形状");
  const poly = d.addPolygon([a, b, e]);
  assert.equal(d.addIntersection("", poly, l1), null, "多边形走采样近似，不解析求交");
});

test("两圆相交的分支顺序：按极角升序，同角再按半径", () => {
  const d = new GeometryDoc();
  const c1 = d.addCircle(d.addPoint("", 0, 0), d.addPoint("", 1, 0));
  const c2 = d.addCircle(d.addPoint("", 1, 0), d.addPoint("", 2, 0));
  const b0 = d.addIntersection("B0", c1, c2, 0)!;
  const b1 = d.addIntersection("B1", c1, c2, 1)!;
  nearPt(at(d, b0), [0.5, -Math.sqrt(3) / 2], 1e-12, "极角 -60° 排在前");
  nearPt(at(d, b1), [0.5, Math.sqrt(3) / 2], 1e-12, "极角 +60° 排在后");
  assert.deepEqual([d.get(b0)!.branch, d.get(b1)!.branch], [0, 1]);
  // 越界分支号回落到离当前位置最近的候选
  const oob = d.addIntersection("OOB", c1, c2, 7)!;
  assert.equal(d.get(oob)!.branch, 0);
  nearPt(at(d, oob), [0.5, -Math.sqrt(3) / 2], 1e-12);
});

test("两圆相离/同心的判定边界", () => {
  const d = new GeometryDoc();
  const mk = (cx: number, r: number) =>
    d.addCircle(d.addPoint("", cx, 0), d.addPoint("", cx + r, 0));
  const o = d.addPoint("", 0, 0);
  const r1 = d.addCircle(o, d.addPoint("", 1, 0));
  // 圆心距 3 > r1+r2=2 → 相离
  assert.equal(d.addIntersection("", r1, mk(3, 1)), null, "相离");
  // 同心：圆心距 0
  assert.equal(d.addIntersection("", r1, d.addCircle(o, d.addPoint("", 3, 0))), null, "同心");
  // 圆与自己
  assert.equal(d.addIntersection("", r1, r1), null, "自交视作无穷多解，直接判无解");
  // 一个是点（零半径），心距落在另一圆内 → 无交点
  const dot = d.addCircle(d.addPoint("", 0, 9), d.addPoint("", 0, 9));
  assert.equal(d.addIntersection("", r1, dot), null, "零半径点在圆内");
});

test("内切圆有解、外切圆也判有解（相切点的两分支重合）", () => {
  const d = new GeometryDoc();
  const o1 = d.addPoint("", 0, 0);
  const o2 = d.addPoint("", 2, 0);
  // 外切：心距 2 = 1+1，切点 (1,0)
  const outer = d.addIntersection("", d.addCircle(o1, d.addPoint("", 1, 0)), d.addCircle(o2, d.addPoint("", 3, 0)));
  assert.ok(outer, "外切被判为相交");
  nearPt(at(d, outer!), [1, 0], 1e-12);
  // 内切：心距 2 = |3−1|，切点在 c1 反方向 (-1,0)
  const inner = d.addIntersection(
    "",
    d.addCircle(o1, d.addPoint("", 1, 0)),
    d.addCircle(o2, d.addPoint("", 5, 0)), // r=3
  );
  assert.ok(inner, "内切被判为相交");
  nearPt(at(d, inner!), [-1, 0], 1e-12);
});

test("圆弧参与求交时按整圆处理（shapeOf 不区分 arc/circle）", () => {
  const d = new GeometryDoc();
  const o = d.addPoint("", 0, 0);
  const arc = d.addArc("A", o, d.addPoint("", 1, 0), d.addPoint("", 0, 1));
  const diag = d.addLine(d.addPoint("", -5, -5), d.addPoint("", 5, 5));
  const i = d.addIntersection("", arc, diag)!;
  // 四分之一圆弧真正覆盖的是第一象限，但解落在 (-√2/2, -√2/2)
  nearPt(at(d, i), [-Math.SQRT1_2, -Math.SQRT1_2], 1e-9, "返回整圆的解");
  assert.ok(Math.atan2(at(d, i)[1], at(d, i)[0]) < 0, "解不在弧的扫角内");
});

test("椭圆与圆/椭圆求交：真实穿越时一律返回 null（实测记录，未修 src）", () => {
  // 源码 intersect() 的牛顿分支用 f() 判收敛，而 f 闭包捕获的是网格初值 q 而不是收敛点 cur，
  // 于是只有"初值本身就压在对方表面上"（相切）才被收下。3×2 椭圆到原点的距离范围是 [2,3]，
  // 圆半径落在这个开区间内时是真穿越，却恒返回 null。
  for (const r of [2.1, 2.5, 2.9]) {
    const d = new GeometryDoc();
    const c = d.addCircle(d.addPoint("", 0, 0), d.addPoint("", r, 0));
    const e = d.addEllipse("E", [0, 0], 3, 2, 0);
    assert.equal(d.addIntersection("", c, e), null, `r=${r} 明明穿越却判无解`);
  }
  // 相切（r 恰为半轴）的两种情形反而"有解"——因为网格点真的落在交点上
  const t2 = new GeometryDoc();
  assert.ok(t2.addIntersection("", t2.addCircle(t2.addPoint("", 0, 0), t2.addPoint("", 2, 0)), t2.addEllipse("E", [0, 0], 3, 2, 0)));
  const t3 = new GeometryDoc();
  assert.ok(t3.addIntersection("", t3.addCircle(t3.addPoint("", 0, 0), t3.addPoint("", 3, 0)), t3.addEllipse("E", [0, 0], 3, 2, 0)));
  // 同心椭圆 3×2 与 1×4 必然交叉 4 点，同样返回 null
  const e2 = new GeometryDoc();
  assert.equal(e2.addIntersection("", e2.addEllipse("E1", [0, 0], 3, 2, 0), e2.addEllipse("E2", [0, 0], 1, 4, 0)), null);
  // 直线×椭圆走解析二次方程，是好的
  const l2 = new GeometryDoc();
  const ok = l2.addIntersection("", l2.addLine(l2.addPoint("", -5, 0), l2.addPoint("", 5, 0)), l2.addEllipse("E", [0, 0], 3, 2, 0));
  assert.ok(ok, "直线×椭圆应可求交");
  near(Math.abs(at(l2, ok!)[0]), 3, 1e-9, "交点在 ±a 上");
});

/* ================================================================= 变换 */

test("旋转/对称/位似/平移：绕点、绕坐标、绕数字中心", () => {
  const d = new GeometryDoc();
  const s = d.addPoint("", 1, 0);
  nearPt(at(d, d.addRotate("", s, [0, 0], Math.PI / 2)), [0, 1], 1e-12);
  nearPt(at(d, d.addRotate("", s, "p1", Math.PI)), [1, 0], 1e-12, "绕自身转 180° 不动");
  nearPt(at(d, d.addRotate("", s, 5, Math.PI / 2)), [10, 1], 1e-12, "数字中心视作 (5,5)");
  // y 轴反射：线由 (0,0)→(0,1) 给出
  const yAxis = d.addLine(d.addPoint("", 0, 0), d.addPoint("", 0, 1));
  nearPt(at(d, d.addReflect("", s, yAxis)), [-1, 0], 1e-12);
  nearPt(at(d, d.addDilate("", s, [0, 0], 3)), [3, 0], 1e-12);
  nearPt(at(d, d.addDilate("", s, [0, 0], -1)), [-1, 0], 1e-12, "负位似 = 中心对称");
  const from = d.addPoint("", 0, 0);
  const to = d.addPoint("", 0, 2);
  nearPt(at(d, d.addTranslate("", s, from, to)), [1, 2], 1e-12, "向量由 from→to 决定");
});

test("作图工具：垂线、平行线、角平分线方向正确", () => {
  const d = new GeometryDoc();
  const a = d.addPoint("", 0, 0);
  const b = d.addPoint("", 4, 0);
  const top = d.addPoint("", 2, 3);
  const seg = d.addSegment(a, b);
  const perp = d.addPerpendicular(top, seg);
  nearVec(d.get(perp)!.dir, [0, 1], 1e-12);
  assert.deepEqual(d.get(perp)!.p1, [2, 3], "过给定点");
  nearVec(d.get(d.addParallel(top, seg))!.dir, [1, 0], 1e-12);
  // 等腰三角形的顶角平分线竖直向下
  nearVec(d.get(d.addBisector(a, top, b))!.dir, [0, -1], 1e-12);
  assert.equal(d.get(d.addPerpendicular(top, "不存在"))!.dir, undefined, "宿主缺失则不计算");
});

test("三点共线的角平分线退化为垂线（u+v=0 → 兜底 perp）", () => {
  // 防的是 norm(add(u,v)) 返回 null 后 dir 变 undefined
  const d = new GeometryDoc();
  const a = d.addPoint("", 0, 0);
  const b = d.addPoint("", 1, 0);
  const c = d.addPoint("", 2, 0);
  const bis = d.addBisector(a, b, c);
  nearVec(d.get(bis)!.dir, [0, -1], 1e-12);
  assert.equal(d.slope(bis), Infinity);
});

test("addTransform 批量变换：默认参数退化为恒等", () => {
  const d = new GeometryDoc();
  const a = d.addPoint("", 1, 1);
  const b = d.addPoint("", 2, 2);
  const rot = d.addTransform("rotate", [a, b], { center: [0, 0], angle: Math.PI / 2 });
  assert.equal(rot.length, 2);
  assert.deepEqual(d.all().slice(-2).map((g) => g.label), ["P1′", "P2′"], "默认标签是原标签 + ′");
  nearPt(at(d, rot[0]), [-1, 1], 1e-12);
  nearPt(at(d, d.addTransform("dilate", [a])[0]), [1, 1], 1e-12, "缺 ratio → 1");
  nearPt(at(d, d.addTransform("translate", [a])[0]), [1, 1], 1e-12, "缺 vector → (0,0)");
  nearPt(at(d, d.addTransform("rotate", [a])[0]), [1, 1], 1e-12, "缺 angle → 0");
  const custom = d.addTransform("dilate", [a], { labels: ["Z"], ratio: 2 });
  assert.equal(d.get(custom[0])!.label, "Z");
  nearPt(at(d, custom[0]), [2, 2], 1e-12);
});

/* ================================================================ 拾取 */

test("hitTest：点优先于线，点的容差放大 1.6 倍，隐藏对象不参与", () => {
  const d = new GeometryDoc();
  const a = d.addPoint("", 0, 0);
  const seg = d.addSegment(a, d.addPoint("", 10, 0));
  // 点在 1.6 倍容差内 → 命中点；1.7 倍 → 落在线的容差（1 倍）之外
  assert.equal(d.hitTest(0, 1.5, 1), a, "点的容差是 1.6×tol");
  assert.equal(d.hitTest(0, 1.7, 1), null, "超出 1.6×tol");
  assert.equal(d.hitTest(5, 0.9, 1), seg, "线按 1×tol");
  assert.equal(d.hitTest(5, 1.1, 1), null);
  // 点线都在容差内时点优先
  assert.equal(d.hitTest(0, 0.1, 1), a);
  // 隐藏点让位给线
  d.setVisible(a, false);
  assert.equal(d.hitTest(0, 0.1, 1), seg, "隐藏的点不参与拾取");
  d.setVisible(a, true);
  assert.equal(d.hitTest(0, 0.1, 1), a);
  assert.equal(d.hitTest(100, 100, 1), null);
});

test("hitTest：圆按 |到心距 − r|、椭圆按隐式残差×短半轴、弧按扫角", () => {
  const d = new GeometryDoc();
  const c = d.addCircle(d.addPoint("", 0, 0), d.addPoint("", 5, 0));
  // 采样点要远离已有对象，否则点优先规则会先命中
  assert.equal(d.hitTest(0, 4.95, 0.1), c, "落在圆周上");
  assert.equal(d.hitTest(0, 2.5, 0.1), null, "圆心附近不算命中");
  const e = d.addEllipse("E", [0, 0], 3, 2, 0);
  assert.equal(d.hitTest(0, -2, 0.05), e, "短轴端点在椭圆上");
  // (−2.4, 1.2) 满足 x²/9 + y²/4 = 0.64 + 0.36 = 1
  assert.equal(d.hitTest(-2.4, 1.2, 0.05), e, "斜向也在椭圆上");
  assert.equal(d.hitTest(-1, 0, 0.05), null, "椭圆内部不命中");
  const arc = d.addArc("A", d.addPoint("", 0, 0), d.addPoint("", 1, 0), d.addPoint("", 0, 1));
  assert.equal(d.hitTest(0.7, 0.7, 0.05), arc, "扫角内");
  assert.equal(d.hitTest(-0.7, 0.7, 0.05), null, "扫角外不命中");
});

test("samplePoints：圆/弧多一个收尾点（首尾重合），点与文本的退化", () => {
  const d = new GeometryDoc();
  const circ = d.addCircle(d.addPoint("", 0, 0), d.addPoint("", 2, 0));
  const pts = d.samplePoints(circ, 8);
  assert.equal(pts.length, 9, "n+1 个点");
  nearPt(pts[0], pts[8], 1e-12, "首尾重合 → 闭合");
  assert.equal(d.samplePoints(circ).length, 129, "默认 n=128");
  const seg = d.addSegment(d.addPoint("", 0, 0), d.addPoint("", 3, 4));
  assert.deepEqual(d.samplePoints(seg), [[0, 0], [3, 4]]);
  const p = d.addPoint("", 1, 2);
  assert.deepEqual(d.samplePoints(p), [[1, 2]]);
  assert.deepEqual(d.samplePoints(d.addText(1, 2, "hi")), [], "文本没有折线");
});

/* ============================================== 对象表 / 撤销 / 序列化 */

test("id 与自动标签按创建序分配", () => {
  const d = new GeometryDoc();
  const a = d.addPoint("", 0, 0);
  const b = d.addPoint("", 1, 1);
  const seg = d.addSegment(a, b, "AB");
  assert.deepEqual(d.ids(), ["p1", "p2", "s3"], "id = 类型首字母 + 全局递增计数");
  assert.deepEqual(d.all().map((g) => g.label), ["P1", "P2", "AB"], "显式标签优先于自动标签");
  assert.deepEqual(d.get(seg)!.p1, [0, 0]);
  assert.deepEqual(d.get(seg)!.p2, [1, 1]);
  near(d.get(seg)!.a!, Math.SQRT2, 1e-12);
  const auto = d.addEllipse("", [0, 0], 1, 1);
  assert.equal(d.get(auto)!.label, "e4", "省略标签时用 kind 的中文/字母前缀 + 计数");
});

test("remove 级联删掉所有下游对象，clear 清空", () => {
  const d = new GeometryDoc();
  const a = d.addPoint("", 0, 0);
  const b = d.addPoint("", 1, 0);
  const seg = d.addSegment(a, b);
  d.addMidpoint("M", seg);
  const far = d.addPoint("", 5, 5);
  d.addPerpendicular(far, seg);
  d.remove(a);
  assert.deepEqual(d.ids(), [b, far], "线段、中点、垂线全部级联删除");
  d.remove("不存在");
  assert.deepEqual(d.ids(), [b, far]);
  d.clear();
  assert.deepEqual(d.ids(), []);
  assert.deepEqual(d.bbox(), [-5, 5, -4, 4]);
});

test("只有自由点/曲线上的点能拖，派生点 move 被忽略", () => {
  const d = new GeometryDoc();
  const a = d.addPoint("", 0, 0);
  const seg = d.addSegment(a, d.addPoint("", 4, 0));
  const mid = d.addMidpoint("", seg);
  assert.equal(d.dragTarget(a), true);
  assert.equal(d.dragTarget(mid), false, "中点是派生的");
  assert.equal(d.dragTarget(seg), false);
  const before = at(d, mid);
  d.move(mid, 99, 99);
  assert.deepEqual(at(d, mid), before, "move 派生点不生效也不报错");
  d.nudge(a, 2, 3);
  nearPt(at(d, a), [2, 3]);
  d.nudge("不存在", 1, 1);
  assert.equal(d.getParam(a), 0, "自由点没有参数域");
});

test("把点拖到曲线上会反解出参数，paramAt/param 双向一致", () => {
  const d = new GeometryDoc();
  const circ = d.addCircle(d.addPoint("", 0, 0), d.addPoint("", 2, 0));
  const q = d.addPointOn("Q", circ, 0);
  nearPt(at(d, q), [2, 0], 1e-12);
  d.move(q, 0, 2);
  near(d.getParam(q), 0.25, 1e-9, "拖到 (0,2) → t=1/4");
  nearPt(at(d, q), [0, 2], 1e-9);
  d.setParam(q, 0.5);
  nearPt(at(d, q), [-2, 0], 1e-9);
  d.setParam(q, -0.25);
  near(d.getParam(q), 0.75, 1e-12, "setParam 走取小数部分");
  d.setParam(q, NaN);
  near(d.getParam(q), 0, 1e-12, "非有限参数归一到 0");
  d.setParam("不存在", 0.5);
  // 宿主不存在的约束点：停在原点，不崩
  const orph = d.addPointOn("O", "不存在", 0.3);
  nearPt(at(d, orph), [0, 0], 1e-12);
  assert.equal(d.canDrive(orph), false, "宿主没了就没有参数域");
});

test("undo/redo：成对回滚，栈空返回 false，栈深封顶 100", () => {
  const d = new GeometryDoc();
  assert.equal(d.undo(), false, "空栈");
  assert.equal(d.redo(), false);
  d.addPoint("", 0, 0);
  d.addPoint("", 1, 1);
  assert.equal(d.undo(), true);
  assert.deepEqual(d.ids(), ["p1"]);
  assert.equal(d.redo(), true);
  assert.deepEqual(d.ids(), ["p1", "p2"]);
  assert.equal(d.redo(), false, "redo 栈已空");
  // 手势期间重复 beginGesture 不额外压栈，一次 undo 撤掉整段
  d.beginGesture();
  d.beginGesture();
  d.addPoint("", 9, 9);
  d.endGesture();
  assert.equal(d.ids().length, 3);
  d.undo();
  assert.equal(d.ids().length, 2, "手势内多次改动只算一步");
  // 栈深上限
  const d2 = new GeometryDoc();
  d2.addPoint("", 0, 0);
  for (let i = 0; i < 150; i++) d2.addPoint("", i, 0);
  let steps = 0;
  while (d2.undo()) steps++;
  assert.equal(steps, 100, "undo 栈封顶 100 条");
  assert.equal(d2.ids().length, 51, "151 个点撤掉 100 步");
});

test("toJSON/fromJSON 往返：只序列化构造配方，派生字段重算出来", () => {
  const d = new GeometryDoc();
  const a = d.addPoint("A", 0, 0);
  const b = d.addPoint("B", 6, 0);
  const seg = d.addSegment(a, b);
  const circ = d.addCircle(a, b);
  const raw = JSON.parse(d.toJSON());
  assert.deepEqual(Object.keys(raw), ["v", "n", "order", "items"]);
  // free 点上 branch/param/host/center/frozen 都是 undefined，JSON 直接省掉
  assert.deepEqual(Object.keys(raw.items[0]), ["id", "kind", "label", "ctor", "visible", "a", "b", "rot"]);
  const back = GeometryDoc.fromJSON(d.toJSON());
  assert.deepEqual(back.ids(), ["p1", "p2", "s3", "c4"]);
  assert.deepEqual(back.all().map((g) => g.label), ["A", "B", "s3", "c4"]);
  near(back.length(seg), 6, 1e-12, "派生标量被重算");
  near(back.get(circ)!.a!, 6, 1e-12);
  assert.equal(back.addPoint("", 0, 0), "p5", "n 被恢复，新对象不撞 id");
  // 构造配方是完整的来源：只存配方也足以重建
  const rebuilt = GeometryDoc.fromJSON(d.toJSON());
  nearPt(at(rebuilt, seg), [3, 0], 1e-12);
});

/* ================================================================= 轨迹 */

test("轨迹采样：开放宿主 480 份、闭合宿主 720 份，采样后驱动点参数复原", () => {
  const d = new GeometryDoc();
  const seg = d.addSegment(d.addPoint("", 0, 0), d.addPoint("", 1, 0));
  const drv = d.addPointOn("D", seg, 0);
  assert.equal(d.canDrive(drv), true, "曲线上的点可以当驱动");
  assert.equal(d.canDrive(seg), false, "线段本身不是驱动点");
  assert.equal(d.canDrive(d.addPoint("", 5, 5)), false, "自由点没有参数域");
  const tgt = d.addTranslate("", drv, drv, d.addPoint("", 0, 0));
  const locus = d.addLocus("", drv, tgt);
  const pts = d.trace(locus);
  assert.equal(pts.length, 480, "开放宿主 480 份");
  assert.equal(d.get(locus)!.pts!.length, 480);
  near(d.getParam(drv), 0, 1e-12, "采样结束驱动点回到原参数");
  assert.equal(d.trace(locus), pts, "同一修订号内结果按引用缓存");
  d.setParam(drv, 0.5);
  assert.notEqual(d.trace(locus), pts, "修订号变了要重采");
  // 闭合宿主
  const d2 = new GeometryDoc();
  const circ = d2.addCircle(d2.addPoint("", 0, 0), d2.addPoint("", 1, 0));
  const drv2 = d2.addPointOn("D", circ, 0);
  assert.equal(d2.trace(d2.addLocus("", drv2, drv2)).length, 720, "闭合宿主 720 份");
  // 非轨迹对象按已有折线返回
  assert.deepEqual(d.trace(seg), [], "线段没有 pts");
  assert.deepEqual(d.trace("不存在"), []);
});

/* ========================================================= 表达式驱动对象 */

test("表达式点/曲线：编译失败时退化而不是抛错", () => {
  const eng = new Engine();
  const d = new GeometryDoc();
  const p = d.addPointExpr("P", "t", "2*t", eng, "t");
  nearPt(at(d, p), [0, 0], 1e-12, "默认 t=0");
  d.setParam(p, 0.25);
  nearPt(at(d, p), [0.25, 0.5], 1e-12);
  // n 个采样点，t 从 t0 均分到 t1
  const curve = d.addCurveExpr("C", "cos(t*6.283185307179586)", "sin(t*6.283185307179586)", eng, 0, 1, 9);
  const pts = d.get(curve)!.pts!;
  assert.equal(pts.length, 9);
  nearPt(pts[0], [1, 0], 1e-9);
  nearPt(pts[8], [1, 0], 1e-6, "t=1 与 t=0 重合（闭合）");
  // n<2 会被 Math.max(2, n) 兜到 2（n=1 时分母 (n−1)=0 会除零）
  const tiny = d.addCurveExpr("S", "t", "t", eng, 0, 1, 1);
  assert.equal(d.get(tiny)!.pts!.length, 2, "n 被兜到 2");
  nearPt(d.get(tiny)!.pts![0], [0, 0], 1e-12, "t=t0");
  nearPt(d.get(tiny)!.pts![1], [1, 1], 1e-12, "t=t1");
});

test("表达式曲线：无法编译的表达式给空点列，曲线不出现", () => {
  const eng = new Engine();
  const d = new GeometryDoc();
  const bad = d.addCurveExpr("Bad", "没有这个函数(t)", "0", eng, 0, 1, 4);
  assert.deepEqual(d.get(bad)!.pts, [], "编译失败 → 空点列");
  // 未绑定 engine 时同样退化为空
  const bare = new GeometryDoc();
  const p = bare.addPointExpr("Q", "t", "0");
  nearPt(at(bare, p), [0, 0], 1e-12, "编译不了就保持原点");
  assert.equal(bare.get(bad!)?.pts, undefined);
});

test("表达式对象随 JSON 往返需要重新绑定 engine", () => {
  const eng = new Engine();
  const d = new GeometryDoc();
  const p = d.addPointExpr("P", "t", "2*t", eng, "t");
  d.setParam(p, 0.5);
  nearPt(at(d, p), [0.5, 1], 1e-12);
  const back = GeometryDoc.fromJSON(d.toJSON());
  // 实测：反序列化出的表达式对象 x/y 归 0（deser 初始化），不是 NaN —— 静默停在原点
  assert.deepEqual(at(back, p), [0, 0], "没有 engine 时表达式对象静默停在原点");
  const eng2 = new Engine();
  back.bindEngine(eng2);
  nearPt(at(back, p), [0.5, 1], 1e-12, "bindEngine 后重算恢复");
});
