/**
 * GeoLab 数学核心回归验证：`npm run verify`
 * 覆盖：表达式解析、复数、向量/矩阵、数值分析、实数快路径一致性
 */
import { Engine, compile, compileReal, evalString, isRealStatic } from "../src/core/machine.ts";
import { parseExpr } from "../src/core/parser.ts";
import { contourLevels, marchingSquares } from "../src/core/contour.ts";
import { integrateSystem, jacobianAt, equilibria, classifyEquilibrium } from "../src/core/field.ts";
import { domainColor, sampleComplex, newtonFractal, jacobianOfMap } from "../src/core/cplane.ts";
import { buildParametricGrid, projectScene, orbitCamera, surfaceNormals, normalAt } from "../src/core/surface.ts";
import { colormap, COLORMAPS } from "../src/core/colormap.ts";
import { niceTicks, Viewport } from "../src/core/view.ts";
import { GeometryDoc } from "../src/core/geometry.ts";
import { F1 } from "../src/render/scene2d.ts";

let pass = 0;
const fails: string[] = [];

function ok(name: string, cond: boolean, extra = "") {
  if (cond) pass++;
  else fails.push(`${name} ${extra}`);
}
function near(name: string, a: number, b: number, eps = 1e-6) {
  const good = Number.isFinite(a) && Number.isFinite(b) && Math.abs(a - b) <= eps * Math.max(1, Math.abs(b));
  ok(name, good, `期望 ${b}，实际 ${a}`);
}

const eng = new Engine();

/* --- 解析与求值 --- */
near("算术优先级", evalString(eng, "2+3*4").re, 14);
near("幂右结合", evalString(eng, "2^3^2").re, 512);
/* 一元负号只吞紧接着的因子：曾经把 -x^2-y^2 解析成 -(x^2-y^2)，符号整体翻错 */
near("负号与减法", evalString(eng, "-5-2").re, -7);
near("负号与加法", evalString(eng, "-5+2").re, -3);
near("乘法里的负号", evalString(eng, "2*-3-1").re, -7);
ok(
  "负号解析树",
  (() => {
    const n = parseExpr("-1-2");
    return n.type === "bin" && n.op === "-" && n.l.type === "num" && n.l.value === -1 && n.r.type === "num" && n.r.value === 2;
  })(),
);
eng.setNum("qa", 1);
eng.setNum("qb", 1);
near("负号不吞减法", evalString(eng, "-qa^2-qb^2").re, -2);
near("负高斯指数", evalString(eng, "3*exp(-qa^2-qb^2)").re, 3 * Math.exp(-2));
near("隐式乘法", evalString(eng, "2*3").re, 6);
near("取模负值", evalString(eng, "mod(-3,5)").re, 2);
near("括号表达式", evalString(eng, "(1+2)*(3+4)").re, 21);
ok("错误捕获", (() => { try { parseExpr("1+"); return false; } catch { return true; } })());
ok("未定义名称报错", (() => { try { evalString(eng, "qaz+1"); return false; } catch { return true; } })());

const v1 = evalString(eng, "[1,2,3] + [4,5,6]");
ok("向量加法", v1.k === 2 && v1.v!.join(",") === "5,7,9");
near("点积", evalString(eng, "dot([1,2,3],[4,5,6])").re, 32);
ok("叉积", (() => { const c = evalString(eng, "cross([1,0,0],[0,1,0])"); return c.v!.join(",") === "0,0,1"; })());
near("范数", evalString(eng, "norm([3,4])").re, 5);
near("单位化", evalString(eng, "norm(unit([3,4]))").re, 1);
near("行列式", evalString(eng, "det([[1,2],[3,4]])").re, -2);
ok("矩阵逆", (() => { const m = evalString(eng, "[[1,2],[3,4]] * inv([[1,2],[3,4]])"); return Math.abs(m.m![0][0] - 1) < 1e-9; })());
near("线性求解", evalString(eng, "solve([[2,1],[1,3]],[3,4])").v![0], 1, 1e-9);
ok("矩阵索引", (() => { const x = evalString(eng, "[[1,2],[3,4]]"); return x.k === 3 && x.m![1][0] === 3; })());
/* 矩阵作用在向量上按线性映射（结果仍是向量）；矩阵加减支持标量广播 */
ok("矩阵乘向量", (() => { const r = evalString(eng, "[[2,1],[1,3]]*[1,1]"); return r.k === 2 && r.v!.join(",") === "3,4"; })());
ok("矩阵加法", (() => { const r = evalString(eng, "[[1,2],[3,4]]+[[1,1],[1,1]]"); return r.m!.map((q) => q.join(",")).join("|") === "2,3|4,5"; })());
ok("矩阵减标量", (() => { const r = evalString(eng, "[[3,4],[5,6]]-1"); return r.m!.map((q) => q.join(",")).join("|") === "2,3|4,5"; })());
ok("尺寸不一致报错", (() => { try { evalString(eng, "[[1,2],[3,4]]+[1,2,3]"); return false; } catch { return true; } })());
near("平方和", evalString(eng, "sum(k, 1, 10, k^2)").re, 385);
ok("虚数单位不能当循环变量", (() => { try { evalString(eng, "sum(i, 1, 3, i)"); return false; } catch { return true; } })());
near("向量投影长度", evalString(eng, "proj([3,4],[1,0])").v![0], 3, 1e-9);

/* --- 复数 --- */
const z = evalString(eng, "(1+2*i)/(1-i)");
near("复数除法实部", z.re, -0.5);
near("复数除法虚部", z.im, 1.5);
near("欧拉恒等式", evalString(eng, "abs(exp(i*pi))+1").re, 2, 1e-12);
near("共轭", evalString(eng, "conj(3-2i)").im, 2);
near("辐角", evalString(eng, "arg(-1+0i)").re, Math.PI);
near("复平方根", evalString(eng, "sqrt(-4)").im, 2);
near("负数开立方取实根", evalString(eng, "(-8)^(1/3)").re, -2);
near("实数幂", evalString(eng, "(-2)^2").re, 4);
near("i 的幂", evalString(eng, "i^2").re, -1);
ok("复数 sin", (() => { const s = evalString(eng, "sin(1+2i)"); return Math.abs(s.re - Math.sin(1) * Math.cosh(2)) < 1e-12 && Math.abs(s.im - Math.cos(1) * Math.sinh(2)) < 1e-12; })());
near("复数模", evalString(eng, "abs(3+4i)").re, 5);
near("极坐标", evalString(eng, "abs(polar(2, pi/3))").re, 2);

/* --- 用户定义与高阶 --- */
eng.define("f(x) = sin(x)*exp(-x/5)");
eng.define("g(x,y) = x^2*y - y^3/3");
eng.define("amp = 2.5");
near("用户函数", evalString(eng, "f(1)").re, Math.sin(1) * Math.exp(-0.2));
near("常量参数", evalString(eng, "amp*f(0)+1").re, 1);
near("二元函数", evalString(eng, "g(2,1)").re, 4 - 1 / 3);
near("数值导数", evalString(eng, "diff(sin, 1)").re, Math.cos(1), 1e-6);
near("导数记号", evalString(eng, "diff(x^3, 2)").re, 12, 1e-5);
near("二阶导", evalString(eng, "diff(x^3+x, 2, 2)").re, 12, 1e-4);
near("复步导数", evalString(eng, "diff(sin, 0.7)").re, Math.cos(0.7), 1e-9);
near("数值积分", evalString(eng, "integrate(sin, 0, pi)").re, 2, 1e-7);
near("积分多项式", evalString(eng, "integrate(x^2, 0, 3)").re, 9, 1e-8);
near("求根", evalString(eng, "fzero(cos, 1.5)").re, Math.PI / 2, 1e-6);
near("区间求根", evalString(eng, "fzero(x^2-2, [1,2])").re, Math.SQRT2, 1e-8);
near("极限", evalString(eng, "limit(sin(x)/x, 0)").re, 1, 1e-5);
near("求和", evalString(eng, "sum(k, 1, 10, k^2)").re, 385);
near("连乘", evalString(eng, "product(k, 1, 5, k)").re, 120);
ok("序列", (() => evalString(eng, "seq(k, 1, 4, k*2)").v!.join(",") === "2,4,6,8")());
ok("range", (() => evalString(eng, "range(0, 1, 0.25)").v!.length === 5)());
ok("linspace", (() => evalString(eng, "linspace(0,1,11)").v!.length === 11)());
near("分段", evalString(eng, "if(-1>0, 1, -1)").re, -1);
near("隐式乘法", evalString(eng, "2*3*(1+2)").re, 18);
near("隐式乘法含虚数单位", evalString(eng, "abs(2i)+1").re, 3);
near("虚部为零的负实数辐角", evalString(eng, "arg(-1-0i)").re, Math.PI);
near("三角函数积分", evalString(eng, "integrate(sin, 0, pi)").re, 2, 1e-7);
near("余弦求根", evalString(eng, "fzero(cos, 1.5)").re, Math.PI / 2, 1e-6);
near("正弦导数", evalString(eng, "diff(sin, 1)").re, Math.cos(1), 1e-9);
ok("逻辑", (() => evalString(eng, "1<2 && 2<3").b)());
near("阶乘", evalString(eng, "5!").re, 120);
near("gamma", evalString(eng, "gamma(5)").re, 24, 1e-9);
near("最小值", evalString(eng, "min(3,1,2)").re, 1);
const fit = evalString(eng, "polyfit([0,1,2,3],[1,3,5,7], 1)");
near("线性拟合斜率", fit.v![0], 2, 1e-6);
near("线性拟合截距", fit.v![1], 1, 1e-6);

/* --- 编译快路径 --- */
const c1 = compile(eng, parseExpr("sin(x)*exp(-x/5)"), ["x"]);
const r1 = compileReal(eng, parseExpr("sin(x)*exp(-x/5)"), ["x"]);
const sl = new Float64Array(2);
sl[0] = 1.3;
near("编译通用路径", c1.run(sl).re, Math.sin(1.3) * Math.exp(-0.26), 1e-9);
ok("实数快路径可用", r1 !== null);
near("编译用户函数", compile(eng, parseExpr("g(x,y)"), ["x", "y"]).run(new Float64Array([2, 0, 1, 0])).re, 4 - 1 / 3);
ok("复数表达式不走实数快路径", compileReal(eng, parseExpr("x+i"), ["x"]) === null);
ok("实数快路径静态判定", isRealStatic(parseExpr("x^2+sin(x)"), ["x"], eng));

// 两条路径的一致性：抽样 2000 点
{
  const node = parseExpr("x^3 - 3*x + sin(x)*amp");
  const fast = compileReal(eng, node, ["x"])!;
  const slow = compile(eng, node, ["x"]);
  let maxDiff = 0;
  for (let k = 0; k < 2000; k++) {
    const x = -6 + (12 * k) / 2000;
    const s = new Float64Array(2);
    s[0] = x;
    const a = fast(s);
    const b = slow.run(s).re;
    maxDiff = Math.max(maxDiff, Math.abs(a - b));
  }
  ok("双路径一致", maxDiff < 1e-12, `最大偏差 ${maxDiff}`);
}

// 倒数三角族：快路径里是 1/cos 这类显式式子，必须与通用（复数）路径逐点相等
{
  const node = parseExpr("sec(x) + csc(x/2) + cot(2*x)");
  const fast = compileReal(eng, node, ["x"]);
  ok("倒数三角走快路径", fast !== null);
  if (fast) {
    const slow = compile(eng, node, ["x"]);
    let maxDiff = 0;
    for (let k = 0; k < 500; k++) {
      const x = -3 + (6 * k) / 500 + 0.011;
      const s = new Float64Array(2);
      s[0] = x;
      maxDiff = Math.max(maxDiff, Math.abs(fast(s) - slow.run(s).re));
    }
    ok("倒数三角双路径一致", maxDiff < 1e-12, `最大偏差 ${maxDiff}`);
    const secFast = compileReal(eng, parseExpr("sec(x)"), ["x"])!;
    near("sec 与 1/cos 相等", 1 / Math.cos(0.7), secFast(new Float64Array([0.7, 0])), 1e-15);
  }
}

/* --- 性能：曲面采样级别的求值吞吐 --- */
{
  const node = parseExpr("sin(sqrt(x^2+y^2)*3)*cos(atan2(y,x)*2)");
  const fast = compileReal(eng, node, ["x", "y"])!;
  const s = new Float64Array(4);
  const t0 = performance.now();
  let acc = 0;
  for (let i = 0; i < 200; i++)
    for (let j = 0; j < 200; j++) {
      s[0] = i / 20;
      s[1] = j / 20;
      acc += fast(s);
    }
  const dt = performance.now() - t0;
  ok("4 万次二元采样 < 200ms", dt < 200, `耗时 ${dt.toFixed(1)}ms acc=${acc.toFixed(2)}`);
  console.log(`  · 求值吞吐：40000 次二元采样 ${dt.toFixed(1)}ms（${(dt / 40000 * 1000).toFixed(2)}µs/次）`);
}

/* --- 视图与刻度 --- */
{
  const v = new Viewport({ cx: 0, cy: 0, scale: 60, width: 800, height: 600 });
  const p = v.toScreen(1, 2);
  const q = v.toWorld(p[0], p[1]);
  near("视图往返 x", q[0], 1, 1e-9);
  near("视图往返 y", q[1], 2, 1e-9);
  const ts = niceTicks(-7.3, 7.3, 8);
  ok("刻度数量合理", ts.values.length >= 4 && ts.values.length <= 14, JSON.stringify(ts.values));
  ok("刻度步长归一", [1, 2, 2.5, 5, 10].some((b) => Math.abs(ts.step / Math.pow(10, Math.round(Math.log10(ts.step))) - b) < 1e-9));
  const piT = niceTicks(-6.5, 6.5, 8, "pi");
  ok("π 刻度", piT.labels.some((l) => l.includes("π")), piT.labels.join(","));
  const zoom = v.zoomAt(400, 300, 2);
  const [ax, ay] = v.toWorld(400, 300);
  const [asx, asy] = zoom.toScreen(ax, ay);
  near("缩放锚点屏幕位置不变 x", asx, 400, 1e-9);
  near("缩放锚点屏幕位置不变 y", asy, 300, 1e-9);
  near("缩放后世界坐标不变", zoom.toWorld(400, 300)[1], ay, 1e-9);
}

/* --- 隐函数：marching squares --- */
{
  const segs = marchingSquares((x, y) => x * x + y * y - 1, -2, 2, -2, 2, 60, 0);
  ok("圆周边线段数量", segs.length > 60, `${segs.length}`);
  const maxR = Math.max(...segs.map((s) => Math.max(Math.hypot(s[0], s[1]), Math.hypot(s[2], s[3]))));
  const minR = Math.min(...segs.map((s) => Math.min(Math.hypot(s[0], s[1]), Math.hypot(s[2], s[3]))));
  ok("圆半径一致性", Math.abs(maxR - 1) < 0.03 && Math.abs(minR - 1) < 0.03, `${minR}..${maxR}`);
  const lv = contourLevels((x, y) => x * x + y * y, -2, 2, -2, 2, 40, [1, 2, 3]);
  ok("多等高线", lv.length === 3 && lv.every((l) => l.segments.length > 40));
}

/* --- 向量场与 ODE --- */
{
  // 简单系统：dy/dt = -y → y(t)=y0*e^-t
  const sol = integrateSystem((_t, y) => [-y[0]], [0, 1], [1, 1e-4, 5000]);
  near("RK4 指数衰减", sol[sol.length - 1][1], Math.exp(-1), 1e-4);
  // 简谐振子：x'' = -x
  const s2 = integrateSystem((_t, y) => [y[1], -y[0]], [0, 1, 0], [2 * Math.PI, 1e-6, 20000]);
  const last = s2[s2.length - 1];
  near("振子周期回位 x", last[1], 1, 1e-4);
  near("振子周期回位 v", last[2], 0, 1e-4);
  // 逻辑斯蒂
  const s3 = integrateSystem((_t, y) => [y[0] * (1 - y[0])], [0, 0.01], [20, 1e-3, 4000]);
  near("逻辑斯蒂收敛", s3[s3.length - 1][1], 1, 1e-4);
}
{
  // 线性系统零点的雅可比与本征分类
  const J = jacobianAt((v) => [v[0] - 2 * v[1], v[0] + 0.5 * v[1]], [0, 0]);
  ok("雅可比形状", J.length === 2 && J[0].length === 2);
  near("雅可比项", J[0][1], -2, 1e-6);
  const cls = classifyEquilibrium([[0.5, 0], [0, -1]]);
  ok("鞍点判定", cls.type === "saddle", cls.type);
  const cls2 = classifyEquilibrium([[-1, 0], [0, -2]]);
  ok("稳定结点判定", cls2.type === "node" && cls2.stable, cls2.type);
  const cls3 = classifyEquilibrium([[0, -1], [1, 0]]);
  ok("中心判定", cls3.type === "center", cls3.type);
  const eqs = equilibria((v) => [v[0] * (1 - v[1]), v[1] * (v[0] - 1)], [
    [-3, 3],
    [-3, 3],
  ]);
  ok("Lotka-Volterra 零点", eqs.some((e) => Math.abs(e[0] - 1) < 0.2 && Math.abs(e[1] - 1) < 0.2), JSON.stringify(eqs));
}

/* --- 复平面 --- */
{
  const sq = (z: { re: number; im: number }) => ({ re: z.re * z.re - z.im * z.im, im: 2 * z.re * z.im });
  const img = sampleComplex(sq, -2, 2, -2, 2, 64, 64);
  ok("复采样尺寸", img.re.length === 64 * 64 && img.shape[0] === 64);
  let bad = 0;
  for (let j = 0; j < 64; j++)
    for (let i = 0; i < 64; i++) {
      const x = -2 + (4 * i) / 63;
      const y = -2 + (4 * j) / 63;
      const idx = j * 64 + i;
      const er = x * x - y * y, ei = 2 * x * y;
      if (Math.abs(img.re[idx] - er) > 1e-9 || Math.abs(img.im[idx] - ei) > 1e-9) bad++;
    }
  ok("z^2 采样正确", bad === 0, `${bad} 处不符`);
  const cub = (z: { re: number; im: number }) => {
    const den = (z.re - 2) * (z.re - 2) + z.im * z.im;
    const nr = z.re * z.re * z.re - 3 * z.re * z.im * z.im - 1;
    const ni = 3 * z.re * z.re * z.im - z.im ** 3;
    // (nr + i ni) / (z - 2)
    return { re: (nr * (z.re - 2) + ni * z.im) / den, im: (ni * (z.re - 2) - nr * z.im) / den };
  };
  const rgb = domainColor(cub, -3, 3, -3, 3, 40, 30, {});
  ok("共形着色输出", rgb.length === 40 * 30 * 3 && rgb.every((c) => c >= 0 && c <= 255));
  const nb = jacobianOfMap((z) => [z.re * z.re - z.im * z.im, 2 * z.re * z.im], 0.7, 0.3);
  near("映射雅可比行列式", nb.det, 4 * (0.7 * 0.7 + 0.3 * 0.3), 1e-4);
  const nz3 = (z: { re: number; im: number }) => ({
    re: z.re ** 3 - 3 * z.re * z.im ** 2 - 1,
    im: 3 * z.re ** 2 * z.im - z.im ** 3,
  });
  const fr = newtonFractal(nz3, -1.5, 1.5, -1.5, 1.5, 32, 32);
  ok("Newton 分形迭代计数", fr.iter.length === 32 * 32 && fr.root.length === 32 * 32);
  ok("Newton 分形收敛到 3 个根", fr.roots.length === 3, JSON.stringify(fr.roots));
  const roots = fr.root.filter((r) => r >= 0 && r < 3).length;
  ok("Newton 分形像素归属根", roots > 500, `${roots}`);
}

/* --- 曲面 --- */
{
  const g = buildParametricGrid((u, v, o) => o.set(u, v, u * u + v * v), [-1, 1, -1, 1, 12, 12]);
  ok("曲面顶点数", g.positions.length === 12 * 12 * 3);
  ok("曲面索引数", g.tris.length === 11 * 11 * 6);
  const n = surfaceNormals(g);
  ok("法向量已归一", Math.abs(Math.hypot(n[3 * (12 * 12 - 1)], n[3 * (12 * 12 - 1) + 1], n[3 * (12 * 12 - 1) + 2]) - 1) < 1e-6);
  // z=x^2+y^2 在 (1,1) 处法向 ∝ (2,2,-1)
  const gi = 12 * 11 + 11;
  const nn = [n[gi * 3], n[gi * 3 + 1], n[gi * 3 + 2]];
  const expect = normalAt((x, y) => x * x + y * y, 1, 1);
  const dot = nn[0] * expect[0] + nn[1] * expect[1] + nn[2] * expect[2];
  ok("法向与解析梯度一致", Math.abs(Math.abs(dot) - 1) < 1e-3, `${dot.toFixed(4)}`);
  const cam = orbitCamera({ azim: 0.6, elev: 0.5, dist: 4, target: [0, 0, 0], width: 400, height: 300, fov: 1 });
  const proj = projectScene(g.positions, g.tris, cam, 64);
  ok("投影面片数", proj.faces.length > 50, `${proj.faces.length}`);
  ok("面片带深度序", proj.faces.every((f) => Number.isFinite(f.depth)));
}

/* --- 色图 --- */
{
  for (const name of COLORMAPS) {
    const c = colormap(name, 0.5);
    ok(`色图 ${name}`, c.length === 3 && c.every((q) => q >= 0 && q <= 255));
  }
  const a = colormap("parula", 0);
  const b = colormap("parula", 1);
  ok("色图端点不同", a.join() !== b.join());
}

/* --- 几何文档 --- */
{
  const doc = new GeometryDoc();
  const A = doc.addPoint("A", 0, 0);
  const B = doc.addPoint("B", 4, 0);
  const C = doc.addPoint("C", 1, 3);
  const ab = doc.addSegment(A, B);
  const mid = doc.addMidpoint("M", ab);
  const perp = doc.addPerpendicular(mid, ab);
  ok("中点坐标", Math.abs(doc.get(mid)!.x - 2) < 1e-12 && Math.abs(doc.get(mid)!.y) < 1e-12);
  const circle = doc.addCircle(A, B);
  const Dp = doc.addPointOn("D", circle, 0.25);
  ok("圆上点半径", Math.abs(Math.hypot(doc.get(Dp)!.x, doc.get(Dp)!.y) - 4) < 1e-9);
  const inter = doc.addIntersection("P", circle, perp);
  ok("交点存在", inter !== null && Math.abs(doc.get(inter)!.x - 2) < 1e-9, String(inter));
  // 拖动 B → 中点/圆/交点联动
  doc.move(A, 0, 0);
  doc.move(B, 6, 0);
  near("拖动联动中点", doc.get(mid)!.x, 3, 1e-12);
  near("拖动联动半径", Math.hypot(doc.get(Dp)!.x, doc.get(Dp)!.y), 6, 1e-9);
  // 面积随拖动变化（几何不变量校验）
  const poly = doc.addPolygon([A, B, C]);
  doc.move(C, 2, 5);
  const area = doc.polygonArea(poly);
  near("多边形面积", area, 15, 1e-9);
  // 轨迹：D 在圆上运动时，D 与定点 C 的中点轨迹应是以 C/2 为心、半径 3 的圆
  const midDC = doc.addMidpoint("N", doc.addSegment(Dp, C));
  const loc = doc.addLocus("L", Dp, midDC);
  const tr = doc.trace(loc);
  ok("轨迹点积累", tr.length > 200, `${tr.length}`);
  /* 采样收尾必须把驱动点留在它自己的参数上：曾经复原成"最后一个采样位置" */
  const drvAfter = doc.get(Dp)!;
  ok(
    "轨迹采样不留残余",
    Math.abs(drvAfter.x) < 1e-9 && Math.abs(drvAfter.y - 6) < 1e-9,
    `${drvAfter.x.toFixed(4)}, ${drvAfter.y.toFixed(4)}`,
  );
  const cxy = [doc.get(C)!.x / 2, doc.get(C)!.y / 2];
  const maxDev = Math.max(...tr.map((q) => Math.abs(Math.hypot(q[0] - cxy[0], q[1] - cxy[1]) - 3)));
  ok("轨迹为解析圆", maxDev < 1e-6, `最大偏差 ${maxDev}`);
  // 变换：旋转 90° 保持长度
  const rot = doc.addRotate("C'", C, 0, Math.PI / 2);
  near("旋转保距", Math.hypot(doc.get(rot)!.x - 0, doc.get(rot)!.y - 0), Math.hypot(2, 5), 1e-9);
  const refl = doc.addReflect("B'", B, perp);
  ok("反射对称", Math.abs(doc.get(refl)!.x) < 1e-9 && Math.abs(doc.get(refl)!.y) < 1e-9, JSON.stringify(doc.get(refl)));
  ok("约束点才能当轨迹驱动", doc.canDrive(Dp) && !doc.canDrive(A) && !doc.canDrive(ab));
  /* 拖动自由点 → 撤销 → 重做：整条依赖链跟着复原 */
  const nBefore = doc.get(midDC)!.y;
  doc.beginGesture();
  doc.move(C, -3, -4);
  doc.endGesture();
  const nAfter = doc.get(midDC)!.y;
  ok("拖动联动下游中点", Math.abs(nAfter - nBefore) > 1e-6, `${nBefore} → ${nAfter}`);
  ok("撤销回到拖动前", doc.undo() && Math.abs(doc.get(C)!.y - 5) < 1e-9 && Math.abs(doc.get(midDC)!.y - nBefore) < 1e-12);
  ok("重做恢复拖动", doc.redo() && Math.abs(doc.get(C)!.x + 3) < 1e-9 && Math.abs(doc.get(midDC)!.y - nAfter) < 1e-12);
  ok("导出导入", (() => {
    const json = doc.toJSON();
    const d2 = GeometryDoc.fromJSON(json);
    return d2.ids().length === doc.ids().length;
  })());
  // 圆锥曲线
  const ell = doc.addEllipse("E", [0, 0], 3, 1.5, Math.PI / 6);
  const ep = doc.get(ell)!;
  ok("椭圆参数", ep.kind === "ellipse" && Math.abs(ep.a! - 3) < 1e-12);
}

/* --- 绘图层求值封装 F1 --- */
{
  const fe = new Engine();
  const f = new F1(fe, "sin(x)+1", 1);
  near("F1 一元求值", f.at(0), 1);
  ok("F1 合法表达式无名字错误", f.badName() === null);
  const g = new F1(fe, "foo(x)", 1);
  ok("F1 拦下拼错的函数名", /未知函数/.test(g.badName() ?? ""), String(g.badName()));
  ok("F1 求值不抛异常（等值线/场采样约定）", Number.isNaN(g.at(1)) && Number.isNaN(f.at(NaN)));
  const s = new F1(fe, "1/(x-1)", 1);
  ok("F1 奇点不误报", s.badName() === null && !Number.isFinite(s.at(1)));
  const p2 = new F1(fe, "x^2+y^2-4", 2);
  near("F1 二元求值", p2.at(1, 1), -2);
  fe.setNum("pa", 3);
  near("F1 取引擎常量", new F1(fe, "pa*x", 1).at(2), 6);
}

console.log(`\n验证结束：通过 ${pass} 项${fails.length ? `，失败 ${fails.length} 项：` : "，全部通过"}`);
for (const f of fails) console.log("  ✗ " + f);
if (fails.length) process.exitCode = 1;
