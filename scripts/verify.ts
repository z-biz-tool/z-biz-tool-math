/**
 * GeoLab 数学核心回归验证：`npm run verify`
 * 覆盖：表达式解析、复数、向量/矩阵、数值分析、实数快路径一致性
 */
import { Engine, compile, compileCplx, compileReal, evalString, isRealStatic } from "../src/core/machine.ts";
import { readFileSync } from "node:fs";
import { parseExpr } from "../src/core/parser.ts";
import { contourLevels, marchingSquares, traceContours } from "../src/core/contour.ts";
import { integrateSystem, jacobianAt, equilibria, classifyEquilibrium } from "../src/core/field.ts";
import { domainColor, sampleComplex, newtonFractal, newtonPlan, jacobianOfMap } from "../src/core/cplane.ts";
import * as CN from "../src/core/cnum.ts";
import { buildParametricGrid, projectScene, orbitCamera, surfaceNormals, normalAt } from "../src/core/surface.ts";
import { colormap, COLORMAPS } from "../src/core/colormap.ts";
import { logTicks, niceTicks, Viewport } from "../src/core/view.ts";
import { GeometryDoc } from "../src/core/geometry.ts";
import { F1, rasterJobFor, rasterKeyOf, stepRasterJob } from "../src/render/scene2d.ts";
import { useStore, type Mode } from "../src/state.ts";
import { parseMatrix, parseVector } from "../src/core/parsemat.ts";
import { PRESETS, applyPreset } from "../src/presets.ts";
import {
  identity, matMul, matScale, transpose, det, solve, inverse, rref, eigen2, eigen, charCoeffs,
  jacobiEigen, isSymmetric, eigenVector, svd, cubicRoots, expMat2, matVec,
} from "../src/core/linalg.ts";
import type { C2, Mat } from "../src/core/linalg.ts";
import {
  rng, createModel, makeCache, forward, activations, makeGrad, zeroGrad, sampleGrad, trainEpoch,
  predict, predictMargin, accuracy, dataLoss, dataset, DATASETS, decisionRow, decisionRaster,
} from "../src/core/nn.ts";
import type { Cache, Model } from "../src/core/nn.ts";

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

/** 复变函数：与渲染层 cfn 同一条编译路径，保证逐位一致 */
const slot = new Float64Array(2);
function cf(src: string): (z: { re: number; im: number }) => { re: number; im: number } {
  const c = compile(eng, parseExpr(src), ["z"]);
  return (z) => {
    slot[0] = z.re;
    slot[1] = z.im;
    const v = c.run(slot);
    return { re: v.re, im: v.im };
  };
}

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

/* --- 对数轴：轴空间（log10）里的线性变换 --- */
{
  const lin = new Viewport({ cx: 0, cy: 0, scale: 60, width: 800, height: 600 });
  ok("线性视口不写 log 标记", !("logX" in lin.toJSON()) && !("logY" in lin.toJSON()));
  ok("线性轴映射为恒等", lin.toAxisX(-3.5) === -3.5 && lin.fromAxisY(0.25) === 0.25);

  const lg = lin.withLog(true, true);
  ok("切换对数轴后中心回到 1", lg.cx === 0 && lg.cy === 0);
  ok("对数轴视口恒正", lg.left > 0 && lg.bottom > 0);
  near("一个十倍频程 = scale 像素", lg.toScreen(10, 0)[0] - lg.toScreen(1, 0)[0], lg.scale, 1e-9);
  const rt = lg.toWorld(...lg.toScreen(1e-4, 7));
  near("对数轴往返 x", rt[0], 1e-4, 1e-18);
  near("对数轴往返 y", rt[1], 7, 1e-12);
  ok("非正的 x 不产生 NaN", Number.isFinite(lg.toScreen(0, 1)[0]) && Number.isFinite(lg.toScreen(-5, 1)[0]));
  ok("0 与负值都落在屏幕左外", lg.toScreen(0, 1)[0] < 0 && lg.toScreen(-5, 1)[0] < 0);
  ok("toJSON 往返保留轴制式", (() => {
    const j = lg.toJSON();
    const r = new Viewport(j);
    return r.logX && r.logY && r.scale === lg.scale && r.right === lg.right;
  })());

  // 缩放与平移在轴空间仍然是线性的：锚点世界值不变，跨度按十倍频程计
  const z = lg.zoomAt(300, 220, 3.7);
  const [wxa, wsp] = [lg.toWorld(300, 220), z.toScreen(...lg.toWorld(300, 220))];
  near("对数轴缩放锚点 x", wsp[0], 300, 1e-6);
  near("对数轴缩放锚点 y", wsp[1], 220, 1e-6);
  near("对数轴缩放保持世界值", z.toWorld(300, 220)[0], wxa[0], 1e-12);
  ok("缩放不丢轴制式", z.logX && z.logY);
  const decades = Math.log10(z.right / z.left);
  near("视口宽度=十倍频程数", decades, z.width / z.scale, 1e-9);

  // 拟合：把 [1e-2, 1e6] × [1e-3, 1e3] 塞进视口
  const f = lg.fit(1e-2, 1e6, 1e-3, 1e3);
  ok("对数轴 fit 覆盖目标区间", f.left <= 1e-2 * 1.001 && f.right >= 1e6 * 0.999, `${f.left}..${f.right}`);
  ok("对数轴 fit 纵向同样", f.bottom <= 1e-3 * 1.001 && f.top >= 1e3 * 0.999, `${f.bottom}..${f.top}`);

  // 采样按等倍率前进（对数轴上等距即等比）
  const u0 = f.toAxisX(f.left);
  const du = (f.toAxisX(f.right) - u0) / 8;
  const ratios: number[] = [];
  for (let i = 1; i <= 8; i++) ratios.push(f.fromAxisX(u0 + du * i) / f.fromAxisX(u0 + du * (i - 1)));
  ok("对数采样等比", ratios.every((r) => Math.abs(r - ratios[0]) < 1e-9), ratios.join(","));
}

/* --- 对数刻度与 loglog 等值线 --- */
{
  const t = niceTicks(1e-3, 1e3, 8, "log");
  ok("主刻度落在整十倍频", t.values.length === 7 && t.values.every((v, i) => Math.abs(Math.log10(v) + 3 - i) < 1e-9), t.values.join(","));
  ok("10ⁿ 标签", t.labels[0] === "10⁻³" && t.labels[3] === "1" && t.labels[6] === "10³", t.labels.join(","));
  ok("次级网格按 2/5", t.minor.length >= 2 && t.minor.some((v) => Math.abs(v - 0.002) < 1e-12), t.minor.slice(0, 4).join(","));
  ok("对数刻度无统一步长", t.step === 0);
  const wide = niceTicks(1e-10, 1e10, 8, "log");
  ok("跨 20 个十倍频时抽稀", wide.values.length <= 9 && wide.values.length >= 7, String(wide.values.length));
  const zoomed = niceTicks(2, 7, 8, "log");
  ok("不足一个十倍频时提倍数为标签", zoomed.values.length === 6 && zoomed.labels.every((l) => l.length > 0), zoomed.labels.join(","));
  ok("log 模式即 logTicks", JSON.stringify(t) === JSON.stringify(logTicks(1e-3, 1e3, 8)));

  // loglog 下 y = x² 的对数像 v = 2u 是直线：等值线映回屏幕后必须共线
  const vp = new Viewport({ cx: 2, cy: 2, scale: 120, width: 900, height: 600, logX: true, logY: true });
  // loglog 下 y = x² 的等值线应当摊成一条直线（斜率即指数），线性轴下明显是弯的
  const isoline = (v: Viewport, res: number) => {
    const ch = traceContours(
      (u, w) => v.fromAxisY(w) - v.fromAxisX(u) ** 2,
      v.axisLeft,
      v.axisRight,
      v.axisBottom,
      v.axisTop,
      res,
      0,
      3,
    );
    return ch
      .flat()
      .flatMap((s) => [
        v.toScreen(v.fromAxisX(s[0]), v.fromAxisY(s[1])),
        v.toScreen(v.fromAxisX(s[2]), v.fromAxisY(s[3])),
      ]);
  };
  /** 以两端点连线为基准，返回最大偏离与斜率 */
  const straightness = (pts: number[][]) => {
    const byX = pts.slice().sort((a, b) => a[0] - b[0]);
    const [p0, p1] = [byX[0], byX[byX.length - 1]];
    const k = (p1[1] - p0[1]) / (p1[0] - p0[0]);
    const b = p1[1] - k * p1[0];
    return { k, dev: Math.max(...pts.map(([x, y]) => Math.abs(y - (k * x + b)))) };
  };
  {
    const pts = isoline(vp, 120);
    const on = straightness(pts);
    ok("loglog 幂律有等值线", pts.length > 200, String(pts.length));
    ok("loglog 幂律近乎直线", on.dev < 1.5, `最大偏离 ${on.dev}`);
    near("直线斜率 = -2（screen 上指数取负）", on.k, -2, 1e-2);
    const lin = new Viewport({ cx: 2, cy: 2, scale: 120, width: 900, height: 600 });
    const off = straightness(isoline(lin, 400));
    ok("线性轴下同一曲线是弯的", off.dev > 20, `最大偏离 ${off.dev}`);
  }
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
  const conv: number[] = [];
  for (let k = 0; k < fr.iter.length; k++) if (fr.root[k] >= 0) conv.push(fr.iter[k]);
  const fast = conv.filter((v) => v < 12).length;
  ok(
    "Newton 记的是真实收敛步数（未被常数拉平）",
    conv.length > 0 && fast / conv.length > 0.5,
    `快收敛 ${fast}/${conv.length}，最大步数 ${Math.max(...conv)}`,
  );
}

/* --- 分带续算：逐行采样必须与整幅一次采样逐字节相同（回补靠这条保证不跳变） --- */
{
  /* 场景约定：图像第 0 行对应世界 y 的上界，所以采样区间传 (top → bottom) */
  const W = 48;
  const H = 26;
  const TOP = 2.4;
  const BOT = -2.4;
  const dy = (BOT - TOP) / (H - 1);
  const fd = cf("z^3-1");
  const dmOpts = { dark: true, alpha: true, levelStep: 1, saturation: 0.85 };
  const dmBad = (() => {
    const ref = domainColor(fd, -3, 3, TOP, BOT, W, H, dmOpts);
    let bad = 0;
    for (let j = 0; j < H; j++) {
      const y = TOP + dy * j;
      const one = domainColor(fd, -3, 3, y, y + dy, W, 1, dmOpts);
      for (let i = 0; i < one.length; i++) if (ref[j * one.length + i] !== one[i]) bad++;
    }
    return bad;
  })();
  ok("共形着色逐行合成 == 整幅采样", dmBad === 0, `差 ${dmBad} 字节 / ${W * H * 3}`);
  const itBad = (() => {
    const ref = domainColor(fd, -3, 3, TOP, BOT, W, H, { dark: true, alpha: true, iterFn: fd });
    let bad = 0;
    for (let j = 0; j < H; j++) {
      const y = TOP + dy * j;
      const one = domainColor(fd, -3, 3, y, y + dy, W, 1, { dark: true, alpha: true, iterFn: fd });
      for (let i = 0; i < one.length; i++) if (ref[j * one.length + i] !== one[i]) bad++;
    }
    return bad;
  })();
  ok("迭代吸引盆逐行合成 == 整幅采样", itBad === 0, `差 ${itBad} 字节 / ${W * H * 3}`);
  const fnw = cf("z^3-1");
  const plan = newtonPlan(fnw, -3, 3, TOP, BOT);
  const nw = 40;
  const nh = 20;
  const nDy = (BOT - TOP) / (nh - 1);
  const nwBad = (usePlan: boolean) => {
    const o = usePlan ? { maxIter: 64, plan } : { maxIter: 64 };
    const ref = newtonFractal(fnw, -3, 3, TOP, BOT, nw, nh, o);
    let iter = 0;
    let root = 0;
    for (let j = 0; j < nh; j++) {
      const y = TOP + nDy * j;
      const one = newtonFractal(fnw, -3, 3, y, y + nDy, nw, 1, o);
      for (let i = 0; i < nw; i++) {
        if (ref.iter[j * nw + i] !== one.iter[i]) iter++;
        if (ref.root[j * nw + i] !== one.root[i]) root++;
      }
    }
    return { iter, root };
  };
  const shared = nwBad(true);
  ok(
    "Newton 逐行合成 == 整幅采样（共用根方案）",
    shared.iter === 0 && shared.root === 0,
    `步数差 ${shared.iter} 格、归属差 ${shared.root} 格`,
  );
  /* 不共用根方案就会漂：每行都按自己的退化包围盒重新发现根 */
  const alone = nwBad(false);
  ok(
    "Newton 逐行必须复用整幅根方案",
    alone.root > shared.root,
    `各自行自采样差 ${alone.root} 格，复用 plan 差 ${shared.root} 格`,
  );
}

/* --- 停手后的分带回补：产品自己的续算任务 --- */
{
  const home = useStore.getState();
  const homeView = home.views.complex;
  home.setMode("complex");
  home.setCplx({ mode: "domain", f: "z^3-1", iterative: false, levelStep: 1, resolution: 2 });
  const s = useStore.getState();
  const vp = s.views.complex;
  const job = rasterJobFor(s, vp.width, vp.height);
  ok("栅格续算任务可建立", !!job && job.rows === 0 && job.rgb.length === job.w * job.h * 3, job ? `${job.w}×${job.h}` : "null");
  if (job) {
    /* 预算 0：一次调用只采一行就让出，正是逐带续算的最坏节奏 */
    let ticks = 0;
    while (stepRasterJob(s, job, 0) === "more" && ticks++ < 100000) {
      /* 逐带推进 */
    }
    ok("逐带采样能画满", job.rows === job.h, `${job.rows}/${job.h} 行、让出 ${ticks} 次`);
    const ref = domainColor(
      cf("z^3-1"),
      vp.left,
      vp.right,
      vp.top,
      vp.bottom,
      job.w,
      job.h,
      { dark: s.settings.dark, alpha: true, levelStep: s.cplx.levelStep, saturation: 0.85 },
    );
    let bad = 0;
    for (let i = 0; i < ref.length; i++) if (ref[i] !== job.rgb[i]) bad++;
    ok("分带续算与整幅一次采样逐字节相同", bad === 0, `差 ${bad} 字节 / ${ref.length}`);
    const key = job.key;
    ok("视口未动时身份仍有效", rasterKeyOf(s, vp.width, vp.height) === key);
    s.setView("complex", vp.panPixels(9, 0));
    ok("视口一挪，续算任务即过期", stepRasterJob(useStore.getState(), job, 1) === "stale");
    ok("视口一挪，采样身份即改变", rasterKeyOf(useStore.getState(), vp.width, vp.height) !== key);
  }
  const back = useStore.getState();
  back.setCplx({ mode: "map" });
  ok("保角映射（矢量描边）不走续算", rasterJobFor(useStore.getState(), 400, 300) === null);
  back.setMode("func");
  back.setCplx({ mode: "domain" });
  ok("非复平面模式不走续算", rasterJobFor(useStore.getState(), 400, 300) === null);
  const rst = useStore.getState();
  rst.setMode(home.mode);
  rst.setView("complex", homeView);
  rst.setCplx({ mode: home.cplx.mode, f: home.cplx.f, iterative: home.cplx.iterative, resolution: home.cplx.resolution, levelStep: home.cplx.levelStep });
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
  /* 参数滑块里的名字不能被当成自变量占槽位，否则求值时会被视口 y 悄悄覆盖，滑块失效 */
  const clob = new F1(fe, "x^2 - pa", 2);
  ok("已定义的全局量不占自变量槽位", clob.names.join(",") === "x,y", clob.names.join(","));
  near("隐函数里的参数取滑块值", clob.at(1, 5), -2);
  /* 反过来：没有定义成全局量的名字仍然按自变量绑定，别把老写法改坏 */
  const free = new F1(fe, "p^2 + q^2 - 4", 2);
  near("陌生名字仍按自变量绑定", free.at(2, 3), 9);
}

/* --- 线性代数内核 linalg --- */
{
  /** 自带的线性同余随机源：内核的 rng 属于 nn 模块，回归里两模块要互不依赖 */
  const laRand = (seed: number): (() => number) => {
    let a = seed >>> 0;
    return () => {
      a = (Math.imul(a, 1664525) + 1013904223) >>> 0;
      return a / 4294967296;
    };
  };
  const laSketch = (rows: number, cols: number, f: () => number, span = 3): Mat =>
    Array.from({ length: rows }, () => Array.from({ length: cols }, () => f() * span - span / 2));
  const laDiff = (a: Mat, b: Mat): number => {
    let d = 0;
    for (let i = 0; i < a.length; i++) for (let j = 0; j < a[i].length; j++) d = Math.max(d, Math.abs(a[i][j] - b[i][j]));
    return d;
  };
  const laTrace = (a: Mat): number => a.reduce((s, row, i) => s + row[i], 0);
  const laThrows = (f: () => unknown): boolean => {
    try {
      f();
    } catch {
      return true;
    }
    return false;
  };
  /** 独立实现：教科书三重循环，用来核对 matMul 的 k-外层累加写法 */
  const laMulRef = (a: Mat, b: Mat): Mat => {
    const ac = a[0].length;
    const bc = b[0].length;
    return Array.from({ length: a.length }, (_, i) =>
      Array.from({ length: bc }, (_, j) => {
        let s = 0;
        for (let k = 0; k < ac; k++) s += a[i][k] * b[k][j];
        return s;
      }),
    );
  };
  /** 独立实现：按第一行余子式展开，用来核对 det 的 LU 路径 */
  const laDetRef = (a: Mat): number => {
    if (a.length === 1) return a[0][0];
    let s = 0;
    for (let j = 0; j < a.length; j++) s += (j % 2 ? -1 : 1) * a[0][j] * laDetRef(a.slice(1).map((row) => row.filter((_, c) => c !== j)));
    return s;
  };
  const laCMul = (a: C2, b: C2): C2 => ({ re: a.re * b.re - a.im * b.im, im: a.re * b.im + a.im * b.re });
  /** 独立实现：复数 Horner，cs 从最高次到常数项 */
  const laCPoly = (cs: number[], z: C2): C2 => {
    let acc: C2 = { re: 0, im: 0 };
    for (const c of cs) {
      const m = laCMul(acc, z);
      acc = { re: m.re + c, im: m.im };
    }
    return acc;
  };
  const laCRes = (cs: number[], roots: C2[]): number =>
    Math.max(...roots.map((z) => { const p = laCPoly(cs, z); return Math.hypot(p.re, p.im); }));

  /* 行列式 */
  const laD3: Mat = [[4, 1, 2], [1, 3, 5], [2, 5, 1]];
  near("det 三阶按手算展开", det(laD3), -81, 1e-12);
  ok("det 奇异矩阵归零", Math.abs(det([[1, 2, 3], [2, 4, 6], [1, 1, 1]])) < 1e-12, `${det([[1, 2, 3], [2, 4, 6], [1, 1, 1]])}`);
  const laD4 = laSketch(4, 4, laRand(7), 2);
  near("det 四阶与余子式展开同值", det(laD4), laDetRef(laD4), 1e-12);

  /* 乘法 */
  const laMR = laRand(11);
  const laA = laSketch(3, 4, laMR);
  const laB = laSketch(4, 2, laMR);
  const la5a = laSketch(5, 5, laMR);
  const la5b = laSketch(5, 5, laMR);
  ok("matMul 对照三重循环（3×4 乘 4×2）", laDiff(matMul(laA, laB), laMulRef(laA, laB)) === 0);
  ok("matMul 对照三重循环（5×5）", laDiff(matMul(la5a, la5b), laMulRef(la5a, la5b)) < 1e-12);
  const laX = laSketch(4, 4, laRand(21));
  const laY = laSketch(4, 4, laRand(22));
  const laZ = laSketch(4, 4, laRand(23));
  const laAsso = laDiff(matMul(laX, matMul(laY, laZ)), matMul(matMul(laX, laY), laZ));
  ok("matMul 结合律 A(BC)=(AB)C", laAsso < 1e-9, `${laAsso.toExponential(2)}`);
  ok("matMul 右乘单位阵不变", laDiff(matMul(laX, identity(4)), laX) === 0);
  ok("matMul 尺寸不匹配抛错", laThrows(() => matMul([[1, 2, 3], [4, 5, 6]], [[1, 2], [3, 4]])));

  /* 逆矩阵 */
  const laInvErr = (a: Mat): number => {
    const i = inverse(a);
    const idn = identity(a.length);
    return Math.max(laDiff(matMul(a, i), idn), laDiff(matMul(i, a), idn));
  };
  const laI2 = laSketch(2, 2, laRand(31));
  const laI3 = laD3;
  ok("inverse 双侧 A·A⁻¹=I（2 阶）", laInvErr(laI2) < 1e-9, `${laInvErr(laI2).toExponential(2)}`);
  ok("inverse 双侧 A·A⁻¹=I（3 阶）", laInvErr(laI3) < 1e-9, `${laInvErr(laI3).toExponential(2)}`);
  ok("inverse 双侧 A·A⁻¹=I（4 阶）", laInvErr(laX) < 1e-9, `${laInvErr(laX).toExponential(2)}`);
  ok("inverse 奇异矩阵抛错", laThrows(() => inverse([[1, 2], [2, 4]])));

  /* 线性方程组：主元为 0 时必须换行，且右端项要跟着置换 */
  const laPA: Mat = [[0, 1, 2], [1, 3, 5], [2, 5, 1]];
  const laPX = [1, -2, 3];
  const laPB = matVec(laPA, laPX);
  const laPS = solve(laPA, laPB);
  const laPRes = Math.max(...matVec(laPA, laPS).map((v, i) => Math.abs(v - laPB[i])));
  ok("solve 首列主元为零仍能解（右端项随行置换）", laPRes < 1e-9, `残差 ${laPRes.toExponential(2)}，解 ${JSON.stringify(laPS)}`);
  const laP5 = laSketch(5, 5, laRand(41), 2);
  laP5[0][0] = 0;
  laP5[1][1] = 1e-9;
  const laX5 = [1, 2, 3, 4, 5];
  const laB5 = matVec(laP5, laX5);
  const laS5 = solve(laP5, laB5);
  ok("solve 5 阶近退化主元残差", Math.max(...matVec(laP5, laS5).map((v, i) => Math.abs(v - laB5[i]))) < 1e-9);

  /* 简化行阶梯形 */
  const laR2 = rref([[1, 2, 3], [2, 4, 6], [1, 1, 1]]);
  ok("rref 秩为 2", laR2.rank === 2, `${laR2.rank}`);
  const laNull = [1, -2, 1];
  ok(
    "rref 阶梯形保零空间",
    laR2.m.every((row) => Math.abs(row.reduce((s, v, i) => s + v * laNull[i], 0)) < 1e-12) && 3 - laR2.rank === 1,
  );
  ok("rref 单位阵不变", laDiff(rref(identity(3)).m, identity(3)) === 0);
  const laRZ = rref([[1, 0, 2], [0, 0, 1], [1, 0, 3]]);
  ok(
    "rref 零列被跳过、主元列正确",
    laRZ.pivots.join(",") === "0,2" && laRZ.rank === 2 && laRZ.m.every((row) => row[1] === 0),
    `pivots=${laRZ.pivots.join(",")}`,
  );

  /* 2×2 闭式特征值 */
  const laE2R = eigen2([[0, -1], [1, 0]]);
  ok(
    "eigen2 90° 旋转给出 ±i",
    laE2R.length === 2 && laE2R.some((z) => z.re === 0 && z.im === 1) && laE2R.some((z) => z.re === 0 && z.im === -1),
    JSON.stringify(laE2R),
  );
  const laE2D = eigen2([[3, 0], [0, -1]]);
  ok("eigen2 对角阵特征值即对角元", laE2D.every((z) => z.im === 0) && laE2D.map((z) => z.re).sort((x, y) => y - x).join(",") === "3,-1");
  const laE2S = eigen2([[1, 1], [0, 1]]);
  ok("eigen2 剪切阵二重实根", laE2S.every((z) => z.re === 1 && z.im === 0), JSON.stringify(laE2S));

  /* 一般特征值：迹、行列式、特征多项式三条独立约束 */
  const laEigenCases: { name: string; a: Mat }[] = [
    { name: "2 阶旋转缩放", a: [[0, -2], [2, 0]] },
    { name: "2 阶实根", a: [[1, 2], [3, -1]] },
    { name: "3 阶对称", a: [[2, -1, 0], [-1, 2, -1], [0, -1, 2]] },
    { name: "3 阶复共轭对", a: [[0, -1, 0], [1, 0, 0], [0, 0, 2]] },
    { name: "4 阶对称", a: [[1, 2, 3, 4], [2, 3, 4, 1], [3, 4, 1, 2], [4, 1, 2, 3]] },
    { name: "4 阶友矩阵", a: [[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [2, 3, 4, 5]] },
    { name: "4 阶两对复根", a: [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 2, -1], [0, 0, 1, 2]] },
  ];
  for (const c of laEigenCases) {
    const ls = eigen(c.a);
    const cs = charCoeffs(c.a);
    const sumRe = ls.reduce((s, z) => s + z.re, 0);
    const prodAbs = ls.reduce((s, z) => s * Math.hypot(z.re, z.im), 1);
    const res = laCRes(cs, ls);
    near(`eigen ${c.name}：ΣRe(λ)=迹`, sumRe, laTrace(c.a), 1e-9);
    near(`eigen ${c.name}：Π|λ|=|det|`, prodAbs, Math.abs(det(c.a)), 1e-9);
    ok(`eigen ${c.name}：每个根满足特征多项式`, ls.length === c.a.length && res < 1e-8, `残差 ${res.toExponential(2)}`);
  }
  /* 排序约定只在多项式路径上成立：对称阵走 Jacobi（降序）、2 阶走闭式，都不做重排 */
  const laSorted = laEigenCases.filter((c) => !isSymmetric(c.a) && c.a.length >= 3);
  ok(
    "eigen 多项式路径按（实部, 虚部）升序",
    laSorted.length === 3 &&
      laSorted.every((c) => {
        const ls = eigen(c.a);
        return ls.every((z, i) => i === 0 || ls[i - 1].re < z.re || (ls[i - 1].re === z.re && ls[i - 1].im <= z.im));
      }),
    laSorted.map((c) => JSON.stringify(eigen(c.a))).join(" "),
  );
  ok(
    "eigen 对称阵沿用 Jacobi 降序",
    eigen(laEigenCases[2].a).every((z, i, all) => i === 0 || all[i - 1].re >= z.re),
    JSON.stringify(eigen(laEigenCases[2].a)),
  );
  ok("isSymmetric 判别", isSymmetric([[1, 2], [2, 1]]) && !isSymmetric([[1, 2], [3, 4]]) && !isSymmetric([[1, 2, 3], [4, 5, 6]]));
  ok("eigen 非方阵抛错", laThrows(() => eigen([[1, 2, 3], [4, 5, 6]])));

  /* Jordan 块：重根也必须给出精确系数 */
  const laJor: Mat = [[1, 1, 0], [0, 1, 1], [0, 0, 1]];
  ok("charCoeffs Jordan 块 = (λ−1)³", charCoeffs(laJor).join(",") === "1,-3,3,-1", charCoeffs(laJor).join(","));
  const laJorR = eigen(laJor);
  ok("eigen 三重根全部落在 1", laJorR.length === 3 && laJorR.every((z) => Math.abs(z.re - 1) < 1e-3), JSON.stringify(laJorR));

  /* 对称阵 Jacobi：A = VΛVᵀ，且 V 正交 */
  const laSymCases: { name: string; a: Mat }[] = [
    { name: "3 阶", a: [[2, -1, 0], [-1, 2, -1], [0, -1, 2]] },
    { name: "4 阶", a: [[4, -2, 1, 0], [-2, 4, -2, 1], [1, -2, 4, -2], [0, 1, -2, 4]] },
  ];
  for (const c of laSymCases) {
    const { values, vecs } = jacobiEigen(c.a);
    const n = c.a.length;
    const rec: Mat = Array.from({ length: n }, (_, i) =>
      Array.from({ length: n }, (_, j) => values.reduce((s, lam, k) => s + vecs[k][i] * lam * vecs[k][j], 0)),
    );
    const recErr = laDiff(rec, c.a);
    ok(`jacobiEigen ${c.name}：A≈VΛVᵀ`, recErr < 1e-9, `最大偏差 ${recErr.toExponential(2)}`);
    const gramErr = laDiff(matMul(vecs, transpose(vecs)), identity(n));
    ok(`jacobiEigen ${c.name}：特征向量标准正交`, gramErr < 1e-9, `Gram 偏差 ${gramErr.toExponential(2)}`);
    ok(`jacobiEigen ${c.name}：特征值降序`, values.every((v, i) => i === 0 || values[i - 1] >= v - 1e-14), JSON.stringify(values));
    const viaEigen = eigen(c.a).map((z) => z.re);
    const mixErr = Math.max(...values.map((v, i) => Math.abs(v - viaEigen[i])));
    ok(`jacobiEigen ${c.name}：eigen 走同一条 Jacobi 路径`, mixErr < 1e-9, `${mixErr.toExponential(2)}`);
  }

  /* 实特征向量 */
  const laEvA: Mat = [[4, -2, 1], [-2, 4, -2], [1, -2, 4]];
  const laEvLs = eigen(laEvA).filter((z) => z.im === 0);
  const laEvVs = laEvLs.map((z) => eigenVector(laEvA, z.re));
  ok("eigenVector 三个实特征值都给出向量", laEvLs.length === 3 && laEvVs.every((v) => v !== null));
  const laEvRes = Math.max(
    ...laEvLs.map((z, i) => {
      const v = laEvVs[i]!;
      return Math.max(...matVec(laEvA, v).map((x, k) => Math.abs(x - z.re * v[k])));
    }),
  );
  ok("eigenVector 满足 A·v=λv", laEvRes < 1e-8, `残差 ${laEvRes.toExponential(2)}`);
  const laEvNrm = Math.max(...laEvVs.map((v) => Math.abs(Math.hypot(...v!) - 1)));
  ok("eigenVector 已归一", laEvNrm < 1e-12, `偏差 ${laEvNrm.toExponential(2)}`);
  ok("eigenVector 复特征值返回 null", eigenVector([[0, -1], [1, 0]], 0) === null);

  /* SVD：A = Σ uᵢₖ sₖ vⱼₖ */
  const laSvdCases: { name: string; a: Mat }[] = [
    { name: "2×3", a: [[1, 2, 3], [4, 5, 7]] },
    { name: "3×2", a: [[1, 2], [3, 4], [5, 7]] },
    { name: "2 阶旋转", a: [[0, -1], [1, 0]] },
    { name: "3 阶对称", a: [[2, -1, 0], [-1, 2, -1], [0, -1, 2]] },
  ];
  for (const c of laSvdCases) {
    const { u, s, v, area } = svd(c.a);
    const m = c.a.length;
    const n = c.a[0].length;
    const rec: Mat = Array.from({ length: m }, (_, i) =>
      Array.from({ length: n }, (_, j) => {
        let r = 0;
        for (let k = 0; k < s.length; k++) r += u[i][k] * s[k] * v[j][k];
        return r;
      }),
    );
    const err = laDiff(rec, c.a);
    ok(`svd ${c.name}：重构 A=Σ u·s·v`, err < 1e-9, `最大偏差 ${err.toExponential(2)}`);
    ok(`svd ${c.name}：奇异值降序且面积=|det|`, s.every((x, i) => i === 0 || s[i - 1] >= x - 1e-14) && area === s.reduce((x, y) => x * y, 1));
    if (m === n) near(`svd ${c.name}：area = |det|`, area, Math.abs(det(c.a)), 1e-9);
  }
  const laSvdD = svd([[3, 0, 0], [0, -5, 0], [0, 0, 2]]);
  ok("svd 对角阵奇异值 = |对角元| 降序", laSvdD.s.join(",") === "5,3,2", laSvdD.s.join(","));

  /* 矩阵指数：与幂级数逐元素对齐 */
  const laExpTaylor = (a: Mat, t: number): Mat => {
    const at = matScale(a, t);
    let term = identity(2);
    const sum = identity(2);
    for (let k = 1; k <= 48; k++) {
      term = matMul(at, term).map((row) => row.map((x) => x / k));
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) sum[i][j] += term[i][j];
    }
    return sum;
  };
  const laExpCases: { name: string; a: Mat }[] = [
    { name: "标量阵（disc≈0）", a: [[2, 0], [0, 2]] },
    { name: "亏损 Jordan 块", a: [[1, 1], [0, 1]] },
    { name: "中心（disc<0）", a: [[0, -2], [2, 0]] },
    { name: "节点（相异实根）", a: [[3, 1], [1, 3]] },
    { name: "一般非对称", a: [[0.5, 1.2], [-0.3, 0.9]] },
  ];
  for (const c of laExpCases) {
    const err = laDiff(expMat2(c.a, 0.4), laExpTaylor(c.a, 0.4));
    ok(`expMat2 ${c.name}：与幂级数一致`, err < 1e-9, `最大偏差 ${err.toExponential(2)}`);
  }
  ok("expMat2 t=0 即单位阵", laExpCases.every((c) => laDiff(expMat2(c.a, 0), identity(2)) === 0));
  const laExpH = 1e-6;
  const laExpDer = Math.max(
    ...laExpCases.map((c) => {
      const up = expMat2(c.a, laExpH);
      const dn = expMat2(c.a, -laExpH);
      return Math.max(...up.map((row, i) => Math.max(...row.map((x, j) => Math.abs((x - dn[i][j]) / (2 * laExpH) - c.a[i][j])))));
    }),
  );
  ok("expMat2 d/dt|₀ = A", laExpDer < 1e-6, `最大偏差 ${laExpDer.toExponential(2)}`);

  /* 三次方程求根 */
  const laCubic = (b: number, c: number, d: number, name: string): C2[] => {
    const roots = cubicRoots(b, c, d);
    const res = laCRes([1, b, c, d], roots);
    ok(`${name}：每个根满足 p(z)=0`, roots.length === 3 && res < 1e-8, `残差 ${res.toExponential(2)}`);
    return roots;
  };
  const laC3 = laCubic(-6, 11, -6, "cubicRoots 三实根 x³−6x²+11x−6");
  ok("cubicRoots 三实根定位到 1/2/3", laC3.every((z) => z.im === 0) && [1, 2, 3].every((m) => laC3.some((z) => Math.abs(z.re - m) < 1e-9)));
  const laC1 = laCubic(-1, 1, -1, "cubicRoots 单实根 + 共轭对 x³−x²+x−1");
  const laC1C = laC1.filter((z) => z.im !== 0);
  ok(
    "cubicRoots 复根严格共轭",
    laC1C.length === 2 && laC1C[0].re === laC1C[1].re && laC1C[0].im === -laC1C[1].im && laC1.some((z) => Math.abs(z.re - 1) < 1e-9),
    JSON.stringify(laC1),
  );
  const laC0 = laCubic(0, 0, 0, "cubicRoots 三重根 x³");
  ok("cubicRoots 三重零根", laC0.every((z) => z.re === 0 && z.im === 0), JSON.stringify(laC0));
}

/* --- 神经网络内核 nn --- */
{
  const nnMean = (v: number[]): number => v.reduce((a, b) => a + b, 0) / v.length;
  const nnThrows = (f: () => unknown): boolean => {
    try {
      f();
    } catch {
      return true;
    }
    return false;
  };
  /** 纯前向损失，用来做中心差分（不经 sampleGrad，避免共享缓存互相污染） */
  const nnLoss = (m: Model, x0: number, x1: number, y: number, cache: Cache): number => {
    const p = forward(m, x0, x1, cache);
    return -Math.log(Math.max(1e-12, p[y]));
  };

  /* 种子确定性 */
  const nnS1 = createModel([2, 5, 3], "tanh", 7);
  const nnS2 = createModel([2, 5, 3], "tanh", 7);
  ok(
    "createModel 同种子逐位复现",
    nnS1.layers.every((L, l) => L.w.every((v, i) => v === nnS2.layers[l].w[i]) && L.b.every((v, i) => v === nnS2.layers[l].b[i])),
  );
  const nnS3 = createModel([2, 5, 3], "tanh", 8);
  ok("createModel 换种子即换权重", nnS1.layers.some((L, l) => L.w.some((v, i) => v !== nnS3.layers[l].w[i])));
  const nnD1 = dataset("moons", 120, 11);
  const nnD2 = dataset("moons", 120, 11);
  ok(
    "dataset 同种子逐位复现",
    nnD1.n === 120 && nnD1.k === 2 && nnD1.xs.every((v, i) => v === nnD2.xs[i]) && nnD1.ys.every((v, i) => v === nnD2.ys[i]),
  );
  ok("dataset 换种子即换样本", dataset("moons", 120, 12).xs.some((v, i) => v !== nnD1.xs[i]));
  ok("createModel 拒绝一维输入与非法形状", nnThrows(() => createModel([1, 3])) && nnThrows(() => createModel([2])));

  /* 前向：softmax 概率分布 */
  const nnPts: [number, number][] = [[0, 0], [1.3, -2.4], [-2.8, 2.6]];
  for (const act of ["tanh", "relu"] as const) {
    const m = createModel([2, 4, 6, 3], act, 7);
    const cache = makeCache(m);
    for (const [x0, x1] of nnPts) {
      const p = forward(m, x0, x1, cache);
      ok(
        `forward(${act}) 输出为概率分布 @${x0},${x1}`,
        p.length === 3 && p.every((v) => v >= 0 && v <= 1) && Math.abs(p.reduce((s, v) => s + v, 0) - 1) < 1e-12,
        `${Array.from(p).join(",")} 和 ${p.reduce((s, v) => s + v, 0)}`,
      );
    }
    const rows = activations(m, 1.3, -2.4, cache);
    ok(
      `activations(${act}) 行数=层数+1 且首行为输入`,
      rows.length === m.layers.length + 1 && rows[0].length === 2 && rows[0][0] === 1.3 && rows[0][1] === -2.4 && rows[1].length === 4,
    );
  }

  /* 梯度检查：解析梯度 vs 中心差分，逐个权重/偏置 */
  const nnGradCheck = (m: Model, x0: number, x1: number, y: number, h: number) => {
    const cache = makeCache(m);
    const g = makeGrad(m);
    zeroGrad(g);
    const loss = sampleGrad(m, x0, x1, y, cache, g);
    let worst = 0;
    let where = "";
    let checked = 0;
    let skipped = 0;
    for (let l = 0; l < m.layers.length; l++) {
      const L = m.layers[l];
      const poke = (arr: Float64Array, idx: number, delta: number): number => {
        const save = arr[idx];
        arr[idx] = save + delta;
        const v = nnLoss(m, x0, x1, y, cache);
        arr[idx] = save;
        return v;
      };
      const one = (arr: Float64Array, an: Float64Array, idx: number, tag: string): void => {
        const num = (poke(arr, idx, h) - poke(arr, idx, -h)) / (2 * h);
        const analytic = an[idx];
        const den = Math.max(Math.abs(num), Math.abs(analytic));
        if (den < 1e-9) {
          skipped++;
          return;
        }
        checked++;
        const rel = Math.abs(num - analytic) / den;
        if (rel > worst) {
          worst = rel;
          where = `${tag}${idx}：数值 ${num.toExponential(6)} 解析 ${analytic.toExponential(6)}`;
        }
      };
      for (let i = 0; i < L.w.length; i++) one(L.w, g.dw[l], i, `第 ${l} 层 w[`);
      for (let j = 0; j < L.b.length; j++) one(L.b, g.db[l], j, `第 ${l} 层 b[`);
    }
    return { loss, worst, where, checked, skipped };
  };
  for (const act of ["tanh", "relu"] as const) {
    const m = createModel([2, 4, 6, 3], act, 7);
    const cache = makeCache(m);
    const gc = nnGradCheck(m, 0.7, -0.4, 2, 1e-6);
    near(`sampleGrad(${act}) 返回值 = −log p[y]`, gc.loss, nnLoss(m, 0.7, -0.4, 2, cache), 1e-12);
    ok(`梯度检查 ${act}：${gc.checked} 个参数全对得上`, gc.worst < 1e-4, `最大相对误差 ${gc.worst.toExponential(2)}（${gc.where}）`);
  }

  /* 训练确实学得会：同种子同打乱顺序，跑几次都是同一条轨迹 */
  const nnRun = (name: (typeof DATASETS)[number], target: number) => {
    const data = dataset(name, 160, 105);
    const m = createModel([2, 8, 8, data.k], "tanh", 5);
    const cache = makeCache(m);
    const g = makeGrad(m);
    const vel = makeGrad(m);
    const r = rng(99);
    const opt = { lr: 0.15, momentum: 0.9, batch: 16 };
    const losses: number[] = [];
    const accs: number[] = [];
    let hit = -1;
    for (let e = 0; e < 60; e++) {
      const st = trainEpoch(m, data, opt, r, cache, g, vel);
      losses.push(dataLoss(m, data, cache));
      accs.push(st.acc);
      if (hit < 0 && st.acc >= target) hit = e + 1;
    }
    return { m, data, losses, accs, hit, target, epochs: losses.length };
  };
  const nnTargets: Record<string, number> = { xor: 0.9, circle: 0.9, moons: 0.9, spiral: 0.85 };
  for (const name of DATASETS) {
    const run = nnRun(name, nnTargets[name]);
    const first5 = nnMean(run.losses.slice(0, 5));
    const last5 = nnMean(run.losses.slice(-5));
    const finalAcc = accuracy(run.m, run.data);
    ok(
      `训练 ${run.data.name} 达标`,
      run.hit > 0 && finalAcc >= run.target && run.accs[run.epochs - 1] >= run.target,
      `${run.data.name} ${run.hit} 轮达标，末轮准确率 ${finalAcc}`,
    );
    ok(`训练 ${run.data.name} 损失整体下降`, last5 < first5, `前 5 轮均值 ${first5.toPrecision(4)} → 后 5 轮 ${last5.toPrecision(4)}`);
    ok(`训练 ${run.data.name} 全程有限`, run.losses.every((v) => Number.isFinite(v)) && run.accs.every((v) => Number.isFinite(v)));
  }

  /* predict / predictMargin 一致性 */
  const nnPm = createModel([2, 6, 2], "tanh", 4);
  const nnPc = makeCache(nnPm);
  for (const [x0, x1] of nnPts.slice(0, 2)) {
    const p = forward(nnPm, x0, x1, nnPc);
    const pm = predictMargin(nnPm, x0, x1, nnPc);
    ok(
      `predictMargin @${x0},${x1} 与 predict 同类`,
      pm.label === predict(nnPm, x0, x1, nnPc) && pm.margin >= 0 && pm.margin <= 1,
      JSON.stringify(pm),
    );
    near(`二分类 margin = |p0−p1| @${x0},${x1}`, pm.margin, Math.abs(p[0] - p[1]), 1e-12);
  }
  const nnPm3 = createModel([2, 6, 3], "tanh", 4);
  const nnPc3 = makeCache(nnPm3);
  const nnP3 = Array.from(forward(nnPm3, 0.9, 0.9, nnPc3)).sort((a, b) => b - a);
  const nnM3 = predictMargin(nnPm3, 0.9, 0.9, nnPc3);
  near("多分类 margin = 第一名减第二名", nnM3.margin, nnP3[0] - nnP3[1], 1e-12);

  /* 决策边界栅格：分带续算赖以成立的采样约定 */
  const nnW = 17;
  const nnH = 11;
  const nnLeft = -3;
  const nnRight = 3;
  const nnTop = 2.5;
  const nnBottom = -2.5;
  const nnRast = decisionRaster(nnPm, nnLeft, nnRight, nnTop, nnBottom, nnW, nnH, true);
  const nnRows = new Uint8ClampedArray(nnW * nnH * 3);
  const nnDy = (nnBottom - nnTop) / (nnH - 1);
  for (let j = 0; j < nnH; j++) decisionRow(nnPm, nnLeft, nnRight, nnTop + nnDy * j, nnW, true, nnRows, j * nnW * 3, nnPc);
  ok(
    "decisionRaster 与逐行 decisionRow 逐字节一致",
    nnRast.length === nnW * nnH * 3 && nnRows.every((v, i) => v === nnRast[i]),
    `差异像素 ${nnRast.filter((v, i) => v !== nnRows[i]).length}`,
  );
  const nnFlip = decisionRaster(nnPm, nnLeft, nnRight, nnBottom, nnTop, nnW, nnH, true);
  const nnBand = nnW * 3;
  ok(
    "row 0 即世界纵坐标 top（翻转上下得镜像）",
    nnFlip.every((v, i) => v === nnRast[(nnH - 1 - Math.floor(i / nnBand)) * nnBand + (i % nnBand)]),
    `差异通道 ${nnFlip.filter((v, i) => v !== nnRast[(nnH - 1 - Math.floor(i / nnBand)) * nnBand + (i % nnBand)]).length}`,
  );
  ok(
    "decisionRaster 退化尺寸不抛错",
    nnThrows(() => decisionRaster(nnPm, nnLeft, nnRight, nnTop, nnTop, 0, 0, true)) === false &&
      decisionRaster(nnPm, nnLeft, nnRight, nnTop, nnTop, 1, 1, false).length === 3 &&
      decisionRaster(nnPm, nnLeft, nnRight, nnTop, nnTop, 1, 4, false).length === 12,
  );
  ok("decisionRaster 明暗底色各一套", decisionRaster(nnPm, nnLeft, nnRight, nnTop, nnBottom, 4, 4, false).some((v, i) => v !== nnRast[i]));

  /* 随机源 */
  const nnR1 = rng(42);
  const nnR2 = rng(42);
  const nnR3 = rng(43);
  const nnSeq1 = Array.from({ length: 64 }, () => nnR1());
  const nnSeq2 = Array.from({ length: 64 }, () => nnR2());
  const nnSeq3 = Array.from({ length: 64 }, () => nnR3());
  ok("rng 取值落在 [0,1)", nnSeq1.every((v) => v >= 0 && v < 1));
  ok("rng 同种子同序列", nnSeq1.every((v, i) => v === nnSeq2[i]));
  ok("rng 异种子异序列", nnSeq1.some((v, i) => v !== nnSeq3[i]));
}

/* --- 新增工作台的 store 契约（lin / nn） --- */
{
  const st = useStore.getState();
  ok("lin 默认元素数与阶数相符", st.lin.a.length === st.lin.dim * st.lin.dim, `${st.lin.a.length}`);
  ok("lin 默认 b 长度与阶数相符", st.lin.b.length === st.lin.dim, `${st.lin.b.length}`);
  ok("lin 视口存在", !!st.views.lin && Number.isFinite(st.views.lin.scale));
  ok("nn 视口存在", !!st.views.nn && Number.isFinite(st.views.nn.scale));
  ok("lin 解析走表达式内核", (() => {
    const m = parseMatrix(st.engine, ["cos(pi/3)", "-sin(pi/3)", "sin(pi/3)", "cos(pi/3)"], 2);
    return Math.abs(m[0][0] - 0.5) < 1e-12 && Math.abs(m[1][1] - 0.5) < 1e-12;
  })());
  ok("lin 空单元格报错而非静默为 0", (() => {
    try {
      parseMatrix(st.engine, ["1", "", "3", "4"], 2);
    } catch (e) {
      return /第 2 格/.test((e as Error).message);
    }
    return false;
  })());
  ok("lin 拒绝复数单元格", (() => {
    try {
      parseMatrix(st.engine, ["1", "2i", "3", "4"], 2);
    } catch {
      return true;
    }
    return false;
  })());
  ok("lin 数量不符时报错", (() => {
    try {
      parseMatrix(st.engine, ["1", "2", "3"], 2);
    } catch {
      return true;
    }
    return false;
  })());
  /* nn 默认超参数必须真能建成网络：面板按 [2, hidden×depth, k] 组装 */
  const nnData = dataset(st.nn.dataset, st.nn.samples, st.nn.seed);
  const nnSizes = [2, ...Array.from({ length: st.nn.depth }, () => st.nn.hidden), nnData.k];
  const nnModel = createModel(nnSizes, st.nn.act, st.nn.seed);
  ok("nn 默认超参可建网", nnModel.sizes.join(",") === nnSizes.join(","), nnSizes.join(","));
  ok("nn 数据集类别数与输出层相符", nnData.k === nnSizes[nnSizes.length - 1] && nnData.n === st.nn.samples);
  const nnR = rng(st.nn.seed + 1);
  const nnOpt = { lr: st.nn.lr, momentum: st.nn.momentum, batch: st.nn.batch };
  const nnS0 = trainEpoch(nnModel, nnData, nnOpt, nnR, makeCache(nnModel), makeGrad(nnModel), makeGrad(nnModel));
  ok("nn 默认一轮训练有效", Number.isFinite(nnS0.loss) && nnS0.acc >= 0 && nnS0.acc <= 1, JSON.stringify(nnS0));
  /* setter 合并 + 递增 revision：面板靠这个触发重绘，少 bump 一次就是画面不动 */
  const rev0 = useStore.getState().revision;
  useStore.getState().setLin({ t: 2.5 });
  ok("setLin 合并并 bump", useStore.getState().revision === rev0 + 1 && useStore.getState().lin.t === 2.5 && useStore.getState().lin.dim === st.lin.dim);
  const rev1 = useStore.getState().revision;
  useStore.getState().setNn({ lr: 0.2, model: nnModel, data: nnData });
  ok("setNn 合并并 bump", useStore.getState().revision === rev1 + 1 && useStore.getState().nn.lr === 0.2 && useStore.getState().nn.model === nnModel);
  useStore.getState().setNn({ lr: st.nn.lr, model: null, data: null });
  useStore.getState().setLin({ t: st.lin.t });
  ok("回归现场已复原", useStore.getState().nn.model === null && useStore.getState().lin.t === st.lin.t);
  /* 三种模式下画布都要能拿到对应视口：setMode 不换 Viewport 就会串图 */
  for (const m of ["lin", "nn"] as Mode[]) {
    useStore.getState().setMode(m);
    ok(`模式 ${m} 有独立视口`, useStore.getState().views[m] === st.views[m] && useStore.getState().mode === m);
  }
  useStore.getState().setMode(st.mode);
}

/* --- 复数快路径 compileCplx / 整数幂 --- */
{
  const cxEng = new Engine();
  type CXFn = ((re: number, im: number) => [number, number]) & {
    wrap?: (z: { re: number; im: number }) => { re: number; im: number };
    cf?: (re: number, im: number, o: { re: number; im: number }) => void;
  };
  /** 通用 AST 路径：逐像素着色的参照实现 */
  const cxAst = (src: string, name = "z"): CXFn => {
    const c = compile(cxEng, parseExpr(src), [name]);
    const s = new Float64Array(2);
    const g: CXFn = ((re: number, im: number) => {
      s[0] = re;
      s[1] = im;
      try {
        const v = c.run(s);
        return typeof v.re === "number" ? [v.re, v.im] : [NaN, NaN];
      } catch {
        return [NaN, NaN];
      }
    }) as CXFn;
    g.wrap = (z) => {
      const [re, im] = g(z.re, z.im);
      return { re, im };
    };
    return g;
  };
  const cxBad: string[] = [];
  let cxPoints = 0;
  const cxCheck = (src: string, name = "z") => {
    const fast = compileCplx(cxEng, parseExpr(src), name);
    if (!fast) {
      cxBad.push(`${src}(未编译)`);
      return;
    }
    const g = cxAst(src, name);
    const o = { re: 0, im: 0 };
    let bad = 0;
    let seed = 7 >>> 0;
    for (let k = 0; k < 400; k++) {
      seed = (Math.imul(seed, 1103515245) + 12345) >>> 0;
      const re = ((seed % 10007) / 10007) * 6.4 - 3.2;
      seed = (Math.imul(seed, 1103515245) + 12345) >>> 0;
      const im = ((((seed >> 3) % 10007) / 10007) * 6.4 - 3.2) * -1;
      const [ar, ai] = g(re, im);
      fast(re, im, o);
      const sameR = Number.isNaN(ar) ? Number.isNaN(o.re) : ar === o.re;
      const sameI = Number.isNaN(ai) ? Number.isNaN(o.im) : ai === o.im;
      if (!(sameR && sameI)) bad++;
      cxPoints++;
    }
    if (bad) cxBad.push(`${src}(${bad}/400)`);
  };
  for (const e of [
    "z^3-1", "sin(z)/z", "exp(-z^2)", "(z^2-1)/(z^2+1)", "log(z)*sqrt(z)", "z^5-z+0.1",
    "tan(z)", "abs(conj(z)*z)", "((z-i)/(z+i))^3", "0.5*(exp(i*z)+exp(-i*z))",
    "z^2 + pi*z + e", "arg(z^3)", "sqrt(1-z^2)/(1+z^2)", "abs2(z)-re(z)^2-im(z)^2",
    "z^-3", "z^16", "(1+z)^7/(1-z)^7",
  ]) cxCheck(e);
  cxCheck("x^3-x+1", "x");
  ok("复数快路径与 AST 路径逐位一致", cxBad.length === 0, cxBad.join(" ") + ` 共 ${cxPoints} 个点`);

  /* 编不出来的构造必须明确回退，而不是给出近似结果 */
  for (const e of ["if(z>0, z, -z)", "[1,2]", "z[1]", "gcd(z,3)", "root(z,3)", "z!", "vec(z,1)", "\"ab\""]) {
    ok(`${e} 回退通用路径`, compileCplx(cxEng, parseExpr(e), "z") === null);
  }
  cxEng.define("g(t) = t^2+1");
  ok("用户自定义函数不参与提升", compileCplx(cxEng, parseExpr("g(z)+1"), "z") === null);

  /* 全局量是活的：滑块一动结果要跟变；纯子树才允许在编译期折叠 */
  const liveEng = new Engine();
  liveEng.setNum("pa", 2);
  const live = compileCplx(liveEng, parseExpr("pa*z + 2*3"), "z");
  ok("含全局量的表达式仍可编译", !!live);
  const lo = { re: 0, im: 0 };
  live!(1, 0, lo);
  near("全局量按当前值求值", lo.re, 8); // pa*z + 折叠常量 2*3
  liveEng.setNum("pa", 5);
  live!(1, 0, lo);
  near("滑块改全局量后同一闭包同步", lo.re, 11);
  liveEng.setNum("pa", 7);
  live!(0, 1, lo);
  ok("虚部随全局量变化", lo.im === 7, `${lo.im}`);

  /* 整数幂：复底走累乘，结果应与二项式展开一致 */
  const cp = (re: number, im: number, e: number): [number, number] => {
    const o = CN.cpow(re, im, e, 0, { re: 0, im: 0 });
    return [o.re, o.im];
  };
  near("(1+i)^8 实部", cp(1, 1, 8)[0], 16, 1e-9); // ((1+i)^2)^4 = (2i)^4
  ok("(1+i)^8 虚部", Math.abs(cp(1, 1, 8)[1]) < 1e-9, `${cp(1, 1, 8)[1]}`);
  near("(2+3i)^3 实部", cp(2, 3, 3)[0], -46, 1e-9);
  near("(2+3i)^3 虚部", cp(2, 3, 3)[1], 9, 1e-9);
  near("(3+4i)^5 模", Math.hypot(...cp(3, 4, 5)), Math.pow(25, 2.5), 1e-9);
  near("(1+i)^-2 虚部", cp(1, 1, -2)[1], -0.5, 1e-12);
  ok("(1+i)^-2 实部", Math.abs(cp(1, 1, -2)[0]) < 1e-12);
  near("^16 走累乘", Math.hypot(...cp(1.1, -0.7, 16)), Math.pow(Math.hypot(1.1, 0.7), 16), 1e-9);
  near("^17 回主值式", Math.hypot(...cp(1.1, -0.7, 17)), Math.pow(Math.hypot(1.1, 0.7), 17), 1e-9);
  near("负底整数幂保持实数", cp(-8, 0, 3)[0], -512, 1e-12);
  ok("负底整数幂不虚增虚部", Math.abs(cp(-8, 0, 3)[1]) < 1e-12);
  near("奇次根仍给实根", cp(-8, 0, 1 / 3)[0], -2, 1e-12);

  /* 着色栅格：快路径与通用路径输出必须逐字节相同 */
  for (const src of ["z^3-1", "sin(z)/z"]) {
    const slow = cxAst(src);
    const fast = cxAst(src);
    const c = compileCplx(cxEng, parseExpr(src), "z");
    if (c) fast.cf = (re, im, o) => c(re, im, o);
    const a = domainColor(slow.wrap!, -2.4, 2.4, 2.4, -2.4, 90, 64, { dark: true });
    const b = domainColor(fast.wrap!, -2.4, 2.4, 2.4, -2.4, 90, 64, { dark: true });
    let diff = 0;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) diff++;
    ok(`${src} 共形着色逐字节一致`, diff === 0, `${diff}/${a.length} 通道不同`);
    const p = newtonPlan(fast.wrap!, -2.4, 2.4, 2.4, -2.4);
    const nF = newtonFractal(fast.wrap!, -2.4, 2.4, 2.4, -2.4, 90, 64, { maxIter: 20, plan: p });
    const nS = newtonFractal(slow.wrap!, -2.4, 2.4, 2.4, -2.4, 90, 64, { maxIter: 20, plan: p });
    let nd = 0;
    for (let i = 0; i < nF.iter.length; i++) if (nF.iter[i] !== nS.iter[i] || nF.root[i] !== nS.root[i]) nd++;
    ok(`${src} Newton 分形逐像素一致`, nd === 0, `${nd} 像素不同`);
  }

  /* 吞吐：只输出信息，不做断言（机器负载会让时间类断言假失败） */
  const cxBench = (fn: () => void, reps: number): number => {
    let best = Infinity;
    for (let k = 0; k < reps; k++) {
      const t0 = Date.now();
      fn();
      best = Math.min(best, Date.now() - t0);
    }
    return best;
  };
  const N = 120000;
  /* 校验和用一个整函数：log/sqrt 在原点附近有非有限值，会把累加和污染成 NaN */
  const gSlow = cxAst("z^5-z+0.1");
  const gFast = compileCplx(cxEng, parseExpr("z^5-z+0.1"), "z")!;
  const oo = { re: 0, im: 0 };
  let accS = 0;
  let accF = 0;
  const tSlow = cxBench(() => {
    accS = 0;
    for (let k = 0; k < N; k++) {
      const [r, i] = gSlow((k % 400) / 200 - 1, ((k >> 8) % 400) / 200 - 1);
      accS += r + i;
    }
  }, 3);
  const tFast = cxBench(() => {
    accF = 0;
    for (let k = 0; k < N; k++) {
      gFast((k % 400) / 200 - 1, ((k >> 8) % 400) / 200 - 1, oo);
      accF += oo.re + oo.im;
    }
  }, 3);
  ok(`12 万次求值累加校验和一致`, accS === accF && Number.isFinite(accS), `${accS} vs ${accF}`);
  console.log(
    `  · 复变求值吞吐：AST ${((tSlow / N) * 1e3).toFixed(3)}µs/次 → 快路径 ${((tFast / N) * 1e3).toFixed(3)}µs/次（${(tSlow / tFast).toFixed(2)}×）`,
  );
}

/* ============================================ 示例库契约：lin / nn 预设拿来就能画、能训 */
{
  /** U Σ Vᵀ 是否真的还原 A：奇异值分解最怕"数值漂亮但乘回去不对" */
  const svdCheck = (a: Mat): number => {
    const { u, s, v } = svd(a);
    const us = matMul(u, s.map((si, i) => s.map((_, j) => (i === j ? si : 0))));
    const rec = matMul(us, transpose(v));
    let err = 0;
    for (let i = 0; i < a.length; i++) for (let j = 0; j < a[i].length; j++) err = Math.max(err, Math.abs(rec[i][j] - a[i][j]));
    return err;
  };

  const linPresets = PRESETS.filter((q) => q.mode === "lin");
  ok("lin 预设至少 3 个", linPresets.length >= 3, `${linPresets.length}`);
  for (const q of linPresets) {
    applyPreset(q.key);
    const st = useStore.getState();
    const lin = st.lin;
    ok(`${q.key}：A 格子数 = dim²`, lin.a.length === lin.dim * lin.dim, `${lin.a.length} vs ${lin.dim * lin.dim}`);
    ok(`${q.key}：b 格子数 = dim`, lin.b.length === lin.dim, `${lin.b.length} vs ${lin.dim}`);
    let A: Mat;
    try {
      A = parseMatrix(st.engine, lin.a, lin.dim);
    } catch (e) {
      ok(`${q.key}：A 可由表达式解析`, false, (e as Error).message);
      continue;
    }
    ok(`${q.key}：A 元素有限`, A.every((r) => r.every((v) => Number.isFinite(v))));
    const b = parseVector(st.engine, lin.b, lin.dim);
    const d = det(A);
    ok(`${q.key}：det 有限`, Number.isFinite(d), `${d}`);
    const ev = eigen(A);
    ok(`${q.key}：特征值个数 = dim`, ev.length === lin.dim, `${ev.length}`);
    ok(`${q.key}：特征值有限`, ev.every((z) => Number.isFinite(z.re) && Number.isFinite(z.im)));
    /* 实特征值必须真的有实特征向量：残差 |Av − λv| 应到机器精度 */
    for (const z of ev) {
      if (Math.abs(z.im) > 1e-9) continue;
      const v = eigenVector(A, z.re);
      if (!v) continue;
      const av = matVec(A, v);
      const res = Math.max(...av.map((w, i) => Math.abs(w - z.re * v[i])));
      ok(`${q.key}：特征残差 ${z.re.toFixed(4)}`, res < 1e-8, `res=${res}`);
    }
    const tr = A.reduce((s2, r, i) => s2 + r[i], 0);
    const sum = ev.reduce((s2, z) => s2 + z.re, 0);
    ok(`${q.key}：迹 = 特征值之和`, Math.abs(tr - sum) < 1e-8, `${tr} vs ${sum}`);
    const prod = ev.reduce((s2, z) => s2 * (z.re * z.re + z.im * z.im), 1);
    ok(`${q.key}：|det| = Π|λ|`, Math.abs(Math.abs(d) - Math.sqrt(prod)) < 1e-7, `${Math.abs(d)} vs ${Math.sqrt(prod)}`);
    const sd = svd(A);
    ok(`${q.key}：奇异值降序非负`, sd.s.every((x) => x >= 0) && sd.s.every((x, i) => i === 0 || x <= sd.s[i - 1] + 1e-12), sd.s.join(","));
    ok(`${q.key}：SVD 还原矩阵`, svdCheck(A) < 1e-10, `err=${svdCheck(A)}`);
    ok(`${q.key}：面积放大率 = |det|`, Math.abs(sd.area - Math.abs(d)) < 1e-9, `${sd.area} vs ${Math.abs(d)}`);
    /* 解 Ax=b 的残差：预设里 b 是给定的，点出来的解必须真的满足方程 */
    if (Math.abs(d) > 1e-12) {
      const x = solve(A, b);
      const r2 = matVec(A, x);
      ok(`${q.key}：Ax=b 残差`, Math.max(...r2.map((w, i) => Math.abs(w - b[i]))) < 1e-9);
    }
    if (lin.dim === 2) {
      /* 矩阵指数的半群性：e^{t1 A}e^{t2 A} = e^{(t1+t2)A}，是 expMat2 唯一像样的自检 */
      const e1 = expMat2(A, 0.6);
      const e2 = expMat2(A, -0.25);
      const rhs = expMat2(A, 0.35);
      let em = 0;
      const p2 = matMul(e1, e2);
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) em = Math.max(em, Math.abs(p2[i][j] - rhs[i][j]));
      ok(`${q.key}：e^{tA} 半群性`, em < 1e-9, `err=${em}`);
      const e0 = expMat2(A, 0);
      ok(`${q.key}：e^{0·A} = I`, Math.max(...e0.map((r, i) => Math.abs(r[i] - 1)), ...e0.map((r, i) => Math.abs(r[1 - i]))) < 1e-12);
    }
  }

  /* 逐个预设的数学承诺：这些是示例标题里向读者保证的东西 */
  const byKey = (k: string) => {
    applyPreset(k);
    const st = useStore.getState();
    return { A: parseMatrix(st.engine, st.lin.a, st.lin.dim), lin: st.lin };
  };
  {
    const { A } = byKey("lin-rot60");
    near("lin-rot60：det = 1", det(A), 1, 1e-12);
    const ev = eigen(A);
    near("lin-rot60：实部 = cos60°", ev[0].re, 0.5, 1e-9);
    near("lin-rot60：模 = 1", Math.hypot(ev[0].re, ev[0].im), 1, 1e-9);
    ok("lin-rot60：是旋转（AᵀA = I）", svdCheck(A) < 1e-12 && Math.max(...matMul(transpose(A), A).map((r) => Math.abs(r[0] + r[1] - 1))) < 1e-12);
    ok("lin-rot60：无实特征向量", ev.every((z) => Math.abs(z.im) > 1e-6) && eigenVector(A, 1) === null);
  }
  {
    const { A } = byKey("lin-shear");
    near("lin-shear：剪切保面积", det(A), 1, 1e-12);
    ok("lin-shear：x 轴方向不动", matVec(A, [1, 0]).join() === "1,0", matVec(A, [1, 0]).join());
    const ev = eigen(A);
    ok("lin-shear：二重特征值 1", ev.every((z) => Math.abs(z.re - 1) < 1e-9 && Math.abs(z.im) < 1e-9), JSON.stringify(ev));
    const v = eigenVector(A, 1);
    ok("lin-shear：只给一个亏损方向", !!v && Math.abs(Math.abs(v[0]) - 1) < 1e-9, JSON.stringify(v));
  }
  {
    const { A } = byKey("lin-spectral");
    ok("lin-spectral：对称", isSymmetric(A));
    const ev = eigen(A).map((z) => z.re);
    near("lin-spectral：λ₁ = 3", ev[0], 3, 1e-9);
    near("lin-spectral：λ₂ = 1", ev[1], 1, 1e-9);
    const v1 = eigenVector(A, ev[0]);
    const v2 = eigenVector(A, ev[1]);
    ok("lin-spectral：主轴正交", !!v1 && !!v2 && Math.abs(v1[0] * v2[0] + v1[1] * v2[1]) < 1e-9);
    const sd = svd(A);
    ok("lin-spectral：奇异值 = |特征值|", Math.abs(sd.s[0] - 3) < 1e-9 && Math.abs(sd.s[1] - 1) < 1e-9, sd.s.join(","));
  }

  const nnPresets = PRESETS.filter((q) => q.mode === "nn");
  ok("nn 预设至少 3 个", nnPresets.length >= 3, `${nnPresets.length}`);
  const KLASS: Record<string, number> = { xor: 2, circle: 2, moons: 2, spiral: 3 };
  for (const q of nnPresets) {
    applyPreset(q.key);
    const st = useStore.getState();
    const nn = st.nn;
    /* 预设刻意把 model 置空：沿用上一次的模型会画出和 dataset/seed 对不上号的边界 */
    ok(`${q.key}：应用后模型为空，等面板重建`, nn.model === null && nn.running === false);
    ok(`${q.key}：数据集名合法`, DATASETS.includes(nn.dataset), `${nn.dataset}`);
    ok(`${q.key}：结构参数为正`, nn.hidden >= 1 && nn.depth >= 1 && nn.samples >= 8 && nn.batch >= 1 && nn.lr > 0);
    ok(`${q.key}：边界分辨率可用`, nn.boundaryRes >= 16 && nn.boundaryRes <= 512, `${nn.boundaryRes}`);
    const data = dataset(nn.dataset, nn.samples, nn.seed);
    ok(`${q.key}：类别数对得上`, data.k === KLASS[nn.dataset], `${data.k}`);
    ok(`${q.key}：样本数对得上`, data.n === nn.samples && data.xs.length === 2 * nn.samples);
    const sizes = [2, ...Array(nn.depth).fill(nn.hidden), data.k];
    const m = createModel(sizes, nn.act, nn.seed);
    ok(`${q.key}：层形状 = sizes`, m.layers.length === sizes.length - 1 && m.sizes.join() === sizes.join());
    const opt = { lr: nn.lr, momentum: nn.momentum, batch: nn.batch };
    const cache = makeCache(m);
    const g = makeGrad(m);
    const vel = makeGrad(m);
    const loss0 = dataLoss(m, data, cache);
    const r = rng(nn.seed);
    const EP = nn.dataset === "spiral" ? 260 : 180;
    const t0 = Date.now();
    let stat = { loss: loss0, acc: 0, steps: 0 };
    for (let e = 0; e < EP; e++) stat = trainEpoch(m, data, opt, r, cache, g, vel);
    const ms = Date.now() - t0;
    ok(`${q.key}：训练不发散`, Number.isFinite(stat.loss) && stat.loss < loss0, `loss ${loss0} → ${stat.loss}`);
    ok(`${q.key}：准确率 ≥ 0.85`, stat.acc >= 0.85, `acc=${stat.acc}`);
    /* 决策边界栅格用预设自己的分辨率：不能一采样就抛或全零 */
    const rgb = decisionRaster(m, -3, 3, 3, -3, 40, 40, true);
    let nz = 0;
    for (let i = 0; i < rgb.length; i += 3) if (rgb[i] !== rgb[3] || rgb[i + 1] !== rgb[4] || rgb[i + 2] !== rgb[5]) nz++;
    ok(`${q.key}：边界栅格有内容`, rgb.length === 40 * 40 * 3 && nz > 100, `变化像素 ${nz}`);
    console.log(`  · ${q.key}：${EP} 轮 ${ms}ms（loss ${loss0.toFixed(3)} → ${stat.loss.toFixed(4)}，acc ${(stat.acc * 100).toFixed(1)}%）`);
    /* 同种子必须同结果：否则"这个示例能训到 98%"是一句无法复现的话 */
    const m2 = createModel(sizes, nn.act, nn.seed);
    const c2 = makeCache(m2);
    const g2 = makeGrad(m2);
    const v2 = makeGrad(m2);
    const r2 = rng(nn.seed);
    const s2 = trainEpoch(m2, data, opt, r2, c2, g2, v2);
    const rr = rng(nn.seed);
    const m3 = createModel(sizes, nn.act, nn.seed);
    const s3 = trainEpoch(m3, data, opt, rr, makeCache(m3), makeGrad(m3), makeGrad(m3));
    ok(`${q.key}：同种子首轮 loss 逐位相同`, s2.loss === s3.loss && s2.acc === s3.acc, `${s2.loss} vs ${s3.loss}`);
  }
  useStore.getState().patch({ mode: "func" });
}

/* ==================== 接线自检：有状态、有示例，还必须真的被画出来、被面板管起来 */
{
  const read = (f: string) => readFileSync(new URL(f, import.meta.url), "utf8");
  const scene = read("../src/render/scene2d.ts");
  const panels = read("../src/ui/panels.tsx");
  const app = read("../src/App.tsx");
  /** 只看 drawScene2D 这一段：别把别的函数里恰好出现的字符串当成已接线。
   *  surf / console 不在其中——它们分别走 Canvas3D 和文字控制台。 */
  const dispatch = scene.slice(scene.indexOf("export function drawScene2D"), scene.indexOf("export function probe"));
  const missing2D = ["func", "geom", "complex", "vector", "lin", "nn"].filter((m) => !dispatch.includes(`case "${m}"`));
  ok("drawScene2D 分支齐全", missing2D.length === 0, `缺 ${missing2D.join(",")}`);
  /** 面板同理：SidePanel 的 switch 才是模式到面板的映射。
   *  console 走 default 分支，所以逐个点名组件而不是点名 case。 */
  const side = panels.slice(panels.indexOf("export default function SidePanel"));
  const missingPanel = ["FuncPanel", "GeoPanel", "ComplexPanel", "VectorPanel", "LinPanel", "NnPanel", "SurfPanel", "ConsolePanel"].filter(
    (c) => !side.includes(`<${c} />`),
  );
  ok("SidePanel 分支齐全", missingPanel.length === 0, `缺 ${missingPanel.join(",")}`);
  for (const m of ["lin", "nn"]) ok(`${m} 出现在模式切换栏`, app.includes(`"${m}"`), "App 模式列表缺少");
}

console.log(`\n验证结束：通过 ${pass} 项${fails.length ? `，失败 ${fails.length} 项：` : "，全部通过"}`);
for (const f of fails) console.log("  ✗ " + f);
if (fails.length) process.exitCode = 1;
