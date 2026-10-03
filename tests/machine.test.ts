// GeoLab 求值机 machine.ts 的纯逻辑回归。
// 测什么：值代数（构造/转换/真值）、线性代数（det/inv/mulMat/solveLin）、
//   数值分析（simpson/bisect/newton/newtonC/polyfit/interp1）、表达式求值与编译、show 格式化。
// 为什么该测：这些函数是"数学库"的地基——det/solveLin 喂给拟合与求根，polyfit/interp1 喂给数据拟合，
//   compileReal 是曲面每帧上万次的热路径。任一处退化（奇异矩阵、区间无根、除零）会静默污染下游结果，
//   这里把每种退化的真实返回值/抛错行为都钉住。
// 跑法：node --experimental-strip-types --test tests/*.test.ts
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  CONSTANTS,
  Engine,
  SLOT_NAMES,
  asFun,
  asMat,
  asReal,
  asVec,
  bisect,
  bool,
  compile,
  compileReal,
  det,
  evalString,
  funVal,
  interp1,
  inv,
  kindName,
  listVal,
  matVal,
  mulMat,
  newton,
  newtonC,
  num,
  numericGlobals,
  polyfit,
  reIm,
  setContextVars,
  show,
  simpson,
  solveLin,
  strVal,
  suggestParams,
  suggestParams as _sp,
  truthy,
  vecVal,
  ZERO,
} from "../src/core/machine.ts";
import { parseExpr, freeNames } from "../src/core/parser.ts";

/* ---------------------------------------------------------------- 断言助手 */

function near(actual: number, expect: number, tol = 1e-9, msg?: string): void {
  assert.ok(
    Math.abs(actual - expect) < tol,
    `${msg ?? "值不相等"}：实际 ${actual}，期望 ${expect}（容差 ${tol}）`,
  );
}
function nearMat(actual: number[][], expect: number[][], tol = 1e-9): void {
  assert.equal(actual.length, expect.length, "行数不符");
  for (let i = 0; i < expect.length; i++) {
    assert.equal(actual[i].length, expect[i].length, `第 ${i} 行列数不符`);
    for (let j = 0; j < expect[i].length; j++) {
      near(actual[i][j], expect[i][j], tol, `M[${i}][${j}]`);
    }
  }
}
/** 引擎求值成实数；解析错误直接抛出未处理异常，测试因此会红——这正是要断言的 */
const ev = (src: string): number => asReal(evalString(new Engine(), src));
/** 求值成显示文本，用于格式化断言 */
const evs = (src: string): string => show(evalString(new Engine(), src));
/** 断言表达式求值抛错，且错误信息里带上原文 */
function evThrows(src: string, needle: string): RegExp {
  try {
    evalString(new Engine(), src);
  } catch (e) {
    assert.match((e as Error).message, needle, "错误信息应说明原因");
    return needle as unknown as RegExp;
  }
  assert.fail(`"${src}" 本应抛错却求出了值`);
}
const fn = (body: string, param = "x") => ({ name: "f", params: [param], body: parseExpr(body) });

/* ============================================================== 值代数 */

test("kindName 按真实类型与维度命名，复数/实数分开", () => {
  assert.equal(kindName(num(3)), "实数");
  assert.equal(kindName(num(3, 2)), "复数", "虚部非 0 即复数");
  assert.equal(kindName(vecVal([1, 2, 3])), "向量(3)");
  assert.equal(kindName(matVal([[1, 2], [3, 4]])), "矩阵2x2");
  assert.equal(kindName(strVal("x")), "文本");
  assert.equal(kindName(bool(true)), "逻辑值");
  assert.equal(kindName(funVal(fn("x", "t"))), "函数");
  assert.deepEqual([ZERO.re, ZERO.im], [0, 0]);
});

test("asReal/asVec/asMat/asFun：能升的就升，不能升的必须报错而不是静默取 0", () => {
  near(asReal(num(3)), 3);
  assert.deepEqual([asReal(bool(true)), asReal(bool(false))], [1, 0], "逻辑值当 1/0");
  assert.throws(() => asReal(vecVal([1]), "参数"), /参数需要实数，实际是 向量\(1\)/);
  assert.throws(() => asReal(num(1, 2), "参数"), /参数需要实数，实际是复数 1\+2i/);
  // 向量：实数升成一维，复数升成 [re,im]，矩阵摊平
  assert.deepEqual(asVec(num(5)), [5]);
  assert.deepEqual(asVec(num(5, 6)), [5, 6]);
  assert.deepEqual(asVec(matVal([[1, 2], [3, 4]])), [1, 2, 3, 4]);
  assert.throws(() => asVec(strVal("a")), /需要向量，实际是 文本/);
  // 矩阵：向量升一行，实数升 1×1
  assert.deepEqual(asMat(vecVal([1, 2])), [[1, 2]]);
  assert.deepEqual(asMat(num(3)), [[3]]);
  assert.throws(() => asMat(strVal("a")), /需要矩阵，实际是 文本/);
  const f = fn("x");
  assert.equal(asFun(funVal(f)), f);
  assert.throws(() => asFun(num(1)), /需要函数，实际是 实数/);
});

test("truthy：0/false/空串/零向量为假；矩阵永远为假（不扫描元素）", () => {
  // 矩阵落进 default 分支是有意的（矩阵真值无定义），但要钉住以免被当成"非空即真"
  assert.equal(truthy(bool(false)), false);
  assert.equal(truthy(num(0)), false);
  assert.equal(truthy(num(-1)), true, "NaN 之外的任意非 0 数为真");
  assert.equal(truthy(vecVal([0, 0, 1])), true);
  assert.equal(truthy(vecVal([0, 0])), false);
  assert.equal(truthy(strVal("")), false);
  assert.equal(truthy(strVal("a")), true);
  assert.equal(truthy(matVal([[0, 1]])), false, "矩阵恒为假");
});

test("reIm 读复平面坐标，越界补 0，文本报错", () => {
  assert.deepEqual(reIm(num(1, 2)), [1, 2]);
  assert.deepEqual(reIm(bool(true)), [1, 0]);
  assert.deepEqual(reIm(vecVal([7, 8, 9])), [7, 8], "一维向量的虚部补 0");
  assert.deepEqual(reIm(vecVal([7])), [7, 0]);
  assert.throws(() => reIm(strVal("a")), /需要数值/);
});

test("listVal：等长行成矩阵，不等长摊平成向量，复数实虚交替", () => {
  assert.deepEqual(listVal([]).v, []);
  assert.deepEqual(listVal([num(1), num(2), num(3)]).v, [1, 2, 3]);
  assert.deepEqual(listVal([num(1, 2), num(3, 4)]).v, [1, 2, 3, 4], "复数按 re,im 依次写入");
  assert.deepEqual(listVal([vecVal([1, 2]), vecVal([3, 4])]).m, [[1, 2], [3, 4]]);
  assert.deepEqual(
    listVal([vecVal([1, 2]), vecVal([3])]).v,
    [1, 2, 3],
    "行不等长 → 摊平成向量",
  );
  assert.deepEqual(listVal([vecVal([1, 2])]).v, [1, 2], "单行不成矩阵");
  assert.deepEqual(listVal([bool(true), bool(false)]).v, [1, 0]);
  assert.throws(() => listVal([strVal("a")]), /列表元素需要是数值/);
});

/* ============================================================== 线性代数 */

test("det：方阵行列式，空阵/非方阵给 NaN，奇异给 0", () => {
  near(det([[5]]), 5, 1e-12);
  near(det([[1, 2], [3, 4]]), -2, 1e-12);
  near(det([[6, 1, 1], [4, -2, 5], [2, 8, 7]]), -306, 1e-9);
  assert.ok(Number.isNaN(det([])), "空阵不是 0 阶行列式");
  assert.ok(Number.isNaN(det([[1, 2, 3], [4, 5]])), "非方阵");
  near(det([[1, 2], [2, 4]]), 0, 1e-12, "行成比例 → 奇异");
  // 行交换不改变 det 的值，只改变消元过程中的符号，末值应一致
  near(det([[0, 1], [1, 0]]), -1, 1e-12);
});

test("inv：奇异/空/非方阵给 null（调用方必须判空而不是直接用）", () => {
  nearMat(inv([[4, 7], [2, 6]])!, [[0.6, -0.7], [-0.2, 0.4]], 1e-12);
  assert.equal(inv([[1, 2], [2, 4]]), null, "奇异");
  assert.equal(inv([]), null);
  assert.equal(inv([[1, 2, 3], [4, 5]]), null, "非方阵");
  // inv(A)·A 必须是单位阵
  const A = [[4, 7], [2, 6]];
  nearMat(mulMat(inv(A)!, A), [[1, 0], [0, 1]], 1e-12);
});

test("mulMat：形状规则；右矩阵为空时抛 TypeError（不做保护）", () => {
  nearMat(mulMat([[1, 2], [3, 4]], [[0, 1], [1, 0]]), [[2, 1], [4, 3]], 1e-12);
  // 1×3 · 3×1 → 1×1
  nearMat(mulMat([[1, 2, 3]], [[1], [2], [3]]), [[14]], 1e-12);
  assert.throws(() => mulMat([[1, 2]], []), TypeError, "空矩阵会读 B[0].length 崩掉");
});

test("solveLin：高斯-约当消元，奇解/空/非方阵给 null", () => {
  const x = solveLin([[2, 1], [1, 3]], [5, 10])!;
  near(x[0], 1, 1e-12);
  near(x[1], 3, 1e-12);
  assert.equal(solveLin([[1, 2], [2, 4]], [1, 2]), null, "奇异无唯一解");
  assert.equal(solveLin([], []), null);
  assert.equal(solveLin([[1, 2, 3], [4, 5]], [1, 2]), null, "非方阵");
  // 3×3 需要换行的情况
  // 首列全 0，必须换行才能推进
  const y = solveLin([[0, 1, 0], [1, 0, 0], [0, 0, 2]], [3, 1, 8])!;
  near(y[0], 1, 1e-12, "x");
  near(y[1], 3, 1e-12, "y");
  near(y[2], 4, 1e-12, "z");
});

/* ============================================================== 数值分析 */

test("simpson 自适应积分：精确结果、退化区间、方向反转都给有限值", () => {
  near(simpson((x) => x * x, 0, 1), 1 / 3, 1e-12);
  near(simpson((x) => Math.sin(x), 0, Math.PI), 2, 1e-9);
  near(simpson(() => 3, 0, 2), 6, 1e-12, "常数函数精确");
  near(simpson((x) => x * x, 1, 1), 0, 1e-12, "零长区间");
  near(simpson((x) => x, 1, 0), -0.5, 1e-12, "反向区间给负值");
  // 实测：极窄尖峰会在递归深度 26 处被截断，无论 tol 多紧都停在 ~4% 偏低；
  // tol 只影响停止条件，不影响深度上限。这里如实记录实测值。
  const narrow = simpson((x) => 1 / (1 + 1000 * (x - 0.5) * (x - 0.5)), 0, 1);
  near(narrow, 0.0953512032284425, 1e-9, "窄尖峰被深度上限截断");
  assert.ok(Math.abs(narrow - Math.PI / Math.sqrt(1000)) < 0.005, "虽不精确但仍在 5% 以内");
  // tol 收紧几乎不改变结果（深度 26 是硬上限）
  near(
    simpson((x) => 1 / (1 + 1000 * (x - 0.5) * (x - 0.5)), 0, 1, 1e-12),
    narrow,
    1e-7,
    "收紧 tol 不再改进",
  );
});

test("bisect：区间有变号才收敛；函数值非实数会直接抛错（isFinite 守卫够不着）", () => {
  const eng = new Engine();
  near(bisect(eng, fn("x^2-2"), 0, 2), Math.SQRT2, 1e-9);
  near(bisect(eng, fn("x^2-2"), 0, 2), Math.SQRT2, 1e-9);
  assert.ok(Number.isNaN(bisect(eng, fn("1"), 0, 2)), "端点同号且放宽搜索仍无解 → NaN");
  near(bisect(eng, fn("x"), 0, 2), 0, 1e-12, "f(lo)=0 直接返回 lo");
  // 实测：0/0 求值为 NaN+NaNi，asReal 先抛错，bisect 的 isFinite 守卫走不到
  assert.throws(() => bisect(eng, fn("0/0"), 0, 2), /需要实数/);
  // cos(x)-x 端点同号，靠向外放宽区间找到根
  near(bisect(eng, fn("cos(x)-x"), 2, 4), 0.7390851332151607, 1e-6, "放宽搜索后仍能收敛");
});

test("newton：带阻尼的数值导数；无根时返回最接近的残差点而非 NaN", () => {
  const eng = new Engine();
  near(newton(eng, fn("cos(x)-x"), 1), 0.7390851332151607, 1e-6);
  near(newton(eng, fn("x^2-4"), 3), 2, 1e-5, "收敛到正根");
  // f≡1 无根：返回残差最小的初值本身
  near(newton(eng, fn("1"), 1.5), 1.5, 1e-12, "无根时返回初值");
  // f=x² 的根是 0，牛顿法停在残差极小处而非精确 0
  const r = newton(eng, fn("x^2"), 1);
  assert.ok(Math.abs(r) < 1e-6, `应收敛到 0 附近，实际 ${r}`);
});

test("newtonC：复牛顿法收敛到 z³=1 的某个根，并返回迭代步数", () => {
  const eng = new Engine();
  const f = fn("z^3-1", "z");
  const [re, im, steps] = newtonC(eng, f, 2, 0);
  near(re, 1, 1e-9, "实轴上出发收敛到 1");
  near(im, 0, 1e-9);
  assert.ok(steps < 60, `应在 60 步内收敛，实际 ${steps}`);
  // 从非实初值出发落到某个复根
  const [re2, im2] = newtonC(eng, f, 0.4, 0.9);
  // z³ = re³ − 3re·im² + i(3re²·im − im³)
  near(re2 ** 3 - 3 * re2 * im2 * im2 - 1, 0, 1e-6, "z³ 的实部应为 1");
  near(3 * re2 * re2 * im2 - im2 ** 3, 0, 1e-6, "z³ 的虚部应为 0");
  assert.ok(Math.hypot(re2, im2) > 1e-6, "不该收敛到原点");
  // 函数值非有限 → [NaN, NaN, k]
  const [nr, ni, ns] = newtonC(eng, fn("1/0", "z"), 1, 1);
  assert.ok(Number.isNaN(nr) && Number.isNaN(ni), "非有限残差返回 NaN");
  near(ns, 0, 1e-12, "第 0 步就退出");
});

test("polyfit：降幂返回系数；欠定/空数据返回整排 NaN", () => {
  const l = polyfit([0, 1, 2, 3], [1, 3, 5, 7], 1)!;
  near(l[0], 2, 1e-9, "斜率");
  near(l[1], 1, 1e-9, "截距");
  // 系数按高次→低次排：y=(x−1)² = x²−2x+1
  const q = polyfit([0, 1, 2, 3, 4], [1, 0, 1, 4, 9], 2)!;
  near(q[0], 1, 1e-8, "x²");
  near(q[1], -2, 1e-8, "x");
  near(q[2], 1, 1e-8, "常数");
  near(polyfit([0, 1, 2, 3], [1, 3, 5, 7], 0)![0], 4, 1e-9, "0 次 = 均值");
  assert.ok(polyfit([0, 1], [1, 2], 3)!.every(Number.isNaN), "2 点拟 3 次 → 欠定");
  assert.ok(polyfit([], [], 1)!.every(Number.isNaN), "空数据");
});

test("interp1：自动按 xs 排序、区间外取端点、重复 xs 取靠前那个", () => {
  near(interp1([0, 1, 2], [10, 20, 30], 1), 20, 1e-12, "命中节点");
  near(interp1([0, 10], [0, 100], 2.5), 25, 1e-12, "线性内插");
  near(interp1([0, 10], [5, 15], -1), 5, 1e-12, "左端外取左端值");
  near(interp1([0, 10], [5, 15], 99), 15, 1e-12, "右端外取右端值");
  near(interp1([10, 0, 5], [100, 0, 50], 7.5), 75, 1e-12, "xs 无序时先排序");
  assert.ok(Number.isNaN(interp1([], [], 1)), "空数组");
  near(interp1([0, 1, 1, 2], [0, 5, 9, 10], 1), 5, 1e-12, "重复 xs 取先出现的");
});

/* ============================================================ 表达式求值 */

test("算术与优先级：^ 右结合，一元负号先于 ^，% 与 / 行为", () => {
  near(ev("1+2*3"), 7);
  near(ev("2^10"), 1024);
  near(ev("7/2"), 3.5);
  near(ev("2^3^2"), 512, 1e-12, "右结合：2^(3^2)");
  near(ev("-2^2"), -4, 1e-12, "-(2^2)");
  near(ev("2^0.5"), Math.SQRT2, 1e-12);
  near(ev("7%3"), 1);
  assert.equal(ev("10^400"), Infinity, "溢出到 Infinity");
  assert.equal(evs("1e400"), "∞", "上屏文本用 ∞");
  assert.equal(evs("0/0"), "NaN+NaNi", "0/0 走复数路径，虚部也是 NaN");
  assert.equal(evs("0/0+1"), "NaN+NaNi", "NaN 继续传播，不是只污染一半");
});

test("复数提升：负数开方、负数取对数、abs/arg", () => {
  assert.equal(evs("sqrt(-1)"), "1i");
  assert.equal(evs("i*i"), "-1");
  assert.equal(evs("abs(3+4i)"), "5");
  near(ev("abs(3+4i)"), 5, 1e-12);
  near(ev("arg(i)"), Math.PI / 2, 1e-12);
  near(ev("sqrt(-1)*sqrt(-1)"), -1, 1e-12);
  // 复数上的超越函数不得退回实数分支
  const z = evs("ln(-1)");
  assert.match(z, /i$/, `ln(-1) 应落在复数轴上，实际 ${z}`);
});

test("向量/矩阵运算：广播补零、向量乘是点积、矩阵乘向量", () => {
  // 实测：标量参与加减时按"零填充到向量长度"处理，因此只改到第 0 个分量；
  // 标量乘法另有逐元素快路径。两者不一致，容易被误当成广播。
  assert.equal(evs("[1,2]+1"), "[2, 2]", "标量 + 只作用于首分量");
  assert.equal(evs("[1,2]+5"), "[6, 2]");
  assert.equal(evs("1+[1,2]"), "[2, 2]", "左右一致");
  assert.equal(evs("[1,2]-1"), "[0, 2]");
  assert.equal(evs("[1,2,3]+1"), "[2, 2, 3]");
  assert.equal(evs("[1,2,3]+[10,20]"), "[11, 22, 3]", "长度不等时右侧零填充");
  assert.equal(evs("[1,2]*3"), "[3, 6]", "标量 * 是逐元素的");
  assert.equal(evs("3*[1,2]"), "[3, 6]", "标量左乘也可");
  near(ev("[1,2]*[3,4]"), 11, 1e-12, "向量 × 向量 = 点积");
  assert.equal(evs("[[1,0],[0,1]]*[[2,3],[4,5]]"), "[2, 3]\n[4, 5]");
  assert.equal(evs("[[1,2],[3,4]]*[1,1]"), "[3, 7]", "矩阵乘向量");
  assert.equal(evs("[10,20,30][1]"), "10", "下标取值");
  assert.equal(evs("[]"), "[]", "空向量");
  assert.equal(evs("[1,0]/0"), "[∞, NaN]", "0/0 在向量里仍是 NaN");
  assert.equal(evs("[1,[2,3]]"), "[1, 2, 3]", "嵌套列表摊平");
});

test("内置数学函数：取整族、组合数、判素、gcd", () => {
  near(ev("abs(-3)"), 3);
  near(ev("max(1,5,3)"), 5);
  near(ev("min([4,2,9])"), 2);
  near(ev("sum([1,2,3])"), 6);
  near(ev("hypot(3,4)"), 5);
  near(ev("factorial(5)"), 120);
  near(ev("gamma(5)"), 24);
  near(ev("nCr(5,2)"), 10);
  assert.equal(evs("isprime(7)"), "真");
  near(ev("gcd(12,18)"), 6);
  near(ev("sign(-2)"), -1);
  near(ev("round(2.5)"), 3);
  near(ev("round(2.4)"), 2);
  near(ev("ceil(-1.5)"), -1, 1e-12, "向 +∞ 取整");
  near(ev("floor(-1.5)"), -2, 1e-12);
  near(ev("mod(-1,3)"), 2, 1e-12, "mod 结果取正");
  near(ev("erf(0)"), 0, 1e-12);
  near(ev("deg(pi)"), 180, 1e-12);
  near(ev("rad(180)"), Math.PI, 1e-12);
  assert.equal(evs("primes(5)"), "[2, 3, 5, 7, 11]", "primes(n) 取前 n 个素数");
  near(ev("det([[1,2],[3,4]])"), -2, 1e-12, "det 走线性代数实现");
  assert.equal(evs("inv([[4,7],[2,6]])"), "[0.6, -0.7]\n[-0.2, 0.4]");
  near(ev("norm([3,4])"), 5, 1e-12);
  near(ev("dot([1,2],[3,4])"), 11, 1e-12);
  assert.equal(evs("cross([1,0,0],[0,1,0])"), "[0, 0, 1]");
});

test("常量与逻辑：CONSTANTS 全可用，布尔可与数值互换", () => {
  near(ev("pi"), Math.PI, 1e-12);
  near(ev("e"), Math.E, 1e-12);
  near(ev("tau"), 2 * Math.PI, 1e-12);
  near(ev("phi"), (1 + Math.sqrt(5)) / 2, 1e-12);
  assert.equal(CONSTANTS.true, 1);
  assert.equal(CONSTANTS.false, 0);
  assert.equal(CONSTANTS.inf, Infinity);
  near(CONSTANTS.euler, 0.5772156649015329, 1e-15);
  assert.deepEqual(Object.keys(CONSTANTS), [
    "pi", "tau", "e", "phi", "euler", "inf", "infinity", "true", "false",
  ]);
  assert.equal(evs("1<2"), "真");
  assert.equal(evs("1>2"), "假");
  assert.equal(evs("1==1"), "真");
  assert.equal(evs("1!=1"), "假");
  assert.equal(evs("true"), "1", "true 是数值 1");
  assert.equal(evs("false"), "0");
  // 短路求值：右侧 1/0 不该被求值
  near(ev("0 && 1/0"), 0, 1e-12);
  near(ev("1 || 1/0"), 1, 1e-12);
});

test("错误路径：未定义名、未知函数、括号不配对、空输入都抛错", () => {
  evThrows("zzz", /未定义的名称/);
  evThrows("nosuchfn(1)", /未知函数/);
  evThrows("(1+2", /期望/);
  evThrows("", /空表达式/);
  evThrows("2**3", /多余的/);
  assert.throws(() => evalString(new Engine(), "1 and 1"), /未定义的名称/, "and 不是关键字，用 &&");
  assert.throws(() => parseExpr("没有这个名字"), /无法识别的字符/, "标识符不接受中文");
});

test("定义与绑定：k = 7 写入 globals，numericGlobals 按名字排序", () => {
  const e = new Engine();
  near(asReal(evalString(e, "k = 7")), 7);
  near(asReal(evalString(e, "k * 2")), 14, 1e-12, "后续表达式能引用");
  e.set("m", num(3));
  e.setNum("n", 4);
  e.setNum("z", 1, 2); // 复数
  assert.deepEqual(
    numericGlobals(e),
    [["k", 7, 0], ["m", 3, 0], ["n", 4, 0], ["z", 1, 2]],
    "复数全局也按 [name, re, im] 上榜，并按名字排序",
  );
  near(asReal(evalString(e, "k = 9")), 9, 1e-12, "可重定义");
  assert.deepEqual(numericGlobals(e)[0], ["k", 9, 0]);
  e.remove("k");
  assert.deepEqual(numericGlobals(e), [["m", 3, 0], ["n", 4, 0], ["z", 1, 2]]);
  // Engine.define 会注册函数；evalString 不会（实测记录）
  const e2 = new Engine();
  e2.define("g(x) = x^2 + 1");
  assert.equal(asReal(evalString(e2, "g(3)")), 10, 1e-12, "define 之后可以调用");
  const e3 = new Engine();
  evalString(e3, "g(x) = x^2 + 1");
  assert.throws(
    () => evalString(e3, "g(3)"),
    /未知函数/,
    "evalString 返回 funVal 但不写入 eng.fns，调用方需自行注册",
  );
  assert.equal(kindName(evalString(e3, "g(x) = x")), "函数", "定义式仍返回一个函数值");
});

/* ================================================================ 编译 */

test("compileReal：静态可判实数的表达式走无分配快路径，参数按 re/im 交错取槽", () => {
  const eng = new Engine();
  const fast = compileReal(eng, parseExpr("x^2 + y"), ["x", "y"])!;
  const s = new Float64Array(4);
  s[0] = 3;
  s[2] = 4;
  near(fast(s), 13, 1e-12, "实部在 s[2i]");
  s[1] = 0;
  near(fast(s), 13, 1e-12, "虚部不参与实数路径");
  // 取实部的表达式照走实数快路径；conj/arg 静态判定不了实数 → 返回 null 让调用方回退
  assert.equal(typeof compileReal(eng, parseExpr("re(z)"), ["z"]), "function", "re(z) 仍是实数");
  assert.equal(compileReal(eng, parseExpr("conj(z)"), ["z"]), null, "conj 静态判定失败 → null");
  assert.equal(compileReal(eng, parseExpr("arg(z)"), ["z"]), null, "arg 静态判定失败 → null");
  // 常量也能编译
  const cst = compileReal(eng, parseExpr("pi"), [])!;
  near(cst(new Float64Array(0)), Math.PI, 1e-12);
  // 未知名不能编译（静态判定失败 → 返回 null，调用方需回退到通用路径）
  assert.equal(compileReal(eng, parseExpr("zzz"), ["x"]), null);
});

test("compile：通用槽位求值，arity 与 names 透传", () => {
  const eng = new Engine();
  const node = parseExpr("x^2 + y");
  const c = compile(eng, node, ["x", "y"]);
  assert.equal(c.arity, 2);
  assert.deepEqual(c.names, ["x", "y"]);
  const s = new Float64Array(4);
  s[0] = 3;
  s[2] = 4;
  assert.equal(show(c.run(s)), "13");
  assert.ok(c.real, "能静态判实数时同时给出快路径");
  // 复用同一个 slots 缓冲必须看到新值
  s[0] = 1;
  s[2] = 1;
  assert.equal(show(c.run(s)), "2");
});

test("suggestParams：常量/函数/全局名不占槽位，上下文名优先，不足时用 SLOT_NAMES 补齐", () => {
  const eng = new Engine();
  assert.deepEqual(suggestParams(parseExpr("x+y"), eng, 2), ["x", "y"]);
  // 全局量已被占用，不能当自变量
  eng.globals.set("q", num(5));
  assert.deepEqual(suggestParams(parseExpr("q+w+e"), eng, 3), ["w", "y", "z"], "e 是常量被排除");
  // 补齐用 SLOT_NAMES[out.length]，所以已占掉 x 后下一个是 y
  assert.deepEqual(suggestParams(parseExpr("x+y"), eng, 20).length, 20);
  assert.deepEqual(suggestParams(parseExpr("x+y"), eng, 0), [], "take=0 给空");
  // 上下文变量排在最前
  setContextVars(["a", "b"]);
  assert.deepEqual(suggestParams(parseExpr("b+a+c"), eng, 3), ["a", "b", "c"]);
  assert.deepEqual(SLOT_NAMES, ["x", "y", "z", "t", "u", "v", "w", "s"]);
  assert.deepEqual([...freeNames(parseExpr("a*b+c"))].sort(), ["a", "b", "c"]);
  assert.equal(_sp, suggestParams, "别名同源");
});

/* ============================================================ show 格式化 */

test("show：定点/科学记数法随量级与位数切换，NaN/∞ 有专用记号", () => {
  assert.equal(show(num(3)), "3");
  assert.equal(show(num(1 / 3)), "0.333333", "默认 6 位有效数字");
  assert.equal(show(num(1 / 3), 2), "0.33");
  assert.equal(show(num(1e21)), "1.00000e+21", "大数切科学记数法");
  assert.equal(show(num(1e-7)), "1.00000e-7");
  assert.equal(show(num(1234567), 3), "1.23e+6");
  assert.equal(show(num(1e-7), 3), "1.00e-7");
  assert.equal(show(num(0)), "0");
  assert.equal(show(num(-0)), "0", "-0 归一成 0");
  assert.equal(show(num(NaN)), "NaN");
  assert.equal(show(num(Infinity)), "∞");
  assert.equal(show(num(1, 2)), "1+2i");
  assert.equal(show(num(0, 1)), "1i", "纯虚数不带实部");
  assert.equal(show(bool(true)), "真");
  assert.equal(show(bool(false)), "假");
  assert.equal(show(vecVal([1.5, 2, 3])), "[1.5, 2, 3]");
  assert.equal(show(vecVal([])), "[]");
  assert.equal(show(matVal([[1, 2], [3, 4]])), "[1, 2]\n[3, 4]", "矩阵每行一行");
  assert.equal(show(strVal("hi")), "hi");
  assert.equal(show(funVal(fn("x", "t"))), "f(t)");
});
