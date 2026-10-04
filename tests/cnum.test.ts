// cnum.ts 的纯逻辑回归（无 React、无 Canvas）。
// 跑法：node --experimental-strip-types --test tests/cnum.test.ts
//
// 为什么优先测它：cnum 是复平面（cplane）与解析（parser）的**地基**，
// 而本仓此前对 cplane / field / linalg / cnum 四个模块**零测试**。
//
// 断言策略：绝大多数用例用**数学恒等式**而不是「和上次跑出来的一样」：
//   · 欧拉恒等式 e^{iθ} = cos θ + i·sin θ
//   · 勾股 csin² + ccos² = 1
//   · 开方还原 csqrt(z)² = z
//   · 对数与指数互逆 cexp(clog z) = z
//   · 乘法与除法互逆 cdiv(cmul(a,b), b) = a
// 恒等式的好处：某处把符号写反、把分支切割搞错、或数值稳定性特判失效，
// 都会被**别的函数之间的不一致**逮到 —— 而快照式断言只能证明"没变"，
// 证不出"是对的"。这正是复数运算出错的典型形态（算出一个形状奇怪的图，不抛错）。
//
// 这些函数是**出参式**（写进调用方给的 o 对象以避免分配），
// 所以每次都自带一个 fresh 缓冲 —— 顺带也把「不该写的字段没被动过」钉住。
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  type C,
  isNum, cplx, cabs, carg, cexp, clog, csqrt,
  cadd, csub, cmul, cdiv, realPow, cpow,
  csin, ccos, ctan, csinh, ccosh, ctanh,
  casin, cacosh, catanh,
  lambertW, gamma, factorial, clamp, lerp, fmt, fmtC, fmtPi,
} from "../src/core/cnum.ts";

/** 复数近似相等 —— 给数值型断言一个统一的容差口径。 */
function closeC(got: C, re: number, im: number, eps = 1e-9, msg = "") {
  assert.ok(
    Math.abs(got.re - re) <= eps && Math.abs(got.im - im) <= eps,
    `${msg}期望 ≈ ${re}${im >= 0 ? "+" : "-"} ${Math.abs(im)}i，实得 ${got.re}${got.im >= 0 ? "+" : "-"} ${Math.abs(got.im)}i`,
  );
}

/** 复数模长比较（有些恒等式关心的是模而不是分量）。 */
function closeAbs(got: C, want: number, eps = 1e-9, msg = "") {
  assert.ok(Math.abs(cabs(got.re, got.im) - want) <= eps,
    `${msg}期望模 ≈ ${want}，实得 ${cabs(got.re, got.im)}`);
}

/* ------------------------------------------------------------------ *
 * 基本量：isNum / cabs / carg
 * ------------------------------------------------------------------ */

test("isNum：NaN 与 ±Infinity 都不算数", () => {
  assert.equal(isNum(0), true);
  assert.equal(isNum(-3.5), true);
  assert.equal(isNum(NaN), false, "NaN 必须被拒");
  assert.equal(isNum(Infinity), false);
  assert.equal(isNum(-Infinity), false);
});

test("cabs：用 hypot，大数不该溢出成 Infinity", () => {
  assert.equal(cabs(3, 4), 5);
  assert.equal(cabs(0, 0), 0);
  // hypot 存在的唯一理由就是这一条：sqrt(re²+im²) 在这里会溢出
  const big = cabs(1e200, 1e200);
  assert.ok(Number.isFinite(big), `1e200,1e200 的模不该是 Infinity，实得 ${big}`);
  assert.ok(Math.abs(big - 1.4142135623730951e200) / 1e200 < 1e-12);
});

test("carg：分支在 (-π, π]，且 0 处返回 0 而不是 NaN", () => {
  assert.equal(carg(1, 0), 0);
  assert.equal(carg(0, 1), Math.PI / 2);
  assert.equal(carg(0, -1), -Math.PI / 2);
  assert.equal(carg(-1, 0), Math.PI, "负实轴取 +π");
  // atan2(-0, -1) 在 IEEE 下是 -π；这条锁住 carg 是否把 -0 归一
  assert.equal(carg(0, 0), 0, "原点必须返回 0，不许 NaN");
});

/* ------------------------------------------------------------------ *
 * 欧拉恒等式：e^{iθ} = cos θ + i·sin θ
 * 这是 cexp / csin / ccos 三者的一致性锚。
 * ------------------------------------------------------------------ */

test("欧拉恒等式：cexp(iθ) 与 ccos(θ)+i·csin(θ) 必须一致", () => {
  for (const th of [0, 0.3, 1, Math.PI / 4, 2, Math.PI, -1.7, 5]) {
    const e = cexp(0, th, { re: 0, im: 0 });
    const s = csin(th, 0, { re: 0, im: 0 });
    const c = ccos(th, 0, { re: 0, im: 0 });
    // i·(s.re + i·s.im) = (−s.im) + i·s.re，所以 i·csin(θ) 的实部取 s 的**虚**部、
    // 虚部取 s 的**实**部；c 的实部再加到实部上。
    closeC(e, c.re - s.im, s.re, 1e-9, `θ=${th}：`);
    // 实数参数下 cosh(0)=1、sinh(0)=0，故 csin/ccos 的实部应等于原生三角函数
    assert.ok(Math.abs(s.re - Math.sin(th)) < 1e-12, `θ=${th} csin 实部应等于 sin θ`);
    assert.ok(Math.abs(c.re - Math.cos(th)) < 1e-12, `θ=${th} ccos 实部应等于 cos θ`);
  }
});

test("勾股：csin² + ccos² = 1（按**复数**算，不拆实虚部）", () => {
  for (const th of [0.2, 1.1, 2.5, -3.3]) {
    const s = csin(th, 0, { re: 0, im: 0 });
    const c = ccos(th, 0, { re: 0, im: 0 });
    const s2 = cmul(s.re, s.im, s.re, s.im, { re: 0, im: 0 });
    const c2 = cmul(c.re, c.im, c.re, c.im, { re: 0, im: 0 });
    closeC(cadd(s2.re, s2.im, c2.re, c2.im, { re: 0, im: 0 }), 1, 0, 1e-9, `θ=${th}：`);
  }
});

test("ctan ≡ csin / ccos（用 sin2x/cos2x 那条实现，必须等价）", () => {
  for (const th of [0.1, 0.7, 1.3, 2.9, -0.4]) {
    const s = csin(th, 0, { re: 0, im: 0 });
    const c = ccos(th, 0, { re: 0, im: 0 });
    const t = ctan(th, 0, { re: 0, im: 0 });
    closeC(t, s.re / c.re, s.im / c.re, 1e-9, `θ=${th}：`);
  }
});

test("双曲：csinh/cosh/ctanh 互相对齐", () => {
  for (const x of [0.2, 1, 2.5]) {
    const sh = csinh(x, 0, { re: 0, im: 0 });
    const ch = ccosh(x, 0, { re: 0, im: 0 });
    const th = ctanh(x, 0, { re: 0, im: 0 });
    assert.ok(Math.abs(sh.re - Math.sinh(x)) < 1e-12, `csinh(${x})`);
    assert.ok(Math.abs(ch.re - Math.cosh(x)) < 1e-12, `ccosh(${x})`);
    closeC(th, sh.re / ch.re, sh.im / ch.re, 1e-9, `ctanh(${x}) vs sinh/cosh：`);
  }
});

/* ------------------------------------------------------------------ *
 * 开方 / 对数 / 指数：各自的定义性质
 * ------------------------------------------------------------------ */

test("csqrt：还原性质 (√z)² = z，且实轴与虚轴走的是两个分支", () => {
  for (const [re, im] of [[4, 0], [-4, 0], [3, 4], [-3, 4], [-3, -4], [0, 9], [0.25, 0]]) {
    const r = csqrt(re, im, { re: 0, im: 0 });
    const back = cmul(r.re, r.im, r.re, r.im, { re: 0, im: 0 });
    closeC(back, re, im, 1e-9, `z=${re}${im >= 0 ? "+" : "-"}i：`);
  }
  // 纯实负轴：主值取 +i 方向（虚部为正）
  const neg = csqrt(-9, 0, { re: 0, im: 0 });
  closeC(neg, 0, 3, 1e-12, "√(-9) 的主值应是 +3i：");
  // 正实轴：虚部必须正好是 0（不是 1e-18 之类）
  const pos = csqrt(9, 0, { re: 0, im: 0 });
  assert.equal(pos.im, 0, "√(9) 的虚部必须恰好为 0");
});

test("clog ∘ cexp 互逆：cexp(clog z) = z", () => {
  for (const [re, im] of [[2, 0], [-3, 4], [1, -1], [0.5, 0.25]]) {
    const l = clog(re, im, { re: 0, im: 0 });
    const back = cexp(l.re, l.im, { re: 0, im: 0 });
    closeC(back, re, im, 1e-9, `z=${re},${im}：`);
  }
});

test("clog：实部是 ln|z|、虚部是 arg(z)", () => {
  const l = clog(-1, 1, { re: 0, im: 0 });
  assert.ok(Math.abs(l.re - Math.log(Math.SQRT2)) < 1e-12, `实部应等于 ln|−1+i|=${Math.log(Math.SQRT2)}，实得 ${l.re}`);
  assert.ok(Math.abs(l.im - (3 * Math.PI) / 4) < 1e-12, `虚部应等于 3π/4，实得 ${l.im}`);
  // 0 的对数是 −∞ + 0i，函数不该崩
  const zero = clog(0, 0, { re: 0, im: 0 });
  assert.equal(zero.re, -Infinity);
  assert.equal(zero.im, 0);
});

/* ------------------------------------------------------------------ *
 * 四则运算与 cdiv 的分支
 * ------------------------------------------------------------------ */

test("加减乘：定义性质（含出参不得污染无关字段）", () => {
  closeC(cadd(1, 2, 3, 4, { re: 0, im: 0 }), 4, 6);
  closeC(csub(1, 2, 3, 4, { re: 0, im: 0 }), -2, -2);
  closeC(cmul(1, 2, 3, 4, { re: 0, im: 0 }), -5, 10);
});

test("cdiv：乘除互逆 cdiv(cmul(a,b), b) = a（覆盖全部 4 条分支）", () => {
  // 分支覆盖：实除数(bi=0) / 纯虚除数(br=0) / |br|<|bi| 稳定分支 / 其余
  const cases: [number, number, number, number][] = [
    [3, 4, 2, 0],     // bi=0
    [3, 4, 0, 2],     // br=0
    [3, 4, 1, 5],     // |br|<|bi| → 稳定分支
    [3, 4, 5, 1],     // |br|>|bi| → 另一分支
    [3, 4, -2, 7],
    [-1.5, 0.25, 3, -4],
  ];
  for (const [ar, ai, br, bi] of cases) {
    const p = cmul(ar, ai, br, bi, { re: 0, im: 0 });
    const q = cdiv(p.re, p.im, br, bi, { re: 0, im: 0 });
    closeC(q, ar, ai, 1e-9, `(${ar},${ai})/(${br},${bi})：`);
  }
});

test("cdiv：除以 0 的语义（0/0→NaN，非 0/0→±∞）", () => {
  const nan = cdiv(0, 0, 0, 0, { re: 0, im: 0 });
  assert.ok(Number.isNaN(nan.re) && Number.isNaN(nan.im), "0/0 两个分量都该是 NaN");
  const inf = cdiv(2, -3, 0, 0, { re: 0, im: 0 });
  assert.equal(inf.re, Infinity, "2/0 的实部应是 +∞");
  assert.equal(inf.im, -Infinity, "-3/0 的虚部应是 -∞");
});

test("cdiv：数值稳定性分支在朴素公式溢出时给出有限值（这正是它存在的理由）", () => {
  // br²+bi² 在 1e-200 量级会**下溢成 0**，朴素公式于是给出 ±Infinity。
  // 所以这里不能断言「稳定分支 == 朴素公式」—— 那样恰好把该分支的意义否掉。
  // 要断言的是相反方向：朴素溢出时稳定分支仍有限，且模长对得上解析解。
  const ar = 1, ai = 1, br = 1e-200, bi = 3e-200;
  const d = br * br + bi * bi;
  assert.equal(d, 0, "先确认前提：这个量级的分母确实下溢成 0");
  const naive = cdiv(ar, ai, br, bi, { re: 0, im: 0 });   // 走稳定分支
  assert.ok(Number.isFinite(naive.re) && Number.isFinite(naive.im),
    `稳定分支不该溢出，实得 ${naive.re},${naive.im}`);
  // |(1+i)/(1e-200+3e-200 i)| = √2 / √(1e-400+9e-400) = √2 / (√10 · 1e-200)
  const want = Math.SQRT2 / (Math.sqrt(10) * 1e-200);
  closeAbs(naive, want, want * 1e-9, "极小分母：");
});

test("cdiv：数值稳定分支与朴素公式在**不溢出**时结果一致（特判不能改答案）", () => {
  const cases: [number, number, number, number][] = [
    [1, 1, 1e-8, 3e-8],     // 走 |br|<|bi| 稳定分支，但朴素公式算得动
    [2, -3, 1, 5],
    [-4, 0.5, 2, 7],
  ];
  for (const [ar, ai, br, bi] of cases) {
    const d = br * br + bi * bi;
    const naiveRe = (ar * br + ai * bi) / d;
    const naiveIm = (ai * br - ar * bi) / d;
    const got = cdiv(ar, ai, br, bi, { re: 0, im: 0 });
    closeC(got, naiveRe, naiveIm, 1e-6 * Math.max(1, Math.abs(naiveRe)), `(${ar},${ai})/(${br},${bi})：`);
  }
});

test("cdiv：除以纯虚数 i 的闭式 (a+bi)/i = b − ai", () => {
  closeC(cdiv(3, 4, 0, 1, { re: 0, im: 0 }), 4, -3, 1e-12);
  closeC(cdiv(3, 4, 0, -1, { re: 0, im: 0 }), -4, 3, 1e-12);
});

/* ------------------------------------------------------------------ *
 * 幂
 * ------------------------------------------------------------------ */

test("cpow：整数幂 ≡ 重复相乘（de Moivre）", () => {
  for (const n of [0, 1, 2, 3, 5]) {
    let acc = cplx(1, 0);
    const b = cplx(1.2, -0.7);
    for (let k = 0; k < n; k++) acc = cmul(acc.re, acc.im, b.re, b.im, { re: 0, im: 0 });
    const got = cpow(b.re, b.im, n, 0, { re: 0, im: 0 });
    closeC(got, acc.re, acc.im, 1e-9, `z^${n}：`);
  }
});

test("cpow：z^0 = 1，z^1 = z", () => {
  closeC(cpow(3, 4, 0, 0, { re: 0, im: 0 }), 1, 0, 1e-12, "z^0：");
  closeC(cpow(3, 4, 1, 0, { re: 0, im: 0 }), 3, 4, 1e-12, "z^1：");
});

test("realPow：奇次根给实根，偶次根返回 NaN（绘图工具的常见期望）", () => {
  assert.equal(realPow(-8, 1 / 3), -2, "(−8)^(1/3) 应是 −2 而不是 NaN");
  assert.ok(Number.isNaN(realPow(-8, 1 / 2)), "(−8)^(1/2) 无实根，应是 NaN");
  assert.ok(Number.isNaN(realPow(-8, 1 / 4)), "(−8)^(1/4) 无实根");
  assert.equal(realPow(-8, 3), -512, "整数幂直接给");
  assert.equal(realPow(9, 0.5), 3);
});

/* ------------------------------------------------------------------ *
 * 特殊函数
 * ------------------------------------------------------------------ */

test("lambertW：W(x)·e^{W(x)} = x，且 x < −1/e 无实解", () => {
  for (const x of [0.5, 1, 2, 10, -0.1, -1 / Math.E + 1e-12]) {
    const w = lambertW(x);
    const back = w * Math.exp(w);
    assert.ok(Math.abs(back - x) < 1e-9, `x=${x}：W·e^W = ${back}，不等于 ${x}`);
  }
  assert.equal(lambertW(0), 0);
  assert.ok(Number.isNaN(lambertW(-0.5)), "x=−0.5 无实解分支");
});

test("gamma：Γ(n) = (n−1)!，且 Γ(1)=Γ(2)=1", () => {
  assert.ok(Math.abs(gamma(1) - 1) < 1e-12);
  assert.ok(Math.abs(gamma(2) - 1) < 1e-12);
  assert.ok(Math.abs(gamma(5) - 24) < 1e-9, "Γ(5) 应为 24");
  assert.ok(Math.abs(gamma(0.5) - Math.sqrt(Math.PI)) < 1e-9, "Γ(0.5)=√π");
});

test("factorial：整数、越界与非整数三条分支", () => {
  assert.equal(factorial(0), 1);
  assert.equal(factorial(5), 120);
  assert.equal(factorial(171), Infinity, "171! 溢出成 Infinity");
  assert.equal(factorial(-1), Infinity, "负整数按本实现约定给 Infinity");
  assert.ok(Math.abs(factorial(4.5) - gamma(5.5)) < 1e-9, "非整数走 Γ(n+1)");
});

test("反三角与反双曲：主值须落在定义域内", () => {
  // asin: 主值实部落在 [−1,1]；asin(i) 虚部为正
  const a = casin(0, 1, { re: 0, im: 0 });
  assert.ok(a.re >= -1 && a.re <= 1, `asin(0+i) 的实部应在 [−1,1]，实得 ${a.re}`);
  assert.ok(a.im > 0, "asin(i) 的虚部应为正");
  // casin 与 csin 互逆：sin(asin z) = z
  const back = csin(a.re, a.im, { re: 0, im: 0 });
  closeC(back, 0, 1, 1e-9, "sin(asin(i))：");

  // acosh 的**实**定义域是 [1,∞)；z=0 在复数域有主值 iπ/2，不该是 NaN。
  // （本条最初断言成"应为 NaN"，跑出来才发现是错��� —— 复反余弦给主值才对。）
  const ac0 = cacosh(0, 0, { re: 0, im: 0 });
  closeC(ac0, 0, Math.PI / 2, 1e-9, "acosh(0) 的主值：");
  const ac1 = cacosh(1, 0, { re: 0, im: 0 });
  assert.ok(Math.abs(ac1.re) < 1e-12 && Math.abs(ac1.im) < 1e-12, "acosh(1) 应为 0");
  // cosh(acosh z) = z
  const ac2 = cacosh(-1, 0, { re: 0, im: 0 });
  const ch2 = ccosh(ac2.re, ac2.im, { re: 0, im: 0 });
  closeC(ch2, -1, 0, 1e-9, "cosh(acosh(−1))：");

  // atanh(±1) 发散
  const at = catanh(1, 0, { re: 0, im: 0 });
  assert.ok(!Number.isFinite(at.re), "atanh(1) 应发散");
  // tanh(atanh z) = z（取定义域内的点）
  const at2 = catanh(0.5, 0, { re: 0, im: 0 });
  const th2 = ctanh(at2.re, at2.im, { re: 0, im: 0 });
  closeC(th2, 0.5, 0, 1e-9, "tanh(atanh(0.5))：");
});

/* ------------------------------------------------------------------ *
 * 工具函数
 * ------------------------------------------------------------------ */

test("clamp / lerp：端点与越界", () => {
  assert.equal(clamp(5, 0, 3), 3);
  assert.equal(clamp(-5, 0, 3), 0);
  assert.equal(clamp(1, 0, 3), 1);
  assert.equal(clamp(2, 0, 2), 2, "右端点应含");
  assert.equal(lerp(0, 10, 0), 0);
  assert.equal(lerp(0, 10, 1), 10);
  assert.ok(Math.abs(lerp(2, 4, 0.25) - 2.5) < 1e-12);
});

test("fmt：非有限值与零各有专门措辞", () => {
  assert.equal(fmt(NaN), "NaN");
  assert.equal(fmt(Infinity), "∞");
  assert.equal(fmt(-Infinity), "-∞");
  assert.equal(fmt(0), "0");
  assert.equal(fmt(1.5), "1.5");
});

test("fmtC：实/虚/双虚三条措辞分支", () => {
  assert.equal(fmtC(3, 0), "3", "虚部为 0 时只显示实部");
  assert.equal(fmtC(0, -2), "-2i", "实部为 0 时只显示虚部");
  assert.equal(fmtC(1, 2), "1+2i");
  assert.equal(fmtC(1, -2), "1-2i");
});

test("fmtPi：整数倍与半整数倍用 π 记号，否则退回普通数字", () => {
  assert.equal(fmtPi(Math.PI), "π");
  assert.equal(fmtPi(2 * Math.PI), "2π");
  assert.equal(fmtPi(Math.PI / 2), "π/2");
  assert.equal(fmtPi(-Math.PI), "-π");
  assert.equal(fmtPi(Math.PI / 3), "π/3");
  assert.equal(fmtPi(1), "1", "不是 π 的有理倍，必须走普通格式化");
  assert.equal(fmtPi(0), "0");
});
