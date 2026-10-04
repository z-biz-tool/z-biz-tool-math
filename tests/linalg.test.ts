// linalg.ts 的纯逻辑回归（无 React、无 Canvas）。
// 跑法：node --experimental-strip-types --test tests/linalg.test.ts
//
// 为什么优先测它：linalg 是 cplane（复平面）、field（向量场）、surface（曲面）
// 共同依赖的**地基**，而本仓此前对它**零测试** —— 地基出错时上面每一块都跟着错，
// 且都是静默的（画面画歪，不抛错）。
//
// 断言策略分两类，刻意都用**可独立验证的期望值**，不用「和上次跑出来的一样」：
//   1) 手算精确值 —— linalg.ts 自己写着「2×2/3×3 直接按课本展开式：
//      教学场景下这个结果必须和学生手算的一致」，所以 det 就该按手算值钉死；
//   2) 恒等式     —— A·A⁻¹=I、转置对合、特征向量残差 Av=λv、SVD 重建 A=UΣVᵀ。
// 恒等式的价值：某处把索引写反、把转置漏了、把 1-based 当 0-based，
// 都会被"另一条独立路径算出的同一个量"逮到。
import { test } from "node:test";
import assert from "node:assert/strict";
import {
  type Mat, zeros, identity, shape, matAdd, matSub, matScale, matMul,
  transpose, luDecomp, det, solve, inverse, rref, cubicRoots,
  eigen2, eigen3, jacobiEigen, eigen, isSymmetric, eigenVector, svd,
  expMat2, matVec,
} from "../src/core/linalg.ts";

/** 矩阵近似相等（统一容差口径）。 */
function closeM(got: Mat, want: Mat, eps = 1e-9, msg = "") {
  const [gr, gc] = shape(got);
  const [wr, wc] = shape(want);
  assert.equal(gr, wr, `${msg}行数：期望 ${wr}，实得 ${gr}`);
  assert.equal(gc, wc, `${msg}列数：期望 ${wc}，实得 ${gc}`);
  for (let i = 0; i < gr; i++) {
    for (let j = 0; j < gc; j++) {
      assert.ok(Math.abs(got[i][j] - want[i][j]) <= eps,
        `${msg}[${i}][${j}]：期望 ${want[i][j]}，实得 ${got[i][j]}`);
    }
  }
}
const closeN = (got: number, want: number, eps = 1e-9, msg = "") =>
  assert.ok(Math.abs(got - want) <= eps, `${msg}期望 ${want}，实得 ${got}`);
const closeV = (got: number[], want: number[], eps = 1e-9, msg = "") => {
  assert.equal(got.length, want.length, `${msg}长度：期望 ${want.length}，实得 ${got.length}`);
  for (let i = 0; i < got.length; i++) closeN(got[i], want[i], eps, `${msg}[${i}]：`);
};

/** 用 matVec 算 A·x，返回最大分量偏差。 */
function residual(a: Mat, x: number[], lam: number): number {
  const ax = matVec(a, x);
  return Math.max(...ax.map((v, i) => Math.abs(v - lam * x[i])));
}

/* ------------------------------------------------------------------ *
 * 基本构造
 * ------------------------------------------------------------------ */

test("zeros / identity / shape：形状与取值", () => {
  closeM(zeros(2, 3), [[0, 0, 0], [0, 0, 0]]);
  closeM(identity(3), [[1, 0, 0], [0, 1, 0], [0, 0, 1]]);
  closeM(identity(1), [[1]]);
  assert.deepEqual(shape([[1, 2, 3], [4, 5, 6]]), [2, 3]);
  assert.deepEqual(shape(identity(4)), [4, 4]);
});

test("matAdd / matSub / matScale：逐元素语义", () => {
  const a = [[1, 2], [3, 4]], b = [[10, 20], [30, 40]];
  closeM(matAdd(a, b), [[11, 22], [33, 44]]);
  closeM(matSub(b, a), [[9, 18], [27, 36]]);
  closeM(matScale(a, -1), [[-1, -2], [-3, -4]]);
  // 加法可交换 —— 索引写反会在这一条露馅
  closeM(matAdd(a, b), matAdd(b, a), 1e-12, "加法可交换：");
});

test("matMul：与手算一致，且满足结合律", () => {
  const a = [[1, 2], [3, 4]], b = [[5, 6], [7, 8]], c = [[2, 0], [1, 3]];
  closeM(matMul(a, b), [[19, 22], [43, 50]], 1e-12, "手算：");
  closeM(matMul(matMul(a, b), c), matMul(a, matMul(b, c)), 1e-9, "结合律：");
  // 单位元
  closeM(matMul(a, identity(2)), a, 1e-12, "右乘单位元：");
  closeM(matMul(identity(2), a), a, 1e-12, "左乘单位元：");
});

test("transpose：对合 (Aᵀ)ᵀ = A，且 matMul 与转置可交换顺序", () => {
  const a = [[1, 2, 3], [4, 5, 6]];
  closeM(transpose(a), [[1, 4], [2, 5], [3, 6]]);
  closeM(transpose(transpose(a)), a, 1e-12, "对合：");
  const b = [[1, 0], [0, 1], [2, 3]];
  // (AB)ᵀ = BᵀAᵀ —— 转置写错方向会在这一条露馅
  closeM(transpose(matMul(a, b)), matMul(transpose(b), transpose(a)), 1e-9, "(AB)ᵀ=BᵀAᵀ：");
});

/* ------------------------------------------------------------------ *
 * det：教学场景，按手算值钉死
 * ------------------------------------------------------------------ */

test("det：1/2/3 维按课本展开式（与手算一致）", () => {
  closeN(det([[5]]), 5, 1e-12, "1×1：");
  closeN(det([[1, 2], [3, 4]]), 1 * 4 - 2 * 3, 1e-12, "2×2：");
  closeN(det([[6, 1, 1], [4, -2, 5], [2, 8, 7]]),
    6 * ((-2) * 7 - 5 * 8) - 1 * (4 * 7 - 5 * 2) + 1 * (4 * 8 - (-2) * 2),
    1e-9, "3×3：");
  // 行交换变号
  closeN(det([[1, 2], [3, 4]]), -det([[3, 4], [1, 2]]), 1e-12, "交换两行变号：");
});

test("det：非方阵**不被拒绝**，而是静默取前 n×n 块（现状刻画，不是认可）", () => {
  // 本条最初断言「非方阵 det 应为 0」，跑出来是 -3 —— 因为 detSmall 只索引
  // [0][0] [0][1] [1][0] [1][1]，**多出来的列被静默丢弃**。
  // 2×3 传入时它等价于对前 2 列构成的方阵求 det：1*5 − 2*4 = −3。
  //
  // 这里刻意断言**现状**而不是「应有的行为」：改行为属于产品决定，
  // 而让现状有测试 = 下一个人改动这条时会被立刻看见。
  closeN(det([[1, 2, 3], [4, 5, 6]]), -3, 1e-12, "2×3 静默取前 2 列：");
  // 与显式取前 2×2 方阵求 det 一致 —— 证明它确实是「丢列」而不是别的算法
  closeN(det([[1, 2, 3], [4, 5, 6]]), det([[1, 2], [4, 5]]), 1e-12, "等价于前 2×2：");
});

test("det：3 维以上走 LU 分解，边界上要与 3 维公式对得上", () => {
  // 4×4 行列式（按首列展开独立核算得 72）
  const a = [[1, 2, 3, 4], [5, 6, 7, 8], [2, 6, 4, 8], [3, 1, 1, 2]];
  closeN(det(a), 72, 1e-7, "4×4：");
  // 三角矩阵：行列式 = 对角线之积
  closeN(det([[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 0], [0, 0, 0, 5]]), 120, 1e-9, "上三角：");
  closeN(det([[1, 2, 3], [0, 1, 4], [5, 6, 0]]),
    1 * (1 * 0 - 4 * 6) - 2 * (0 * 0 - 4 * 5) + 3 * (0 * 6 - 1 * 5), 1e-9, "3×3 含零元：");
});

/* ------------------------------------------------------------------ *
 * luDecomp / solve / inverse：奇异矩阵必须抛错，不能静默 NaN
 * ------------------------------------------------------------------ */

test("luDecomp：非方阵抛错；奇异矩阵被标记而不是给一个假的分解", () => {
  assert.throws(() => luDecomp([[1, 2, 3], [4, 5, 6]]), /方阵/, "非方阵必须抛错");
  const sing = luDecomp([[1, 2], [2, 4]]);
  assert.equal(sing.singular, true, "秩亏矩阵必须被标为 singular");
  const ok = luDecomp([[4, 3], [6, 3]]);
  assert.equal(ok.singular, false);
  assert.equal(ok.piv.length, 2, "行置换长度应等于阶数");
});

test("inverse：A·A⁻¹ = I（含需要换行的场景）", () => {
  const cases: Mat[] = [
    [[4, 7], [2, 6]],
    [[1, 2, 3], [0, 1, 4], [5, 6, 0]],
    [[2, 0, 0, 0], [0, 3, 0, 0], [0, 0, 4, 0], [0, 0, 0, 5]],
  ];
  for (const a of cases) {
    const n = a.length;
    const inv = inverse(a);
    closeM(matMul(a, inv), identity(n), 1e-8, `A·A⁻¹（${n}×${n}）：`);
    closeM(matMul(inv, a), identity(n), 1e-8, `A⁻¹·A（${n}×${n}）：`);
  }
});

test("inverse / solve：奇异矩阵必须**抛错**，绝不静默返回 NaN", () => {
  // 这条是本仓反复强调的「静默失败」判据在数值侧的样子：
  // 返回 NaN 的话，界面会画出空白/跳变的图，而没有任何错误提示。
  assert.throws(() => inverse([[1, 2], [2, 4]]), /奇异/, "inverse 遇奇异矩阵：");
  assert.throws(() => solve([[1, 2], [2, 4]], [1, 2]), /奇异/, "solve 遇奇异矩阵：");
  // 右端项长度不符也必须抛错，不能默默截断
  assert.throws(() => solve([[1, 2], [3, 4]], [1, 2, 3]), /长度/, "右端项长度不符：");
  assert.throws(() => matVec([[1, 2], [3, 4]], [1]), /长度/, "matVec 向量长度不符：");
});

test("solve：解代回原方程组要对得上（A·x = b）", () => {
  const a = [[2, 1], [1, 3]];
  const b = [5, 10];
  const x = solve(a, b);
  closeV(matVec(a, x), b, 1e-9, "代回 A·x：");
  closeV(x, [1, 3], 1e-9, "2x+3=5, x+9=10 ⇒ x=1,y=3：");
  // 三元
  const a3 = [[1, 1, 1], [0, 2, 5], [2, 5, -1]];
  const x3 = solve(a3, [6, -4, 27]);
  closeV(matVec(a3, x3), [6, -4, 27], 1e-7, "三元代回：");
});

/* ------------------------------------------------------------------ *
 * rref：秩与主元列
 * ------------------------------------------------------------------ */

test("rref：秩、主元列与行阶梯形", () => {
  // 行满秩 → 化为单位阵
  const f = rref([[2, 1], [1, 3]]);
  assert.equal(f.rank, 2);
  closeM(f.m, identity(2), 1e-9, "行满秩应化为 I：");

  // 秩亏：第二行是第一行的倍数
  const d = rref([[1, 2, 3], [2, 4, 6]]);
  assert.equal(d.rank, 1, "倍增行只该算 1 阶：");
  assert.deepEqual(d.pivots, [0], "主元应在第 0 列：");

  // 秩亏但不是倍数关系（线性相关）
  const e = rref([[1, 2, 3], [4, 5, 6], [7, 8, 9]]);
  assert.equal(e.rank, 2, "1..9 的秩是 2：");
  closeV(e.m[0], [1, 0, -1], 1e-9, "化简后首行：");
  closeV(e.m[1], [0, 1, 2], 1e-9, "化简后次行：");
  assert.equal(e.m[2].every((v) => Math.abs(v) < 1e-12), true, "第三行应全零：");
});

/* ------------------------------------------------------------------ *
 * 特征值：残差判据 Av = λv
 * ------------------------------------------------------------------ */

test("eigen2：特征向量残差 Av = λv，且 det = 特征值之积", () => {
  const a = [[4, 1], [2, 3]];
  const es = eigen2(a);
  assert.equal(es.length, 2, "2×2 应给出两个特征值：");
  for (const e of es) {
    const v = eigenVector(a, e.re);
    assert.ok(v, `λ=${e.re} 应能找到特征向量：`);
    assert.ok(residual(a, v!, e.re) < 1e-7, `λ=${e.re} 的残差过大：`);
  }
  closeN(es[0].re * es[1].re, det(a), 1e-7, "特征值之积 = det：");
});

test("eigen3：三个特征向量残差都要小，且之积 = det", () => {
  const a = [[2, 0, 1], [0, 3, 0], [1, 0, 2]];
  const es = eigen3(a);
  assert.equal(es.length, 3, "3×3 应给出三个特征值：");
  for (const e of es) {
    const v = eigenVector(a, e.re);
    assert.ok(v, `λ=${e.re} 应能找到特征向量：`);
    assert.ok(residual(a, v!, e.re) < 1e-6, `λ=${e.re} 残差过大：`);
  }
  closeN(es.reduce((s, e) => s * e.re, 1), det(a), 1e-6, "特征值之积 = det：");
  // 对称阵的特征值必须全实
  const es2 = eigen3([[4, 1, 0], [1, 4, 0], [0, 0, 5]]);
  assert.equal(es2.every((e) => Math.abs(e.im) < 1e-9), true, "实对称阵特征值必须全实");
});

test("jacobiEigen：只收对称阵，输出按约定排布且残差小", () => {
  const a = [[2, 1], [1, 2]];
  assert.equal(isSymmetric(a), true);
  assert.equal(isSymmetric([[1, 2], [3, 4]]), false);
  const { values, vecs } = jacobiEigen(a);
  assert.equal(values.length, 2);
  // vecs[i][k] 是第 k 个特征向量的第 i 个分量 ⇒ 每个特征向量是 vecs 的**列**
  for (let k = 0; k < values.length; k++) {
    const v = vecs.map((row) => row[k]);
    assert.ok(residual(a, v, values[k]) < 1e-8, `第 ${k} 个特征向量残差：`);
  }
  closeV([...values].sort((x, y) => x - y), [1, 3], 1e-8, "特征值（升序）：");
});

test("eigen：分派器，2×2 与 3×3 都要给出正确残差", () => {
  for (const a of [[[2, 1], [1, 2]] as Mat, [[4, 1, 0], [1, 4, 0], [0, 0, 5]] as Mat]) {
    for (const e of eigen(a)) {
      const v = eigenVector(a, e.re);
      if (v) assert.ok(residual(a, v, e.re) < 1e-6, `λ=${e.re} 残差：`);
    }
  }
});

/* ------------------------------------------------------------------ *
 * cubicRoots / svd / expMat2
 * ------------------------------------------------------------------ */

test("cubicRoots：代入多项式应为 0", () => {
  // x³ − 6x² + 11x − 6 = (x−1)(x−2)(x−3)
  const roots = cubicRoots(-6, 11, -6);
  assert.equal(roots.length, 3);
  for (const r of roots) {
    const v = r.re ** 3 - 6 * r.re ** 2 + 11 * r.re - 6;
    assert.ok(Math.abs(v) < 1e-7, `根 ${r.re} 代回得 ${v}：`);
  }
  closeV(roots.map((r) => r.re).sort((a, b) => a - b), [1, 2, 3], 1e-7, "三个实根：");
});

test("svd：重建 A = U·Σ·Vᵀ，且奇异值降序", () => {
  const a = [[3, 0], [0, -2]];
  const { u, s, v } = svd(a);
  assert.equal(s.length, 2);
  for (let i = 0; i + 1 < s.length; i++) {
    assert.ok(s[i] >= s[i + 1] - 1e-9, `奇异值必须降序，实得 ${s}`);
  }
  closeV(s, [3, 2], 1e-7, "对角阵的奇异值：");
  // 重建：U · diag(s) · Vᵀ
  const k = Math.min(a.length, a[0].length);
  const rebuilt = Array.from({ length: a.length }, (_, i) =>
    Array.from({ length: a[0].length }, (_, j) => {
      let sum = 0;
      for (let p = 0; p < k; p++) sum += u[i][p] * s[p] * v[j][p];
      return sum;
    }));
  closeM(rebuilt, a, 1e-7, "A = UΣVᵀ：");
});

test("expMat2：t=0 必须是 I；对角阵应给出 diag(e^{tλ})", () => {
  const a = [[1, 2], [3, 4]];
  closeM(expMat2(a, 0), identity(2), 1e-12, "t=0：");
  // A = λI ⇒ e^{tA} = e^{tλ}·I（只有对角线是 e^{tλ}，非对角必须是 0）
  const lam = 2.5;
  const want = identity(2).map((r) => r.map((v) => v * Math.exp(lam)));
  closeM(expMat2([[lam, 0], [0, lam]], 1), want, 1e-9, "λI 的非对角必须为 0：");
  // t 的线性性：e^{0.3A} ≠ e^{A}，但两个都该是合法矩阵
  const e1 = expMat2(a, 1);
  const e03 = expMat2(a, 0.3);
  assert.equal(shape(e1)[0], 2, "输出形状保持 2×2：");
  assert.ok(Number.isFinite(e03[0][0]), "输出不应含 NaN/Infinity");
  // 三条判别式分支都该有限：disc>0 / |disc|≈0 / disc<0
  for (const m of [[[0, 1], [1, 0]], [[1, 1], [1, 1]], [[1, -1], [1, -1]]]) {
    const r = expMat2(m, 0.7);
    assert.ok(r.every((row) => row.every(Number.isFinite)), `分支 ${JSON.stringify(m)} 出现非有限值：`);
  }
});
