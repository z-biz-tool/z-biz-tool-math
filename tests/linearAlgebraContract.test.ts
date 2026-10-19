// 「同一个数学操作、两套实现」的**契约关系**测试。
// 跑法：node --experimental-strip-types --test tests/linearAlgebraContract.test.ts
//
// 为什么要有这个文件（01 的 A4）：
//   `linalg.ts`（库，给 UI 面板用）与 `machine.ts`（表达式求值器）**各有一份**
//   线性代数实现：det / inv / mulMat / solveLin。
//   2026-10-04 那轮审计把它记成「没量到：两者在方阵上是否逐例一致、该不该合并」。
//   本轮**量完了**，结论是：
//
//   ① 方阵上两者**数学等价**，差异只是浮点舍入（实测最大绝对差 3e-13）。
//   ② 但**错误策略不同**，而且在若干边界上**行为真的不同**：
//      非方阵 det、空矩阵 inv、右端向量长度不对时的 solveLin ——
//      这些地方**合并会改变可观察行为**，属于产品决定。
//   ⇒ 本文件把「哪些地方一致、哪些地方不一致」**钉成可执行的断言**，
//     这样将来若要合并，diff 会直接显示"哪一条行为被改了"，而不是悄悄漂掉。
//
// 本文件里标了【现状刻画】的断言锁的是**当前真实行为（含疑似缺陷）**，
// 不是"认可"—— 疑似缺陷在 01 的 A4 里逐条列了复现与建议。
import { test } from "node:test";
import assert from "node:assert/strict";
import { det as lDet, inverse, matMul, solve as lSolve } from "../src/core/linalg.ts";
import { det as mDet, inv as mInv, mulMat, solveLin } from "../src/core/machine.ts";

/** 线性同余伪随机：固定种子 ⇒ 结果可复现（不依赖 Math.random） */
function lcg(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

/** 造一个 n×n 整数矩阵（元素落在 [-10, 10]） */
function randMat(n: number, rnd: () => number): number[][] {
  return Array.from({ length: n }, () =>
    Array.from({ length: n }, () => Math.floor(rnd() * 21) - 10)
  );
}

function closeN(a: number, b: number, tol: number, msg = ""): void {
  assert.ok(
    Number.isFinite(a) && Math.abs(a - b) <= tol,
    `${msg} 期望 ${b}，实得 ${a}（容差 ${tol}）`
  );
}

// ── 方阵：两者数学等价 ────────────────────────────────────────────────

test("det：方阵上两套实现数学等价（差异只是浮点舍入）", () => {
  const rnd = lcg(20261019);
  let maxAbs = 0;
  for (let n = 1; n <= 6; n++) {
    for (let k = 0; k < 120; k++) {
      const a = randMat(n, rnd);
      // 掺入奇异矩阵：两套实现在奇异点都应给 0
      if (k % 5 === 0 && n > 1) a[n - 1] = a[0].slice();
      const x = lDet(a);
      const y = mDet(a);
      const abs = Math.abs(x - y);
      if (abs > maxAbs) maxAbs = abs;
      const scale = Math.max(1, Math.abs(x), Math.abs(y));
      assert.ok(
        abs <= 1e-9 * scale,
        `n=${n} 矩阵 ${JSON.stringify(a)}：linalg=${x} machine=${y}，差 ${abs}`
      );
    }
  }
  // 实测最大绝对差约 3e-13（浮点舍入量级），给一个宽松但有意义的上界
  assert.ok(maxAbs < 1e-8, `最大绝对差 ${maxAbs} 应在浮点舍入量级`);
});

test("inv：方阵上「有解/无解」的判定一致，数值一致", () => {
  const rnd = lcg(20261020);
  for (let n = 1; n <= 5; n++) {
    for (let k = 0; k < 120; k++) {
      const a = randMat(n, rnd);
      if (k % 5 === 0 && n > 1) a[n - 1] = a[0].slice();
      const x = mInv(a);
      let y: number[][] | null = null;
      let threw = false;
      try {
        y = inverse(a);
      } catch {
        threw = true;
      }
      // 奇异矩阵：machine 返回 null，linalg 抛错 —— 判定"都没有逆"是一致的
      if (x === null || threw) {
        assert.ok(x === null, `n=${n}：machine 说奇异时必须返回 null`);
        continue;
      }
      assert.ok(y !== null, `n=${n}：machine 说有解时 linalg 也不能说没有`);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          closeN(x[i][j], y[i][j], 1e-8 * Math.max(1, Math.abs(y[i][j])), `inv[${i}][${j}]：`);
        }
      }
    }
  }
});

test("solveLin：方阵上「有解/无解」的判定一致，数值一致", () => {
  const rnd = lcg(20261021);
  for (let n = 1; n <= 5; n++) {
    for (let k = 0; k < 120; k++) {
      const a = randMat(n, rnd);
      if (k % 5 === 0 && n > 1) a[n - 1] = a[0].slice();
      const b = Array.from({ length: n }, () => Math.floor(rnd() * 21) - 10);
      const x = solveLin(a, b);
      let y: number[] | null = null;
      let threw = false;
      try {
        y = lSolve(a, b);
      } catch {
        threw = true;
      }
      if (x === null || threw) {
        assert.ok(x === null, `n=${n}：machine 说无解时必须返回 null`);
        continue;
      }
      assert.ok(y !== null, `n=${n}：machine 说有解时 linalg 也不能说没有`);
      for (let i = 0; i < n; i++) {
        closeN(x[i], y[i], 1e-8 * Math.max(1, Math.abs(y[i])), `解[${i}]：`);
      }
    }
  }
});

test("mulMat：两套实现逐元素完全相同", () => {
  const rnd = lcg(20261022);
  for (let n = 1; n <= 5; n++) {
    for (let k = 0; k < 80; k++) {
      const a = randMat(n, rnd);
      const b = randMat(n, rnd);
      const x = mulMat(a, b);
      const y = matMul(a, b);
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          assert.equal(x[i][j], y[i][j], `mul[${i}][${j}]：必须逐位相同`);
        }
      }
    }
  }
});

// ── 边界：两者行为**真的不同**的地方（现状刻画） ─────────────────────

test("det【现状刻画】：非方阵时两套实现给的是**两个不同的东西**", () => {
  // linalg：2×3 ⇒ 静默丢掉第三列，取左上 2×2 求 det = 1×5 − 2×4 = −3
  // machine：2×3 ⇒ NaN
  // 这条差异是**产品行为**：linalg 那条已被 linalg.test.ts 钉住
  // （"非方阵不被拒绝，而是静默取前 n×n 块 —— 现状刻画，不是认可"）。
  closeN(lDet([[1, 2, 3], [4, 5, 6]]), -3, 1e-12, "linalg 的非方阵 det：");
  assert.ok(Number.isNaN(mDet([[1, 2, 3], [4, 5, 6]])), "machine 的非方阵 det 必须是 NaN");
});

test("det【现状刻画】：行长不一致的畸形矩阵，linalg 抛错而 machine 给 NaN", () => {
  assert.throws(() => lDet([[1, 2], [3]]), "linalg 对畸形输入抛错");
  assert.ok(Number.isNaN(mDet([[1, 2], [3]])), "machine 对畸形输入给 NaN");
});

test("det【现状刻画】：3 行 2 列时两者都给 NaN", () => {
  assert.ok(Number.isNaN(lDet([[1, 2], [3, 4], [5, 6]])));
  assert.ok(Number.isNaN(mDet([[1, 2], [3, 4], [5, 6]])));
});

test("inv【现状刻画】：空矩阵时 linalg 返回空矩阵、machine 返回 null", () => {
  // linalg.inverse([]) 走 LU 分解，0×0 直接返回 []
  assert.deepEqual(inverse([]), []);
  assert.equal(mInv([]), null);
});

test("solveLin【现状刻画，疑似缺陷】：右端向量长度不对时，machine 返回的是 **NaN 数组**而不是 null", () => {
  // machine.solveLin 用 b[i] 取值，b 短于 n ⇒ 取到 undefined ⇒ 消元出 NaN。
  // 函数末尾的有限性检查在**每个分量**上判 ⇒ 整个数组被 NaN 填满，
  // 但**数组本身是 truthy** ⇒ 调用方的 `if (!sol)` 判不出来。
  // 调用点（machine.ts 的 STRICT.solve / polyfit）因此会拿到 NaN 向量。
  const sol = solveLin([[1, 2], [3, 4]], []);
  assert.ok(Array.isArray(sol), "返回的是数组");
  assert.equal(sol.length, 2);
  assert.ok(sol.every((v) => Number.isNaN(v)), "每个分量都是 NaN");
  assert.ok(sol, "⚠️ 但数组是 truthy —— 调用方的 `if (!sol)` 判不出来（这就是疑似缺陷）");

  // linalg 走的是另一条路：直接抛"右端向量长度应为 2"
  assert.throws(
    () => lSolve([[1, 2], [3, 4]], []),
    /右端向量长度/,
    "linalg 明确报错"
  );
});

test("solveLin【现状刻画】：右端向量过长时，machine 默默忽略多余项", () => {
  const sol = solveLin([[1, 2], [3, 4]], [1, 2, 3]);
  assert.ok(sol !== null);
  closeN(sol[0], 0, 1e-9, "x：");
  closeN(sol[1], 0.5, 1e-9, "y：");
  assert.throws(() => lSolve([[1, 2], [3, 4]], [1, 2, 3]), /右端向量长度/);
});

// ── 结论：合并的前提 ──────────────────────────────────────────────────

test("【结论】两套实现的差异只在错误策略与边界；方阵上的数值差异是浮点舍入", () => {
  // 这条不是断言某个具体值，而是把"合并的前提"写成可核对的：
  //   ① 方阵数值一致（上面几条已逐例验证）
  //   ② 边界行为不同（本文件已逐条列出）
  // ⇒ 合并的前提是**先统一错误策略**（表达式求值器遇错返回 null/NaN，
  //   还是抛错），那是产品决定，不是重构能顺手带过去的。
  const square = [[1, 2], [3, 4]];
  closeN(lDet(square), mDet(square), 1e-12, "方阵上必须一致：");
  assert.ok(
    Number.isNaN(mDet([[1, 2, 3], [4, 5, 6]])) && !Number.isNaN(lDet([[1, 2, 3], [4, 5, 6]])),
    "非方阵上两者必须仍然不同（否则上面那条「现状刻画」就过期了）"
  );
});
