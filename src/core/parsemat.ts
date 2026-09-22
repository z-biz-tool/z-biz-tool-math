/** 界面上的矩阵/向量输入格：每格一个表达式串，与函数模式共用同一个表达式内核 */
import { Engine, evalString } from "./machine.ts";
import { VK } from "./types.ts";
import type { Mat } from "./linalg.ts";
import type { Val } from "./types.ts";

export function parseCell(eng: Engine, src: string, i: number): number {
  const where = `第 ${i + 1} 格`;
  const t = src.trim();
  if (!t) throw new Error(`${where}为空`);
  let v: Val;
  try {
    v = evalString(eng, t);
  } catch (e) {
    throw new Error(`${where}：${(e as Error).message}`);
  }
  if (v.k !== VK.Num) throw new Error(`${where}需要实数`);
  if (!Number.isFinite(v.re) || v.im !== 0) throw new Error(`${where}需要有限实数`);
  return v.re;
}

/** 行优先展开的 dim² 个串 → 矩阵 */
export function parseMatrix(eng: Engine, cells: string[], dim: number): Mat {
  if (cells.length !== dim * dim) throw new Error(`需要 ${dim * dim} 个元素，实际 ${cells.length}`);
  const out: Mat = [];
  for (let i = 0; i < dim; i++) {
    const row: number[] = [];
    for (let j = 0; j < dim; j++) row.push(parseCell(eng, cells[i * dim + j], i * dim + j));
    out.push(row);
  }
  return out;
}

export function parseVector(eng: Engine, cells: string[], dim: number): number[] {
  if (cells.length !== dim) throw new Error(`需要 ${dim} 个元素，实际 ${cells.length}`);
  return cells.map((c, i) => parseCell(eng, c, i));
}
