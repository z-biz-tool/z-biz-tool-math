/** GeoLab 数学核心：表达式语言与场景对象的类型定义 */

export type Vec2 = [number, number];
export type Vec3 = [number, number, number];

/** 值类型标签（不使用 const enum，兼容 isolatedModules） */
export const VK = { Num: 0, Bool: 1, Vec: 2, Mat: 3, Fun: 4, Str: 5 } as const;
export type ValKind = (typeof VK)[keyof typeof VK];

export interface FunValue {
  name: string;
  params: string[];
  body: Node;
}

/** 运行期值：数值（含复数）走 re/im，其它类型走引用字段 */
export interface Val {
  k: ValKind;
  re: number;
  im: number;
  b: boolean;
  /** 向量分量 */
  v: number[] | null;
  /** 矩阵：行优先 */
  m: number[][] | null;
  fn: FunValue | null;
  /** 文本（标注用） */
  s: string | null;
}

export type BinOp =
  | "+"
  | "-"
  | "*"
  | "/"
  | "\\"
  | "%"
  | "^"
  | "=="
  | "!="
  | "<"
  | "<="
  | ">"
  | ">="
  | "&&"
  | "||";

export type UnOp = "-" | "+" | "!";

export type Node =
  | { type: "num"; value: number }
  | { type: "str"; value: string }
  | { type: "ident"; name: string }
  | { type: "bin"; op: BinOp; l: Node; r: Node }
  | { type: "un"; op: UnOp; e: Node }
  | { type: "post"; op: "!" | "°"; e: Node }
  | { type: "call"; name: string; args: Node[] }
  | { type: "idx"; e: Node; i: Node }
  | { type: "list"; items: Node[] }
  | { type: "cond"; c: Node; a: Node; b: Node };

/** 一行工作区定义 */
export interface Definition {
  kind: "expr" | "let" | "fn";
  name: string;
  params: string[];
  node: Node;
  src: string;
}

export class GeoError extends Error {
  pos: number;
  constructor(msg: string, pos = -1) {
    super(pos >= 0 ? `${msg}（位置 ${pos}）` : msg);
    this.name = "GeoError";
    this.pos = pos;
  }
}
