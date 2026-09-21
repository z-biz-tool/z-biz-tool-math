/**
 * GeoLab 表达式语言：词法 + Pratt 递归下降语法分析
 *
 * 设计目标：贴近教科书书写习惯
 *  - 隐式乘法：`2x`、`3sin(x)`、`(x+1)(x-1)`、`2(x+1)`
 *  - 复数字面量：`3+2i`、`(1+i)/(1-i)`
 *  - 多种幂写法：`x^2`、`x**2`、`x²`
 *  - 向量/矩阵字面量：`[1,2,3]`、`[[1,2],[3,4]]`、`<1,2,3>`
 *  - 派生记号：`f'(x)`、`f''(x)`
 *  - 后缀单位：`90°`、`3!`
 */

import type { Node, Definition, BinOp, UnOp } from "./types.ts";
export type { Definition } from "./types.ts";
import { GeoError } from "./types.ts";

export type TokKind =
  | "num"
  | "ident"
  | "op"
  | "("
  | ")"
  | "["
  | "]"
  | "<"
  | ">"
  | ","
  | ";"
  | ":"
  | "="
  | "eof";

export interface Tok {
  kind: TokKind;
  text: string;
  value: number;
  pos: number;
}

/** 已知函数名：其后紧跟左括号时按调用解析，否则按隐式乘法处理 */
export const BUILTIN_NAMES = new Set([
  "sin", "cos", "tan", "sec", "csc", "cot",
  "asin", "acos", "atan", "asec", "acsc", "acot", "atan2",
  "sinh", "cosh", "tanh", "coth", "asinh", "acosh", "atanh",
  "exp", "exp2", "exp10", "log", "ln", "log2", "log10", "lg", "logb",
  "sqrt", "cbrt", "root", "abs", "sign", "arg", "conj", "re", "im", "polar",
  "floor", "ceil", "round", "trunc", "frac", "mod", "rem", "min", "max",
  "gcd", "lcm", "factorial", "gamma", "lgamma", "beta", "erf",
  "sinpi", "cospi", "sinc", "step", "dirac", "clamp", "hypot",
  "if", "when", "sum", "product", "seq", "list", "vec", "mat", "range", "linspace",
  "dot", "cross", "norm", "len", "magnitude", "angle", "unit", "proj", "component",
  "det", "inv", "transpose", "matmul", "identity", "solve",
  "diff", "derivative", "integrate", "quad", "fzero", "roots", "fsolve",
  "polyfit", "polyval", "interp1", "mean", "var", "std", "median", "sort",
  "rand", "randn", "primes", "isprime", "nCr", "nPr", "binomial",
  "piecewise", "minimize", "maximize", "limit", "series",
  "normc", "abs2", "phase", "complex", "reim", "compress",
]);

const SUPERSCRIPT: Record<string, string> = {
  "⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4",
  "⁵": "5", "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9",
  "⁻": "-", "·": "*", "×": "*", "∗": "*", "÷": "/",
  "≤": "<=", "≥": ">=", "≠": "!=", "≪": "<<", "‑": "-",
  "–": "-", "—": "-", "－": "-", "（": "(", "）": ")",
  "［": "[", "］": "]", "，": ",", "；": ";", "：": ":", "＿": "_",
};

const GREEK: Record<string, string> = {
  "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
  "ζ": "zeta", "η": "eta", "θ": "theta", "λ": "lambda", "μ": "mu",
  "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau",
  "φ": "phi", "χ": "chi", "ψ": "psi", "ω": "omega", "Δ": "Delta",
  "Φ": "Phi", "Ω": "Omega", "Σ": "Sigma", "∞": "inf", "ℯ": "e",
};

export function normalize(src: string): string {
  let out = "";
  for (const ch of src) {
    if (SUPERSCRIPT[ch] !== undefined) out += SUPERSCRIPT[ch];
    else if (GREEK[ch] !== undefined) out += " " + GREEK[ch] + " ";
    else out += ch;
  }
  return out;
}

export function tokenize(src: string): Tok[] {
  const s = normalize(src);
  const toks: Tok[] = [];
  let i = 0;
  const n = s.length;
  const isIdStart = (c: string) => /[A-Za-z_]/.test(c);
  const isIdPart = (c: string) => /[A-Za-z_0-9]/.test(c);
  const isDigit = (c: string) => c >= "0" && c <= "9";

  while (i < n) {
    const c = s[i];
    if (c === " " || c === "\t" || c === "\n" || c === "\r") {
      i++;
      continue;
    }
    // 行内注释：# 或 //
    if (c === "#" || (c === "/" && s[i + 1] === "/")) {
      while (i < n && s[i] !== "\n") i++;
      continue;
    }
    const start = i;
    if (isDigit(c) || (c === "." && isDigit(s[i + 1] ?? ""))) {
      let j = i;
      if (c === "0" && (s[1 + i] === "x" || s[1 + i] === "X")) {
        j = i + 2;
        while (j < n && /[0-9a-fA-F]/.test(s[j])) j++;
        toks.push({ kind: "num", text: s.slice(i, j), value: parseInt(s.slice(i + 2, j), 16), pos: start });
        i = j;
        continue;
      }
      while (j < n && (isDigit(s[j]) || s[j] === "." || s[j] === "_")) j++;
      if (j < n && (s[j] === "e" || s[j] === "E")) {
        let k = j + 1;
        if (k < n && (s[k] === "+" || s[k] === "-")) k++;
        if (k < n && isDigit(s[k])) {
          k++;
          while (k < n && isDigit(s[k])) k++;
          j = k;
        }
      }
      const text = s.slice(i, j).replace(/_/g, "");
      toks.push({ kind: "num", text, value: Number(text), pos: start });
      i = j;
      continue;
    }
    if (isIdStart(c)) {
      let j = i;
      while (j < n && isIdPart(s[j])) j++;
      let text = s.slice(i, j);
      // 吸收尾部下标：theta1 -> theta_1 之类保持原样即可
      toks.push({ kind: "ident", text, value: NaN, pos: start });
      i = j;
      continue;
    }
    const two = s.slice(i, i + 2);
    if (two === "**" || two === "==" || two === "!=" || two === "<=" || two === ">=" || two === "&&" || two === "||" || two === "<>") {
      toks.push({ kind: "op", text: two === "<>" ? "!=" : two, value: NaN, pos: start });
      i += 2;
      continue;
    }
    if ("+-*/^%!<>".includes(c)) {
      toks.push({ kind: "op", text: c, value: NaN, pos: start });
      i++;
      continue;
    }
    if (c === "(") { toks.push({ kind: "(", text: c, value: NaN, pos: start }); i++; continue; }
    if (c === ")") { toks.push({ kind: ")", text: c, value: NaN, pos: start }); i++; continue; }
    if (c === "[") { toks.push({ kind: "[", text: c, value: NaN, pos: start }); i++; continue; }
    if (c === "]") { toks.push({ kind: "]", text: c, value: NaN, pos: start }); i++; continue; }
    if (c === "{") { toks.push({ kind: "[", text: c, value: NaN, pos: start }); i++; continue; }
    if (c === "}") { toks.push({ kind: "]", text: c, value: NaN, pos: start }); i++; continue; }
    if (c === ",") { toks.push({ kind: ",", text: c, value: NaN, pos: start }); i++; continue; }
    if (c === ";") { toks.push({ kind: ";", text: c, value: NaN, pos: start }); i++; continue; }
    if (c === ":") { toks.push({ kind: ":", text: c, value: NaN, pos: start }); i++; continue; }
    if (c === "=") { toks.push({ kind: "=", text: c, value: NaN, pos: start }); i++; continue; }
    if (c === "'") { toks.push({ kind: "op", text: "'", value: NaN, pos: start }); i++; continue; }
    if (c === "°") { toks.push({ kind: "op", text: "°", value: NaN, pos: start }); i++; continue; }
    if (c === "|") {
      // `|` 只在求值上下文（如范数）出现，这里忽略为分隔
      i++;
      continue;
    }
    if (c === '"') {
      let j = i + 1;
      while (j < n && s[j] !== '"') j++;
      toks.push({ kind: "ident", text: "@str:" + s.slice(i + 1, j), value: NaN, pos: start });
      i = Math.min(j + 1, n);
      continue;
    }
    throw new GeoError(`无法识别的字符 “${c}”`, start);
  }
  toks.push({ kind: "eof", text: "", value: NaN, pos: n });
  return toks;
}

/** 合并相邻上标，让 `x^2^3` 成为右结合链 */
const POW: BinOp = "^";

function prec(op: string): number {
  switch (op) {
    case "||": return 1;
    case "&&": return 2;
    case "==": case "!=": return 3;
    case "<": case "<=": case ">": case ">=": return 4;
    case "+": case "-": return 5;
    case "*": case "/": case "%": case "mod": return 6;
    case "°": return 7;
    case POW: return 8;
    default: return 0;
  }
}

class P {
  toks: Tok[];
  i = 0;
  constructor(toks: Tok[]) {
    this.toks = toks;
  }
  peek(): Tok {
    return this.toks[this.i];
  }
  next(): Tok {
    return this.toks[this.i++];
  }
  eat(kind: TokKind, text?: string): Tok | null {
    const t = this.peek();
    if (t.kind === kind && (text === undefined || t.text === text)) {
      this.i++;
      return t;
    }
    return null;
  }
  expect(kind: TokKind, text?: string): Tok {
    const t = this.eat(kind, text);
    if (!t) throw new GeoError(`期望 “${text ?? kind}”，实际是 “${this.peek().text || "结尾"}”`, this.peek().pos);
    return t;
  }

  /** 顶层：允许 `a,b,c` 表达式序列（用于参数曲线） */
  top(): Node {
    const first = this.expr(0);
    if (this.peek().kind === ",") {
      const items = [first];
      while (this.eat(",")) items.push(this.expr(0));
      return { type: "list", items };
    }
    return first;
  }

  expr(minp: number): Node {
    let lhs = this.unary();
    for (;;) {
      const t = this.peek();
      let op: string | null = null;
      let implicit = false;
      if (t.kind === "op") op = t.text;
      else if (t.kind === "ident" && t.text === "mod") op = "mod";
      else if (t.kind === "num" || t.kind === "(" || t.kind === "[" || t.kind === "ident") {
        // 隐式乘法：下一 token 开启了一个新原子，例如 2x、3sin(x)、(a+1)(a-1)、2i
        if (this.canImplicit()) {
          op = "*";
          implicit = true;
        }
      }
      if (op === null) break;
      const p = prec(op);
      if (p < minp || p === 0) break;
      // 隐式乘法没有运算符 token 可消耗
      if (!implicit) this.next();
      const minp2 = op === POW ? p : p + 1;
      const rhs = this.expr(minp2);
      lhs = { type: "bin", op: (op === "mod" ? "%" : op) as BinOp, l: lhs, r: rhs };
    }
    return lhs;
  }

  /** 是否允许隐式乘法：下一 token 自成一个原子即可，如 `2x`、`2(x)`、`(a)(b)`、`x y` */
  canImplicit(): boolean {
    const t = this.peek();
    if (t.kind === "(" || t.kind === "[" || t.kind === "num" || t.kind === "ident") return true;
    return false;
  }

  unary(): Node {
    const t = this.peek();
    if (t.kind === "op" && (t.text === "-" || t.text === "+")) {
      this.next();
      /* 一元负号只吞掉紧接着的因子：运算元的最小优先级取加法级 +1，
         否则 `-x^2-y^2` 会被解析成 `-(x^2-y^2)` */
      const e = this.expr(prec("-") + 1);
      if (t.text === "-") {
        if (e.type === "num") return { type: "num", value: -e.value };
        return { type: "un", op: "-" as UnOp, e };
      }
      return e;
    }
    if (t.kind === "ident" && (t.text === "not" || t.text === "NOT")) {
      this.next();
      return { type: "un", op: "!" as UnOp, e: this.expr(3) };
    }
    return this.postfix();
  }

  postfix(): Node {
    let e = this.atom();
    for (;;) {
      const t = this.peek();
      if (t.kind === "op" && (t.text === "!" || t.text === "°")) {
        this.next();
        e = { type: "post", op: t.text, e };
        continue;
      }
      if (t.kind === "op" && t.text === "'") {
        // 导数记号：f' → 函数；f'(x) → 数值导数；f'' → 二阶
        this.next();
        const primes = this.countPrimes();
        const asCall = this.peek().kind === "(";
        if (asCall) {
          this.next();
          const args: Node[] = [];
          if (this.peek().kind !== ")") {
            for (;;) {
              args.push(this.expr(0));
              if (!this.eat(",")) break;
            }
          }
          this.expect(")");
          e = { type: "call", name: "diff", args: [e, args[0] ?? { type: "ident", name: "x" }] };
          for (let k = 1; k < primes; k++) {
            e = { type: "call", name: "diff", args: [e, args[0] ?? { type: "ident", name: "x" }] };
          }
        } else {
          e = { type: "call", name: "derivative", args: [e] };
          for (let k = 1; k < primes; k++) e = { type: "call", name: "derivative", args: [e] };
        }
        continue;
      }
      if (t.kind === "[") {
        this.next();
        const idx = this.expr(0);
        this.expect("]");
        e = { type: "idx", e, i: idx };
        continue;
      }
      break;
    }
    return e;
  }

  countPrimes(): number {
    let n = 1;
    while (this.peek().kind === "op" && this.peek().text === "'") {
      this.next();
      n++;
    }
    return n;
  }

  atom(): Node {
    const t = this.next();
    switch (t.kind) {
      case "num":
        return { type: "num", value: t.value };
      case "(": {
        const e = this.top();
        this.expect(")");
        return e;
      }
      case "[": {
        const items: Node[] = [];
        if (this.peek().kind !== "]") {
          // `[a, b; c, d]` —— 分号分行
          let row: Node[] = [];
          const rows: Node[] = [];
          for (;;) {
            row.push(this.expr(0));
            if (this.eat(",")) continue;
            if (this.eat(";")) {
              rows.push({ type: "list", items: row });
              row = [];
              if (this.peek().kind === "]") break;
              continue;
            }
            break;
          }
          if (row.length) rows.push({ type: "list", items: row });
          if (rows.length === 1) {
            const only = rows[0];
            if (only.type === "list") items.push(...only.items);
          } else {
            items.push(...rows);
          }
        }
        this.expect("]");
        return { type: "list", items };
      }
      case "<": {
        // `<1,2,3>` 向量
        const items: Node[] = [];
        for (;;) {
          items.push(this.expr(0));
          if (!this.eat(",")) break;
        }
        this.expect(">");
        return { type: "list", items };
      }
      case "ident": {
        const name = t.text;
        if (name.startsWith("@str:")) return { type: "str", value: name.slice(5) };
        if (this.peek().kind === "(") {
          this.next();
          const args: Node[] = [];
          if (this.peek().kind !== ")") {
            for (;;) {
              args.push(this.expr(0));
              if (!this.eat(",")) break;
            }
          }
          this.expect(")");
          return { type: "call", name, args };
        }
        // 虚数单位：`i` / `j`
        if (name === "i" || name === "j") {
          return { type: "call", name: "__imag", args: [] };
        }
        return { type: "ident", name };
      }
      default:
        throw new GeoError(`意外的符号 “${t.text || "结尾"}”`, t.pos);
    }
  }
}

export function parseExpr(src: string): Node {
  const p = new P(tokenize(src));
  if (p.peek().kind === "eof") throw new GeoError("空表达式", 0);
  const node = p.top();
  const rest = p.peek();
  if (rest.kind !== "eof") throw new GeoError(`多余的 “${rest.text}”`, rest.pos);
  return node;
}

/**
 * 解析一行输入：
 *   `f(x) = x^2`        → fn
 *   `a = 2.5` / `a:=2.5` → let
 *   `sin(x)+1`          → expr
 * 尾部 `;` 静默，`,` 表达式序列作为 list。
 */
/**
 * 解析一行输入：
 *   `f(x) = x^2`         → fn      （左侧必须是 名称(参数列表) 且顶层有单个 =）
 *   `a = 2.5` / `pi2 = 2` → let
 *   `sin(x)+1`           → expr
 * 只有当 = 出现在顶层（不在任何括号内）且左侧形式合法时才当作定义，
 * 否则整行按表达式解析——这样 `sqrt(-4)`、`f(1)` 不会被误判成函数声明。
 */
export function parseDefinition(src: string): Definition {
  const trimmed = src.trim();
  if (!trimmed) throw new GeoError("空表达式", 0);
  const toks = tokenize(trimmed);
  const def = matchDefinition(toks);
  if (def) {
    const p = new P(def.rest);
    const node = p.top();
    const tail = p.peek();
    if (tail.kind !== "eof") {
      throw new GeoError(`定义右侧有多余内容 “${tail.text}”`, tail.pos);
    }
    return { kind: def.params.length ? "fn" : "let", name: def.name, params: def.params, node, src: trimmed };
  }
  return { kind: "expr", name: "", params: [], node: parseExpr(trimmed), src: trimmed };
}

/** 尝试把 token 序列识别为 `name = expr` 或 `name(p1,p2) = expr` */
function matchDefinition(toks: Tok[]): { name: string; params: string[]; rest: Tok[] } | null {
  if (toks[0]?.kind !== "ident") return null;
  if (toks[0].text.startsWith("@str:")) return null;
  let k = 1;
  const params: string[] = [];
  if (toks[k]?.kind === "(") {
    k++;
    if (toks[k]?.kind === ")") return null;
    for (;;) {
      if (toks[k]?.kind !== "ident" || toks[k].text.startsWith("@str:")) return null;
      params.push(toks[k].text);
      k++;
      if (toks[k]?.kind === ",") {
        k++;
        continue;
      }
      break;
    }
    if (toks[k]?.kind !== ")") return null;
    k++;
  }
  // 顶层赋值号：单个 = （排除 ==、<=、>=、!=）
  if (toks[k]?.kind !== "=") return null;
  return { name: toks[0].text, params, rest: toks.slice(k + 1) };
}

/** 表达式中出现的所有自由标识符（用于自动发现参数） */
export function freeNames(node: Node, out: Set<string> = new Set()): Set<string> {
  const skip = new Set(["x", "y", "z", "t", "u", "v", "n", "i", "j"]);
  walk(node);
  function walk(nd: Node) {
    switch (nd.type) {
      case "ident":
        out.add(nd.name);
        break;
      case "bin":
        walk(nd.l);
        walk(nd.r);
        break;
      case "un":
        walk(nd.e);
        break;
      case "post":
        walk(nd.e);
        break;
      case "idx":
        walk(nd.e);
        walk(nd.i);
        break;
      case "list":
        nd.items.forEach(walk);
        break;
      case "cond":
        walk(nd.c);
        walk(nd.a);
        walk(nd.b);
        break;
      case "call": {
        if (!BUILTIN_NAMES.has(nd.name) && !skip.has(nd.name)) out.add(nd.name);
        nd.args.forEach(walk);
        break;
      }
      default:
        break;
    }
  }
  return out;
}
