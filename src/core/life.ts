/** 元胞自动机内核：方格世界 + B/S 记法的可换规则（生命游戏家族） */

export interface Rule {
  birth: boolean[];
  survive: boolean[];
  /** 原始记法，如 B3/S23 */
  src: string;
}

/** 解析 "B3/S23" 记法；不合法返回 null（面板据此报错，运行回落到上一条合法规则） */
export function parseRule(src: string): Rule | null {
  const m = /^\s*[Bb]?([0-8]*)\s*\/\s*[Ss]?([0-8]*)\s*$/.exec(src);
  if (!m) return null;
  const birth = new Array<boolean>(9).fill(false);
  const survive = new Array<boolean>(9).fill(false);
  for (const c of m[1]) birth[Number(c)] = true;
  for (const c of m[2]) survive[Number(c)] = true;
  return { birth, survive, src: src.trim() };
}

export const RULE_PRESETS: { value: string; label: string }[] = [
  { value: "B3/S23", label: "生命游戏 B3/S23" },
  { value: "B36/S23", label: "HighLife B36/S23（复制体）" },
  { value: "B2/S", label: "Seeds B2/S（ seeds 爆炸）" },
  { value: "B34/S34", label: "34 Life B34/S34" },
  { value: "B35678/S5678", label: "Diamoeba B35678/S5678" },
  { value: "B368/S245", label: "Morley B368/S245（慢速脉冲）" },
  { value: "B3/S1234", label: "Mazectric B3/S1234（迷宫）" },
];

/** 缓存最近一次解析：运行循环里逐帧用，字符串没变就不再重 parse */
let ruleCacheSrc = "";
let ruleCache: Rule | null = null;
export function ruleOf(src: string): Rule | null {
  if (src !== ruleCacheSrc) {
    ruleCacheSrc = src;
    ruleCache = parseRule(src);
  }
  return ruleCache;
}

/**
 * 生命世界：一个固定大小的环形方格（环绕边界），格子坐标 = 世界坐标（一格一单位）。
 * 实例活在模块作用域——与神经网络模型同理，模式切走再回来，世界还得是原来那个。
 * 初始布局不进工程 JSON：工程只带规则与速度等配置，载入后是空盘（面板可一键撒种）。
 */
export class LifeWorld {
  readonly w: number;
  readonly h: number;
  gen = 0;
  population = 0;
  /** 当前格子盘（渲染与批量撒种直读直写；step 内部用双缓冲换盘） */
  cells: Uint8Array;
  private buf: Uint8Array;

  constructor(w = 360, h = 240) {
    this.w = w;
    this.h = h;
    this.cells = new Uint8Array(w * h);
    this.buf = new Uint8Array(w * h);
  }

  inGrid(x: number, y: number): boolean {
    return x >= 0 && y >= 0 && x < this.w && y < this.h;
  }

  get(x: number, y: number): number {
    return this.inGrid(x, y) ? this.cells[y * this.w + x] : 0;
  }

  set(x: number, y: number, v: number): void {
    if (!this.inGrid(x, y)) return;
    const i = y * this.w + x;
    if (this.cells[i] === (v ? 1 : 0)) return;
    this.cells[i] = v ? 1 : 0;
    this.population += v ? 1 : -1;
  }

  toggle(x: number, y: number): void {
    this.set(x, y, this.get(x, y) ? 0 : 1);
  }

  clear(): void {
    this.cells.fill(0);
    this.gen = 0;
    this.population = 0;
  }

  /** 环形边界的一步：邻居数查 B/S 表 */
  step(rule: Rule): void {
    const { w, h } = this;
    const cells = this.cells;
    const next = this.buf;
    let pop = 0;
    for (let y = 0; y < h; y++) {
      const yu = (y === 0 ? h - 1 : y - 1) * w;
      const ym = y * w;
      const yd = (y === h - 1 ? 0 : y + 1) * w;
      for (let x = 0; x < w; x++) {
        const xl = x === 0 ? w - 1 : x - 1;
        const xr = x === w - 1 ? 0 : x + 1;
        const n =
          cells[yu + xl] + cells[yu + x] + cells[yu + xr] +
          cells[ym + xl] + cells[ym + xr] +
          cells[yd + xl] + cells[yd + x] + cells[yd + xr];
        const i = ym + x;
        const v = cells[i] ? (rule.survive[n] ? 1 : 0) : rule.birth[n] ? 1 : 0;
        next[i] = v;
        pop += v;
      }
    }
    this.buf = cells;
    this.cells = next;
    this.gen++;
    this.population = pop;
  }
}

/** 全局唯一的世界实例（渲染、交互、运行循环共用） */
export const world = new LifeWorld();

export type LifeSeed =
  | { kind: "gun" }
  | { kind: "rpentomino" }
  | { kind: "acorn" }
  | { kind: "random"; density: number };

/** 把相对坐标图案画进世界中央 */
function stamp(cells: [number, number][]): void {
  const cx = Math.floor(world.w / 2) - 4;
  const cy = Math.floor(world.h / 2) - 2;
  for (const [x, y] of cells) world.set(cx + x, cy + y, 1);
}

/** Gosper 滑翔机枪（36 个活格的标准坐标） */
export function seedGun(): void {
  world.clear();
  stamp([
    [24, 0], [22, 1], [24, 1], [12, 2], [13, 2], [20, 2], [21, 2], [34, 2], [35, 2],
    [11, 3], [15, 3], [20, 3], [21, 3], [34, 3], [35, 3],
    [0, 4], [1, 4], [10, 4], [16, 4], [20, 4], [21, 4],
    [0, 5], [1, 5], [10, 5], [14, 5], [16, 5], [17, 5], [22, 5], [24, 5],
    [10, 6], [16, 6], [24, 6],
    [11, 7], [15, 7],
    [12, 8], [13, 8],
  ]);
}

/** R-五连块：最小的不稳定图案里增长最猛的其一 */
export function seedRpentomino(): void {
  world.clear();
  stamp([
    [1, 0], [2, 0], [0, 1], [1, 1], [1, 2],
  ]);
}

/** Acorn（橡果）：5206 代才稳定 */
export function seedAcorn(): void {
  world.clear();
  stamp([
    [1, 0], [3, 1], [0, 2], [1, 2], [4, 2], [5, 2], [6, 2],
  ]);
}

export function seedRandom(density: number): void {
  world.clear();
  const d = Math.min(0.9, Math.max(0.02, density));
  for (let i = 0; i < world.w * world.h; i++) if (Math.random() < d) world.cells[i] = 1;
  // 直接改了内部数组，population 重新数一遍
  let pop = 0;
  for (let i = 0; i < world.cells.length; i++) pop += world.cells[i];
  world.population = pop;
}

export function applySeed(seed: LifeSeed): void {
  if (seed.kind === "gun") seedGun();
  else if (seed.kind === "rpentomino") seedRpentomino();
  else if (seed.kind === "acorn") seedAcorn();
  else seedRandom(seed.density);
}
