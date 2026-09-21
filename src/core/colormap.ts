/** 色图（MATLAB colormap 家族）与几何画板风格的高亮配色 */

export type RGB = [number, number, number];

export const COLORMAPS = [
  "parula",
  "jet",
  "hsv",
  "hot",
  "cool",
  "gray",
  "twilight",
  "viridis",
  "plasma",
  "inferno",
  "turbo",
  "spring",
  "summer",
  "autumn",
] as const;

export type ColormapName = (typeof COLORMAPS)[number];

function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** 关键点线性插值型色图 */
function ramp(stops: [number, number, number][], t: number): RGB {
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  const a = stops[i];
  const b = stops[i + 1];
  return [
    Math.round(lerp(a[0], b[0], f) * 255),
    Math.round(lerp(a[1], b[1], f) * 255),
    Math.round(lerp(a[2], b[2], f) * 255),
  ];
}

const PARULA: [number, number, number][] = [
  [0.208, 0.166, 0.314], [0.234, 0.377, 0.557], [0.191, 0.555, 0.639],
  [0.229, 0.714, 0.502], [0.541, 0.839, 0.318], [0.891, 0.969, 0.231],
];
const JET: [number, number, number][] = [
  [0, 0, 0.5], [0, 0, 1], [0, 1, 1], [1, 1, 0], [1, 0, 0], [0.5, 0, 0],
];
const HOT: [number, number, number][] = [
  [0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1],
];
const COOL: [number, number, number][] = [
  [0, 1, 1], [1, 0, 1],
];
const TURBO: [number, number, number][] = [
  [0.18995, 0.07176, 0.23214], [0.25561, 0.24704, 0.75124], [0.19612, 0.56481, 0.90848],
  [0.05799, 0.85143, 0.70845], [0.32644, 0.96528, 0.35063], [0.69867, 0.97211, 0.19019],
  [0.94570, 0.84379, 0.25166], [0.99018, 0.58337, 0.25295], [0.85913, 0.29363, 0.19926],
  [0.61033, 0.07409, 0.13270],
];
const VIRIDIS: [number, number, number][] = [
  [0.267004, 0.004874, 0.329415], [0.282327, 0.140926, 0.457517], [0.221285, 0.289889, 0.539405],
  [0.126838, 0.423995, 0.553674], [0.070802, 0.537099, 0.514357], [0.133399, 0.652473, 0.432469],
  [0.385721, 0.755765, 0.293909], [0.741388, 0.849614, 0.192040], [0.993248, 0.906157, 0.143936],
];
const PLASMA: [number, number, number][] = [
  [0.050383, 0.029803, 0.527975], [0.382736, 0.006049, 0.784029], [0.649400, 0.123561, 0.720207],
  [0.857154, 0.281926, 0.567379], [0.978147, 0.483216, 0.404495], [0.999813, 0.728863, 0.239509],
  [0.940729, 0.975542, 0.131358],
];
const INFERNO: [number, number, number][] = [
  [0.012119, 0.011206, 0.073456], [0.254267, 0.033517, 0.540070], [0.532450, 0.187284, 0.611682],
  [0.787400, 0.302022, 0.508945], [0.964653, 0.439120, 0.327322], [0.997152, 0.679618, 0.171794],
  [0.998072, 0.906015, 0.147497], [0.988648, 0.998364, 0.644924],
];
const TWILIGHT: [number, number, number][] = [
  [0.885, 0.87, 0.885], [0.62, 0.6, 0.75], [0.25, 0.22, 0.35], [0.12, 0.1, 0.16],
  [0.25, 0.22, 0.35], [0.62, 0.6, 0.75], [0.885, 0.87, 0.885],
];
const SPRING: [number, number, number][] = [[1, 0, 1], [1, 1, 0]];
const SUMMER: [number, number, number][] = [[0, 0.5, 0.4], [1, 1, 0.4]];
const AUTUMN: [number, number, number][] = [[1, 0, 0], [1, 1, 0]];

const TABLES: Record<string, [number, number, number][]> = {
  parula: PARULA, jet: JET, hot: HOT, cool: COOL, turbo: TURBO, viridis: VIRIDIS,
  plasma: PLASMA, inferno: INFERNO, twilight: TWILIGHT, spring: SPRING, summer: SUMMER,
  autumn: AUTUMN,
};

/** t∈[0,1] → 0..255 RGB */
export function colormap(name: string, t: number): RGB {
  if (name === "gray") {
    const g = Math.round(Math.min(1, Math.max(0, t)) * 255);
    return [g, g, g];
  }
  if (name === "hsv") {
    const h = (Math.min(1, Math.max(0, t)) * 360) / 60;
    const i = Math.floor(h) % 6;
    const f = h - Math.floor(h);
    const r = [1, 1 - f, f, 0, 0, 1][i];
    const g = [f, 0, 0, 0, 1, 1][i];
    const b = [0, 0, 1, 1 - f, 1 - f, 0][i];
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
  }
  const stops = TABLES[name] ?? PARULA;
  return ramp(stops, t);
}

/** 连续色标 → 采样到查找表，供热循环使用（每帧只做一次） */
export function buildLUT(name: string, n = 256): Uint8ClampedArray {
  const lut = new Uint8ClampedArray(n * 3);
  for (let i = 0; i < n; i++) {
    const c = colormap(name, i / (n - 1));
    lut[i * 3] = c[0];
    lut[i * 3 + 1] = c[1];
    lut[i * 3 + 2] = c[2];
  }
  return lut;
}

export function rgbCss(c: RGB, a = 1): string {
  return a >= 1 ? `rgb(${c[0]},${c[1]},${c[2]})` : `rgba(${c[0]},${c[1]},${c[2]},${a})`;
}

/** 曲线调色板：几何画板风格的稳定取色（按图层序号循环） */
export const PALETTE = [
  "#5b6bd6", "#e5533c", "#12a594", "#d98c17", "#a855f7",
  "#2563eb", "#db2777", "#059669", "#b45309", "#7c3aed",
  "#dc2626", "#0891b2", "#65a30d", "#c026d3", "#0284c7",
];

export function paletteAt(i: number): string {
  return PALETTE[((i % PALETTE.length) + PALETTE.length) % PALETTE.length];
}
