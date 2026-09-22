/** 复数 / 实数数学工具：GeoLab 的数值底座 */

export interface C {
  re: number;
  im: number;
}

export const TAU = Math.PI * 2;
export const DEG = Math.PI / 180;

export const INF = Infinity;
export const NAN = NaN;

export function isNum(x: number): boolean {
  return x === x && x !== Infinity && x !== -Infinity;
}

export function cplx(re: number, im: number): C {
  return { re, im };
}

export function cabs(re: number, im: number): number {
  // hypot 避免溢出
  return Math.hypot(re, im);
}

export function carg(re: number, im: number): number {
  if (re === 0 && im === 0) return 0;
  return Math.atan2(im, re);
}

export function cexp(re: number, im: number, o: C): C {
  const m = Math.exp(re);
  o.re = m * Math.cos(im);
  o.im = m * Math.sin(im);
  return o;
}

export function clog(re: number, im: number, o: C): C {
  o.re = Math.log(cabs(re, im));
  o.im = carg(re, im);
  return o;
}

export function csqrt(re: number, im: number, o: C): C {
  if (im === 0) {
    if (re >= 0) {
      o.re = Math.sqrt(re);
      o.im = 0;
    } else {
      o.re = 0;
      o.im = Math.sqrt(-re);
    }
    return o;
  }
  const m = cabs(re, im);
  if (re >= 0) {
    const t = Math.sqrt((m + re) / 2);
    o.re = t;
    o.im = im / (2 * t);
  } else {
    const t = Math.sqrt((m - re) / 2);
    o.re = Math.abs(im) / (2 * t);
    o.im = im < 0 ? -t : t;
  }
  return o;
}

export function cadd(ar: number, ai: number, br: number, bi: number, o: C): C {
  o.re = ar + br;
  o.im = ai + bi;
  return o;
}

export function csub(ar: number, ai: number, br: number, bi: number, o: C): C {
  o.re = ar - br;
  o.im = ai - bi;
  return o;
}

export function cmul(ar: number, ai: number, br: number, bi: number, o: C): C {
  o.re = ar * br - ai * bi;
  o.im = ar * bi + ai * br;
  return o;
}

export function cdiv(ar: number, ai: number, br: number, bi: number, o: C): C {
  if (br === 0 && bi === 0) {
    o.re = ar === 0 ? NaN : Math.sign(ar) * Infinity;
    o.im = ai === 0 ? NaN : Math.sign(ai) * Infinity;
    return o;
  }
  if (bi === 0) {
    o.re = ar / br;
    o.im = ai / br;
    return o;
  }
  if (br === 0) {
    o.re = ai / bi;
    o.im = -ar / bi;
    return o;
  }
  // 数值稳定除法（防止平方溢出）
  if (Math.abs(br) < Math.abs(bi)) {
    const r = br / bi;
    const d = bi + r * br;
    o.re = (ar * r + ai) / d;
    o.im = (ai * r - ar) / d;
  } else {
    const r = bi / br;
    const d = br + r * bi;
    o.re = (ar + ai * r) / d;
    o.im = (ai - ar * r) / d;
  }
  return o;
}

/** 有理指数检测：(-8)^(1/3) 应给出实根 -2（绘图工具的常见期望） */
export function realPow(b: number, e: number): number {
  if (b >= 0) return Math.pow(b, e);
  if (Number.isInteger(e)) return Math.pow(b, e);
  const inv = 1 / e;
  const r = Math.round(inv);
  // e ≈ 1/奇数 → 视为奇次根
  if (Math.abs(r) % 2 === 1 && Math.abs(inv - r) < 1e-9 * Math.max(1, Math.abs(inv))) {
    return -Math.pow(-b, e);
  }
  if (Math.abs(r) % 2 === 1 && Math.abs(1 / r - e) < 1e-12) {
    return -Math.pow(-b, e);
  }
  return NaN;
}

export function cpow(ar: number, ai: number, br: number, bi: number, o: C): C {
  if (bi === 0) {
    if (ai === 0) {
      const r = realPow(ar, br);
      if (!Number.isNaN(r) || ar >= 0) {
        o.re = r;
        o.im = 0;
        return o;
      }
    }
    if (Number.isInteger(br)) {
      if (br === 0) {
        o.re = 1;
        o.im = 0;
        return o;
      }
      if (br === 1) {
        o.re = ar;
        o.im = ai;
        return o;
      }
      if (br === 2) {
        o.re = ar * ar - ai * ai;
        o.im = 2 * ar * ai;
        return o;
      }
      if (br === -1) return cdiv(1, 0, ar, ai, o);
      if (Math.abs(br) <= 16) {
        // 复底的整数幂走二进制累乘：主值式 exp(e·Log z) 要对辐角取模，
        // 既慢（两次超越函数）又在 ±π 边界丢符号精度
        let qr = ar;
        let qi = ai;
        let wr = 1;
        let wi = 0;
        for (let e = br < 0 ? -br : br; e > 0; e >>= 1) {
          if (e & 1) {
            const t = wr * qr - wi * qi;
            wi = wr * qi + wi * qr;
            wr = t;
          }
          const t = qr * qr - qi * qi;
          qi = 2 * qr * qi;
          qr = t;
        }
        if (br < 0) return cdiv(1, 0, wr, wi, o);
        o.re = wr;
        o.im = wi;
        return o;
      }
    }
  }
  if (ar === 0 && ai === 0) {
    o.re = br === 0 ? 1 : 0;
    o.im = 0;
    return o;
  }
  // 主值：b^e = exp(e * Log b)
  const lnr = Math.log(cabs(ar, ai));
  const ang = carg(ar, ai);
  // exp((br+i bi)(lnr + i ang))
  const rr = Math.exp(br * lnr - bi * ang);
  const aa = br * ang + bi * lnr;
  o.re = rr * Math.cos(aa);
  o.im = rr * Math.sin(aa);
  return o;
}

export function csin(re: number, im: number, o: C): C {
  o.re = Math.sin(re) * Math.cosh(im);
  o.im = Math.cos(re) * Math.sinh(im);
  return o;
}

export function ccos(re: number, im: number, o: C): C {
  o.re = Math.cos(re) * Math.cosh(im);
  o.im = -Math.sin(re) * Math.sinh(im);
  return o;
}

export function ctan(re: number, im: number, o: C): C {
  const d = Math.cos(2 * re) + Math.cosh(2 * im);
  o.re = Math.sin(2 * re) / d;
  o.im = Math.sinh(2 * im) / d;
  return o;
}

export function casin(re: number, im: number, o: C): C {
  // asin z = -i Log(i z + sqrt(1 - z^2))
  const sr = -im, si = re; // i*z
  // 1 - z^2
  const qr = 1 - (re * re - im * im);
  const qi = -2 * re * im;
  const t = csqrt(qr, qi, TMP[0]);
  const ur = sr + t.re;
  const ui = si + t.im;
  const l = clog(ur, ui, TMP[1]);
  o.re = l.im;
  o.im = -l.re;
  return o;
}

export function cacos(re: number, im: number, o: C): C {
  const r = casin(re, im, TMP[2]);
  o.re = Math.PI / 2 - r.re;
  o.im = -r.im;
  return o;
}

export function catan(re: number, im: number, o: C): C {
  // atan z = (i/2) Log((i+z)/(i-z))
  const nr = -im, ni = 1 + re; // i + z
  const dr = im, di = 1 - re; // i - z
  const q = cdiv(nr, ni, dr, di, TMP[3]);
  const l = clog(q.re, q.im, TMP[4]);
  o.re = l.im / 2;
  o.im = -l.re / 2;
  return o;
}

export function csinh(re: number, im: number, o: C): C {
  o.re = Math.sinh(re) * Math.cos(im);
  o.im = Math.cosh(re) * Math.sin(im);
  return o;
}

export function ccosh(re: number, im: number, o: C): C {
  o.re = Math.cosh(re) * Math.cos(im);
  o.im = Math.sinh(re) * Math.sin(im);
  return o;
}

export function ctanh(re: number, im: number, o: C): C {
  const d = Math.cosh(2 * re) + Math.cos(2 * im);
  o.re = Math.sinh(2 * re) / d;
  o.im = Math.sin(2 * im) / d;
  return o;
}

export function casinh(re: number, im: number, o: C): C {
  // asinh z = Log(z + sqrt(z^2 + 1))
  const qr = re * re - im * im + 1;
  const qi = 2 * re * im;
  const t = csqrt(qr, qi, TMP[5]);
  return clog(re + t.re, im + t.im, o);
}

export function cacosh(re: number, im: number, o: C): C {
  const qr = re * re - im * im - 1;
  const qi = 2 * re * im;
  const t = csqrt(qr, qi, TMP[6]);
  return clog(re + t.re, im + t.im, o);
}

export function catanh(re: number, im: number, o: C): C {
  // atanh z = 0.5 * Log((1+z)/(1-z))
  const q = cdiv(1 + re, im, 1 - re, -im, TMP[7]);
  const l = clog(q.re, q.im, TMP[8]);
  o.re = l.re / 2;
  o.im = l.im / 2;
  return o;
}

/** Lambert W 主支（Newton 迭代），用于求解 x*e^x = z 类型 */
export function lambertW(x: number): number {
  if (x < -1 / Math.E) return NaN;
  if (x === 0) return 0;
  let w = x >= 1 ? Math.log(x) - Math.log(Math.log(x) + 1) : 0;
  for (let k = 0; k < 60; k++) {
    const ew = Math.exp(w);
    const f = w * ew - x;
    const fp = ew * (w + 1);
    if (fp === 0) break;
    const nw = w - f / fp;
    if (!Number.isFinite(nw)) break;
    if (Math.abs(nw - w) < 1e-15 * Math.max(1, Math.abs(w))) {
      w = nw;
      break;
    }
    w = nw;
  }
  return w;
}

/** Lanczos 近似的 Gamma 函数（实轴，含负数反射） */
export function gamma(x: number): number {
  if (Number.isInteger(x)) {
    if (x <= 0) return Infinity;
    if (x <= 171) {
      let r = 1;
      for (let k = 2; k < x; k++) r *= k;
      return r;
    }
    return Infinity;
  }
  if (x < 0.5) {
    return Math.PI / (Math.sin(Math.PI * x) * gamma(1 - x));
  }
  const g = [
    676.5203681218851, -1259.1392167224028, 771.32342877765313, -176.61502916214059,
    12.507343278686905, -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  const z = x - 1;
  let a = 0.99999999999980993;
  const t = z + 7.5;
  for (let i = 0; i < g.length; i++) a += g[i] / (z + i + 1);
  return Math.sqrt(2 * Math.PI) * Math.pow(t, z + 0.5) * Math.exp(-t) * a;
}

export function factorial(n: number): number {
  if (n < 0) {
    if (Number.isInteger(n)) return Infinity;
    return gamma(n + 1);
  }
  if (Number.isInteger(n)) {
    if (n > 170) return Infinity;
    let r = 1;
    for (let k = 2; k <= n; k++) r *= k;
    return r;
  }
  return gamma(n + 1);
}

/** 用于求根/拟合的 1D 插值（单调三次，避免震荡） */
export function clamp(x: number, lo: number, hi: number): number {
  return x < lo ? lo : x > hi ? hi : x;
}

export function lerp(a: number, b: number, t: number): number {
  return a + (b - a) * t;
}

/** 阶乘/幂等热路径复用池 */
export const TMP: C[] = Array.from({ length: 16 }, () => ({ re: 0, im: 0 }));

/** 数值格式化：用于读数面板与刻度标签 */
export function fmt(x: number, digits = 4): string {
  if (!Number.isFinite(x)) return x > 0 ? "∞" : x < 0 ? "-∞" : "NaN";
  if (x === 0) return "0";
  const a = Math.abs(x);
  if (a >= 1e6 || a < 1e-4) return x.toExponential(Math.max(1, digits - 1));
  let s = x.toFixed(digits);
  if (s.includes(".")) s = s.replace(/\.?0+$/, "");
  return s;
}

export function fmtC(re: number, im: number, digits = 4): string {
  if (Math.abs(im) < 1e-12) return fmt(re, digits);
  if (Math.abs(re) < 1e-12) return `${fmt(im, digits)}i`;
  return `${fmt(re, digits)}${im < 0 ? "-" : "+"}${fmt(Math.abs(im), digits)}i`;
}

/** π 的整数/半整数倍显示为 π 记号，贴近数学教科书 */
export function fmtPi(x: number, digits = 4): string {
  if (!Number.isFinite(x) || x === 0) return fmt(x, digits);
  const r = x / Math.PI;
  for (const d of [1, 2, 3, 4, 6, 8, 12]) {
    const n = Math.round(r * d);
    if (n !== 0 && Math.abs(r * d - n) < 1e-9) {
      const sign = n < 0 ? "-" : "";
      const an = Math.abs(n);
      const num = an === 1 ? "π" : `${an}π`;
      return d === 1 ? `${sign}${num}` : `${sign}${num}/${d}`;
    }
  }
  return fmt(x, digits);
}
