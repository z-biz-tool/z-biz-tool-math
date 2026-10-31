/**
 * zbiz:// 深链接的 **math 语义层**。
 *
 * ## 分工
 *
 * | 层 | 在哪 | 管什么 |
 * |----|----|--------|
 * | 协议 | `z-biz-tool-shared/protocol` | 字节级编解码（金标 = Rust `cap-hub`） |
 * | 机制 | `z-biz-tool-shared/protocol` 的 `startDeepLinkReceiver` | 冷启动补取、事件监听、不静默吞掉 |
 * | **语义** | **本文件** | **收到深链后 math 该做什么** |
 *
 * ## math 支持的动作
 *
 * | action | 含义 |
 * |--------|------|
 * | `open` | 进入/切换到本应用（缺省 action） |
 *
 * `open` 不带参数时只是把 math 唤到前台（no-op，不报错）；带参数时定位到具体内容。
 *
 * ## 支持的参数
 *
 * | 参数 | 作用 |
 * |------|------|
 * | `preset` | 应用一个预设（key 形如 `func-basic` / `cpx-newton` / `vec-quiver` / `lin-rot60` / `nn-xor` / `surf-graph`），必须真实存在，否则 problem 说「没有这个预设 X」 |
 * | `mode`   | 直接切工作模式：`func` \| `geom` \| `complex` \| `vector` \| `lin` \| `nn` \| `surf` \| `console` |
 * | `from`   | 发送方，仅用于日志与回执 |
 *
 * `preset` 与 `mode` 同时存在时：`applyPreset` 会按其自身定义切到对应 mode，随后再用
 * 显式的 `mode` 覆盖（显式意图优先）。
 */

import { PRESETS } from "../presets.ts";
import type { Mode } from "../state.ts";
import type { Handoff } from "z-biz-tool-shared/protocol";

/** math 的工作模式白名单（与 `src/state.ts` 的 `Mode` 一致） */
export const MATH_MODES: readonly Mode[] = [
  "func",
  "geom",
  "complex",
  "vector",
  "lin",
  "nn",
  "surf",
  "life",
  "frac",
  "mech",
  "console",
];

/**
 * math 的深链结果。
 *
 * `preset` / `mode` 非 null 表示要应用那个预设 / 切到那个模式；
 * `problem` 非 null 表示接不住 —— 前端必须给可见提示（不静默吞掉）。
 */
export interface MathDeepLinkOutcome {
  /** 要应用的预设 key；null 表示不带预设意图 */
  preset: string | null;
  /** 要切换的工作模式；null 表示不带切模式意图 */
  mode: Mode | null;
  /** 发送方（from），仅用于提示与日志 */
  from: string | null;
  /** 无法适配时的原因 */
  problem: string | null;
}

/**
 * 把一条深链翻译成 math 内的动作。
 *
 * 纯函数 —— 不碰 store、不碰 Tauri，所以可以直接单测。
 *
 * @param item 深链
 */
export function resolveMathDeepLink(item: Handoff): MathDeepLinkOutcome {
  const outcome: MathDeepLinkOutcome = {
    preset: null,
    mode: null,
    from: item.params.from ?? null,
    problem: null,
  };

  const action = item.action.trim() || "open";
  if (action !== "open") {
    outcome.problem = `不认识的动作 ${action}`;
    return outcome;
  }

  const rawPreset = item.params.preset?.trim() ?? "";
  const rawMode = item.params.mode?.trim() ?? "";

  // 预设：给出来就必须真实存在，否则是「id 不存在」
  if (rawPreset) {
    if (PRESETS.some((p) => p.key === rawPreset)) {
      outcome.preset = rawPreset;
    } else {
      outcome.problem = `没有这个预设 ${rawPreset}`;
      return outcome;
    }
  }

  // 模式：白名单校验
  if (rawMode) {
    if ((MATH_MODES as readonly string[]).includes(rawMode)) {
      outcome.mode = rawMode as Mode;
    } else {
      outcome.problem = `不认识的 mode ${rawMode}`;
      return outcome;
    }
  }

  return outcome;
}
