/**
 * 2D 画布：函数 / 复平面 / 向量场 / 动态几何的统一交互层。
 *
 * 分两层 canvas：底层画场景（可能有几十万像素的栅格，较重），
 * 上层只画十字光标与读数板，鼠标移动时仅重画这一层。
 */
import { useCallback, useEffect, useRef, useState } from "react";
import * as P from "../render/plot2d.ts";
import {
  applyGeoTool,
  drawScene2D,
  probe,
  rasterJobFor,
  rasterKeyOf,
  stepRasterJob,
  type RasterJob,
} from "../render/scene2d.ts";
import { niceTicks } from "../core/view.ts";
import { useStore } from "../state.ts";

interface Size {
  w: number;
  h: number;
  dpr: number;
}

function same(a: string[], b: string[]): boolean {
  return a.length === b.length && a.every((s, i) => s === b[i]);
}

function fmt(v: number): string {
  const a = Math.abs(v);
  if (a >= 1e6 || (a > 0 && a < 1e-4)) return v.toExponential(2);
  return String(Number(v.toFixed(4)));
}

/** 两次输入间隔短于此值即判定为连续交互（拖动 / 缩放 / 参数动画） */
const INTERACTIVE_MS = 130;
/** 一帧预算：上一帧超出此值说明画面本身够重，新来的输入先按降质档画 */
const FRAME_MS = 16;
/** 停手后逐带回补时留给采样器的时间；留出余量给合成与叠加层 */
const REFINE_BUDGET_MS = 8;

export default function Canvas2D() {
  const wrap = useRef<HTMLDivElement>(null);
  const base = useRef<HTMLCanvasElement>(null);
  const over = useRef<HTMLCanvasElement>(null);
  const size = useRef<Size>({ w: 900, h: 620, dpr: 1 });
  const hover = useRef<{ sx: number; sy: number } | null>(null);
  const drag = useRef<{ type: "pan"; lx: number; ly: number } | { type: "point"; id: string } | null>(null);
  const frame = useRef(0);
  const settle = useRef(0);
  const lastAt = useRef(0);
  const pendingDraft = useRef(false);
  const lastCost = useRef(0);
  /** 栅格续算：跨 rAF 逐行采样的缓冲，画完之后本帧直接贴用它 */
  const job = useRef<RasterJob | null>(null);
  const refine = useRef(0);
  const [hud, setHud] = useState<{ errors: string[]; info: string[] }>({ errors: [], info: [] });
  const [note, setNote] = useState<string | null>(null);
  const [cursor, setCursor] = useState("crosshair");
  const mode = useStore((s) => s.mode);
  const revision = useStore((s) => s.revision);
  const viewSig = useStore((s) => {
    const v = s.views[s.mode];
    return `${s.mode}|${v.cx}|${v.cy}|${v.scale}|${v.width}|${v.height}|${v.logX ? 1 : 0}${v.logY ? 1 : 0}`;
  });

  /** 只画叠加层：十字光标、吸附点、读数板、几何悬停高亮 */
  const overlay = useCallback(() => {
    const cv = over.current;
    if (!cv) return;
    const { w, h, dpr } = size.current;
    if (cv.width !== Math.floor(w * dpr) || cv.height !== Math.floor(h * dpr)) {
      cv.width = Math.floor(w * dpr);
      cv.height = Math.floor(h * dpr);
    }
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const st = useStore.getState();
    const hv = hover.current;
    if (!hv || hv.sx < 0 || hv.sy < 0 || hv.sx > w || hv.sy > h) return;
    const p: P.Paper = { ctx, vp: st.views[st.mode], w, h, dark: st.settings.dark };
    const [wx, wy] = p.vp.toWorld(hv.sx, hv.sy);
    const lines: string[] = [];
    if (st.mode === "geom") {
      const doc = st.geo.doc;
      const id = doc.hitTest(wx, wy, 10 / p.vp.scale);
      const g = id ? doc.get(id) : null;
      if (g) {
        lines.push(`${g.label} · ${g.kind === "point" ? `(${fmt(g.x)}, ${fmt(g.y)})` : "点击选中"}`);
        if (g.kind === "point") {
          ctx.save();
          ctx.strokeStyle = "#f0abfc";
          ctx.lineWidth = 2;
          ctx.beginPath();
          ctx.arc(...P.vpScreen(p, g.x, g.y), 9, 0, Math.PI * 2);
          ctx.stroke();
          ctx.restore();
        }
      }
    } else {
      const r = probe(st, wx, wy);
      lines.push(...r.lines);
      if (st.settings.showCrosshair) P.drawCrosshair(p, wx, wy);
      if (r.snap) {
        const [sx, sy] = P.vpScreen(p, r.snap[0], r.snap[1]);
        ctx.save();
        ctx.fillStyle = "#22d3ee";
        ctx.beginPath();
        ctx.arc(sx, sy, 3.6, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      }
    }
    if (lines.length) P.drawReadout(p, lines);
  }, []);

  /** 重画场景（底层），尺寸未同步时先写回视口再等下一次通知 */
  const draw = useCallback(
    (draft: boolean) => {
      const el = wrap.current;
      const cv = base.current;
      if (!el || !cv) return;
      const rect = el.getBoundingClientRect();
      const w = Math.max(1, Math.floor(rect.width));
      const h = Math.max(1, Math.floor(rect.height));
      const dpr = Math.min(2.5, window.devicePixelRatio || 1);
      const st = useStore.getState();
      const vp = st.views[st.mode];
      if (vp.width !== w || vp.height !== h) {
        st.setView(st.mode, vp.with({ width: w, height: h }));
        return;
      }
      size.current = { w, h, dpr };
      const pw = Math.floor(w * dpr);
      const ph = Math.floor(h * dpr);
      if (cv.width !== pw || cv.height !== ph) {
        cv.width = pw;
        cv.height = ph;
      }
      const ctx = cv.getContext("2d");
      if (!ctx) return;
      // 改写 width/height 会把变换重置为单位矩阵，必须每次重新按 dpr 缩放
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const out = drawScene2D({ ctx, vp, w, h, dark: st.settings.dark }, st, {
        draft,
        raster: draft ? null : job.current,
      });
      setHud((prev) => (same(prev.errors, out.errors) && same(prev.info, out.info) ? prev : { errors: out.errors, info: out.info }));
      overlay();
    },
    [overlay],
  );

  /* 停手后的全质量回补：整幅栅格一帧干不完（几百毫秒），
     于是逐带采样、每带用完 8ms 预算就让出一帧，画满再一次性换上锐利图像。 */
  const refineStep = useCallback(() => {
    refine.current = 0;
    const st = useStore.getState();
    const j = job.current;
    if (!j) return;
    const r = stepRasterJob(st, j, REFINE_BUDGET_MS);
    if (r === "done") {
      draw(false);
      return;
    }
    if (r === "stale") {
      job.current = rasterJobFor(st, size.current.w, size.current.h);
      if (!job.current) return;
    }
    refine.current = requestAnimationFrame(refineStep);
  }, [draw]);

  const stopRefine = useCallback(() => {
    if (refine.current) cancelAnimationFrame(refine.current);
    refine.current = 0;
  }, []);

  const startRefine = useCallback(() => {
    const st = useStore.getState();
    const { w, h } = size.current;
    const key = rasterKeyOf(st, w, h);
    const kept = job.current;
    // 非栅格模式一帧就够；栅格身份没变且已画完则直接贴现有结果
    if (key === null || (kept && kept.key === key && kept.rows === kept.h)) {
      draw(false);
      return;
    }
    if (!kept || kept.key !== key) job.current = rasterJobFor(st, w, h);
    if (!job.current) {
      draw(false);
      return;
    }
    stopRefine();
    refine.current = requestAnimationFrame(refineStep);
  }, [draw, refineStep, stopRefine]);

  /* 一帧内可能有多个 wheel / pointermove，逐次同步重画会把几十毫秒的活干好几遍；
     所以统一排到 requestAnimationFrame，每帧最多画一次。
     降质档必须在「输入到达」时判定：一次全质量本身就要几百毫秒，
     等到真正开画时任何时间窗都已经过期，就再也进不去降质了。 */
  const step = useCallback(() => {
    frame.current = 0;
    const draft0 = pendingDraft.current;
    pendingDraft.current = false;
    /* 栅格一帧采不完，所以「整幅现算」这种帧根本不该出现：改一个表达式、打开工作台的首帧，
       都会在主线程里算掉几百毫秒到几秒（分辨率 1 时五秒）。只要有栅格可续算，就先画降质档，
       停手再交给分带回补——拖动路径本来就是这么走的，这里补上非交互的那一半。 */
    const draft = draft0 || rasterKeyOf(useStore.getState(), size.current.w, size.current.h) !== null;
    const t0 = performance.now();
    draw(draft);
    lastCost.current = performance.now() - t0;
    if (!draft) return;
    window.clearTimeout(settle.current);
    settle.current = window.setTimeout(() => {
      settle.current = 0;
      if (!frame.current && !refine.current) startRefine();
    }, INTERACTIVE_MS);
  }, [draw, startRefine]);

  const schedule = useCallback(() => {
    const now = performance.now();
    pendingDraft.current = now - lastAt.current < INTERACTIVE_MS || lastCost.current > FRAME_MS;
    lastAt.current = now;
    // 输入还在继续：回补得立刻让路，否则锐利图会卡在拖动中间盖住新画面
    stopRefine();
    if (settle.current) window.clearTimeout(settle.current);
    settle.current = 0;
    if (!frame.current) frame.current = requestAnimationFrame(step);
  }, [step, stopRefine]);

  useEffect(() => {
    schedule();
  }, [schedule, mode, revision, viewSig]);

  useEffect(() => {
    const el = wrap.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver(() => schedule());
    ro.observe(el);
    return () => ro.disconnect();
  }, [schedule]);

  useEffect(
    () => () => {
      if (frame.current) cancelAnimationFrame(frame.current);
      if (settle.current) window.clearTimeout(settle.current);
      /* 分带回补的续算令牌同样要收回：换模式卸载后它还会再跑一帧，去写已废弃画布的缓存 */
      if (refine.current) cancelAnimationFrame(refine.current);
      frame.current = 0;
      settle.current = 0;
      refine.current = 0;
    },
    [],
  );

  /** 滚轮缩放需要 passive:false 才能 preventDefault */
  useEffect(() => {
    const cv = over.current;
    if (!cv) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const st = useStore.getState();
      const r = cv.getBoundingClientRect();
      const vp = st.views[st.mode];
      const k = Math.exp(-e.deltaY * (e.shiftKey ? 0.006 : 0.0016));
      st.setView(st.mode, vp.zoomAt(e.clientX - r.left, e.clientY - r.top, k));
    };
    cv.addEventListener("wheel", onWheel, { passive: false });
    return () => cv.removeEventListener("wheel", onWheel);
  }, []);

  const localPoint = (e: React.PointerEvent | React.MouseEvent): [number, number] => {
    const r = (over.current as HTMLCanvasElement).getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  };

  /** 世界坐标 → 是否吸附到当前刻度 */
  const snapTo = (x: number, y: number): [number, number] => {
    const st = useStore.getState();
    if (st.mode !== "geom" || !st.geo.snap) return [x, y];
    const vp = st.views.geom;
    const step = niceTicks(vp.left, vp.right, 10).step;
    return [Math.round(x / step) * step, Math.round(y / step) * step];
  };

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    const st = useStore.getState();
    const vp = st.views[st.mode];
    const [sx, sy] = localPoint(e);
    (e.target as Element).setPointerCapture?.(e.pointerId);
    if (st.mode !== "geom") {
      drag.current = { type: "pan", lx: sx, ly: sy };
      setCursor("grabbing");
      return;
    }
    const doc = st.geo.doc;
    const [wx0, wy0] = vp.toWorld(sx, sy);
    const [wx, wy] = snapTo(wx0, wy0);
    const picked = doc.hitTest(wx0, wy0, 10 / vp.scale);
    if (st.geo.tool === "select") {
      if (picked && doc.dragTarget(picked)) {
        doc.beginGesture();
        drag.current = { type: "point", id: picked };
        st.setGeo({ selected: picked, pending: [] });
        setCursor("grabbing");
      } else {
        st.setGeo({ selected: picked ?? null, pending: [] });
        drag.current = { type: "pan", lx: sx, ly: sy };
        setCursor("grabbing");
      }
      return;
    }
    const r = applyGeoTool(doc, st.geo.tool, st.geo.pending, picked, [wx, wy]);
    st.setGeo({ pending: r.pending, selected: r.done ? null : st.geo.selected });
    if (r.error) {
      setNote(r.error);
      window.setTimeout(() => setNote(null), 2600);
    }
  };

  const onPointerMove = (e: React.PointerEvent) => {
    const [sx, sy] = localPoint(e);
    const st = useStore.getState();
    const vp = st.views[st.mode];
    const d = drag.current;
    if (d?.type === "pan") {
      st.setView(st.mode, vp.panPixels(sx - d.lx, sy - d.ly));
      drag.current = { type: "pan", lx: sx, ly: sy };
      return;
    }
    if (d?.type === "point") {
      const [wx, wy] = snapTo(...vp.toWorld(sx, sy));
      st.geo.doc.move(d.id, wx, wy);
      st.bump();
      return;
    }
    hover.current = { sx, sy };
    if (st.mode === "geom" && st.geo.tool === "select") {
      const [wx, wy] = vp.toWorld(sx, sy);
      const id = st.geo.doc.hitTest(wx, wy, 10 / vp.scale);
      const next = id && st.geo.doc.dragTarget(id) ? "grab" : id ? "pointer" : "crosshair";
      setCursor((c) => (c === next ? c : next));
    }
    overlay();
  };

  const endDrag = () => {
    const d = drag.current;
    if (d?.type === "point") {
      useStore.getState().geo.doc.endGesture();
      useStore.getState().bump();
    }
    drag.current = null;
    setCursor("crosshair");
  };

  const onPointerLeave = () => {
    hover.current = null;
    endDrag();
    overlay();
  };

  const onDoubleClick = () => {
    const st = useStore.getState();
    if (st.mode !== "geom" || st.geo.tool !== "polygon" || st.geo.pending.length < 3) return;
    st.geo.doc.addPolygon(st.geo.pending, "多边形");
    st.setGeo({ pending: [], selected: null });
  };

  return (
    <div className="gl-stage" ref={wrap}>
      <canvas ref={base} className="gl-canvas" style={{ position: "absolute", inset: 0 }} />
      <canvas
        ref={over}
        className="gl-canvas"
        data-cursor={cursor}
        style={{ position: "absolute", inset: 0 }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onPointerLeave={onPointerLeave}
        onDoubleClick={onDoubleClick}
      />
      {hud.errors.length > 0 && <div className="gl-err">{hud.errors.slice(0, 5).join("\n")}</div>}
      <div className="gl-hud">
        {note && <div className="gl-chip" style={{ color: "#fbbf24" }}>{note}</div>}
        {hud.info.map((l) => (
          <div className="gl-chip" key={l}>
            {l}
          </div>
        ))}
        <div className="gl-chip">滚轮缩放 · 拖动平移{mode === "geom" ? " · 拖动点改图形" : ""}</div>
      </div>
    </div>
  );
}
