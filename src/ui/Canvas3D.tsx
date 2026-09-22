/**
 * 3D 画布：轨道旋转、滚轮推拉、双击复位，右上角提供「适配视野」。
 * 网格构建在 scene3d 内按签名缓存，因此转动视角只走投影与 painter 排序。
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { drawScene3D } from "../render/scene3d.ts";
import { useStore } from "../state.ts";

export default function Canvas3D() {
  const wrap = useRef<HTMLDivElement>(null);
  const cv = useRef<HTMLCanvasElement>(null);
  const drag = useRef<{ x: number; y: number } | null>(null);
  const frame = useRef(0);
  const [hud, setHud] = useState<{ errors: string[]; info: string[] }>({ errors: [], info: [] });
  const [fit, setFit] = useState(0);
  const revision = useStore((s) => s.revision);
  const camSig = useStore((s) => {
    const c = s.surf.cam;
    return `${c.azim}|${c.elev}|${c.dist}|${c.target.join(",")}|${s.surf.box.join(",")}|${s.settings.dark}|${s.surf.showAxes}|${s.surf.layers.length}`;
  });

  const draw = useCallback(() => {
    const el = wrap.current;
    const canvas = cv.current;
    if (!el || !canvas) return;
    const rect = el.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));
    const dpr = Math.min(2.5, window.devicePixelRatio || 1);
    const pw = Math.floor(w * dpr);
    const ph = Math.floor(h * dpr);
    if (canvas.width !== pw || canvas.height !== ph) {
      canvas.width = pw;
      canvas.height = ph;
    }
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const st = useStore.getState();
    const out = drawScene3D(ctx, w, h, st, dpr);
    setFit((f) => (Math.abs(f - out.suggestDist) > 1e-6 ? out.suggestDist : f));
    setHud((p) =>
      p.errors.length === out.errors.length &&
      p.errors.every((s, i) => s === out.errors[i]) &&
      p.info.length === out.info.length &&
      p.info.every((s, i) => s === out.info[i])
        ? p
        : { errors: out.errors, info: out.info },
    );
    const vp = st.views.surf;
    if (vp.width !== w || vp.height !== h) st.setView("surf", vp.with({ width: w, height: h }));
  }, []);

  /** 轨道拖动时 pointermove 比帧还快，合并到每帧一次 */
  const schedule = useCallback(() => {
    if (frame.current) return;
    frame.current = requestAnimationFrame(() => {
      frame.current = 0;
      draw();
    });
  }, [draw]);

  useEffect(() => {
    schedule();
  }, [schedule, revision, camSig]);

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
      frame.current = 0;
    },
    [],
  );

  useEffect(() => {
    const canvas = cv.current;
    if (!canvas) return;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const st = useStore.getState();
      const cam = st.surf.cam;
      const next = Math.min(1e5, Math.max(1e-3, cam.dist * Math.exp(e.deltaY * 0.0012)));
      st.setSurf({ cam: { ...cam, dist: next } });
    };
    canvas.addEventListener("wheel", onWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", onWheel);
  }, []);

  const onPointerDown = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    const r = (e.target as Element).getBoundingClientRect();
    drag.current = { x: e.clientX - r.left, y: e.clientY - r.top };
    (e.target as Element).setPointerCapture?.(e.pointerId);
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (!drag.current) return;
    const r = (e.target as Element).getBoundingClientRect();
    const x = e.clientX - r.left;
    const y = e.clientY - r.top;
    const dx = x - drag.current.x;
    const dy = y - drag.current.y;
    drag.current = { x, y };
    const st = useStore.getState();
    const cam = st.surf.cam;
    if (e.shiftKey) {
      /* 沿相机基向量平移视点：内容跟随光标，故 target 取反方向 */
      const k = cam.dist * 0.0016;
      const ce = Math.cos(cam.elev),
        se = Math.sin(cam.elev);
      const cp = Math.cos(cam.azim),
        sp = Math.sin(cam.azim);
      const right: [number, number, number] = [-cp, sp, 0];
      const up: [number, number, number] = [-se * sp, -se * cp, ce];
      const t = cam.target;
      st.setSurf({
        cam: {
          ...cam,
          target: [
            t[0] - right[0] * dx * k + up[0] * dy * k,
            t[1] - right[1] * dx * k + up[1] * dy * k,
            t[2] - right[2] * dx * k + up[2] * dy * k,
          ],
        },
      });
      return;
    }
    st.setSurf({
      cam: {
        ...cam,
        azim: cam.azim + dx * 0.007,
        elev: Math.min(1.5, Math.max(-1.5, cam.elev + dy * 0.007)),
      },
    });
  };

  const end = () => {
    drag.current = null;
  };

  const reset = () => {
    const st = useStore.getState();
    st.setSurf({ cam: { ...st.surf.cam, azim: 0.7, elev: 0.5, target: [0, 0, 0] } });
  };

  return (
    <div className="gl-stage" ref={wrap}>
      <canvas
        ref={cv}
        className="gl-canvas"
        data-cursor={drag.current ? "grabbing" : "grab"}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={end}
        onPointerCancel={end}
        onPointerLeave={end}
      />
      <div className="gl-stage-tools">
        <button
          type="button"
          className="gl-tool"
          style={{ height: 30, width: 78 }}
          disabled={!fit}
          onClick={() => useStore.getState().setSurf({ cam: { ...useStore.getState().surf.cam, dist: fit } })}
        >
          适配视野
        </button>
        <button type="button" className="gl-tool" style={{ height: 30, width: 78 }} onClick={reset}>
          复位视角
        </button>
      </div>
      {hud.errors.length > 0 && <div className="gl-err">{hud.errors.slice(0, 5).join("\n")}</div>}
      <div className="gl-hud">
        {hud.info.map((l) => (
          <div className="gl-chip" key={l}>
            {l}
          </div>
        ))}
        <div className="gl-chip">拖动旋转 · 滚轮推拉 · Shift+拖动平移视点</div>
      </div>
    </div>
  );
}
