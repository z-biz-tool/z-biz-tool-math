/**
 * 左侧面板：按工作模式提供各自的编辑界面。
 * 所有写回都走 useStore.getState() 的动作，组件只订阅自己渲染需要的切片。
 */
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Button, Input, InputNumber, Segmented, Select, Slider, Switch, Tooltip } from "antd";
import {
  DeleteOutlined,
  EyeInvisibleOutlined,
  EyeOutlined,
  PauseOutlined,
  PlayCircleOutlined,
  PlusOutlined,
  RedoOutlined,
  ReloadOutlined,
  UndoOutlined,
} from "@ant-design/icons";
import { COLORMAPS } from "../core/colormap.ts";
import { fmt, fmtC } from "../core/cnum.ts";
import { GeometryDoc } from "../core/geometry.ts";
import { det, eigen, identity, isSymmetric, matMul, rref, shape, solve, svd, transpose } from "../core/linalg.ts";
import type { Mat } from "../core/linalg.ts";
import { createModel, dataset as sampleDataset, makeCache, makeGrad, rng, trainEpoch } from "../core/nn.ts";
import type { Act, Cache, Dataset, DatasetName, GradPack, Model } from "../core/nn.ts";
import { evalString, show } from "../core/machine.ts";
import { parseMatrix, parseVector } from "../core/parsemat.ts";
import { clearSurfCache } from "../render/scene3d.ts";
import { kindZh, toolZh } from "../render/scene2d.ts";
import { useStore, uid } from "../state.ts";
import type {
  ConsoleLine,
  Layer,
  LayerKind,
  NnState,
  SurfKind,
  SurfLayer,
  SurfStyle,
  ToolKind,
} from "../state.ts";

/* ------------------------------------------------------------ 通用小件 */

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="gl-card">
      <div className="gl-card-title">{title}</div>
      {children}
    </div>
  );
}

function Row({ children }: { children?: React.ReactNode }) {
  return <div className="gl-row">{children}</div>;
}

function Num(props: {
  value: number;
  onChange: (v: number) => void;
  min?: number;
  max?: number;
  step?: number;
  w?: number;
}) {
  return (
    <InputNumber
      size="small"
      value={props.value}
      min={props.min}
      max={props.max}
      step={props.step ?? 0.1}
      style={{ width: props.w ?? 72 }}
      onChange={(v) => {
        const n = typeof v === "number" && Number.isFinite(v) ? v : props.min ?? 0;
        props.onChange(n);
      }}
    />
  );
}

function Swatch({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  return (
    <input
      type="color"
      className="gl-swatch"
      value={/^#[0-9a-f]{6}$/i.test(value) ? value : "#667eea"}
      onChange={(e) => onChange(e.target.value)}
    />
  );
}

function Toggle({ label, on, onChange }: { label: string; on: boolean; onChange: (v: boolean) => void }) {
  return (
    <Row>
      <span className="gl-grow" style={{ fontSize: 12 }}>
        {label}
      </span>
      <Switch size="small" checked={on} onChange={onChange} />
    </Row>
  );
}

function Slide({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div>
      <div style={{ fontSize: 11.5, opacity: 0.72 }}>
        {label} = {Number(value.toFixed(4))}
      </div>
      <Slider min={min} max={max} step={step} value={value} onChange={onChange} tooltip={{ open: false }} />
    </div>
  );
}

function Eye({ on, onClick }: { on: boolean; onClick: () => void }) {
  return (
    <Button
      size="small"
      type="text"
      icon={on ? <EyeOutlined /> : <EyeInvisibleOutlined />}
      onClick={onClick}
      style={{ opacity: on ? 0.95 : 0.45 }}
    />
  );
}

const exprStyle: React.CSSProperties = { fontFamily: "SF Mono, Menlo, monospace", fontSize: 12.5 };

/** 读数行：左标签右数值，数值一律等宽，多个读数并排时不会跳动 */
function Stat({ k, v }: { k: string; v: string }) {
  return (
    <div className="gl-row" style={{ gap: 8, alignItems: "baseline" }}>
      <span style={{ fontSize: 11.5, width: 84, flex: "none", opacity: 0.72 }}>{k}</span>
      <span className="gl-mono gl-grow" style={{ textAlign: "right", wordBreak: "break-word" }}>
        {v}
      </span>
    </div>
  );
}

/** 面板内联的解析错误：与控制台同一条红，但不占用画布上的公告位 */
function ErrNote({ msg }: { msg: string }) {
  return (
    <div style={{ fontSize: 11.5, lineHeight: 1.7, color: "#fca5a5", whiteSpace: "pre-wrap" }}>{msg}</div>
  );
}

const HintNote = ({ children }: { children: React.ReactNode }) => (
  <div style={{ fontSize: 11.5, opacity: 0.66, lineHeight: 1.7 }}>{children}</div>
);

/* ------------------------------------------------------------ 函数模式 */

const LAYER_KINDS: { value: LayerKind; label: string }[] = [
  { value: "cartesian", label: "y = f(x)" },
  { value: "polar", label: "极坐标 r(θ)" },
  { value: "param", label: "参数方程 x(t),y(t)" },
  { value: "implicit", label: "隐函数 F(x,y)=c" },
  { value: "inequality", label: "不等式 F(x,y)≶0" },
  { value: "derivative", label: "导数 f '(x)" },
  { value: "integral", label: "积分 ∫f" },
  { value: "sequence", label: "数列 a(n)" },
];

const RELS = ["<", "<=", ">", ">="].map((r) => ({ value: r as Layer["rel"], label: r }));

function LayerCard({ l }: { l: Layer }) {
  const active = useStore((s) => s.activeLayer === l.id);
  const st = useStore.getState();
  const upd = (p: Partial<Layer>) => useStore.getState().updateLayer(l.id, p);
  const kind = l.kind;
  return (
    <div className="gl-layer" data-active={active} onClick={() => useStore.getState().patch({ activeLayer: l.id })}>
      <Row>
        <Eye
          on={l.visible}
          onClick={() => {
            upd({ visible: !l.visible });
            st.bump();
          }}
        />
        <Swatch value={l.color} onChange={(v) => upd({ color: v })} />
        <Input
          className="gl-grow"
          size="small"
          variant="borderless"
          style={exprStyle}
          value={l.expr}
          placeholder={kind === "polar" ? "sin(4*theta)" : kind === "sequence" ? "1 + 1/n" : "sin(x)*exp(-x/6)"}
          onChange={(e) => upd({ expr: e.target.value })}
        />
        <Tooltip title="删除图层">
          <Button
            size="small"
            type="text"
            danger
            icon={<DeleteOutlined />}
            onClick={() => useStore.getState().removeLayer(l.id)}
          />
        </Tooltip>
      </Row>
      {active && (
        <>
          <Row>
            <Select
              size="small"
              className="gl-grow"
              value={kind}
              options={LAYER_KINDS}
              onChange={(v: LayerKind) => upd({ kind: v })}
            />
          </Row>
          {(kind === "param" || kind === "integral") && (
            <Row>
              <span style={{ fontSize: 11.5, width: 42 }}>{kind === "param" ? "y(t)" : "上限"}</span>
              <Input
                className="gl-grow"
                size="small"
                style={exprStyle}
                value={l.expr2 ?? ""}
                placeholder={kind === "param" ? "1 - cos(t)" : "b"}
                onChange={(e) => upd({ expr2: e.target.value })}
              />
            </Row>
          )}
          {(kind === "implicit" || kind === "inequality") && (
            <Row>
              <span style={{ fontSize: 11.5, width: 42 }}>{kind === "implicit" ? "= c" : "方向"}</span>
              {kind === "implicit" ? (
                <Num value={l.level ?? 0} onChange={(v) => upd({ level: v })} step={0.1} w={80} />
              ) : (
                <Select size="small" style={{ width: 90 }} value={l.rel ?? "<"} options={RELS} onChange={(v) => upd({ rel: v })} />
              )}
            </Row>
          )}
          <Row>
            <span style={{ fontSize: 11.5, width: 42 }}>x 范围</span>
            <Num value={l.domain?.[0] ?? -6} onChange={(v) => upd({ domain: [v, l.domain?.[1] ?? 6] })} step={0.5} />
            <Num value={l.domain?.[1] ?? 6} onChange={(v) => upd({ domain: [l.domain?.[0] ?? -6, v] })} step={0.5} />
            <Button size="small" type="text" onClick={() => upd({ domain: undefined })}>
              视口
            </Button>
          </Row>
          <Row>
            <span style={{ fontSize: 11.5, width: 42 }}>标签</span>
            <Input
              className="gl-grow"
              size="small"
              value={l.label}
              placeholder="留空则用表达式"
              onChange={(e) => upd({ label: e.target.value })}
            />
          </Row>
          <Slide label="线宽" value={l.width} min={0.6} max={6} step={0.1} onChange={(v) => upd({ width: v })} />
          <Slide label="采样" value={l.samples} min={120} max={6000} step={20} onChange={(v) => upd({ samples: v })} />
          <Row>
            <Toggle label="虚线" on={l.dashed} onChange={(v) => upd({ dashed: v })} />
            <Toggle label={kind === "inequality" ? "填充区域" : "填充到轴"} on={l.fill} onChange={(v) => upd({ fill: v })} />
          </Row>
          <Row>
            <Button size="small" onClick={() => useStore.getState().reorder(l.id, -1)}>
              ↑ 上移
            </Button>
            <Button size="small" onClick={() => useStore.getState().reorder(l.id, 1)}>
              ↓ 下移
            </Button>
          </Row>
        </>
      )}
    </div>
  );
}

function FuncPanel() {
  const layers = useStore((s) => s.layers);
  const settings = useStore((s) => s.settings);
  const setSettings = useStore((s) => s.setSettings);
  const logX = useStore((s) => s.views.func.logX);
  const logY = useStore((s) => s.views.func.logY);
  /**
   * 对数轴长在视口上：投影本身要跟着变换走，做成全局开关会让其它模式失配。
   * 另一轴的现值取自 store，连点两个开关时不能沿用渲染时的闭包旧值。
   */
  const setLog = (axis: "x" | "y", on: boolean) => {
    const st = useStore.getState();
    const v = st.views.func;
    st.setView("func", axis === "x" ? v.withLog(on, v.logY) : v.withLog(v.logX, on));
  };
  return (
    <div className="gl-panel">
      <Card title="函数图层">
        {layers.map((l) => (
          <LayerCard key={l.id} l={l} />
        ))}
        <Row>
          <Button
            size="small"
            icon={<PlusOutlined />}
            className="gl-grow"
            onClick={() => useStore.getState().addLayer("cartesian", "cos(x)")}
          >
            添加函数
          </Button>
          <Button size="small" onClick={() => useStore.getState().addLayer("implicit", "x^2 + y^2 - 4")}>
            隐函数
          </Button>
          <Button size="small" onClick={() => useStore.getState().addLayer("polar", "1 + cos(theta)")}>
            极坐标
          </Button>
        </Row>
      </Card>
      <Card title="坐标轴">
        <Toggle label="对数刻度（x）" on={logX} onChange={(v) => setLog("x", v)} />
        <Toggle label="对数刻度（y）" on={logY} onChange={(v) => setLog("y", v)} />
        <Toggle label="π 刻度（x）" on={settings.piTicksX} onChange={(v) => setSettings({ piTicksX: v })} />
        <Toggle label="π 刻度（y）" on={settings.piTicksY} onChange={(v) => setSettings({ piTicksY: v })} />
        <Toggle label="次级网格" on={settings.showMinorGrid} onChange={(v) => setSettings({ showMinorGrid: v })} />
        <div style={{ fontSize: 11.5, opacity: 0.66, lineHeight: 1.7 }}>
          对数轴即 MATLAB 的 semilogx / semilogy / loglog：主刻度落在 10 的整数幂，次级网格按倍率铺开，
          采样按等倍率前进。该轴上不存在 0 与负值，中心会回到 1；开启后 π 刻度让位。
        </div>
        <div style={{ fontSize: 11.5, opacity: 0.66, lineHeight: 1.7 }}>
          支持 sin/cos/tan/asin/…、ln/log/exp、abs、sqrt、gamma、erf、floor/ceil、if(cond,a,b)、
          向量与复数；用户函数在控制台用 f(x) = … 定义。
        </div>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------ 几何模式 */

const TOOLS: ToolKind[] = [
  "select",
  "point",
  "segment",
  "line",
  "ray",
  "vector",
  "circle",
  "arc",
  "ellipse",
  "polygon",
  "midpoint",
  "intersection",
  "perpendicular",
  "parallel",
  "bisector",
  "pointOn",
  "locus",
  "rotate",
  "reflect",
  "dilate",
  "angle",
  "area",
  "text",
  "erase",
];

/** 每个工具一个可辨识字形（均已在 PingFang SC / system-ui 下实测有墨，不会退化成豆腐块） */
const TOOL_GLYPH: Record<ToolKind, string> = {
  select: "↖",
  point: "●",
  segment: "─",
  line: "╱",
  ray: "→",
  vector: "⇀",
  circle: "○",
  arc: "⌒",
  ellipse: "⬭",
  polygon: "△",
  midpoint: "⊙",
  intersection: "⊗",
  perpendicular: "⊥",
  parallel: "∥",
  bisector: "⊾",
  pointOn: "∈",
  locus: "∿",
  rotate: "↻",
  reflect: "⇄",
  dilate: "⤢",
  angle: "∠",
  area: "▦",
  text: "T",
  erase: "✕",
};

function GeoPanel() {
  const geo = useStore((s) => s.geo);
  const doc = geo.doc;
  const objects = doc.all();
  const label = (id: string) => doc.get(id)?.label ?? id;
  return (
    <div className="gl-panel">
      <Card title="工具">
        <div className="gl-tools">
          {TOOLS.map((t) => (
            <button
              key={t}
              type="button"
              className="gl-tool"
              data-on={geo.tool === t}
              title={toolZh(t)}
              onClick={() => useStore.getState().setGeo({ tool: t, pending: [] })}
            >
              <span style={{ fontSize: 15, lineHeight: "18px" }}>{TOOL_GLYPH[t]}</span>
              <span>{toolZh(t)}</span>
            </button>
          ))}
        </div>
        <div style={{ fontSize: 11.5, opacity: 0.72 }}>
          当前：<b style={{ color: "var(--gl-accent)" }}>{toolZh(geo.tool)}</b>
          {geo.pending.length > 0 && ` · 已选 ${geo.pending.map(label).join("、")}`}
        </div>
        <Row>
          <Button size="small" onClick={() => useStore.getState().setGeo({ pending: [] })}>
            清空待选
          </Button>
          <Button
            size="small"
            icon={<UndoOutlined />}
            onClick={() => {
              doc.undo();
              useStore.getState().bump();
            }}
          >
            撤销
          </Button>
          <Button
            size="small"
            icon={<RedoOutlined />}
            onClick={() => {
              doc.redo();
              useStore.getState().bump();
            }}
          >
            重做
          </Button>
        </Row>
      </Card>
      <Card title="显示">
        <Toggle label="轨迹（追踪点）" on={geo.showTrace} onChange={(v) => useStore.getState().setGeo({ showTrace: v })} />
        <Toggle label="标签" on={geo.showLabels} onChange={(v) => useStore.getState().setGeo({ showLabels: v })} />
        <Toggle label="网格" on={geo.grid} onChange={(v) => useStore.getState().setGeo({ grid: v })} />
        <Toggle label="吸附到格点" on={geo.snap} onChange={(v) => useStore.getState().setGeo({ snap: v })} />
      </Card>
      <Card title={`对象 ${objects.length}`}>
        {objects.map((g) => (
          <Row key={g.id}>
            <button
              type="button"
              className="gl-tool"
              style={{ height: 24, width: 26, padding: 0, fontSize: 11 }}
              data-on={g.visible !== false}
              title="显示/隐藏"
              onClick={() => {
                doc.setVisible(g.id, g.visible === false);
                useStore.getState().bump();
              }}
            >
              ●
            </button>
            <span className="gl-mono gl-grow" style={{ opacity: 0.9 }}>
              {g.label}
            </span>
            <span style={{ fontSize: 11, opacity: 0.6 }}>{kindZh(g.kind)}</span>
            <Button
              size="small"
              type="text"
              danger
              icon={<DeleteOutlined />}
              onClick={() => {
                doc.remove(g.id);
                useStore.getState().setGeo({ selected: null, pending: [] });
              }}
            />
          </Row>
        ))}
        <Row>
          <Button
            size="small"
            className="gl-grow"
            onClick={() => {
              const fresh = new GeometryDoc();
              fresh.engine = useStore.getState().engine;
              useStore.getState().setGeo({ doc: fresh, pending: [], selected: null });
            }}
          >
            新建空白画板
          </Button>
        </Row>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------ 复平面模式 */

const CPLX_MODES = [
  { value: "domain", label: "共形着色" },
  { value: "map", label: "保角映射" },
  { value: "newton", label: "Newton 分形" },
  { value: "log", label: "迭代吸引盆" },
];

const CURVES = [
  { value: "grid", label: "方格栅格" },
  { value: "circle", label: "同心圆" },
  { value: "ray", label: "射线" },
  { value: "polar", label: "圆 + 射线" },
  { value: "both", label: "圆 + 方格" },
];

function ComplexPanel() {
  const c = useStore((s) => s.cplx);
  const set = (p: Partial<typeof c>) => useStore.getState().setCplx(p);
  return (
    <div className="gl-panel">
      <Card title="复变函数 f(z)">
        <Input
          size="small"
          style={exprStyle}
          value={c.f}
          onChange={(e) => set({ f: e.target.value })}
          placeholder="(z^3 - 1)/(z^2 + 2i)"
        />
        <Segmented size="small" block value={c.mode} options={CPLX_MODES} onChange={(v) => set({ mode: v as typeof c.mode })} />
        {c.mode === "map" && (
          <Row>
            <span style={{ fontSize: 11.5, width: 56 }}>曲线族</span>
            <Select size="small" className="gl-grow" value={c.curve} options={CURVES} onChange={(v) => set({ curve: v })} />
          </Row>
        )}
        {c.mode === "domain" && (
          <Toggle label="叠加迭代收敛（z←f(z)）" on={c.iterative} onChange={(v) => set({ iterative: v })} />
        )}
      </Card>
      <Card title="着色">
        <Row>
          <span style={{ fontSize: 11.5, width: 56 }}>色标</span>
          <Select
            size="small"
            className="gl-grow"
            value={c.colormap}
            options={COLORMAPS.map((n) => ({ value: n, label: n }))}
            onChange={(v) => set({ colormap: v })}
          />
        </Row>
        <Slide label="等值线间隔" value={c.levelStep} min={0.2} max={4} step={0.1} onChange={(v) => set({ levelStep: v })} />
        <Slide label="栅格步长（1 最细）" value={c.resolution} min={1} max={4} step={1} onChange={(v) => set({ resolution: v })} />
        <div style={{ fontSize: 11.5, opacity: 0.66, lineHeight: 1.7 }}>
          着色约定：色相 = arg f(z)，明度 = log|f|。零点为完整色环，极点为反向色环。
        </div>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------ 向量模式 */

const FIELD_MODES = [
  { value: "quiver", label: "向量场" },
  { value: "slope", label: "斜率场" },
  { value: "stream", label: "流线" },
  { value: "phase", label: "相图" },
];

function VectorPanel() {
  const v = useStore((s) => s.vec);
  const set = (p: Partial<typeof v>) => useStore.getState().setVec(p);
  return (
    <div className="gl-panel">
      <Card title="场">
        <Segmented size="small" block value={v.fieldMode} options={FIELD_MODES} onChange={(x) => set({ fieldMode: x as typeof v.fieldMode })} />
        {v.fieldMode === "slope" ? (
          <Row>
            <span style={{ fontSize: 11.5, width: 34 }}>y' =</span>
            <Input className="gl-grow" size="small" style={exprStyle} value={v.dfxy} onChange={(e) => set({ dfxy: e.target.value })} />
          </Row>
        ) : (
          <>
            <Row>
              <span style={{ fontSize: 11.5, width: 26 }}>F.x</span>
              <Input className="gl-grow" size="small" style={exprStyle} value={v.fx} onChange={(e) => set({ fx: e.target.value })} />
            </Row>
            <Row>
              <span style={{ fontSize: 11.5, width: 26 }}>F.y</span>
              <Input className="gl-grow" size="small" style={exprStyle} value={v.fy} onChange={(e) => set({ fy: e.target.value })} />
            </Row>
          </>
        )}
        <Slide label="密度" value={v.density} min={6} max={44} step={1} onChange={(x) => set({ density: x })} />
        {(v.fieldMode === "stream" || v.fieldMode === "phase") && (
          <>
            <Slide label="流线条数" value={v.streamlineCount} min={4} max={48} step={1} onChange={(x) => set({ streamlineCount: x })} />
            <Slide label="积分步数" value={v.steps} min={100} max={2000} step={50} onChange={(x) => set({ steps: x })} />
          </>
        )}
        <Row>
          <span style={{ fontSize: 11.5, width: 56 }}>色标</span>
          <Select
            size="small"
            className="gl-grow"
            value={v.colormap}
            options={COLORMAPS.map((n) => ({ value: n, label: n }))}
            onChange={(x) => set({ colormap: x })}
          />
        </Row>
      </Card>
      <Card title="自由向量">
        {v.arrows.map((a) => (
          <Row key={a.id}>
            <Swatch
              value={a.color}
              onChange={(c) => set({ arrows: v.arrows.map((q) => (q.id === a.id ? { ...q, color: c } : q)) })}
            />
            <Input
              size="small"
              style={{ width: 34, ...exprStyle }}
              value={a.label}
              onChange={(e) => set({ arrows: v.arrows.map((q) => (q.id === a.id ? { ...q, label: e.target.value } : q)) })}
            />
            <Input
              className="gl-grow"
              size="small"
              style={exprStyle}
              value={a.vec}
              placeholder="[3,1]"
              onChange={(e) => set({ arrows: v.arrows.map((q) => (q.id === a.id ? { ...q, vec: e.target.value } : q)) })}
            />
            <Button
              size="small"
              type="text"
              danger
              icon={<DeleteOutlined />}
              onClick={() => set({ arrows: v.arrows.filter((q) => q.id !== a.id) })}
            />
          </Row>
        ))}
        <Row>
          <Button
            size="small"
            icon={<PlusOutlined />}
            className="gl-grow"
            onClick={() =>
              set({
                arrows: [
                  ...v.arrows,
                  { id: uid("V"), tail: "[0,0]", vec: "[1,1]", color: COLORMAPS[v.arrows.length % COLORMAPS.length], label: `w${v.arrows.length + 1}` },
                ],
              })
            }
          >
            添加向量
          </Button>
        </Row>
        <div style={{ fontSize: 11.5, opacity: 0.66, lineHeight: 1.7 }}>
          起点与分量都可写表达式（如 [a, b]）。两个向量时自动画出平行四边形法则 u+v、u−v。
        </div>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------ 线性代数模式 */

const LIN_DIMS = [
  { value: 2, label: "2 × 2" },
  { value: 3, label: "3 × 3" },
];

/** 扩维时按单位矩阵补格：原来的二维变换作为左上块保留，新方向先是恒等 */
const LIN_PAD = ["1", "0", "0", "0", "1", "0", "0", "0", "1"];

interface LinReadout {
  n: number;
  det: number;
  trace: number;
  rank: number;
  eig: string;
  sing: string;
  /** 奇异值之积：方阵时应与 |det| 相等，这条对照是给学生自查的 */
  area: number;
  fro: number;
  sol: string;
  sym: boolean;
  /** |A^T A − I| 的 F 范数：为 0 才是正交阵，只报实算值不下结论 */
  orth: number;
}

function linReadout(a: Mat, b: number[]): LinReadout {
  const n = shape(a)[0];
  const { rank } = rref(a);
  /**
   * eigen() 各条路径的次序并不统一（对称阵走 Jacobi 降序、2×2 先给 (tr+√)/2），
   * 直接列出来会让改一个格子时整个列表重排，所以展示前统一按实部降序。
   */
  const eig = eigen(a)
    .slice()
    .sort((x, y) => y.re - x.re || y.im - x.im)
    .map((z) => fmtC(z.re, z.im, 3))
    .join("，");
  const { s, area } = svd(a);
  const eye = identity(n);
  const ata = matMul(transpose(a), a);
  let sol = "无唯一解（秩亏）";
  if (rank === n) {
    try {
      sol = `x = (${solve(a, b).map((v) => fmt(v, 3)).join("，")})`;
    } catch (e) {
      sol = e instanceof Error ? e.message : "求解失败";
    }
  }
  return {
    n,
    det: det(a),
    trace: a.reduce((t, row, i) => t + row[i], 0),
    rank,
    eig,
    sing: s.map((v) => fmt(v, 3)).join("，"),
    area,
    fro: Math.hypot(...a.flat()),
    sol,
    sym: isSymmetric(a),
    orth: Math.hypot(...ata.flatMap((row, i) => row.map((v, j) => v - eye[i][j]))),
  };
}

function LinPanel() {
  const lin = useStore((s) => s.lin);
  const engine = useStore((s) => s.engine);
  const setLin = useStore((s) => s.setLin);
  /** 格子里可以引用参数滑块，滑块一动读数值就变了，所以签名里要带上现值 */
  const paramSig = useStore((s) => s.params.map((p) => `${p.name}=${p.value}`).join("|"));

  const setCell = (i: number, v: string) => {
    const a = lin.a.slice();
    a[i] = v;
    setLin({ a });
  };
  const setElem = (i: number, v: string) => {
    const b = lin.b.slice();
    b[i] = v;
    setLin({ b });
  };
  const resize = (dim: 2 | 3) => {
    /**
     * 换维必须按行列重排：行优先串的长度一变化旧矩阵的下标就整体错位，
     * 直接 slice/push 会把原矩阵的第二行抖到新矩阵的第一行末尾。
     */
    const old = lin.dim;
    const a: string[] = [];
    for (let i = 0; i < dim; i++)
      for (let j = 0; j < dim; j++)
        a.push(i < old && j < old ? (lin.a[i * old + j] ?? LIN_PAD[i * dim + j]) : LIN_PAD[i * dim + j]);
    const b: string[] = [];
    for (let i = 0; i < dim; i++) b.push(i < old ? lin.b[i] : "0");
    // 画布只有 2×2 的闭式矩阵指数，升到三维必须回到离散幂次
    if (dim === 3) setLin({ dim, a, b, useExp: false });
    else setLin({ dim, a, b });
  };
  const setExp = (on: boolean) => {
    // t 的含义在两种模式下不同（整数幂次 ↔ 流时间），切换时把值夹进新范围
    const t = on ? Math.max(-2, Math.min(2, lin.t)) : Math.max(0, Math.min(6, lin.t));
    setLin({ useExp: on, t });
  };

  /** 依赖取「输入指纹」而不是数组本身：LU/SVD 不该在每次按键触发的重渲染里重跑 */
  const sig = `${lin.dim}|${lin.a.join("\u0001")}|${lin.b.join("\u0001")}|${paramSig}`;
  const parsed = useMemo(() => {
    try {
      return {
        ro: linReadout(parseMatrix(engine, lin.a, lin.dim), parseVector(engine, lin.b, lin.dim)),
        err: "",
      };
    } catch (e) {
      return { ro: null as LinReadout | null, err: e instanceof Error ? e.message : String(e) };
    }
  }, [sig]);
  /** 改一格可能只是表达式写了一半，这时保留上一次读数而不是把整块清空 */
  const good = useRef<LinReadout | null>(parsed.ro);
  if (parsed.ro) good.current = parsed.ro;
  const ro = good.current;

  const grid: React.CSSProperties = {
    display: "grid",
    gridTemplateColumns: `repeat(${lin.dim}, minmax(0, 1fr))`,
    gap: 6,
    flex: 1,
    minWidth: 0,
  };
  const bracket: React.CSSProperties = { fontSize: 34, fontWeight: 300, lineHeight: 1, color: "var(--gl-muted)" };
  return (
    <div className="gl-panel">
      <Card title="矩阵 A（每格一个表达式）">
        <Segmented size="small" block value={lin.dim} options={LIN_DIMS} onChange={(v) => resize(v as 2 | 3)} />
        <Row>
          <span style={bracket}>[</span>
          <div style={grid}>
            {lin.a.map((v, i) => (
              <Input
                key={i}
                size="small"
                style={exprStyle}
                value={v}
                placeholder="0"
                onChange={(e) => setCell(i, e.target.value)}
              />
            ))}
          </div>
          <span style={bracket}>]</span>
        </Row>
        <HintNote>
          行优先排列，格子里可写 sin(pi/3)、2^3 这类表达式，也可直接引用底部参数滑块 a、b（拖动即时重算）。
        </HintNote>
      </Card>
      <Card title="右端项 b">
        <Row>
          <span style={{ fontSize: 11.5, width: 34, flex: "none" }}>b =</span>
          <div style={grid}>
            {lin.b.map((v, i) => (
              <Input
                key={i}
                size="small"
                style={exprStyle}
                value={v}
                placeholder="0"
                onChange={(e) => setElem(i, e.target.value)}
              />
            ))}
          </div>
        </Row>
      </Card>
      <Card title="画什么">
        <Toggle label="整数网格线的像" on={lin.showGrid} onChange={(v) => setLin({ showGrid: v })} />
        <Toggle label="单位圆（原像）" on={lin.showCircle} onChange={(v) => setLin({ showCircle: v })} />
        <Toggle label="单位圆的像" on={lin.showEllipse} onChange={(v) => setLin({ showEllipse: v })} />
        <Toggle label="实特征向量" on={lin.showEigen} onChange={(v) => setLin({ showEigen: v })} />
        <Toggle label="SVD 奇异向量与主轴" on={lin.showSVD} onChange={(v) => setLin({ showSVD: v })} />
        <Toggle label="轨道 p ↦ A·p" on={lin.showFlow} onChange={(v) => setLin({ showFlow: v })} />
        <Slide
          label={lin.useExp ? "t（流时间）" : "k（离散幂次）"}
          value={lin.t}
          min={lin.useExp ? -2 : 0}
          max={lin.useExp ? 2 : 6}
          step={lin.useExp ? 0.02 : 1}
          onChange={(v) => setLin({ t: v })}
        />
        <Row>
          <Tooltip
            title={
              lin.dim === 3
                ? "三维矩阵指数尚未实现：画布只用 2×2 的 Cayley–Hamilton 闭式解，三维请回到离散幂次 A^k"
                : "关：画 A 的整数幂 A^k；开：画连续流 e^(tA)，即线性系统 dx/dt = A·x"
            }
          >
            <span className="gl-grow" style={{ fontSize: 12 }}>
              连续流 e^(tA)（矩阵指数）
            </span>
          </Tooltip>
          <Switch size="small" checked={lin.useExp} disabled={lin.dim === 3} onChange={setExp} />
        </Row>
        {lin.dim === 3 && <HintNote>三维下矩阵指数不可用，开关已锁定为离散幂次 A^k。</HintNote>}
      </Card>
      <Card title="读数">
        {parsed.err && <ErrNote msg={`解析失败：${parsed.err}`} />}
        {ro ? (
          <>
            <Stat k="det A" v={fmt(ro.det, 4)} />
            <Stat k="迹 tr(A)" v={fmt(ro.trace, 4)} />
            <Stat k="秩 rank" v={`${ro.rank} / ${ro.n}`} />
            <Stat k="特征值 λ" v={ro.eig} />
            <Stat k="奇异值 σ" v={ro.sing} />
            <Stat k="Frobenius" v={fmt(ro.fro, 4)} />
            <Stat k="Ax = b" v={ro.sol} />
            <Stat k="对称 A = A^T" v={ro.sym ? "是" : "否"} />
            <Stat k="|A^TA − I|" v={fmt(ro.orth, 4)} />
            <HintNote>
              面积放大率 |det A| = {fmt(Math.abs(ro.det), 4)}，σ 之积 = {fmt(ro.area, 4)}（两者相等说明 SVD 与行列式自洽）。
              |A^TA − I| = {fmt(ro.orth, 3)} {ro.orth < 1e-6 ? "，即正交阵：保长度与夹角。" : "，不为 0，非正交阵。"}
            </HintNote>
          </>
        ) : (
          <HintNote>把每一格填成实数表达式后显示读数。</HintNote>
        )}
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------ 神经网络模式 */

const NN_DATASETS: { value: DatasetName; label: string }[] = [
  { value: "xor", label: "异或" },
  { value: "circle", label: "同心圆" },
  { value: "moons", label: "双月" },
  { value: "spiral", label: "三臂螺旋" },
];

const NN_ACTS: { value: Act; label: string }[] = [
  { value: "tanh", label: "tanh 双曲正切" },
  { value: "relu", label: "ReLU 线性修正" },
  { value: "sigmoid", label: "Sigmoid S 型" },
];

/** 迷你曲线保留的点数上限 */
const CURVE_MAX = 240;
/** 一帧里训练的时间预算（ms），剩下的时间还给画布与输入事件 */
const EPOCH_BUDGET_MS = 8;

/**
 * 曲线封顶的方式是隔点抽稀而不是滑窗丢弃：本机一帧能跑 20+ 轮，
 * 严格保留「最后 240 轮」时曲线里永远只剩收敛末尾的一条平线，
 * 抽稀后第一轮的 0.7 仍在，整条下降过程都看得见，而点数依旧 ≤ CURVE_MAX。
 */
function pushCurve(curve: number[], add: number[]): number[] {
  let out = curve.length ? curve.concat(add) : add.slice();
  while (out.length > CURVE_MAX) {
    const thin = out.filter((_, i) => i % 2 === 0);
    out = thin.length >= 2 ? thin : out.slice(-CURVE_MAX);
  }
  return out;
}

interface TrainKit {
  cache: Cache;
  g: GradPack;
  vel: GradPack;
  r: () => number;
}

/**
 * 模型与训练缓冲按身份键活在模块作用域：面板切到别的模式会卸载，
 * 但训练成果必须还在，回来接着练。
 */
let nnKey = "";
let nnKit: TrainKit | null = null;

type NnConfig = Pick<NnState, "dataset" | "samples" | "hidden" | "depth" | "act" | "seed">;

/** 决定「是不是同一张网络」的只有这六项；lr/动量/batch 是优化器的量，改它们不能重建 */
const nnIdent = (c: NnConfig): string => `${c.dataset}|${c.samples}|${c.hidden}|${c.depth}|${c.act}|${c.seed}`;

function buildNn(c: NnConfig): { model: Model; data: Dataset; kit: TrainKit } {
  const data = sampleDataset(c.dataset, c.samples, c.seed);
  const model = createModel([2, ...Array(Math.max(1, Math.round(c.depth))).fill(Math.max(1, c.hidden)), data.k], c.act, c.seed);
  return {
    model,
    data,
    kit: { cache: makeCache(model), g: makeGrad(model), vel: makeGrad(model), r: rng(c.seed * 7919 + 13) },
  };
}

function nnParamCount(m: Model | null): number {
  return m ? m.layers.reduce((s, L) => s + L.w.length + L.b.length, 0) : 0;
}

/** 迷你 loss 曲线：内联小画布，用外壳那条紫色渐变 */
function LossCurve({ data }: { data: number[] }) {
  const ref = useRef<HTMLCanvasElement | null>(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const w = cv.clientWidth;
    const h = cv.clientHeight;
    if (w < 4 || h < 4) return;
    const dpr = Math.min(2.5, window.devicePixelRatio || 1);
    const pw = Math.floor(w * dpr);
    const ph = Math.floor(h * dpr);
    if (cv.width !== pw || cv.height !== ph) {
      cv.width = pw;
      cv.height = ph;
    }
    const ctx = cv.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    if (data.length < 2) return;
    let hi = -Infinity;
    let lo = Infinity;
    for (const v of data) {
      if (v > hi) hi = v;
      if (v < lo) lo = v;
    }
    /** 交叉熵从 0.7 掉到 0.001 跨三个数量级，线性纵轴会把后期收敛压成一条贴底直线 */
    const log = hi > 0 && lo > 0 && hi / lo > 40;
    const tf = (v: number) => (log ? Math.log(Math.max(v, 1e-12)) : v);
    const [th, tl] = [tf(hi), tf(lo)];
    const span = th - tl || 1;
    const px = (i: number) => (i / (data.length - 1)) * w;
    const py = (v: number) => h - 3 - ((tf(v) - tl) / span) * (h - 6);
    const stroke = ctx.createLinearGradient(0, 0, w, 0);
    stroke.addColorStop(0, "#667eea");
    stroke.addColorStop(1, "#764ba2");
    ctx.beginPath();
    data.forEach((v, i) => (i ? ctx.lineTo(px(i), py(v)) : ctx.moveTo(px(i), py(v))));
    ctx.strokeStyle = stroke;
    ctx.lineWidth = 1.6;
    ctx.lineJoin = "round";
    ctx.stroke();
    const fill = ctx.createLinearGradient(0, 0, 0, h);
    fill.addColorStop(0, "rgba(102, 126, 234, 0.28)");
    fill.addColorStop(1, "rgba(118, 75, 162, 0.02)");
    ctx.lineTo(w, h);
    ctx.lineTo(0, h);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();
  }, [data]);
  return (
    <canvas
      ref={ref}
      style={{ display: "block", width: "100%", height: 56, borderRadius: 8, border: "1px solid var(--gl-line)" }}
    />
  );
}

function NnPanel() {
  const nn = useStore((s) => s.nn);
  const setNn = useStore((s) => s.setNn);
  const rafId = useRef(0);
  const burstId = useRef(0);
  const [burst, setBurst] = useState<number | null>(null);
  const key = nnIdent(nn);

  /** 新建同尺寸的随机初始化模型并接管训练缓冲；身份变化与「重置」走同一条路径 */
  const rebuild = useCallback(() => {
    const cur = useStore.getState().nn;
    const built = buildNn(cur);
    nnKey = nnIdent(cur);
    nnKit = built.kit;
    setNn({ model: built.model, data: built.data, epochs: 0, loss: 0, acc: 0, curve: [] });
  }, [setNn]);

  /**
   * 训练是原地改 model 权重的，重建等于把训练清零：
   * 所以这里只在身份键变化（或模型丢失）时重建，其余渲染一律不动。
   */
  useEffect(() => {
    const cur = useStore.getState().nn;
    if (nnKey === key && nnKit && cur.model && cur.data) return;
    rebuild();
  }, [key, rebuild]);

  /* 唯一的训练循环：一帧内跑满 8ms 预算、只提交一次 store，
     既不让每个控件各起一个 rAF，也不会一帧写几十次 revision。 */
  useEffect(() => {
    if (!nn.running) return;
    const tick = () => {
      const st = useStore.getState();
      const kit = nnKit;
      if (!st.nn.running || !kit || !st.nn.model || !st.nn.data) return;
      const opt = { lr: st.nn.lr, momentum: st.nn.momentum, batch: st.nn.batch };
      const t0 = performance.now();
      const losses: number[] = [];
      let acc = st.nn.acc;
      do {
        const s = trainEpoch(st.nn.model, st.nn.data, opt, kit.r, kit.cache, kit.g, kit.vel);
        losses.push(s.loss);
        acc = s.acc;
      } while (performance.now() - t0 < EPOCH_BUDGET_MS);
      const curve = pushCurve(st.nn.curve, losses);
      st.setNn({
        epochs: st.nn.epochs + losses.length,
        loss: losses[losses.length - 1],
        acc,
        curve,
        running: true,
      });
      rafId.current = requestAnimationFrame(tick);
    };
    rafId.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId.current);
  }, [nn.running]);

  /* 卸载时收回两条 rAF：训练循环靠 running 变假自动停，补练定时器得显式取消 */
  useEffect(
    () => () => {
      cancelAnimationFrame(rafId.current);
      cancelAnimationFrame(burstId.current);
    },
    [],
  );

  /** 单跑 100 轮：按帧切片推进，界面能看到轮次在涨而不是白屏卡住 */
  const burst100 = () => {
    cancelAnimationFrame(burstId.current);
    const total = 100;
    let done = 0;
    let loss = 0;
    let acc = 0;
    const step = () => {
      const st = useStore.getState();
      const kit = nnKit;
      if (!kit || !st.nn.model || !st.nn.data) {
        setBurst(null);
        return;
      }
      const opt = { lr: st.nn.lr, momentum: st.nn.momentum, batch: st.nn.batch };
      const t0 = performance.now();
      const losses: number[] = [];
      while (done < total && performance.now() - t0 < EPOCH_BUDGET_MS) {
        const s = trainEpoch(st.nn.model, st.nn.data, opt, kit.r, kit.cache, kit.g, kit.vel);
        losses.push(s.loss);
        loss = s.loss;
        acc = s.acc;
        done++;
      }
      st.setNn({
        epochs: st.nn.epochs + losses.length,
        loss,
        acc,
        curve: pushCurve(st.nn.curve, losses),
      });
      setBurst(done < total ? done : null);
      if (done < total) burstId.current = requestAnimationFrame(step);
      else burstId.current = 0;
    };
    burstId.current = requestAnimationFrame(step);
  };

  const toggleRun = () => {
    cancelAnimationFrame(burstId.current);
    burstId.current = 0;
    setBurst(null);
    setNn({ running: !nn.running });
  };

  return (
    <div className="gl-panel">
      <Card title="数据与网络">
        <Row>
          <span style={{ fontSize: 11.5, width: 42, flex: "none" }}>数据集</span>
          <Select size="small" className="gl-grow" value={nn.dataset} options={NN_DATASETS} onChange={(v: DatasetName) => setNn({ dataset: v })} />
        </Row>
        <Row>
          <span style={{ fontSize: 11.5, width: 42, flex: "none" }}>激活</span>
          <Select size="small" className="gl-grow" value={nn.act} options={NN_ACTS} onChange={(v: Act) => setNn({ act: v })} />
        </Row>
        <Slide label="样本数" value={nn.samples} min={40} max={600} step={20} onChange={(v) => setNn({ samples: v })} />
        <Slide label="隐藏层宽度" value={nn.hidden} min={2} max={24} step={1} onChange={(v) => setNn({ hidden: v })} />
        <Slide label="隐藏层数" value={nn.depth} min={1} max={4} step={1} onChange={(v) => setNn({ depth: v })} />
        <Slide label="随机种子" value={nn.seed} min={1} max={200} step={1} onChange={(v) => setNn({ seed: v })} />
        <HintNote>
          结构 [2, {Array.from({ length: Math.max(1, nn.depth) }).fill(nn.hidden).join(", ")}, 类别数]；
          改数据集、宽度、层数、激活或种子都会重建网络（训练清零），改学习率一类只换优化器节奏。
        </HintNote>
      </Card>
      <Card title="优化器">
        <Slide label="学习率 lr" value={nn.lr} min={0.002} max={0.3} step={0.002} onChange={(v) => setNn({ lr: v })} />
        <Slide label="动量" value={nn.momentum} min={0} max={0.99} step={0.01} onChange={(v) => setNn({ momentum: v })} />
        <Slide label="批大小" value={nn.batch} min={1} max={64} step={1} onChange={(v) => setNn({ batch: v })} />
        <Slide label="边界栅格分辨率" value={nn.boundaryRes} min={24} max={200} step={4} onChange={(v) => setNn({ boundaryRes: v })} />
        <Toggle label="决策边界着色" on={nn.showBoundary} onChange={(v) => setNn({ showBoundary: v })} />
      </Card>
      <Card title="训练">
        <Row>
          <Button
            size="small"
            type="primary"
            className="gl-grow"
            icon={nn.running ? <PauseOutlined /> : <PlayCircleOutlined />}
            onClick={toggleRun}
          >
            {nn.running ? "暂停" : "训练"}
          </Button>
          <Button size="small" icon={<ReloadOutlined />} onClick={rebuild}>
            重置
          </Button>
        </Row>
        <Row>
          <Tooltip title="与连续训练互斥：一次补 100 轮，按帧切片跑完">
            <Button size="small" className="gl-grow" disabled={nn.running || burst !== null} onClick={burst100}>
              {burst === null ? "训练 100 轮" : `补练中 ${burst}/100`}
            </Button>
          </Tooltip>
        </Row>
        <Stat k="轮次" v={String(nn.epochs)} />
        <Stat k="损失 loss" v={nn.curve.length ? fmt(nn.loss, 5) : "—"} />
        <Stat k="正确率" v={nn.model ? `${fmt(nn.acc * 100, 2)} %` : "—"} />
        <Stat k="可训练参数" v={nn.model ? String(nnParamCount(nn.model)) : "—"} />
        <LossCurve data={nn.curve} />
        <HintNote>曲线最多 {CURVE_MAX} 点，超出后隔点抽稀（首点必定保留）；纵轴自动取对数。</HintNote>
        {nn.model === null && <ErrNote msg="模型尚未建立。" />}
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------ 曲面模式 */

const SURF_KINDS: { value: SurfKind; label: string }[] = [
  { value: "graph", label: "z = f(x,y)" },
  { value: "param", label: "参数曲面" },
  { value: "implicit", label: "隐式曲面" },
  { value: "revolve", label: "旋转体" },
  { value: "spacecurve", label: "空间曲线" },
];

const SURF_STYLES: { value: SurfStyle; label: string }[] = [
  { value: "surf", label: "surf" },
  { value: "mesh", label: "mesh" },
  { value: "wire", label: "wire" },
  { value: "contour", label: "contour3" },
  { value: "surfc", label: "surfc" },
];

function SurfCard({ l }: { l: SurfLayer }) {
  const active = useStore((s) => s.activeLayer === l.id);
  const set = (p: Partial<SurfLayer>) =>
    useStore.getState().setSurf({
      layers: useStore.getState().surf.layers.map((q) => (q.id === l.id ? { ...q, ...p } : q)),
    });
  const rangeLabel =
    l.kind === "graph" ? "x 范围 / y 范围" : l.kind === "param" ? "u 范围 / v 范围" : l.kind === "revolve" ? "t 范围 · — · 段数" : "t 范围";
  const [r0, r1, r2, r3] = l.range;
  return (
    <div className="gl-layer" data-active={active} onClick={() => useStore.getState().patch({ activeLayer: l.id })}>
      <Row>
        <Eye on={l.visible} onClick={() => set({ visible: !l.visible })} />
        <Swatch value={l.color} onChange={(v) => set({ color: v })} />
        <Input
          className="gl-grow"
          size="small"
          variant="borderless"
          style={exprStyle}
          value={l.expr}
          onChange={(e) => set({ expr: e.target.value })}
        />
        <Button size="small" type="text" danger icon={<DeleteOutlined />} onClick={() => useStore.getState().setSurf({ layers: useStore.getState().surf.layers.filter((q) => q.id !== l.id) })} />
      </Row>
      {l.kind !== "graph" && (
        <Row>
          <span style={{ fontSize: 11.5, width: 34 }}>expr2</span>
          <Input
            className="gl-grow"
            size="small"
            style={exprStyle}
            value={l.expr2 ?? ""}
            placeholder={l.kind === "revolve" ? "x 或 y（旋转轴）" : l.kind === "spacecurve" ? "y(t)" : "y(u,v)"}
            onChange={(e) => set({ expr2: e.target.value })}
          />
        </Row>
      )}
      {(l.kind === "param" || l.kind === "spacecurve") && (
        <Row>
          <span style={{ fontSize: 11.5, width: 34 }}>expr3</span>
          <Input className="gl-grow" size="small" style={exprStyle} value={l.expr3 ?? ""} placeholder="z(u,v)" onChange={(e) => set({ expr3: e.target.value })} />
        </Row>
      )}
      <Row>
        <Select size="small" className="gl-grow" value={l.kind} options={SURF_KINDS} onChange={(v: SurfKind) => set({ kind: v })} />
        <Select size="small" style={{ width: 96 }} value={l.style} options={SURF_STYLES} onChange={(v: SurfStyle) => set({ style: v })} />
      </Row>
      <div style={{ fontSize: 11, opacity: 0.6 }}>{rangeLabel}</div>
      <Row>
        <Num value={r0} step={0.5} onChange={(v) => set({ range: [v, r1, r2, r3] })} />
        <Num value={r1} step={0.5} onChange={(v) => set({ range: [r0, v, r2, r3] })} />
        <Num value={r2} step={0.5} onChange={(v) => set({ range: [r0, r1, v, r3] })} />
        <Num value={r3} step={l.kind === "revolve" ? 6 : 0.5} onChange={(v) => set({ range: [r0, r1, r2, v] })} />
      </Row>
      <Row>
        <span style={{ fontSize: 11.5, width: 56 }}>色标</span>
        <Select
          size="small"
          className="gl-grow"
          value={l.colormap}
          options={COLORMAPS.map((n) => ({ value: n, label: n }))}
          onChange={(v) => set({ colormap: v })}
        />
      </Row>
      <Slide label="分辨率" value={l.res} min={8} max={200} step={2} onChange={(v) => set({ res: v })} />
      <Slide label="不透明度" value={l.opacity} min={0.15} max={1} step={0.05} onChange={(v) => set({ opacity: v })} />
      <Row>
        <Toggle label="光照" on={l.lit} onChange={(v) => set({ lit: v })} />
        <span className="gl-grow" />
        <Input size="small" style={{ width: 110 }} value={l.label} placeholder="图例名" onChange={(e) => set({ label: e.target.value })} />
      </Row>
    </div>
  );
}

function SurfPanel() {
  const surf = useStore((s) => s.surf);
  const set = (p: Partial<typeof surf>) => useStore.getState().setSurf(p);
  return (
    <div className="gl-panel">
      <Card title="曲面图层">
        {surf.layers.map((l) => (
          <SurfCard key={l.id} l={l} />
        ))}
        <Row>
          <Button
            size="small"
            icon={<PlusOutlined />}
            className="gl-grow"
            onClick={() =>
              set({
                layers: [
                  ...surf.layers,
                  {
                    id: uid("S"),
                    kind: "graph",
                    expr: "sin(x)*cos(y)",
                    style: "surf",
                    colormap: "parula",
                    res: 70,
                    range: [-3, 3, -3, 3],
                    visible: true,
                    color: "#667eea",
                    opacity: 1,
                    lit: true,
                    label: "新曲面",
                  },
                ],
              })
            }
          >
            添加曲面
          </Button>
        </Row>
      </Card>
      <Card title="视角与包围盒">
        <Slide label="方位角" value={surf.cam.azim} min={-Math.PI} max={Math.PI} step={0.02} onChange={(v) => set({ cam: { ...surf.cam, azim: v } })} />
        <Slide label="仰角" value={surf.cam.elev} min={-1.5} max={1.5} step={0.02} onChange={(v) => set({ cam: { ...surf.cam, elev: v } })} />
        <Slide label="距离" value={surf.cam.dist} min={1} max={40} step={0.2} onChange={(v) => set({ cam: { ...surf.cam, dist: v } })} />
        <Toggle label="自动包围盒" on={surf.autoBox} onChange={(v) => set({ autoBox: v })} />
        <Toggle label="显示坐标框" on={surf.showAxes} onChange={(v) => set({ showAxes: v })} />
        {!surf.autoBox && (
          <>
            <div style={{ fontSize: 11, opacity: 0.6 }}>世界包围盒 x₀ x₁ y₀ y₁ z₀ z₁</div>
            <Row>
              <Num value={surf.box[0]} step={0.5} onChange={(v) => set({ box: [v, surf.box[1], surf.box[2], surf.box[3], surf.box[4], surf.box[5]] })} />
              <Num value={surf.box[1]} step={0.5} onChange={(v) => set({ box: [surf.box[0], v, surf.box[2], surf.box[3], surf.box[4], surf.box[5]] })} />
              <Num value={surf.box[2]} step={0.5} onChange={(v) => set({ box: [surf.box[0], surf.box[1], v, surf.box[3], surf.box[4], surf.box[5]] })} />
            </Row>
            <Row>
              <Num value={surf.box[3]} step={0.5} onChange={(v) => set({ box: [surf.box[0], surf.box[1], surf.box[2], v, surf.box[4], surf.box[5]] })} />
              <Num value={surf.box[4]} step={0.5} onChange={(v) => set({ box: [surf.box[0], surf.box[1], surf.box[2], surf.box[3], v, surf.box[5]] })} />
              <Num value={surf.box[5]} step={0.5} onChange={(v) => set({ box: [surf.box[0], surf.box[1], surf.box[2], surf.box[3], surf.box[4], v] })} />
            </Row>
          </>
        )}
        <div style={{ fontSize: 11.5, opacity: 0.66, lineHeight: 1.7 }}>
          隐式曲面与自动包围盒共用这里的范围；拖动画布旋转，Shift+拖动平移，滚轮推拉。
        </div>
      </Card>
    </div>
  );
}

/* ------------------------------------------------------------ 控制台 */

const HELP = [
  "f(x) = sin(x)*x      定义用户函数（所有模式可用）",
  "k = 2.5              定义常量；参数滑块会自动写入同名变量",
  "integrate(f, 0, 1)   数值积分    quad(f, 0, 1) 同义",
  "diff(f, x0)          某点数值导数    derivative(f) → 可求值的导函数",
  "limit(f, p)          极限    minimize(f, a, b)    fzero(f, x0 或 [a,b])",
  "grad(f, x, y) / div(F) / curl(F)   数值微分算子",
  "solve(A, b) 解线性方程组；roots(f, seed) 找复根；矩阵 [1,2;3,4]、A*v、A\\b、inv、det、eye",
  "sum(k, 1, 10, k^2)   求和    seq(k, 0, 20, k/10)   数列",
  "复数：z = 3+4i、arg、re、im、abs；复平面模式自变量写作 z",
  "注意 i 恒为虚数单位，不能作为循环变量（用 k、n）。",
];

function ConsolePanel() {
  const lines = useStore((s) => s.console.lines);
  const [input, setInput] = useState("");
  const push = (l: ConsoleLine) =>
    useStore.getState().patch({ console: { lines: [...useStore.getState().console.lines, l].slice(-200), input: "" } });
  const run = (raw: string) => {
    const src = raw.trim();
    if (!src) return;
    const eng = useStore.getState().engine;
    if (src === "clear") {
      useStore.getState().patch({ console: { lines: [], input: "" } });
      return;
    }
    if (src === "help" || src === "?") {
      push({ src, out: HELP.join("\n"), ok: true });
      return;
    }
    if (src === "vars") {
      const g = [...eng.globals.entries()].map(([k, v]) => `${k} = ${show(v)}`);
      const f = [...eng.fns.keys()];
      push({ src, out: `变量: ${g.join(", ") || "（无）"}\n函数: ${f.join(", ") || "（无）"}`, ok: true });
      return;
    }
    try {
      const d = eng.define(src);
      if (d.kind === "fn") push({ src, out: `已定义 ${d.name}(${d.params.join(", ")})`, ok: true });
      else if (d.kind === "let") {
        const v = eng.globals.get(d.name);
        push({ src, out: `${d.name} = ${v ? show(v) : "?"}`, ok: true });
      } else push({ src, out: show(evalString(eng, src)), ok: true });
    } catch (e) {
      push({ src, out: e instanceof Error ? e.message : String(e), ok: false });
    }
    clearSurfCache();
    useStore.getState().bump();
  };
  return (
    <div className="gl-panel" style={{ height: "100%" }}>
      <div className="gl-console">
        <div className="gl-console-out">
          {lines.length === 0 && <span style={{ opacity: 0.6 }}>输入表达式或定义，回车执行。试试 f(x) = sin(x)*exp(-x/6)，然后回到函数模式画 f(x)。</span>}
          {lines.map((l, i) => (
            <div className="gl-console-line" key={i}>
              <span className="in">›</span>
              <span className="out" style={{ whiteSpace: "pre-wrap" }}>
                {l.src}
                {"\n"}
                {l.out}
              </span>
            </div>
          ))}
        </div>
        <Input
          size="small"
          style={exprStyle}
          value={input}
          placeholder="help · vars · clear · f(x) = x^2*sin(x)"
          onChange={(e) => setInput(e.target.value)}
          onPressEnter={() => {
            run(input);
            setInput("");
          }}
        />
        <Row>
          <Button size="small" type="primary" className="gl-grow" onClick={() => { run(input); setInput(""); }}>
            执行
          </Button>
          <Button size="small" onClick={() => setInput("help")}>
            help
          </Button>
        </Row>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------ 导出 */

export default function SidePanel() {
  const mode = useStore((s) => s.mode);
  switch (mode) {
    case "func":
      return <FuncPanel />;
    case "geom":
      return <GeoPanel />;
    case "complex":
      return <ComplexPanel />;
    case "vector":
      return <VectorPanel />;
    case "lin":
      return <LinPanel />;
    case "nn":
      return <NnPanel />;
    case "surf":
      return <SurfPanel />;
    default:
      return <ConsolePanel />;
  }
}
