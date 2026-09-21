/**
 * 左侧面板：按工作模式提供各自的编辑界面。
 * 所有写回都走 useStore.getState() 的动作，组件只订阅自己渲染需要的切片。
 */
import { useState } from "react";
import { Button, Input, InputNumber, Segmented, Select, Slider, Switch, Tooltip } from "antd";
import {
  DeleteOutlined,
  EyeInvisibleOutlined,
  EyeOutlined,
  PlusOutlined,
  RedoOutlined,
  UndoOutlined,
} from "@ant-design/icons";
import { COLORMAPS } from "../core/colormap.ts";
import { GeometryDoc } from "../core/geometry.ts";
import { evalString, show } from "../core/machine.ts";
import { clearSurfCache } from "../render/scene3d.ts";
import { kindZh, toolZh } from "../render/scene2d.ts";
import { useStore, uid } from "../state.ts";
import type {
  ConsoleLine,
  Layer,
  LayerKind,
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
      value={/^#[0-9a-f]{6}$/i.test(value) ? value : "#8b7cf6"}
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
        <Toggle label="π 刻度（x）" on={settings.piTicksX} onChange={(v) => setSettings({ piTicksX: v })} />
        <Toggle label="π 刻度（y）" on={settings.piTicksY} onChange={(v) => setSettings({ piTicksY: v })} />
        <Toggle label="次级网格" on={settings.showMinorGrid} onChange={(v) => setSettings({ showMinorGrid: v })} />
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
              <span style={{ fontSize: 14 }}>{t === "select" ? "↖" : t === "erase" ? "✕" : "·"}</span>
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
                    color: "#8b7cf6",
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
    case "surf":
      return <SurfPanel />;
    default:
      return <ConsolePanel />;
  }
}
