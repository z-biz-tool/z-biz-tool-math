/**
 * GeoLab 外壳：顶栏切模式与载入示例，左侧模式面板，中间画布，底部参数条。
 * 参数动画集中在这里的单个 rAF 循环，避免每个滑块各自驱动重绘。
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { Button, ConfigProvider, Popover, Segmented, Select, Switch, Tooltip, theme as antdTheme } from "antd";
import zhCN from "antd/locale/zh_CN";
import {
  DownloadOutlined,
  EnvironmentOutlined,
  FileImageOutlined,
  FunctionOutlined,
  GlobalOutlined,
  LineChartOutlined,
  SettingOutlined,
  ThunderboltOutlined,
  UploadOutlined,
} from "@ant-design/icons";
import { PRESETS, applyPreset } from "./presets.ts";
import { useStore, type Mode } from "./state.ts";
import Canvas2D from "./ui/Canvas2D.tsx";
import Canvas3D from "./ui/Canvas3D.tsx";
import ParamsBar from "./ui/ParamsBar.tsx";
import SidePanel from "./ui/panels.tsx";
import { applyProject, exportPng, saveProject } from "./ui/exporters.ts";

const MODES: { value: Mode; label: string; icon: React.ReactNode }[] = [
  { value: "func", label: "函数图像", icon: <FunctionOutlined /> },
  { value: "geom", label: "动态几何", icon: <EnvironmentOutlined /> },
  { value: "complex", label: "复平面", icon: <GlobalOutlined /> },
  { value: "vector", label: "向量与场", icon: <LineChartOutlined /> },
  { value: "surf", label: "三维曲面", icon: <ThunderboltOutlined /> },
  { value: "console", label: "控制台", icon: <SettingOutlined /> },
];

const PRESET_GROUPS: { label: string; options: { value: string; label: string }[] }[] = (
  ["func", "geom", "complex", "vector", "surf"] as Mode[]
).map((m) => ({
  label: MODES.find((q) => q.value === m)?.label ?? m,
  options: PRESETS.filter((p) => p.mode === m).map((p) => ({ value: p.key, label: `${p.title}｜${p.hint}` })),
}));

export default function App() {
  const mode = useStore((s) => s.mode);
  const dark = useStore((s) => s.settings.dark);
  const setMode = useStore((s) => s.setMode);
  const settings = useStore((s) => s.settings);
  const setSettings = useStore((s) => s.setSettings);
  const [preset, setPreset] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);
  /** 顶栏右侧的轻量回执：导出/载入这类一次性动作不值得弹窗打断 */
  const [hint, setHint] = useState("");
  const say = (m: string) => {
    setHint(m);
    window.setTimeout(() => setHint((q) => (q === m ? "" : q)), 4000);
  };

  const openProject = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    try {
      applyProject(await file.text());
      say(`已载入工程 ${file.name}`);
    } catch (err) {
      say(`载入失败：${err instanceof Error ? err.message : String(err)}`);
    }
  };

  /** 把参数初值写进引擎，之后 setParam 会持续覆盖 */
  useEffect(() => {
    const st = useStore.getState();
    for (const p of st.params) st.engine.setNum(p.name, p.value);
  }, []);

  /** 参数动画：正弦往复，起始相位对齐当前值，按下播放不会跳变 */
  const animating = useStore((s) => s.params.some((p) => p.animate));
  useEffect(() => {
    if (!animating) return;
    const phases = new Map<string, number>();
    let raf = 0;
    let last = performance.now();
    const tick = (now: number) => {
      const dt = Math.min(0.06, (now - last) / 1000);
      last = now;
      const st = useStore.getState();
      const params = st.params.map((p) => {
        if (!p.animate) {
          phases.delete(p.name);
          return p;
        }
        const mid = (p.min + p.max) / 2;
        const amp = (p.max - p.min) / 2;
        let ph = phases.get(p.name);
        if (ph === undefined) {
          const r = amp === 0 ? 0 : Math.max(-1, Math.min(1, (p.value - mid) / amp));
          ph = Math.asin(r);
        }
        const next = ph + dt * Math.PI * 0.5 * Math.max(0.01, p.speed);
        phases.set(p.name, next);
        const v = mid + amp * Math.sin(next);
        st.engine.setNum(p.name, v);
        return { ...p, value: v };
      });
      st.patch({ params, revision: st.revision + 1 });
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [animating]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable)) return;
      const st = useStore.getState();
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "z") {
        e.preventDefault();
        if (e.shiftKey ? st.geo.doc.redo() : st.geo.doc.undo()) st.bump();
        return;
      }
      if (e.key === "Escape") {
        if (st.geo.pending.length) st.setGeo({ pending: [] });
        return;
      }
      if ((e.key === "Delete" || e.key === "Backspace") && st.mode === "geom" && st.geo.selected) {
        st.geo.doc.remove(st.geo.selected);
        st.setGeo({ selected: null, pending: [] });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const stage = useMemo(() => {
    if (mode === "surf") return <Canvas3D />;
    if (mode === "console")
      return (
        <div style={{ padding: "12px 14px", height: "100%", display: "flex", flexDirection: "column" }}>
          <SidePanel />
        </div>
      );
    return <Canvas2D />;
  }, [mode]);

  return (
    <ConfigProvider
      locale={zhCN}
      theme={{
        algorithm: dark ? antdTheme.darkAlgorithm : antdTheme.defaultAlgorithm,
        token: { colorPrimary: "#8b7cf6", borderRadius: 8, fontSize: 13 },
      }}
    >
      <div className="gl-root" data-dark={dark}>
        <div className="gl-header">
          <div className="gl-logo">
            <span className="gl-logo-mark">∮</span>
            <span>
              GeoLab 数学实验室
              <div className="gl-logo-sub">几何画板 · 复平面 · 场与曲面</div>
            </span>
          </div>
          <Segmented
            value={mode}
            onChange={(v) => setMode(v as Mode)}
            options={MODES.map((m) => ({
              value: m.value,
              label: (
                <span style={{ display: "inline-flex", alignItems: "center", gap: 5 }}>
                  {m.icon}
                  {m.label}
                </span>
              ),
            }))}
          />
          <Select
            size="small"
            showSearch
            allowClear
            placeholder="载入示例"
            style={{ width: 260 }}
            value={preset}
            options={PRESET_GROUPS}
            onChange={(v?: string) => {
              setPreset(v ?? null);
              if (v) applyPreset(v);
            }}
            filterOption={(q, o) => String((o as { label: string })?.label ?? "").toLowerCase().includes(q.toLowerCase())}
          />
          <span className="gl-spacer" />
          {hint && (
            <span style={{ fontSize: 11.5, opacity: 0.72, maxWidth: 230, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
              {hint}
            </span>
          )}
          <Tooltip title="导出当前画布为 PNG">
            <Button
              size="small"
              icon={<FileImageOutlined />}
              disabled={mode === "console"}
              onClick={() => say(exportPng() ? "已导出 PNG" : "画布不可见，无法导出")}
            />
          </Tooltip>
          <Tooltip title="保存工程（JSON）">
            <Button size="small" icon={<DownloadOutlined />} onClick={() => say(`已保存 ${saveProject()}`)} />
          </Tooltip>
          <Tooltip title="打开工程">
            <Button size="small" icon={<UploadOutlined />} onClick={() => fileRef.current?.click()} />
          </Tooltip>
          <input
            ref={fileRef}
            type="file"
            accept="application/json,.json"
            style={{ display: "none" }}
            onChange={(e) => void openProject(e)}
          />
          <Popover
            trigger="click"
            placement="bottomRight"
            title="显示"
            content={
              <div style={{ width: 190, display: "flex", flexDirection: "column", gap: 2 }}>
                <SwitchRow label="深色主题" on={settings.dark} onChange={(v) => setSettings({ dark: v })} />
                <SwitchRow label="次级网格" on={settings.showMinorGrid} onChange={(v) => setSettings({ showMinorGrid: v })} />
                <SwitchRow label="十字光标" on={settings.showCrosshair} onChange={(v) => setSettings({ showCrosshair: v })} />
                <SwitchRow label="π 刻度 x" on={settings.piTicksX} onChange={(v) => setSettings({ piTicksX: v })} />
                <SwitchRow label="π 刻度 y" on={settings.piTicksY} onChange={(v) => setSettings({ piTicksY: v })} />
                <SwitchRow label="栅格平滑" on={settings.antialias} onChange={(v) => setSettings({ antialias: v })} />
                <div style={{ fontSize: 11.5, opacity: 0.66, lineHeight: 1.7, marginTop: 6 }}>
                  几何模式：⌘Z 撤销 / ⇧⌘Z 重做，Delete 删除选中，Esc 取消构造队列
                </div>
              </div>
            }
          >
            <Tooltip title="显示设置">
              <Button size="small" icon={<SettingOutlined />} />
            </Tooltip>
          </Popover>
        </div>
        <div className="gl-body">
          {mode !== "console" && (
            <aside className="gl-sider">
              <SidePanel />
            </aside>
          )}
          <main className="gl-main">
            <div className="gl-stage-box">{stage}</div>
            <ParamsBar />
          </main>
        </div>
      </div>
    </ConfigProvider>
  );
}

function SwitchRow({ label, on, onChange }: { label: string; on: boolean; onChange: (v: boolean) => void }) {
  return (
    <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: 8 }}>
      <span style={{ fontSize: 12 }}>{label}</span>
      <Switch size="small" checked={on} onChange={onChange} />
    </div>
  );
}
