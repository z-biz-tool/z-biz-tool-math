/**
 * GeoLab 外壳：顶栏切模式与载入示例，左侧模式面板，中间画布，底部参数条。
 * 参数动画集中在这里的单个 rAF 循环，避免每个滑块各自驱动重绘。
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { Button, ConfigProvider, Modal, Popover, Segmented, Select, Switch, Tooltip, theme as antdTheme } from "antd";
import zhCN from "antd/locale/zh_CN";
import {
  DownloadOutlined,
  EnvironmentOutlined,
  FileImageOutlined,
  FunctionOutlined,
  GlobalOutlined,
  LineChartOutlined,
  ClusterOutlined,
  SettingOutlined,
  TableOutlined,
  ThunderboltOutlined,
  UploadOutlined,
} from "@ant-design/icons";
import { PRESETS, applyPreset } from "./presets.ts";
import { useStore, type Mode } from "./state.ts";
import Canvas2D from "./ui/Canvas2D.tsx";
import Canvas3D from "./ui/Canvas3D.tsx";
import ParamsBar from "./ui/ParamsBar.tsx";
import SidePanel from "./ui/panels.tsx";
import { applyProject, exportPng, saveProject, serializeProject } from "./ui/exporters.ts";

const MODES: { value: Mode; label: string; icon: React.ReactNode }[] = [
  { value: "func", label: "函数图像", icon: <FunctionOutlined /> },
  { value: "geom", label: "动态几何", icon: <EnvironmentOutlined /> },
  { value: "complex", label: "复平面", icon: <GlobalOutlined /> },
  { value: "vector", label: "向量与场", icon: <LineChartOutlined /> },
  { value: "lin", label: "矩阵与线性", icon: <TableOutlined /> },
  { value: "nn", label: "神经网络", icon: <ClusterOutlined /> },
  { value: "surf", label: "三维曲面", icon: <ThunderboltOutlined /> },
  { value: "console", label: "控制台", icon: <SettingOutlined /> },
];

/* 会话快照：复用工程 JSON 的格式与校验（Project.v 已是版本标记），不另起一套形状 */
const SESSION_KEY = "z-biz-tool-math-session";
/** StrictMode 下挂载跑两遍，回位只能发生一次 */
let sessionRestored = false;

const PRESET_GROUPS: { label: string; options: { value: string; label: string }[] }[] = (
  ["func", "geom", "complex", "vector", "lin", "nn", "surf"] as Mode[]
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
  /** 已读出但尚未被确认覆盖的工程：整个工作台会被替换，不能选完文件就默默动手 */
  const [pending, setPending] = useState<{ name: string; text: string } | null>(null);
  const say = (m: string) => {
    setHint(m);
    window.setTimeout(() => setHint((q) => (q === m ? "" : q)), 4000);
  };

  const doExportPng = () => say(exportPng() ? "已导出 PNG" : "画布不可见，无法导出");

  const openProject = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    try {
      /** 先只读不套：结构不合法就没必要再问一次「要不要覆盖」 */
      setPending({ name: file.name, text: await file.text() });
    } catch (err) {
      say(`读取失败：${err instanceof Error ? err.message : String(err)}`);
    }
  };

  const confirmProject = () => {
    const p = pending;
    setPending(null);
    if (!p) return;
    try {
      applyProject(p.text);
      say(`已载入工程 ${p.name}`);
    } catch (err) {
      say(`载入失败：${err instanceof Error ? err.message : String(err)}`);
    }
  };

  /** 会话自动落盘：关窗前没手动「保存工程」的图层、表达式、几何构造不该被 reload 抹掉 */
  useEffect(() => {
    let timer = 0;
    const unsubscribe = useStore.subscribe((s, prev) => {
      if (typeof window === "undefined" || s.revision === prev.revision) return;
      clearTimeout(timer);
      timer = window.setTimeout(() => {
        try {
          localStorage.setItem(SESSION_KEY, serializeProject());
        } catch {
          /* 隐私模式或配额满：下次开不了窗，但也不能把当前的画布弄崩 */
        }
      }, 1200);
    });
    return () => {
      clearTimeout(timer);
      unsubscribe();
    };
  }, []);

  /** 冷启动回位：老工程结构不对时静默放弃，让默认示例照常打开 */
  useEffect(() => {
    if (sessionRestored || typeof window === "undefined") return;
    sessionRestored = true;
    let text: string | null = null;
    try {
      text = localStorage.getItem(SESSION_KEY);
    } catch {
      return;
    }
    if (!text) return;
    try {
      /** 会话只是画布留底，主题跟着全家的 z-tool-theme 走 */
      applyProject(text, { keepDark: true });
      say("已恢复上次会话");
    } catch {
      try {
        localStorage.removeItem(SESSION_KEY);
      } catch {
        /* 清不掉就算了，下一次启动仍会走静默放弃这条路 */
      }
    }
  }, []);

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
      const st = useStore.getState();
      const t = e.target as HTMLElement | null;
      const inField = !!t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
      const mod = e.metaKey || e.ctrlKey;
      /** 字段里只放行「动作键」：⌘Z 交给浏览器做文本撤销，退格得能删字符，Esc 得能收自动补全 */
      if (mod) {
        const k = e.key.toLowerCase();
        const digit = "12345678".indexOf(e.key);
        if (digit >= 0 && !e.shiftKey && !e.altKey) {
          e.preventDefault();
          setMode(MODES[digit].value);
          return;
        }
        if (k === "z") {
          /** 只有几何文档有撤销栈：函数模式下按 ⌘Z 悄悄改几何图会让人找不到东西去哪了 */
          if (inField || st.mode !== "geom") return;
          e.preventDefault();
          if (e.shiftKey ? st.geo.doc.redo() : st.geo.doc.undo()) st.bump();
          return;
        }
        if (k === "s") {
          e.preventDefault();
          say(`已保存 ${saveProject()}`);
          return;
        }
        if (k === "e") {
          e.preventDefault();
          doExportPng();
          return;
        }
        return;
      }
      if (inField) return;
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
        token: { colorPrimary: "#667eea", borderRadius: 8, fontSize: 13 },
      }}
    >
      <div className="gl-root" data-dark={dark}>
        <div className="gl-header">
          <div className="gl-logo">
            <span className="gl-logo-mark">∮</span>
            <span>
              <span className="gl-logo-title">GeoLab 数学实验室</span>
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
          <Tooltip title="导出当前画布为 PNG（⌘/Ctrl+E）">
            <Button
              size="small"
              icon={<FileImageOutlined />}
              disabled={mode === "console"}
              onClick={doExportPng}
            />
          </Tooltip>
          <Tooltip title="保存工程（JSON，⌘/Ctrl+S）">
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
                <Button
                  size="small"
                  style={{ marginTop: 8 }}
                  onClick={() => {
                    try {
                      localStorage.removeItem(SESSION_KEY);
                    } catch {
                      /* 读不到的时候也谈不上清除 */
                    }
                    say("已清除上次会话，下次打开回到默认示例");
                  }}
                >
                  清除已存会话
                </Button>
                <div style={{ fontSize: 11.5, opacity: 0.66, lineHeight: 1.7, marginTop: 6 }}>
                  ⌘1…⌘8 切模式，⌘S 保存工程，⌘E 导出 PNG
                  <br />
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
        <Modal
          open={pending !== null}
          title="覆盖当前工作台？"
          okText="覆盖"
          cancelText="取消"
          okButtonProps={{ danger: true }}
          onCancel={() => setPending(null)}
          onOk={confirmProject}
        >
          <p style={{ margin: 0 }}>
            载入「{pending?.name}」会替换当前八种模式里的全部图层、参数、几何构造与视口，且这一步无法撤销。
            想保留现在这份，先按 ⌘/Ctrl+S 保存工程。
          </p>
        </Modal>
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
