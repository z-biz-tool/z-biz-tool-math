/**
 * 底部参数条：几何画板式的数值滑块。动画由 App 里唯一的 rAF 循环推进，
 * 这里只负责编辑参数本身（改名、调范围、播放/暂停、删除）。
 */
import { Button, Input, InputNumber, Slider, Tooltip } from "antd";
import { DeleteOutlined, PauseOutlined, PlayCircleOutlined, PlusOutlined } from "@ant-design/icons";
import { useStore, type Param } from "../state.ts";

/** i 永远是虚数单位，不能当参数名 */
const RESERVED = new Set(["i", "pi", "tau", "inf", "NaN", "true", "false"]);
const VALID = /^[a-zA-Z_][a-zA-Z_0-9]*$/;
const LETTERS = "abcdefghijklmnopqrstuvw";

function freeName(taken: Set<string>): string {
  for (const c of LETTERS) if (!taken.has(c) && !RESERVED.has(c)) return c;
  let n = 1;
  while (taken.has(`v${n}`) || RESERVED.has(`v${n}`)) n++;
  return `v${n}`;
}

function rd(v: number): number {
  return Number(v.toFixed(4));
}

function ParamChip({ p }: { p: Param }) {
  const setParam = useStore((s) => s.setParam);
  const removeParam = useStore((s) => s.removeParam);
  const toggleAnimate = useStore((s) => s.toggleAnimate);
  const rename = (raw: string) => {
    const name = raw.trim();
    if (!VALID.test(name) || RESERVED.has(name)) return;
    const st = useStore.getState();
    if (name === p.name || st.params.some((q) => q.name === name)) return;
    st.engine.remove(p.name);
    st.engine.setNum(name, p.value);
    st.patch({ params: st.params.map((q) => (q.name === p.name ? { ...q, name } : q)), revision: st.revision + 1 });
  };
  return (
    <div className="gl-param">
      <Tooltip title="参数名，表达式里直接写这个名字；i 已被虚数单位占用">
        <Input
          size="small"
          value={p.name}
          onChange={(e) => rename(e.target.value)}
          style={{ width: 44 }}
          className="gl-param-label"
        />
      </Tooltip>
      <Slider
        style={{ width: 150 }}
        min={p.min}
        max={p.max}
        step={p.step}
        value={p.value}
        onChange={(v) => setParam(p.name, v)}
        tooltip={{ open: false }}
      />
      <InputNumber
        size="small"
        value={rd(p.value)}
        step={p.step}
        style={{ width: 70 }}
        onChange={(v) => {
          if (typeof v === "number" && Number.isFinite(v)) setParam(p.name, v);
        }}
      />
      <Tooltip title="动画：在当前范围内往复扫描">
        <Button
          size="small"
          type={p.animate ? "primary" : "text"}
          icon={p.animate ? <PauseOutlined /> : <PlayCircleOutlined />}
          onClick={() => toggleAnimate(p.name)}
        />
      </Tooltip>
      {p.animate && (
        <InputNumber
          size="small"
          min={0.05}
          max={8}
          step={0.05}
          value={p.speed}
          style={{ width: 62 }}
          onChange={(v) => {
            if (typeof v === "number" && Number.isFinite(v)) {
              const st = useStore.getState();
              st.patch({
                params: st.params.map((q) => (q.name === p.name ? { ...q, speed: v } : q)),
              });
            }
          }}
        />
      )}
      <Tooltip title="删除参数">
        <Button size="small" type="text" danger icon={<DeleteOutlined />} onClick={() => removeParam(p.name)} />
      </Tooltip>
    </div>
  );
}

export default function ParamsBar() {
  const params = useStore((s) => s.params);
  const add = () => {
    const st = useStore.getState();
    const name = freeName(new Set(st.params.map((q) => q.name)));
    const p: Param = { name, value: 0, min: -5, max: 5, step: 0.01, animate: false, speed: 1 };
    st.engine.setNum(name, 0);
    st.addParam(p);
  };
  return (
    <div className="gl-params">
      <span className="gl-card-title" style={{ flex: "none" }}>
        参数
      </span>
      {params.map((p) => (
        <ParamChip key={p.name} p={p} />
      ))}
      <Button size="small" icon={<PlusOutlined />} onClick={add}>
        参数
      </Button>
      {params.length === 0 && (
        <span style={{ fontSize: 11.5, opacity: 0.6 }}>
          添加参数后，在函数表达式里直接写 a、b、c… 即可拖动或动画
        </span>
      )}
    </div>
  );
}
