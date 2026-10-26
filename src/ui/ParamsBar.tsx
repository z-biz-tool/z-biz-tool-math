/**
 * 底部参数条：几何画板式的数值滑块。动画由 App 里唯一的 rAF 循环推进，
 * 这里只负责编辑参数本身（改名、调范围、播放/暂停、删除）。
 */
import { useEffect, useState } from "react";
import { App as AntdApp, Button, Input, InputNumber, Slider, Tooltip } from "antd";
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
  const { message, modal } = AntdApp.useApp();
  const setParam = useStore((s) => s.setParam);
  const removeParam = useStore((s) => s.removeParam);
  const toggleAnimate = useStore((s) => s.toggleAnimate);
  /**
   * 名字先落在草稿里、失焦或回车才提交：
   * 直接受控改名时，删到空串就会被拒绝并弹回上一个字，等于没法重打一个名字。
   */
  const [draft, setDraft] = useState(p.name);
  useEffect(() => setDraft(p.name), [p.name]);

  const commitName = () => {
    const name = draft.trim();
    if (name === p.name) {
      setDraft(p.name);
      return;
    }
    if (!VALID.test(name) || RESERVED.has(name)) {
      message.warning(`「${name || "空"}」不能作参数名：需字母开头，且不能占用 i、pi 等保留名`);
      setDraft(p.name);
      return;
    }
    const st = useStore.getState();
    if (st.params.some((q) => q.name === name)) {
      message.warning(`参数 ${name} 已存在`);
      setDraft(p.name);
      return;
    }
    st.engine.remove(p.name);
    st.engine.setNum(name, p.value);
    st.patch({ params: st.params.map((q) => (q.name === p.name ? { ...q, name } : q)), revision: st.revision + 1 });
    setDraft(name);
    message.success(`已把参数 ${p.name} 改名为 ${name}；表达式里写 ${p.name} 的地方要手动改成 ${name}`);
  };

  const askRemove = () =>
    modal.confirm({
      title: `删除参数「${p.name}」？`,
      content: `引用 ${p.name} 的表达式会立刻变成未定义并在画布上报错。参数可以重新添加，但当前值不会保留。`,
      okText: "删除",
      cancelText: "取消",
      okButtonProps: { danger: true },
      focusable: { autoFocusButton: "cancel" },
      onOk: () => removeParam(p.name),
    });

  return (
    <div className="gl-param">
      <Tooltip title="参数名，表达式里直接写这个名字；i 已被虚数单位占用">
        <Input
          size="small"
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onBlur={commitName}
          onPressEnter={(e) => (e.target as HTMLInputElement).blur()}
          aria-label={`参数 ${p.name} 的名称`}
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
        aria-label={`参数 ${p.name} 的值`}
        onChange={(v) => {
          if (typeof v === "number" && Number.isFinite(v)) setParam(p.name, v);
        }}
      />
      <Tooltip title="动画：在当前范围内往复扫描">
        <Button
          size="small"
          type={p.animate ? "primary" : "text"}
          icon={p.animate ? <PauseOutlined /> : <PlayCircleOutlined />}
          aria-label={p.animate ? `暂停参数 ${p.name} 的动画` : `播放参数 ${p.name} 的动画`}
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
          aria-label={`参数 ${p.name} 的动画速度`}
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
        <Button
          size="small"
          type="text"
          danger
          icon={<DeleteOutlined />}
          aria-label={`删除参数 ${p.name}`}
          onClick={askRemove}
        />
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
