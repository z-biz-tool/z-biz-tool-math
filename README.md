# z-biz-tool-math · GeoLab 数学实验室

一个纯前端的数学可视化工作台：几何画板式的**动态几何** + **函数图像** + **复平面** + **向量场** + MATLAB 风格的**三维曲面**，共享同一个表达式内核。

无后端、无账号、可离线；渲染全部走 Canvas 2D，不依赖任何绘图库。

## 五种工作模式

| 模式 | 能画什么 |
| --- | --- |
| 函数图像 | 直角坐标、极坐标、参数方程、隐函数 `f(x,y)=c`、不等式区域、导数/积分曲线、数列散点；多图层、线宽/虚线/填充/采样数逐个可调；x/y 可各自切**对数轴**（semilogx / semilogy / loglog） |
| 动态几何 | 24 种工具（含选择与擦除）：点/线段/直线/射线/向量/圆/弧/椭圆/多边形、中点、交点、垂线、平行线、角平分线、轨迹（locus）、旋转/反射/位似、角度与面积测量；拖动自由点即时重算，快照式撤销重做 |
| 复平面 | Needham 共形着色（色相 = arg f，明度 = \|f\|）、保角映射网格、Newton 分形、迭代吸引盆；自变量写作 `z` |
| 向量与场 | 自由向量平行四边形法则、向量场 quiver、方向场 dy/dx、流线、相图（含平衡点分类），箭头按模长上色 |
| 三维曲面 | surf / mesh / wire / contour3 / surfc，参数曲面、隐式等值面（marching cubes）、旋转曲面、空间曲线；轨道相机、painter 排序、Gouraud 光照、14 种色图 |

顶栏「载入示例」内置 31 个预设，覆盖上述每一类；控制台模式可直接写 `f(x) = …`、`k = …` 定义用户函数与常量，全部模式共享。

顶栏右侧另有三个动作：**导出 PNG**（把舞台上的场景层与读数层按当前分辨率合成）、**保存工程**、**打开工程**（JSON，含全部图层、参数、五种模式的视口、三维相机与几何文档）。

## 表达式内核

`src/core/` 是一个手写的解析 + 求值内核（无第三方依赖）：

- 复数原生：`i` 恒为虚数单位（因此不能当循环变量），支持 `arg/conj/re/im/abs`
- 向量与矩阵：`[1,2,3]`、`[[1,2],[3,4]]`、`A*v` 按线性映射、`A\b`、`inv/det/eye/transpose`、标量广播加减
- 数值分析（**数值而非符号**）：`integrate(f,lo,hi)`、`diff(f,x0)`、`derivative(f)`、`limit(f,p)`、`minimize(f,a,b)`、`fzero`、`roots`、`grad/div/curl`、`sum(k,…)`、`seq(k,…)`
- 133 个内置函数名（`STRICT` 105 + 惰性求值的 `lazy` 28，含 `log`/`ln`、`re`/`real` 这类别名）；隐函数/等值面/等高线由 `contour.ts`（marching squares）与 `surface.ts`（marching cubes）提供
- 纯实数表达式走 `compileReal` 无分配快路径，与通用 (re,im) 双槽路径结果一致性有回归测试保障

## 工程结构

```
src/core/      解析器、求值机、复数/场/曲面几何、动态几何文档、视口、色图
src/render/    plot2d（画布原语）、scene2d（四种 2D 模式）、scene3d（带缓存的三维场景）
src/ui/        Canvas2D（底层场景 + 顶层读数双画布）、Canvas3D、panels（各模式面板）、ParamsBar、exporters（PNG 与工程 JSON）
src/state.ts   zustand store：模式、图层、参数、五种模式各自的视口
src/presets.ts 31 个示例预设
scripts/       verify.ts —— 183 项内核与视口回归断言
```

## 命令

```bash
npm install
npm run dev        # http://localhost:5199
npm run verify     # 内核回归验证（183 项）
npm run typecheck  # tsc --noEmit（strict + noUnusedLocals）
npm run build
```

## 交互速查

- 滚轮以光标为中心缩放，拖动平移；三维模式拖动转视角、Shift+拖动平移、滚轮推拉
- 几何模式：⌘/Ctrl+Z 撤销，⇧⌘/Ctrl+Z 重做，Delete 删除选中，Esc 清空构造队列
- 底部参数条的滑块即全局变量，表达式里直接写 `a`、`b`…；点播放键做正弦往复动画
- 函数模式的「坐标轴」卡片可把 x / y 各自切成对数轴：视口在对数轴上按**十倍频程**记，主刻度落在 10 的整数幂，次级网格按 2/5（放大到不足两个十倍频时展开为 2…9）铺开，采样与等值线也按等倍率前进。对数轴上没有 0 与负值，切换后该轴中心回到 1，π 刻度让位
- 工程 JSON 是纯文本快照，可版本管理；打开工程会整体替换当前五种模式的画布状态（表达式内核仍是同一个）
