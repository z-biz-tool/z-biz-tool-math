# Changelog

z-biz-tool-math 版本变更记录。格式遵循 [Conventional Commits](https://www.conventionalcommits.org/)。

## [0.2.1] - 2026-10-29

更新通道修复版（无功能变更）。

### Fixes

- 更新通道 CI 两处修复：片段目录被 `upload-artifact@v4` 默认隐藏跳过（latest.json 从未成功推出去）；latest.json 下载链接因双 macOS 包被 tauri-action 重命名而 404 —— 现回查 release 资产按「扩展名+架构词」取真实名，并远端/本地签名逐字节比对

## [0.2.0] - 2026-10-25

自 v0.1.0（2026-09-24）以来共 16 个提交：1 个 feat、3 个 fix。特性级测试网从 89 条补齐到 286 条，写测试的过程查出「两套 det 行为不一致」并记账；主题深浅色落盘统一。注意：v0.1.0 之后曾长期未产出安装包（发布资产断档），本版恢复四平台产物；**2026-10-25 裁定正式发布**（免费个人版 + 机构授权）。

### Features

- 深浅色主题落盘共用 z-tool-theme 键，订阅里双写路径统一

### Fixes

- 工作台崩一次不再丢掉整轮手活；自动回位不再顺手改主题
- 记录「两套 det、行为不一致」缺陷的正确修法边界（椭圆求交）

### Tests

- 几何 / 状态机纯逻辑回归 89 条 → cnum/linalg 地基 135 条 → 特性级测试网 286 条

### Chore / CI / Docs

- MIT 许可证统一；换家族矢量图标源；图标挂进 build script 重跑指纹
- `tests/*.test.ts` 接进流水线；解除 npmmirror 地域锁定
- 补 Rust 侧 fmt 静态门；首份优化方案 + 05 落地率回填（20/21）
