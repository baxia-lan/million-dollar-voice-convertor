# Claude Code 主提示词（一步到位交付版）

把下面整段给 Claude Code；同时把根目录 `CLAUDE.md` 放进去。

---

你要把这个仓库做成一个**真正能运行的本地桌面产品**，不是 demo，不是架构建议，不是半成品。你必须直接改文件、跑测试、补文档、补脚本。

## 产品定义
这个产品是 **zero-shot / few-shot speech-to-singing voice conversion**：

输入：
- 一首完整歌曲
- 用户任意内容的说话样本

输出：
- 保留原曲 vocal 的歌词、音高、节奏、滑音、颤音、强弱与表情
- 只替换 vocal timbre 为用户音色
- 不要求用户唱歌
- 不要求用户跟唱
- 不允许把任务改写成修音或对齐软件

## 非谈判约束
- source vocal 是唯一 performance carrier
- preserve_f0 = true
- preserve_duration = true
- preserve_lyrics = true
- preserve_expression = true
- 不允许直接对 full mix 做 voice conversion
- 不允许关键路径留空实现
- 不允许只交 CLI
- 必须有 GUI
- 必须有合规与 provenance
- 必须有测试
- 必须有打包脚本

## 你必须落地的模块
- GUI（PySide6）
- 音频导入与预处理
- vocal/instrumental 分离适配
- zero-shot/few-shot speech-reference SVC backend adapter
- judge
- 合规与授权
- 导出
- 日志
- 测试
- 文档

## 你要优先做的工程决策
- 纯 Python 单仓库
- adapter 架构
- 可替换后端
- 商业默认后端与高质量演示后端隔离
- README 清晰到新机器可复现

## 结束时必须给我
- 功能完成清单
- 运行命令
- 测试命令
- 打包命令
- 风险与限制
- 目录树
