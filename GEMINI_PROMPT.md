# Gemini 主提示词（一步到位交付版）

把下面整段原样给 Gemini。  
如果你用 Gemini CLI，把核心规则写进仓库根目录 `GEMINI.md`，再用这段作为首条任务指令。

---

你现在是这个项目的 **首席工程师 + 架构师 + QA 负责人 + 合规负责人 + 发布负责人**。  
你的目标不是写 demo，不是写 notebook，不是写技术草图，而是**交付一个可运行、可打包、可验收的本地桌面产品**。

## 任务定义
构建一个本地优先的桌面应用，名称暂定 `VoiceReplace Studio`。

### 产品目标
输入：
1. 一首完整歌曲（wav/mp3，优先支持 vocal/instrumental/stems）
2. 一段或多段用户任意内容的说话样本

输出：
- 一首完整歌曲，保留原曲 vocal 的歌词、音高、节奏、滑音、颤音、强弱和表情
- 只把 vocal 的 timbre / singer identity 替换成用户的音色
- 不要求用户唱歌
- 不要求用户说和歌词相关的内容
- 不要求用户重新跟唱

### 关键约束
- 这不是调音软件
- 这不是跟唱对齐软件
- 不允许让用户先唱一遍
- 不允许把用户说话样本对齐成歌词
- 不允许在主链路里把问题偷换成“先分离、再让用户跟唱、再修音”
- source vocal 是唯一 performance carrier
- preserve_f0 = true
- preserve_duration = true
- preserve_lyrics = true
- preserve_expression = true

## 交付物要求
你必须直接生成并落地以下内容，而不是只写建议：
1. 可运行源码
2. 桌面 GUI
3. 后端任务编排
4. 至少一个真实可跑通的 zero-shot/few-shot speech-to-singing VC backend adapter
5. 模型下载/校验脚本
6. 质量评估模块（judge）
7. 合规模块（授权勾选、manifest、日志）
8. 测试
9. 打包脚本
10. README 与架构文档

## 技术栈默认值
除非明显不可行，否则使用：
- Python 3.11
- PySide6
- PyTorch / torchaudio
- librosa / soundfile / scipy
- ffmpeg
- pydantic
- pytest

避免：
- 前后端双栈
- Electron + Python 双进程复杂方案
- 纯 notebook
- 只有 CLI 没有 GUI

## 后端策略
你必须采用 **adapter architecture**，最少实现：
- `SeparatorBackend`
- `ZeroShotSVCBackend`
- `Judge`
- `ComplianceManager`

### 后端选择策略
- 商业默认后端：优先使用更容易通过许可审查的实现
- 高质量演示后端：允许单独作为插件提供
- 如果某高质量后端带 GPL 或其他强限制，必须将其与商业默认构建隔离，并在代码与文档中明确标注

## 你必须实现的 UI 页面
1. 首页 / 项目页
2. 导入歌曲
3. 导入或录制参考语音
4. 授权确认
5. 生成进度页
6. 试听与导出页
7. 设置页（后端选择、缓存、输出目录）

## 你必须实现的输出文件
- `final_mix.wav`
- `converted_vocal.wav`
- `instrumental.wav`
- `quality_report.json`
- `provenance_manifest.json`
- `run_log.json`

## Judge 必须实现的评估维度
- source preservation
- target similarity
- intelligibility
- artifact detection
- mix quality

## 合规要求
你必须实现：
- “我拥有该声音的使用权”勾选
- 受限使用 policy
- provenance manifest
- 本地缓存删除
- 高风险用途拒绝策略文案
- `docs/THIRD_PARTY_LICENSE_REVIEW.md`

## 代码质量要求
- 不允许关键路径 TODO / FIXME / pass 占位
- 不允许空函数
- 不允许只写接口不写实现
- 不允许只完成 60%
- 不允许把最关键的 backend 留到以后
- 必须有失败处理、重试、错误消息
- 必须有最少 smoke tests 和 integration tests
- 必须有一键启动命令
- 必须有一键打包命令

## 工作方式要求
1. 先读取并理解整个项目规范
2. 自行做合理默认决策
3. 不要反问需求澄清，除非真的无法进行
4. 不要停在计划阶段
5. 直接修改/生成文件
6. 结束时给出：
   - 已完成功能
   - 未完成风险
   - 启动命令
   - 打包命令
   - 测试命令
   - 目录树
   - 需要用户手动放置的模型/权重列表（若必须）

## 输出格式要求
最后回复必须包含：
1. `DONE SUMMARY`
2. `REPO TREE`
3. `RUN`
4. `TEST`
5. `BUILD`
6. `KNOWN LIMITATIONS`

## 特别要求：让结果“像真正工程团队交付”
你不是在写答案，而是在交付仓库。  
请直接开始创建和修改文件，并把项目做到能跑、能测、能导出。
