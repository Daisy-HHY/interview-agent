# Interview Agent

Interview Agent 是一个 VS Code 侧边栏 AI 技术面试官插件。打开任意目标项目后，它会结合岗位 JD、简历和当前工作区代码，围绕真实项目技术栈进行追问。

本插件只用于技术面试练习，不会自动修改、提交或发布你的代码。

## 功能

- 粘贴岗位 JD，上传 `.pdf` / `.docx` / `.txt` / `.md` / 图片简历
- 自动读取当前 VS Code 工作区项目结构
- Demo Mode 零配置体验完整面试流程
- OpenAI 兼容模型调用，支持自定义 `interview.baseUrl` 和 `interview.model`
- 设置页一键“测试模型连接”
- 历史会话可继续、删除，并自动生成可读标题
- 对话结束后可导出 Markdown 面试报告到 `.interview-agent/reports`
- 扫描版 PDF 和图片简历可选 OCR 识别，OCR 依赖按需安装

## 从 GitHub Release 安装

1. 到 GitHub Release 下载 `interview-agent-0.2.0.vsix`
2. VS Code 执行 `Extensions: Install from VSIX...`
3. 选择下载的 `.vsix`
4. 打开要准备面试的目标项目文件夹
5. 点击左侧 Activity Bar 的 `Interview Agent`

## 快速体验 Demo Mode

1. 打开任意项目文件夹
2. 进入 `Interview Agent`
3. 打开 VS Code 设置，搜索 `interview.demoMode`
4. 勾选 Demo Mode
5. 回到面板，粘贴岗位 JD
6. 可选上传简历或填写简历补充
7. 点击“开始面试”

Demo Mode 使用内置 FakeLLM，不需要 API Key，也不会调用真实模型。

## 真实模型配置

在 VS Code 设置中配置：

| 配置项 | 说明 |
|---|---|
| `interview.apiKey` | OpenAI 兼容服务的 API Key |
| `interview.baseUrl` | OpenAI 兼容端点。留空使用官方 OpenAI |
| `interview.model` | 模型名，例如 `gpt-4o-mini`、`deepseek-chat`、`glm-4-flash` |
| `interview.pythonPath` | Python 解释器路径，默认 `python` |
| `interview.demoMode` | 演示模式开关 |
| `interview.agentRuntime` | Agent 运行时。默认 `native`；可选 `langchain` 做 LangChain / LangGraph 运行时验证，也可选 `pi` 使用 Python pi-style 运行时 |
| `interview.piMaxSteps` | Pi runtime 的单轮最大模型回合数，默认 32；模型不再请求工具时立即结束，该值仅作为防止无限工具循环的安全阀 |
| `interview.enabledTools` | 启用的 Agent 工具名列表。默认启用 `list_directory`、`search_code`、`read_file`、`lookup_questions` |
| `interview.compactionEnabled` | Pi checkpoint 上下文压缩开关，默认关闭；失败时自动回退 |
| `interview.compactionTriggerTokens` | Pi checkpoint 压缩触发的估算 token 数 |
| `interview.compactionKeepMessages` | 压缩后保留的最近消息数量，系统提示始终保留 |

配置后先点击“测试模型连接”。模型不存在时请检查 `interview.model` 是否与 `interview.baseUrl` 对应的服务商匹配。

## Python 依赖

基础运行依赖只需要：

```powershell
python -m pip install openai
```

如果上传的是扫描版 PDF 或图片简历，并且需要 OCR，再安装可选依赖：

```powershell
python -m pip install PyMuPDF numpy rapidocr onnxruntime
```

普通文字层 PDF、DOCX、TXT、MD 简历不需要 OCR 依赖。

如需验证 LangChain runtime，再安装可选框架依赖：

```powershell
python -m pip install -e ".[agent-framework]"
```

0.2.0 仍默认使用 `native` runtime。0.2.1 继续保持这个默认值，并集中完善 `pi` 的底层边界。`pi` 是运行在 Python 子进程内的 Pi-style runtime，不是官方 Pi SDK，也不直接依赖 TypeScript 版 Pi 包。只有 runtime benchmark 在统一场景下持续达标后，才应考虑把默认值从 `native` 切换到 `pi` 或 `langchain`。

### Pi-style runtime 边界

项目只借鉴 Pi 的组织思想，不复制 Pi 的 TypeScript `agent_loop`、CLI、TUI 或完整扩展生态：

| Pi 思想 | 当前项目实现 | 状态 |
|---|---|---|
| 外置上下文 | `AgentContext` + `transform_context` / `convert_to_llm` | 已实现，作为项目自己的 context pipeline |
| 工具生命周期 | `PiToolExecutor` + before/after hook | 已实现，支持显式进程内注册 |
| 事件流 | Pi 内部事件 + `agent_event` 脱敏协议通知 | 0.2.2 增加限量 Webview 时间线 |
| 上下文压缩 | checkpoint + `compress_history` / `enforce_token_limit` | 0.2.2 可选、默认关闭、失败回退 |
| 会话持久化 | `.sessions/{id}.json` 线性消息数组 | 保持现有格式，session tree 后续规划 |
| MCP | 本地 `McpToolAdapter` 合同和 fake 隔离测试 | 仅接口设计，不接真实传输 |
| 外部 hook 文件 | `piHooksFile` 安全评估 | 0.2.2 默认不加载、不执行 |

Pi runtime 的外部边界仍是 [agent/runtime.py](agent/runtime.py) 中的 `AgentRuntime`；会话文件由 `SessionStore` 管理，runtime 不直接读写文件。

无 API Key 的底层回归可以使用 FakeLLM：

```powershell
.\.venv\Scripts\python.exe -m agent.runtime_benchmark --runtime pi --rounds 3 --fake-llm
```

工具注册已和 runtime 解耦：`native`、`langchain`、`pi` 都只接收最终的 `ToolRegistry`。当前 0.2.0 先支持基础工具的启用和禁用，后续可在同一入口接入更多工具 provider 或 MCP adapter。

0.2.2 的 FakeLLM 回归不需要 API Key 或网络：

```powershell
.\.venv\Scripts\python.exe -m agent.runtime_benchmark --runtime all --rounds 1 --fake-llm
```

0.2.3 的压缩对照评测需要真实模型时，使用同一模型分别运行 baseline/compaction，输出脱敏 JSONL 和聚合指标：

```powershell
.\.venv\Scripts\python.exe -m agent.runtime_benchmark --runtime pi --rounds 3 --compaction-mode compare --compaction-seed-messages 12 --output .tmp\pi-compaction.jsonl --summary-output .tmp\pi-compaction-summary.json
```

`compression_quality_status=manual_review_required` 表示质量仍需人工按评测样本检查；FakeLLM 只验证协议、上下文合法性和回退行为，不能替代真实模型质量结论。0.2.3 不默认启用 LLM summarizer，也不默认持久化 runtime trace。

checkpoint 压缩默认关闭，只在 `interview.compactionEnabled` 开启且达到阈值时影响本次模型请求上下文，不会隐式改写 `.sessions` 历史。运行中的脱敏事件会显示在诊断区域；事件不包含完整 prompt、工具参数值或工具结果。MCP adapter 目前只作为未来工具 provider 的隔离合同，未建立远程连接。

## 导出报告

完成一轮面试后，点击顶栏“导出报告”。报告会保存到当前工作区：

```text
.interview-agent/reports/
```

Markdown 报告包含 JD 摘要、项目摘要、考察技术点、回答表现、薄弱点和复习建议。

## 隐私说明

- `.sessions` 会保存本地历史会话，可能包含 JD、简历、代码片段和面试对话
- `.interview-agent/reports` 会保存导出的 Markdown 报告
- 插件不会自动提交、上传、发布或修改你的代码
- 使用真实模型时，对话内容会发送到你配置的 OpenAI 兼容服务商

## 故障排查

| 问题 | 处理方式 |
|---|---|
| 面板空白或提示没有数据提供程序 | 确认安装的是 `0.2.0` VSIX，执行 `Developer: Reload Window` |
| 缺少 Python 依赖 | 在面板点击“安装 Agent 依赖”，或手动执行 `python -m pip install openai` |
| Python 路径不对 | 在设置里填写 `interview.pythonPath` 为目标解释器完整路径 |
| API Key 错误 | 检查 `interview.apiKey` 和账户额度 |
| 模型不存在 | 检查 `interview.model` 是否拼写正确，并与 `interview.baseUrl` 服务商匹配 |
| Base URL 或网络错误 | 检查 `interview.baseUrl`、代理和网络连通性 |
| OCR 失败 | 扫描版 PDF 或图片简历需要安装 OCR 依赖；也可以改用文本粘贴 |

## 开发验证

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check agent

cd vscode-extension
npm test
npx tsc -p ./ --noEmit
npm run compile
npm run package
```
