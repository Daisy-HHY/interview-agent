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

1. 到 GitHub Release 下载 `interview-agent-0.1.12.vsix`
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
| `interview.agentRuntime` | Agent 运行时。默认 `native`；可选 `langchain` 做 LangChain / LangGraph 运行时验证 |

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

0.1.12 仍默认使用 `native` runtime。只有真实 runtime benchmark 达标后，才应把默认值切换到 `langchain`。

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
| 面板空白或提示没有数据提供程序 | 确认安装的是 `0.1.12` VSIX，执行 `Developer: Reload Window` |
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
