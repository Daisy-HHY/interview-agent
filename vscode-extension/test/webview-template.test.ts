import { readFileSync } from "fs";
import { join } from "path";
import { describe, expect, it } from "vitest";

const webviewRoot = join(__dirname, "..", "src", "webview");
const html = readFileSync(join(webviewRoot, "index.html"), "utf-8");
const script = readFileSync(join(webviewRoot, "main.js"), "utf-8").replace(/\r\n/g, "\n");
const styles = readFileSync(join(webviewRoot, "styles.css"), "utf-8");
const panelSource = readFileSync(join(__dirname, "..", "src", "webviewPanel.ts"), "utf-8");

describe("Webview 面试入口模板", () => {
  it("输入区只有一个状态按钮，不再保留独立停止按钮", () => {
    expect(html).toContain('id="action"');
    expect(html).not.toContain('id="stop"');
    expect(script).toContain('vscode.postMessage({ type: "stop" })');
    expect(styles).toContain(".composer__action");
    expect(styles).toContain("right: 20px");
    expect(styles).toContain("bottom: 20px");
  });

  it("首屏包含简历上传和当前项目自动读取信息", () => {
    expect(html).toContain("上传简历附件");
    expect(html).toContain("支持 .pdf / .docx / .txt / .md / 图片");
    expect(html).toContain('class="resume-upload"');
    expect(html).toContain('id="resumeFileInput"');
    expect(html).not.toContain('for="resumeFileInput"');
    expect(html).toContain('type="file"');
    expect(html).toContain('accept=".pdf,.docx,.txt,.md,.markdown,.png,.jpg,.jpeg,.webp');
    expect(html).toContain('id="resumeSupplement"');
    expect(html).toContain('id="workspaceInfo"');
    expect(html).not.toContain("简历 / 项目背景");
    expect(script).toContain("resumeFileInputEl.files?.[0]");
    expect(script).toContain("请自动读取当前 VS Code 工作区下的项目情况");
  });

  it("展示 OCR 处理进度", () => {
    expect(script).toContain('msg.type === "resumeOcrProgress"');
    expect(script).toContain("formatOcrProgress");
    expect(script).toContain("elapsedMs");
    expect(script).toContain("currentPage");
    expect(script).toContain("totalPages");
    expect(script).toContain("已耗时");
  });

  it("简历上传区支持拖拽上传", () => {
    expect(script).toContain('resumeFileInputEl.addEventListener("change"');
    expect(script).toContain('pickResumeBtn.addEventListener("click"');
    expect(script).toContain("event.preventDefault()");
    expect(script).toContain('vscode.postMessage({ type: "pickResume" })');
    expect(script).not.toContain('console.info("[resume-debug]"');
    expect(panelSource).toContain("[resume]");
    expect(script).toContain("function onResumeDrag");
    expect(script).toContain("function armResumeFileDrop");
    expect(script).toContain('"armResumeFileDrop"');
    expect(script).toContain("function setResumeCaptureState");
    expect(script).toContain('"resumeCaptureState"');
    expect(script).toContain("setResumeCaptureState(true)");
    expect(script).toContain("setResumeCaptureState(false)");
    expect(script).toContain("function isSystemFileDrag");
    expect(script).toContain('includes("Files")');
    expect(script).toContain("[pickResumeBtn, resumeFileInputEl].forEach");
    expect(script).toContain('dropTarget.addEventListener("dragenter"');
    expect(script).toContain('dropTarget.addEventListener("dragover"');
    expect(script).toContain('resumeFileInputEl.addEventListener("drop"');
    expect(script).toContain('pickResumeBtn.addEventListener("drop"');
    expect(script).toContain("event.target === resumeFileInputEl");
    expect(script).toContain('document.addEventListener("dragenter"');
    expect(script).toContain('document.addEventListener("dragover"');
    expect(script).toContain('document.addEventListener("drop"');
    expect(script).toContain("readDroppedResume");
    expect(script).toContain("function getDroppedResumePayload");
    expect(script).toContain("getAsFile()");
    expect(script).toContain("function getDroppedFilePath");
    expect(script).toContain('"text/uri-list"');
    expect(script).toContain('"application/vnd.code.tree.resourceuris"');
    expect(script).toContain('startsWith("application/vnd.code.tree.")');
    expect(script).toContain('"pickResumePath"');
    expect(script).toContain("new FileReader()");
    expect(script).toContain("readAsDataURL(file)");
    expect(script).toContain('"pickResumeUpload"');
    expect(script).not.toContain("file?.path");
    expect(styles).toContain(".resume-upload.is-dragover");
    expect(styles).toContain(".resume-upload__input");
    expect(styles).toContain("z-index: 2");
    expect(styles).toContain("pointer-events: none");
    expect(panelSource).toContain("enableForms: true");
  });

  it("处理 cancelled 通知并显示已停止状态", () => {
    expect(script).toContain('case "cancelled"');
    expect(script).toContain("onCancelled");
    expect(script).toContain("已停止");
  });

  it("等待回复时按截图样式显示思考球，收到输出后移除", () => {
    expect(script).toContain("function showThinkingOrb");
    expect(script).toContain("function removeThinkingOrb");
    expect(script).toContain("thinking-orb");
    expect(script).toContain("Waiting for model...");
    expect(script).toContain("showThinkingOrb()");
    expect(script).toContain("removeThinkingOrb()");
    expect(styles).toContain(".thinking-orb");
    expect(styles).toContain("clamp(18px, 7vw, 26px)");
    expect(styles).toContain("@keyframes orbSpin");
  });

  it("思考状态右侧折叠展示工具调用内容", () => {
    expect(script).toContain("document.createElement(\"details\")");
    expect(script).toContain("document.createElement(\"summary\")");
    expect(script).toContain("thinking-details");
    expect(script).toContain("查看运行命令和工具调用");
    expect(script).toContain("function updateThinkingDetails");
    expect(script).toContain("function formatThinkingDetails");
    expect(styles).toContain(".thinking-details");
    expect(styles).toContain(".thinking-details__body");
  });

  it("一轮面试官流式输出复用同一个气泡，工具调用期间仍显示思考球", () => {
    expect(script).toContain("if (!thinkingBubble)");
    expect(script).toContain("showThinkingOrb();");
    expect(script).toContain("if (!delta)");
    expect(script).not.toContain("updateThinkingDetails();\n      currentInterviewerBubble = null;");
    expect(script).toContain("function getCurrentInterviewerSegment");
    expect(script).toContain("segment.innerHTML = renderMarkdown(segment.__raw);");
    expect(script).toContain("showThinkingOrb({ resetLogs: false });");
    expect(script).toContain("if (options.resetLogs !== false)");
  });

  it("工具调用按 Codex 风格显示活动行并保留小思考球", () => {
    expect(script).toContain("function appendActivityRow");
    expect(script).toContain("function updateActivityRow");
    expect(script).toContain("function toolActionLabel");
    expect(script).toContain("activity-row__orb");
    expect(styles).toContain(".activity-row");
    expect(styles).toContain(".activity-row.is-running .activity-row__orb");
    expect(styles).toContain(".activity-row.is-running .activity-row__icon");
  });

  it("面试官流式输出不显示块状光标效果", () => {
    expect(script).not.toContain('classList.add("cursor")');
    expect(styles).not.toContain(".cursor::after");
    expect(styles).not.toContain("@keyframes blink");
  });

  it("面试官消息渲染 Markdown，用户/工具消息保持纯文本", () => {
    expect(script).toContain("function renderMarkdown");
    expect(script).toContain("function escapeHtml");
    expect(script).toContain("function renderInline");
    // 面试官气泡走 Markdown 渲染（先转义再转换，无注入面）
    expect(script).toContain("segment.innerHTML = renderMarkdown");
    // 行内代码防二次转换的占位机制
    expect(script).toContain("md-code");
    // 取消提示同样经 Markdown 渲染
    expect(script).toContain("segment.__raw");
  });

  it("面试官回答和思考状态不显示外框", () => {
    expect(styles).toContain(".bubble--interviewer");
    expect(styles).toContain("background: transparent");
    expect(styles).toContain("border-color: transparent");
    expect(styles).toContain(".bubble--thinking");
  });

  it("会话条目按钮不被挤压换行（窄侧边栏回归）", () => {
    expect(styles).toContain(".session-item__buttons");
    expect(styles).toContain("flex: 0 0 auto");
    expect(styles).toContain("white-space: nowrap");
    expect(styles).toContain(".session-item__buttons .secondary-button");
  });

  it("0.1.7 普通聊天发送不再保存配置重启 Agent", () => {
    expect(script).toContain("function send()");
    expect(script).toContain("sendChat(text, text);");
    expect(script).not.toContain("saveConfig(() => sendChat(text, text))");
  });

  it("模型配置区只显示摘要和设置入口，面板不展示敏感配置表单", () => {
    expect(html).toContain('id="settings"');
    expect(html).toContain("模型配置");
    expect(html).toContain('id="modelConfigSummary"');
    expect(html).toContain('id="openModelSettings"');
    expect(html).toContain('id="testModelConnection"');
    expect(html).not.toContain('id="provider"');
    expect(html).not.toContain('id="model"');
    expect(html).not.toContain('id="baseUrl"');
    expect(html).not.toContain('id="apiKey"');
    expect(html).not.toContain('id="demoMode"');
    expect(html).not.toContain('id="saveConfig"');
    expect(script).not.toContain("function saveConfig");
    expect(script).not.toContain('type: "updateConfig"');
  });

  it("首屏包含 runtime 诊断面板并展示 runtime 配置", () => {
    expect(html).toContain('id="runtimeDiagnostics"');
    expect(html).toContain("诊断");
    expect(script).toContain('case "runtime_metric"');
    expect(script).toContain("function renderRuntimeMetric");
    expect(script).toContain("metric.model_elapsed_ms");
    expect(script).toContain("config.agentRuntime");
    expect(styles).toContain(".diagnostics__body");
  });

  it("包含依赖安装和历史会话入口", () => {
    expect(html).toContain('id="dependencyPanel"');
    expect(html).toContain('id="installDependencies"');
    expect(html).toContain('id="exportReport"');
    expect(html).toContain('id="historyPanel"');
    expect(html).toContain('id="newSession"');
    expect(script).toContain('"installDependencies"');
    expect(script).toContain('"installOcrDependencies"');
    expect(script).toContain('"testModelConnection"');
    expect(script).toContain('"exportReport"');
    expect(script).toContain('type: "resumeSession"');
    expect(script).toContain('type: "deleteSession"');
  });

  it("导出报告前提示报告包含隐私内容", () => {
    expect(script).toContain("导出的 Markdown 报告会包含 JD、简历和面试对话内容");
    expect(script).toContain("reportExported");
    expect(script).toContain("reportError");
  });

  it("对话区滚动，输入框固定在面板最底部", () => {
    expect(html).toContain('id="chat"');
    expect(html).toContain('class="chat is-hidden"');
    // 纵向 flex 骨架：设置页/对话页各自占满剩余区域
    expect(styles).toContain("flex-direction: column");
    expect(styles).toContain(".setup");
    expect(styles).toContain(".chat");
    expect(styles).toContain(".messages");
    expect(styles).toContain(".composer");
    // 设置页和消息流占满剩余空间（可滚动），输入框不参与压缩（钉在底部）
    expect(styles).toContain("flex: 1 1 auto");
    expect(styles).toContain("flex: 0 0 auto");
    expect(styles).toContain("overflow-y: auto");
    expect(styles).toContain("min-height: 0");
    expect(styles).not.toContain("max-height: 38vh");
    expect(styles).toContain("min-height: 92px");
    expect(styles).toContain("width: 32px");
    expect(styles).toContain("height: 32px");
  });

  it("初始进入设置页，开始/继续会话进入对话页，新建会话回设置页", () => {
    expect(script).toContain('const chatEl = document.getElementById("chat")');
    expect(script).toContain("function showSetupPage");
    expect(script).toContain("function showChatPage");
    expect(script).toContain('chatEl.classList.add("is-hidden")');
    expect(script).toContain('chatEl.classList.remove("is-hidden")');
    expect(script).toContain("showChatPage();\n    sendChat(text");
    expect(script).toContain('if (msg.type === "sessionNew")');
    expect(script).toContain('showSetupPage("已新建会话")');
  });

  it("顶栏为品牌标识布局（logo + 标题 + 副标题）", () => {
    expect(html).toContain("topbar__brand");
    expect(html).toContain("topbar__logo");
    expect(html).toContain("topbar__title");
    expect(html).toContain("topbar__subtitle");
    expect(styles).toContain(".topbar__logo");
    expect(styles).toContain(".topbar__subtitle");
  });

  it("JD 与简历区为带步骤编号的卡片分区", () => {
    expect(html).toContain('class="card"');
    expect(html).toContain("card__step");
    expect(html).toContain("card__header");
    expect(html).toContain("card__hint");
    expect(html).toContain("岗位 JD");
    expect(html).toContain('id="jd"');
    expect(html).toContain('id="pickResume"');
    expect(html).toContain('id="resumeSupplement"');
    expect(styles).toContain(".card__step");
    expect(styles).toContain(".card");
  });

  it("依赖检测通过后隐藏依赖面板", () => {
    expect(script).toContain('dependencyPanelEl.classList.toggle("is-hidden", !status.message)');
  });
});
