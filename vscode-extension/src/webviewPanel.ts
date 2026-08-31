/**
 * Webview 面试视图管理。
 *
 * Extension Host 只做三件事：
 * - 把 Webview 的聊天消息转成 Python Agent 请求
 * - 把 Python Agent 通知转发给 Webview
 * - 读取/保存模型配置并在必要时重启 Python 子进程
 */

import { randomBytes } from "crypto";
import {
  existsSync,
  mkdirSync,
  readFileSync,
  readdirSync,
  rmSync,
  statSync,
  writeFileSync,
} from "fs";
import { tmpdir } from "os";
import { basename, extname, join } from "path";
import {
  CancellationToken,
  OutputChannel,
  type Tab,
  Uri,
  Webview,
  WebviewView,
  WebviewViewProvider,
  WebviewViewResolveContext,
  commands,
  window,
} from "vscode";
import { AgentClient } from "./agentClient";
import { buildChat, buildStop } from "./protocol";
import { locatePython } from "./pythonLocator";

/** 共享的调试输出通道（整个插件一个，所有 Python 日志都写这里）。 */
let debugChannel: OutputChannel | null = null;
function getDebugChannel(): OutputChannel {
  if (!debugChannel) {
    debugChannel = window.createOutputChannel("Interview Agent", { log: true });
  }
  return debugChannel;
}

function writeDebugLine(message: string, show = false): void {
  const logger = getDebugChannel();
  if (show) {
    logger.show(true);
  }
  logger.appendLine(`${new Date().toISOString()} ${message}`);
}

/** 从配置构造 AgentClient 需要的参数。 */
export interface PanelOptions {
  pythonPath: string;
  vscodePythonPath?: string;
  scriptPath: string;
  /** 被面试的项目根（工具翻代码的根）。 */
  workspace: string;
  workspaceName: string;
  hasWorkspace: boolean;
  /** agent 包所在根（PYTHONPATH 用），通常 = bundled-agent 根。 */
  pythonPathRoot: string;
  /** 打包进插件的 Python 依赖清单。 */
  requirementsPath: string;
  /** 扫描版 PDF OCR 的可选依赖清单。 */
  requirementsOcrPath: string;
  apiKey: string;
  model: string;
  baseUrl?: string;
  resume?: string;
  /** 演示模式：用 FakeLLM，零费用。 */
  demoMode?: boolean;
  maxSteps?: number;
  maxHistoryTokens?: number;
  maxKeptFull?: number;
  agentRuntime?: "native" | "langchain";
}

/** 发给 Webview 的配置快照；不回传 API Key 明文。 */
export interface WebviewConfigSnapshot {
  model: string;
  baseUrl: string;
  demoMode: boolean;
  hasApiKey: boolean;
  workspaceName: string;
  workspacePath: string;
  hasWorkspace: boolean;
}

/** Webview 发来的配置变更。 */
export interface WebviewConfigUpdate {
  model?: string;
  baseUrl?: string;
  apiKey?: string;
  demoMode?: boolean;
}

type WebviewToHostMessage =
  | { type: "ready" }
  | { type: "chat"; text: string }
  | { type: "stop" }
  | { type: "pickResume" }
  | { type: "armResumeFileDrop" }
  | { type: "resumeCaptureState"; enabled: boolean }
  | { type: "pickResumePath"; path: string }
  | { type: "pickResumeUpload"; fileName: string; dataBase64: string }
  | { type: "installDependencies" }
  | { type: "installOcrDependencies" }
  | { type: "checkDependencies" }
  | { type: "testModelConnection" }
  | { type: "exportReport" }
  | { type: "listSessions" }
  | { type: "newSession" }
  | { type: "resumeSession"; session: string }
  | { type: "deleteSession"; session: string }
  | { type: "openSettings" }
  | { type: "updateConfig"; config: WebviewConfigUpdate };

export interface ResumeParseResult {
  fileName: string;
  content: string;
  truncated: boolean;
}

interface SessionSummary {
  id: string;
  title: string;
  updatedAt: number;
  messageCount: number;
  preview: string;
}

interface DisplayMessage {
  role: "user" | "assistant";
  content: string;
}

const RESUME_MAX_CHARS = 80_000;
const RESUME_PARSE_TIMEOUT_MS = 150_000;
const PDF_TEXT_PARSE_TIMEOUT_MS = 8_000;
const PDF_TEXT_FALLBACK_TIMEOUT_MS = 20_000;

interface ReportInput {
  sessionId: string;
  title: string;
  workspaceName: string;
  workspacePath: string;
  createdAt: Date;
  messages: DisplayMessage[];
}

interface ModelTestInput {
  pythonPath: string;
  pythonPathRoot: string;
  apiKey: string;
  model: string;
  baseUrl?: string;
}

interface ModelTestResult {
  ok: boolean;
  kind: string;
  message: string;
}

export class InterviewViewProvider implements WebviewViewProvider {
  private view: WebviewView | undefined;
  private agent: AgentClient | null = null;
  private sessionId = makeSessionId();
  private resumeDropArmedUntil = 0;
  private resumeCaptureReady = true;

  constructor(
    private readonly htmlBasePath: Uri,
    private readonly buildOptions: () => PanelOptions,
    private readonly saveConfig: (config: WebviewConfigUpdate) => Promise<void>,
  ) {}

  /** 注册给 VS Code 的侧边栏 Webview View 创建入口。 */
  resolveWebviewView(
    webviewView: WebviewView,
    _context: WebviewViewResolveContext,
    _token: CancellationToken,
  ): void {
    this.disposeAgent();
    this.view = webviewView;
    webviewView.webview.options = {
      enableScripts: true,
      enableForms: true,
      localResourceRoots: [this.htmlBasePath],
    };
    webviewView.webview.html = this.buildHtml(webviewView.webview);
    this.wireMessages(webviewView.webview);
    this.postConfig(webviewView.webview);
    this.postSessions(webviewView.webview);

    webviewView.onDidDispose(() => {
      this.disposeAgent();
      if (this.view === webviewView) {
        this.view = undefined;
      }
    });
  }

  /** 聚焦面试视图；命令面板入口复用它，不再新开编辑器 Tab。 */
  focus(): void {
    void commands.executeCommand("workbench.view.extension.interview-agent");
    void commands.executeCommand("interview.chatView.focus");
  }

  /** 系统文件拖入 Webview 被 VS Code 打开成 Tab 时，短时间内兜底接管为简历上传。 */
  captureOpenedResumeTab(tab: Tab): void {
    const canCapture = this.resumeCaptureReady || Date.now() <= this.resumeDropArmedUntil;
    if (!this.view?.visible || !canCapture) {
      return;
    }
    const filePath = getResumeFilePathFromTabInput(tab.input);
    if (!filePath) {
      return;
    }
    this.resumeDropArmedUntil = 0;
    void this.parsePickedResume(this.view.webview, filePath);
  }

  private armResumeFileDrop(): void {
    this.resumeDropArmedUntil = Date.now() + 10_000;
  }

  /** 选中代码提问命令：聚焦面板并预填一条面试追问。 */
  prefillSelectionQuestion(): void {
    this.focus();
    void this.view?.webview.postMessage({
      type: "prefill",
      text: "请针对我当前选中的这段代码进行面试追问。",
    });
  }

  private wireMessages(webview: Webview): void {
    webview.onDidReceiveMessage((msg: WebviewToHostMessage) => {
      if (msg.type === "ready") {
        this.postConfig(webview);
        return;
      }
      if (msg.type === "chat") {
        if (!this.ensureAgentStarted(webview)) {
          return;
        }
        const attached = this.readSelection();
        this.agent?.send(
          buildChat({
            session: this.sessionId,
            text: msg.text,
            attached_code: attached,
          }),
        );
        return;
      }
      if (msg.type === "stop") {
        this.agent?.send(buildStop(this.sessionId));
        return;
      }
      if (msg.type === "pickResume") {
        writeDebugLine("[resume-debug] received pickResume");
        void this.pickResume(webview);
        return;
      }
      if (msg.type === "armResumeFileDrop") {
        this.armResumeFileDrop();
        return;
      }
      if (msg.type === "resumeCaptureState") {
        this.resumeCaptureReady = msg.enabled;
        return;
      }
      if (msg.type === "pickResumePath") {
        this.resumeDropArmedUntil = 0;
        writeDebugLine(`[resume-debug] received pickResumePath path=${msg.path}`, true);
        void this.parsePickedResume(webview, msg.path, "drop-path");
        return;
      }
      if (msg.type === "pickResumeUpload") {
        this.resumeDropArmedUntil = 0;
        writeDebugLine(
          `[resume-debug] received pickResumeUpload file=${msg.fileName} base64Chars=${msg.dataBase64.length}`,
          true,
        );
        void this.parseUploadedResume(webview, msg.fileName, msg.dataBase64);
        return;
      }
      if (msg.type === "installDependencies") {
        this.installDependencies(webview);
        return;
      }
      if (msg.type === "installOcrDependencies") {
        this.installOcrDependencies(webview);
        return;
      }
      if (msg.type === "checkDependencies") {
        this.checkDependencies(webview);
        return;
      }
      if (msg.type === "testModelConnection") {
        void this.testModelConnection(webview);
        return;
      }
      if (msg.type === "exportReport") {
        void this.exportReport(webview);
        return;
      }
      if (msg.type === "listSessions") {
        this.postSessions(webview);
        return;
      }
      if (msg.type === "newSession") {
        this.newSession(webview);
        return;
      }
      if (msg.type === "resumeSession") {
        this.resumeSession(webview, msg.session);
        return;
      }
      if (msg.type === "deleteSession") {
        this.deleteSession(webview, msg.session);
        return;
      }
      if (msg.type === "openSettings") {
        void commands.executeCommand("workbench.action.openSettings", "interview");
        return;
      }
      if (msg.type === "updateConfig") {
        void this.handleConfigUpdate(webview, msg.config);
      }
    });
  }

  /** 通过 VS Code 文件选择器读取简历附件。 */
  private async pickResume(webview: Webview): Promise<void> {
    writeDebugLine("[resume-debug] opening VS Code file dialog", true);
    let selected: Uri[] | undefined;
    try {
      selected = await window.showOpenDialog({
        canSelectFiles: true,
        canSelectFolders: false,
        canSelectMany: false,
        filters: {
          "简历文件": ["pdf", "docx", "txt", "md", "markdown", "png", "jpg", "jpeg", "webp"],
        },
        title: "选择简历文件",
      });
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      writeDebugLine(`[resume-debug] file dialog failed error=${message}`, true);
      void webview.postMessage({
        type: "resumeError",
        message: `打开文件选择器失败：${message}`,
      });
      return;
    }
    const file = selected?.[0];
    if (!file) {
      writeDebugLine("[resume-debug] file dialog canceled");
      return;
    }

    writeDebugLine(`[resume-debug] selected path=${file.fsPath}`, true);
    await this.parsePickedResume(webview, file.fsPath, "dialog");
  }

  /** 读取用户选择或拖入的简历附件。 */
  private async parsePickedResume(
    webview: Webview,
    filePath: string,
    source = "picked",
  ): Promise<void> {
    await this.parseResumePath(webview, filePath, undefined, source);
  }

  /** 读取 Webview 拖拽传来的简历内容，写入临时文件后复用原解析流程。 */
  private async parseUploadedResume(
    webview: Webview,
    fileName: string,
    dataBase64: string,
  ): Promise<void> {
    const safeName = safeUploadedFileName(fileName);
    const dir = join(tmpdir(), "interview-agent-resume-drop");
    const filePath = join(dir, `${Date.now()}-${safeName}`);
    mkdirSync(dir, { recursive: true });
    writeFileSync(filePath, Buffer.from(stripDataUrlPrefix(dataBase64), "base64"));
    writeDebugLine(
      `[resume-debug] wrote dropped temp file path=${filePath} bytes=${safeStatSize(filePath)}`,
      true,
    );
    try {
      await this.parseResumePath(webview, filePath, safeName, "drop-upload");
    } finally {
      rmSync(filePath, { force: true });
      writeDebugLine(`[resume-debug] removed dropped temp file path=${filePath}`);
    }
  }

  private async parseResumePath(
    webview: Webview,
    filePath: string,
    displayFileName?: string,
    source = "unknown",
  ): Promise<void> {
    const options = this.buildOptions();
    const pythonLookup = locatePython({
      configuredPath: options.pythonPath,
      workspacePath: options.workspace,
      vscodePythonPath: options.vscodePythonPath,
      requireOpenAI: false,
    });
    const startedAt = Date.now();
    const ext = extname(filePath).toLowerCase() || "unknown";
    writeDebugLine(
      `[resume-debug] parse start source=${source} path=${filePath} ext=${ext} bytes=${safeStatSize(filePath)} python=${pythonLookup.pythonPath}`,
      true,
    );
    try {
      void webview.postMessage({ type: "resumeStatus", message: "正在读取简历..." });
      const resume = await withResumeParseTimeout(
        parseResumeFile(filePath, {
          ocr: (ocrPath) => runResumeOcr({
            filePath: ocrPath,
            pythonPath: pythonLookup.pythonPath,
            scriptPath: join(options.pythonPathRoot, "agent", "resume_ocr.py"),
            pythonPathRoot: options.pythonPathRoot,
            onProgress: (progress) => {
              void webview.postMessage({ type: "resumeOcrProgress", progress });
            },
          }),
          pdfTextFallback: (pdfPath, reason) => runResumePdfText({
            filePath: pdfPath,
            pythonPath: pythonLookup.pythonPath,
            pythonPathRoot: options.pythonPathRoot,
            reason,
          }),
          onStatus: (message) => {
            writeDebugLine(`[resume-debug] status ${message}`);
            void webview.postMessage({ type: "resumeStatus", message });
          },
          onDebug: (message) => {
            writeDebugLine(`[resume-debug] ${message}`);
          },
        }),
      );
      if (displayFileName) {
        resume.fileName = displayFileName;
      }
      writeDebugLine(
        `[resume-debug] parse success file=${resume.fileName} chars=${resume.content.length} truncated=${resume.truncated} elapsedMs=${Date.now() - startedAt}`,
        true,
      );
      void webview.postMessage({
        type: "resumePicked",
        resume,
      });
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      writeDebugLine(
        `[resume-debug] parse failed path=${filePath} elapsedMs=${Date.now() - startedAt} error=${message}`,
        true,
      );
      if (isOcrDependencyError(message)) {
        this.postOcrDependencyError(webview, message, pythonLookup.pythonPath);
        return;
      }
      void webview.postMessage({
        type: "resumeError",
        message: `读取简历失败：${message}`,
      });
    }
  }

  /** 在 VS Code Terminal 中显式安装 Agent 依赖。 */
  private installDependencies(webview: Webview): void {
    const options = this.buildOptions();
    const pythonLookup = locatePython({
      configuredPath: options.pythonPath,
      workspacePath: options.workspace,
      vscodePythonPath: options.vscodePythonPath,
      requireOpenAI: false,
    });
    const command = buildInstallCommand(pythonLookup.pythonPath, options.requirementsPath);
    const logger = getDebugChannel();
    logger.appendLine(`[deps] ${command}`);
    const terminal = window.createTerminal("Interview Agent 依赖安装");
    terminal.show();
    terminal.sendText(command);
    void webview.postMessage({
      type: "dependencyStatus",
      message: "已在 Terminal 启动依赖安装。安装完成后点击重新检测或重新开始面试。",
      command,
      canInstall: true,
    });
  }

  /** 在 VS Code Terminal 中显式安装扫描版 PDF OCR 可选依赖。 */
  private installOcrDependencies(webview: Webview): void {
    const options = this.buildOptions();
    const pythonLookup = locatePython({
      configuredPath: options.pythonPath,
      workspacePath: options.workspace,
      vscodePythonPath: options.vscodePythonPath,
      requireOpenAI: false,
    });
    const command = buildOcrInstallCommand(
      pythonLookup.pythonPath,
      options.requirementsOcrPath,
    );
    const logger = getDebugChannel();
    logger.appendLine(`[ocr-deps] ${command}`);
    const terminal = window.createTerminal("Interview Agent OCR 依赖安装");
    terminal.show();
    terminal.sendText(command);
    void webview.postMessage({
      type: "dependencyStatus",
      message: "已在 Terminal 启动 OCR 依赖安装。安装完成后重新上传扫描版 PDF。",
      command,
      canInstall: true,
      installType: "ocr",
      buttonLabel: "安装 OCR 依赖",
    });
  }

  /** 重新检测真实模式运行依赖。 */
  private checkDependencies(webview: Webview): void {
    const options = this.buildOptions();
    const pythonLookup = locatePython({
      configuredPath: options.pythonPath,
      workspacePath: options.workspace,
      vscodePythonPath: options.vscodePythonPath,
      requireOpenAI: !options.demoMode,
    });
    if (pythonLookup.error) {
      this.postDependencyError(webview, pythonLookup.error, pythonLookup.pythonPath);
      return;
    }
    void webview.postMessage({
      type: "dependencyStatus",
      message: "",
      canInstall: false,
    });
  }

  /** 读取目标工作区本地历史会话并发给 Webview。 */
  private postSessions(webview: Webview): void {
    const options = this.buildOptions();
    void webview.postMessage({
      type: "sessions",
      sessions: listSessionSummaries(options.workspace),
      current: this.sessionId,
    });
  }

  private newSession(webview: Webview): void {
    this.restartAgent();
    void webview.postMessage({ type: "sessionNew", session: this.sessionId });
    this.postSessions(webview);
  }

  private resumeSession(webview: Webview, session: string): void {
    const options = this.buildOptions();
    const loaded = loadSessionMessages(options.workspace, session);
    this.disposeAgent();
    this.sessionId = session;
    void webview.postMessage({
      type: "sessionLoaded",
      session,
      messages: loaded,
    });
    this.postSessions(webview);
  }

  private deleteSession(webview: Webview, session: string): void {
    const options = this.buildOptions();
    deleteSessionFile(options.workspace, session);
    if (session === this.sessionId) {
      this.restartAgent();
      void webview.postMessage({ type: "sessionNew", session: this.sessionId });
    }
    this.postSessions(webview);
  }

  /** 测试当前真实模型配置；只发一条极短请求，不写入面试会话。 */
  private async testModelConnection(webview: Webview): Promise<void> {
    const options = this.buildOptions();
    if (options.demoMode) {
      void webview.postMessage({
        type: "modelTestResult",
        ok: true,
        message: "Demo Mode 使用内置 FakeLLM，无需测试真实模型连接。",
      });
      return;
    }
    if (!options.apiKey) {
      void webview.postMessage({
        type: "modelTestResult",
        ok: false,
        message: "还未配置 API Key。请填写 interview.apiKey，或开启 Demo Mode。",
      });
      return;
    }
    if (!options.model.trim()) {
      void webview.postMessage({
        type: "modelTestResult",
        ok: false,
        message: "还未配置模型名。请填写 interview.model 后再测试连接。",
      });
      return;
    }

    const pythonLookup = locatePython({
      configuredPath: options.pythonPath,
      workspacePath: options.workspace,
      vscodePythonPath: options.vscodePythonPath,
      requireOpenAI: true,
    });
    if (pythonLookup.error) {
      this.postDependencyError(webview, pythonLookup.error, pythonLookup.pythonPath);
      void webview.postMessage({
        type: "modelTestResult",
        ok: false,
        message: pythonLookup.error,
      });
      return;
    }

    void webview.postMessage({
      type: "modelTestStatus",
      message: "正在测试模型连接...",
    });
    try {
      const result = await runModelConnectionTest({
        pythonPath: pythonLookup.pythonPath,
        pythonPathRoot: options.pythonPathRoot,
        apiKey: options.apiKey,
        model: options.model,
        baseUrl: options.baseUrl,
      });
      void webview.postMessage({ type: "modelTestResult", ...result });
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      void webview.postMessage({
        type: "modelTestResult",
        ok: false,
        message,
      });
    }
  }

  /** 将当前会话导出为工作区本地 Markdown 报告。 */
  private async exportReport(webview: Webview): Promise<void> {
    const options = this.buildOptions();
    if (!options.hasWorkspace || !options.workspace) {
      void webview.postMessage({
        type: "reportError",
        message: "请先打开目标项目文件夹，再导出面试报告。",
      });
      return;
    }

    const messages = loadSessionMessages(options.workspace, this.sessionId);
    if (!canExportReport(messages)) {
      void webview.postMessage({
        type: "reportError",
        message: "当前会话还没有可导出的面试内容。",
      });
      return;
    }

    const title = makeSessionTitle(messages, this.sessionId);
    const reportsDir = join(options.workspace, ".interview-agent", "reports");
    const fileName = `${formatReportTimestamp(new Date())}-${sanitizeReportFileName(title)}.md`;
    const filePath = join(reportsDir, fileName);
    mkdirSync(reportsDir, { recursive: true });
    writeFileSync(
      filePath,
      generateMarkdownReport({
        sessionId: this.sessionId,
        title,
        workspaceName: options.workspaceName,
        workspacePath: options.workspace,
        createdAt: new Date(),
        messages,
      }),
      "utf-8",
    );
    await commands.executeCommand("vscode.open", Uri.file(filePath));
    void webview.postMessage({
      type: "reportExported",
      message: `报告已导出：${filePath}`,
      filePath,
    });
  }

  /** 读当前编辑器选中的代码，作为下一轮面试追问的上下文。 */
  private readSelection(): { file: string; content: string } | undefined {
    const editor = window.activeTextEditor;
    if (!editor || editor.selection.isEmpty) {
      return undefined;
    }
    const text = editor.document.getText(editor.selection);
    if (!text.trim()) {
      return undefined;
    }
    return {
      file: editor.document.fileName,
      content: text,
    };
  }

  /**
   * 保存模型配置并重启 Agent。
   *
   * 参数：
   * - config：Webview 发来的模型名、Base URL、API Key 或 Demo Mode 更新
   * 返回值：无；保存成功后把最新配置快照发回 Webview
   */
  private async handleConfigUpdate(
    webview: Webview,
    config: WebviewConfigUpdate,
  ): Promise<void> {
    try {
      await this.saveConfig(config);
      this.restartAgent();
      this.postConfig(webview);
      void webview.postMessage({ type: "configSaved" });
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      void webview.postMessage({
        method: "error",
        params: { session: this.sessionId, message: `保存配置失败：${message}` },
      });
    }
  }

  /** 按当前配置懒启动 Python Agent；无 API Key 时阻止真实调用。 */
  private ensureAgentStarted(webview: Webview): boolean {
    if (this.agent) {
      return true;
    }

    const options = this.buildOptions();
    if (!options.hasWorkspace || !options.workspace) {
      void webview.postMessage({
        method: "error",
        params: {
          session: this.sessionId,
          message: "请先打开要面试的目标项目文件夹，再开始面试。",
        },
      });
      return false;
    }

    if (!options.demoMode && !options.apiKey) {
      void webview.postMessage({
        method: "error",
        params: {
          session: this.sessionId,
          message: "还未配置 API Key。请填写 API Key，或开启 Demo Mode。",
        },
      });
      return false;
    }

    if (!options.model.trim()) {
      void webview.postMessage({
        method: "error",
        params: {
          session: this.sessionId,
          message: "还未配置模型名。请填写 interview.model 后再开始面试。",
        },
      });
      return false;
    }

    const pythonLookup = locatePython({
      configuredPath: options.pythonPath,
      workspacePath: options.workspace,
      vscodePythonPath: options.vscodePythonPath,
      requireOpenAI: !options.demoMode,
    });

    const logger = getDebugChannel();
    for (const line of pythonLookup.diagnostics) {
      logger.appendLine(line);
    }
    logger.appendLine(
      `[llm] ${options.demoMode ? "Demo Mode（固定脚本）" : "真实模型"} model=${options.model} baseUrl=${options.baseUrl || "OpenAI 默认"} runtime=${options.agentRuntime || "native"}`,
    );
    if (!options.demoMode && pythonLookup.error) {
      this.postDependencyError(webview, pythonLookup.error, pythonLookup.pythonPath);
      return false;
    }
    void webview.postMessage({
      type: "dependencyStatus",
      message: "",
      canInstall: false,
    });

    this.agent = new AgentClient({
      pythonPath: pythonLookup.pythonPath,
      scriptPath: options.scriptPath,
      workspace: options.workspace,
      pythonPathRoot: options.pythonPathRoot,
      apiKey: options.apiKey || "demo",
      model: options.model,
      baseUrl: options.baseUrl,
      resume: options.resume,
      session: this.sessionId,
      demoMode: options.demoMode,
      maxSteps: options.maxSteps,
      maxHistoryTokens: options.maxHistoryTokens,
      maxKeptFull: options.maxKeptFull,
      agentRuntime: options.agentRuntime,
    });
    this.wireAgent(webview);
    this.agent.start();
    return true;
  }

  private wireAgent(webview: Webview): void {
    const logger = getDebugChannel();

    this.agent?.onLog((message) => {
      logger.appendLine(message);
    });

    this.agent?.onNotification((n) => {
      void webview.postMessage({ method: n.method, params: n.params });
      if (n.method === "done" || n.method === "cancelled") {
        this.postSessions(webview);
      }
    });

    this.agent?.onError((message) => {
      logger.appendLine(`[error] ${message}`);
      void webview.postMessage({
        method: "error",
        params: { session: this.sessionId, message },
      });
    });
  }

  private restartAgent(): void {
    this.disposeAgent();
    this.sessionId = makeSessionId();
  }

  private disposeAgent(): void {
    this.agent?.dispose();
    this.agent = null;
  }

  private postConfig(webview: Webview): void {
    const options = this.buildOptions();
    const config: WebviewConfigSnapshot = {
      model: options.model,
      baseUrl: options.baseUrl ?? "",
      demoMode: Boolean(options.demoMode),
      hasApiKey: Boolean(options.apiKey),
      workspaceName: options.workspaceName,
      workspacePath: options.workspace,
      hasWorkspace: options.hasWorkspace,
    };
    void webview.postMessage({ type: "config", config });
  }

  private postDependencyError(
    webview: Webview,
    message: string,
    pythonPath?: string,
  ): void {
    const options = this.buildOptions();
    const command = buildInstallCommand(
      pythonPath || options.pythonPath,
      options.requirementsPath,
    );
    void webview.postMessage({
      type: "dependencyStatus",
      message,
      command,
      canInstall: true,
    });
    void webview.postMessage({
      method: "error",
      params: { session: this.sessionId, message },
    });
  }

  private postOcrDependencyError(
    webview: Webview,
    message: string,
    pythonPath?: string,
  ): void {
    const options = this.buildOptions();
    const command = buildOcrInstallCommand(
      pythonPath || options.pythonPath,
      options.requirementsOcrPath,
    );
    void webview.postMessage({
      type: "dependencyStatus",
      message,
      command,
      canInstall: true,
      installType: "ocr",
      buttonLabel: "安装 OCR 依赖",
    });
    void webview.postMessage({
      type: "resumeError",
      message: `读取简历失败：${message}`,
    });
  }

  private buildHtml(webview: Webview): string {
    const nonce = getNonce();
    const stylesUri = webview.asWebviewUri(
      Uri.joinPath(this.htmlBasePath, "styles.css"),
    );
    const scriptUri = webview.asWebviewUri(
      Uri.joinPath(this.htmlBasePath, "main.js"),
    );
    const template = readFileSync(
      Uri.joinPath(this.htmlBasePath, "index.html").fsPath,
      "utf-8",
    );

    return template
      .replaceAll("${nonce}", nonce)
      .replaceAll("${cspSource}", webview.cspSource)
      .replaceAll("${stylesUri}", String(stylesUri))
      .replaceAll("${scriptUri}", String(scriptUri));
  }
}

interface ResumeParseOptions {
  onStatus?: (message: string) => void;
  onDebug?: (message: string) => void;
  ocr?: (filePath: string) => Promise<string>;
  pdfText?: (filePath: string) => Promise<string>;
  pdfTextFallback?: (filePath: string, reason: string) => Promise<string>;
  pdfTextTimeoutMs?: number;
}

interface ResumeOcrProgress {
  stage: string;
  message: string;
  elapsedMs: number;
  currentPage?: number;
  totalPages?: number;
}

interface ResumeOcrInput {
  filePath: string;
  pythonPath: string;
  scriptPath: string;
  pythonPathRoot: string;
  onProgress?: (progress: ResumeOcrProgress) => void;
}

interface ResumePdfTextInput {
  filePath: string;
  pythonPath: string;
  pythonPathRoot: string;
  reason: string;
}

/** 读取并解析简历附件，返回可注入首轮上下文的纯文本。 */
export async function parseResumeFile(
  filePath: string,
  options: ResumeParseOptions = {},
): Promise<ResumeParseResult> {
  const ext = extname(filePath).toLowerCase();
  const startedAt = Date.now();
  options.onDebug?.(
    `parseResumeFile start file=${basename(filePath)} ext=${ext || "unknown"} bytes=${safeStatSize(filePath)}`,
  );
  let raw = "";

  if ([".txt", ".md", ".markdown"].includes(ext)) {
    raw = readFileSync(filePath, "utf-8");
    options.onDebug?.(`text read chars=${raw.length} elapsedMs=${Date.now() - startedAt}`);
  } else if (ext === ".docx") {
    options.onDebug?.("docx import mammoth start");
    const mammoth = await import("mammoth");
    options.onDebug?.("docx extractRawText start");
    const result = await mammoth.extractRawText({ path: filePath });
    raw = result.value;
    options.onDebug?.(`docx extractRawText done chars=${raw.length} elapsedMs=${Date.now() - startedAt}`);
  } else if (ext === ".pdf") {
    try {
      raw = await readPdfTextLayer(filePath, options, startedAt);
    } catch (e) {
      const reason = e instanceof Error ? e.message : String(e);
      options.onDebug?.(`pdf primary text extraction failed reason=${reason}`);
      if (!options.pdfTextFallback) {
        throw e;
      }
      options.onStatus?.("正在使用备用 PDF 解析...");
      options.onDebug?.("pdf fallback text extractor start");
      raw = await options.pdfTextFallback(filePath, reason);
      options.onDebug?.(`pdf fallback text extractor done chars=${raw.length} elapsedMs=${Date.now() - startedAt}`);
    }
    if (!raw.trim()) {
      if (!options.ocr) {
        options.onDebug?.("pdf text empty and OCR handler missing");
        throw new Error(
          "扫描版 PDF 需要 OCR 依赖。请安装 OCR 依赖，或改用文本粘贴。",
        );
      }
      options.onStatus?.("正在识别扫描版 PDF...");
      options.onDebug?.("pdf text empty, OCR start");
      raw = await options.ocr(filePath);
      options.onDebug?.(`pdf OCR done chars=${raw.length} elapsedMs=${Date.now() - startedAt}`);
    }
  } else if (isSupportedResumeImageExt(ext)) {
    if (!options.ocr) {
      options.onDebug?.("image OCR handler missing");
      throw new Error("图片简历需要 OCR 依赖。请安装 OCR 依赖，或改用文本粘贴。");
    }
    options.onStatus?.("正在识别图片简历...");
    options.onDebug?.("image OCR start");
    raw = await options.ocr(filePath);
    options.onDebug?.(`image OCR done chars=${raw.length} elapsedMs=${Date.now() - startedAt}`);
  } else {
    options.onDebug?.(`unsupported file ext=${ext || "unknown"}`);
    throw new Error("当前只支持 .pdf、.docx、.txt、.md、.markdown、.png、.jpg、.jpeg、.webp 简历。");
  }

  const normalized = raw.trim();
  if (!normalized) {
    options.onDebug?.(`normalized text empty elapsedMs=${Date.now() - startedAt}`);
    throw new Error("未从简历附件中提取到文字内容。扫描版 PDF 或图片请改用文本粘贴。");
  }
  options.onDebug?.(
    `parseResumeFile done chars=${normalized.length} truncated=${normalized.length > RESUME_MAX_CHARS} elapsedMs=${Date.now() - startedAt}`,
  );

  return {
    fileName: basename(filePath),
    content: normalized.slice(0, RESUME_MAX_CHARS),
    truncated: normalized.length > RESUME_MAX_CHARS,
  };
}

async function readPdfTextLayer(
  filePath: string,
  options: ResumeParseOptions,
  startedAt: number,
): Promise<string> {
  const timeoutMs = options.pdfTextTimeoutMs ?? PDF_TEXT_PARSE_TIMEOUT_MS;
  if (options.pdfText) {
    options.onDebug?.("pdf custom text extractor start");
    const text = await withPdfTextTimeout(options.pdfText(filePath), timeoutMs);
    options.onDebug?.(`pdf custom text extractor done chars=${text.length} elapsedMs=${Date.now() - startedAt}`);
    return text;
  }

  options.onDebug?.("pdf import pdf-parse start");
  const pdfParse = await importPdfParse();
  options.onDebug?.("pdf read file start");
  const pdfBuffer = readFileSync(filePath);
  options.onDebug?.(`pdf read file done bytes=${pdfBuffer.length}`);
  options.onDebug?.("pdf text extraction start");
  const result = await withPdfTextTimeout(pdfParse(pdfBuffer), timeoutMs);
  options.onDebug?.(`pdf text extraction done chars=${result.text.length} elapsedMs=${Date.now() - startedAt}`);
  return result.text;
}

function withPdfTextTimeout<T>(promise: Promise<T>, timeoutMs: number): Promise<T> {
  let timer: NodeJS.Timeout | undefined;
  const timeout = new Promise<T>((_, reject) => {
    timer = setTimeout(() => {
      reject(new Error(`PDF 文字层解析超时（${timeoutMs}ms）`));
    }, timeoutMs);
  });
  return Promise.race([promise, timeout]).finally(() => {
    if (timer) {
      clearTimeout(timer);
    }
  });
}

export function withResumeParseTimeout<T>(
  promise: Promise<T>,
  timeoutMs = RESUME_PARSE_TIMEOUT_MS,
): Promise<T> {
  let timer: NodeJS.Timeout | undefined;
  const timeout = new Promise<T>((_, reject) => {
    timer = setTimeout(() => {
      reject(new Error("读取简历超时。请换用更小或更清晰的文件，或改用文本粘贴。"));
    }, timeoutMs);
  });
  return Promise.race([promise, timeout]).finally(() => {
    if (timer) {
      clearTimeout(timer);
    }
  });
}

function safeStatSize(filePath: string): string {
  try {
    return String(statSync(filePath).size);
  } catch {
    return "unknown";
  }
}

async function importPdfParse(): Promise<(data: Buffer) => Promise<{ text: string }>> {
  const mod = await import("pdf-parse/lib/pdf-parse.js");
  return (mod.default ?? mod) as unknown as (data: Buffer) => Promise<{ text: string }>;
}

export function buildInstallCommand(pythonPath: string, requirementsPath: string): string {
  if (process.platform === "win32") {
    return `& ${quotePowerShell(pythonPath)} -m pip install -r ${quotePowerShell(requirementsPath)}`;
  }
  return `${quoteShell(pythonPath)} -m pip install -r ${quoteShell(requirementsPath)}`;
}

export function buildOcrInstallCommand(
  pythonPath: string,
  requirementsOcrPath: string,
): string {
  return buildInstallCommand(pythonPath, requirementsOcrPath);
}

function quotePowerShell(value: string): string {
  return `"${value.replaceAll('"', '`"')}"`;
}

function quoteShell(value: string): string {
  return `'${value.replaceAll("'", "'\\''")}'`;
}

export async function runResumePdfText(input: ResumePdfTextInput): Promise<string> {
  const { execFile } = await import("child_process");
  const startedAt = Date.now();
  const logger = getDebugChannel();
  logger.appendLine(
    `[pdf-text] fallback start file=${basename(input.filePath)} reason=${input.reason}`,
  );
  const code = [
    "import sys",
    "path = sys.argv[1]",
    "try:",
    "    import pymupdf as fitz",
    "except ModuleNotFoundError:",
    "    try:",
    "        import fitz",
    "    except ModuleNotFoundError:",
    "        sys.stderr.write('PDF 备用解析依赖未安装：PyMuPDF。请点击“安装 OCR 依赖”后重试。')",
    "        raise SystemExit(3)",
    "max_chars = 100000",
    "parts = []",
    "total = 0",
    "with fitz.open(path) as doc:",
    "    for page in doc:",
    "        text = page.get_text()",
    "        parts.append(text)",
    "        total += len(text)",
    "        if total >= max_chars:",
    "            break",
    "sys.stdout.write('\\n'.join(parts)[:max_chars])",
  ].join("\n");

  return new Promise((resolve, reject) => {
    const env = { ...process.env };
    const existing = env.PYTHONPATH ?? "";
    env.PYTHONPATH = existing
      ? `${input.pythonPathRoot}${process.platform === "win32" ? ";" : ":"}${existing}`
      : input.pythonPathRoot;
    env.PYTHONIOENCODING = "utf-8";

    execFile(
      input.pythonPath,
      ["-c", code, input.filePath],
      {
        encoding: "utf-8",
        env,
        timeout: PDF_TEXT_FALLBACK_TIMEOUT_MS,
        windowsHide: true,
        maxBuffer: RESUME_MAX_CHARS * 8,
      },
      (error, stdout, stderr) => {
        const elapsedMs = Date.now() - startedAt;
        const detail = stderr.trim();
        if (detail) {
          logger.appendLine(`[pdf-text] stderr ${detail}`);
        }
        if (error) {
          logger.appendLine(`[pdf-text] failed elapsedMs=${elapsedMs}`);
          reject(new Error(
            `PDF 备用解析失败。${detail ? `\n${detail}` : `\n${error.message}`}`,
          ));
          return;
        }
        const text = stdout.trim();
        logger.appendLine(`[pdf-text] success chars=${text.length} elapsedMs=${elapsedMs}`);
        resolve(text);
      },
    );
  });
}

async function runResumeOcr(input: ResumeOcrInput): Promise<string> {
  const { spawn } = await import("child_process");
  return new Promise((resolve, reject) => {
    const env = { ...process.env };
    const startedAt = Date.now();
    const logger = getDebugChannel();
    const fileType = extname(input.filePath).toLowerCase() || "unknown";
    logger.appendLine(`[ocr] start file=${basename(input.filePath)} type=${fileType}`);
    const existing = env.PYTHONPATH ?? "";
    env.PYTHONPATH = existing
      ? `${input.pythonPathRoot}${process.platform === "win32" ? ";" : ":"}${existing}`
      : input.pythonPathRoot;
    // OCR 结果含中文，强制 Python stdout 用 UTF-8（脚本内也有 reconfigure 兜底）
    env.PYTHONIOENCODING = "utf-8";

    const child = spawn(
      input.pythonPath,
      [input.scriptPath, input.filePath],
      {
        env,
        windowsHide: true,
      },
    );
    let stdout = "";
    let stderrBuffer = "";
    const stderrLines: string[] = [];
    let settled = false;
    let timedOut = false;
    const timeout = setTimeout(() => {
      timedOut = true;
      child.kill();
    }, 120_000);

    const finishReject = (message: string): void => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(timeout);
      reject(new Error(message));
    };
    const handleStderrLine = (line: string): void => {
      const progress = parseResumeOcrProgressLine(line);
      if (progress) {
        logger.appendLine(
          `[ocr] progress stage=${progress.stage} page=${progress.currentPage ?? "-"}`
            + `/${progress.totalPages ?? "-"} elapsedMs=${progress.elapsedMs}`,
        );
        input.onProgress?.(progress);
        return;
      }
      if (line.trim()) {
        stderrLines.push(line.trim());
      }
    };

    child.stdout.setEncoding("utf-8");
    child.stderr.setEncoding("utf-8");
    child.stdout.on("data", (chunk) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk) => {
      stderrBuffer += chunk;
      const lines = stderrBuffer.split(/\r?\n/);
      stderrBuffer = lines.pop() || "";
      lines.forEach(handleStderrLine);
    });
    child.on("error", (error) => {
      logger.appendLine(`[ocr] failed type=${fileType} elapsedMs=${Date.now() - startedAt}`);
      finishReject(`OCR 识别失败。请先安装 OCR 依赖，或改用文本粘贴。\n${error.message}`);
    });
    child.on("close", (code) => {
      if (settled) {
        return;
      }
      if (stderrBuffer.trim()) {
        handleStderrLine(stderrBuffer);
      }
      const elapsedMs = Date.now() - startedAt;
      clearTimeout(timeout);
      if (timedOut) {
        logger.appendLine(`[ocr] failed type=${fileType} elapsedMs=${elapsedMs}`);
        reject(new Error("OCR 识别超时。请改用更清晰或页数更少的文件，或改用文本粘贴。"));
        return;
      }
      if (code !== 0) {
        logger.appendLine(`[ocr] failed type=${fileType} elapsedMs=${elapsedMs}`);
        const detail = stderrLines.join("\n").trim();
        reject(new Error(
          `OCR 识别失败。请先安装 OCR 依赖，或改用文本粘贴。${detail ? `\n${detail}` : ""}`,
        ));
        return;
      }
      const text = stdout.trim();
      if (!text) {
        logger.appendLine(`[ocr] empty type=${fileType} elapsedMs=${elapsedMs}`);
        reject(new Error("OCR 未识别到文字。请改用文本粘贴。"));
        return;
      }
      logger.appendLine(`[ocr] success type=${fileType} chars=${text.length} elapsedMs=${elapsedMs}`);
      resolve(text);
    });
  });
}

export function parseResumeOcrProgressLine(line: string): ResumeOcrProgress | null {
  try {
    const parsed = JSON.parse(line) as Partial<ResumeOcrProgress> & { kind?: string };
    if (parsed.kind !== "ocr_progress" || typeof parsed.message !== "string") {
      return null;
    }
    return {
      stage: typeof parsed.stage === "string" ? parsed.stage : "progress",
      message: parsed.message,
      elapsedMs: typeof parsed.elapsedMs === "number" ? parsed.elapsedMs : 0,
      currentPage: typeof parsed.currentPage === "number" ? parsed.currentPage : undefined,
      totalPages: typeof parsed.totalPages === "number" ? parsed.totalPages : undefined,
    };
  } catch {
    return null;
  }
}

async function runModelConnectionTest(input: ModelTestInput): Promise<ModelTestResult> {
  const { execFile } = await import("child_process");
  const code = [
    "import json, os, sys",
    "from agent.llm_client import test_model_connection",
    "result = test_model_connection(api_key=os.environ.get('INTERVIEW_TEST_API_KEY', ''), model=sys.argv[1], base_url=sys.argv[2] or None)",
    "print(json.dumps(result, ensure_ascii=False))",
  ].join("; ");

  return new Promise((resolve, reject) => {
    const env = { ...process.env };
    const existing = env.PYTHONPATH ?? "";
    env.PYTHONPATH = existing
      ? `${input.pythonPathRoot}${process.platform === "win32" ? ";" : ":"}${existing}`
      : input.pythonPathRoot;
    env.PYTHONIOENCODING = "utf-8";
    env.INTERVIEW_TEST_API_KEY = input.apiKey;

    execFile(
      input.pythonPath,
      ["-c", code, input.model, input.baseUrl || ""],
      {
        encoding: "utf-8",
        env,
        timeout: 30_000,
        windowsHide: true,
        maxBuffer: 1024 * 1024,
      },
      (error, stdout, stderr) => {
        if (error) {
          reject(new Error(
            `模型连接测试失败。请检查 interview.baseUrl、interview.model 和网络。${stderr ? `\n${stderr.trim()}` : ""}`,
          ));
          return;
        }
        try {
          resolve(JSON.parse(stdout.trim()) as ModelTestResult);
        } catch {
          reject(new Error("模型连接测试返回了无法解析的结果。"));
        }
      },
    );
  });
}

function isOcrDependencyError(message: string): boolean {
  return message.includes("OCR 依赖")
    || message.includes("PyMuPDF")
    || message.includes("rapidocr")
    || message.includes("onnxruntime");
}

export function getResumeFilePathFromTabInput(input: unknown): string {
  const uri = (input as { uri?: Uri } | undefined)?.uri;
  if (uri?.scheme !== "file" || !isSupportedResumeFilePath(uri.fsPath)) {
    return "";
  }
  return uri.fsPath;
}

function isSupportedResumeFilePath(filePath: string): boolean {
  const ext = extname(filePath).toLowerCase();
  return [".pdf", ".docx", ".txt", ".md", ".markdown"].includes(ext)
    || isSupportedResumeImageExt(ext);
}

function isSupportedResumeImageExt(ext: string): boolean {
  return [".png", ".jpg", ".jpeg", ".webp"].includes(ext);
}

function safeUploadedFileName(fileName: string): string {
  return basename(fileName || "resume.txt")
    .replace(/[<>:"/\\|?*\x00-\x1F]/g, "-")
    .replace(/\s+/g, "-") || "resume.txt";
}

function stripDataUrlPrefix(value: string): string {
  const comma = value.indexOf(",");
  return value.startsWith("data:") && comma >= 0 ? value.slice(comma + 1) : value;
}

function sessionsDir(workspacePath: string): string {
  return join(workspacePath, ".sessions");
}

function safeSessionId(session: string): string {
  return Array.from(session).filter((c) => /[a-zA-Z0-9_-]/.test(c)).join("");
}

function sessionPath(workspacePath: string, session: string): string {
  return join(sessionsDir(workspacePath), `${safeSessionId(session)}.json`);
}

function listSessionSummaries(workspacePath: string): SessionSummary[] {
  if (!workspacePath) {
    return [];
  }
  const dir = sessionsDir(workspacePath);
  if (!existsSync(dir)) {
    return [];
  }
  return readdirSync(dir)
    .filter((name) => name.endsWith(".json"))
    .map((name) => readSessionSummary(dir, name))
    .filter((item): item is SessionSummary => Boolean(item))
    .sort((a, b) => b.updatedAt - a.updatedAt);
}

function readSessionSummary(dir: string, fileName: string): SessionSummary | null {
  try {
    const path = join(dir, fileName);
    const messages = JSON.parse(readFileSync(path, "utf-8"));
    if (!Array.isArray(messages)) {
      return null;
    }
    const display = toDisplayMessages(messages);
    const last = display.at(-1);
    const stat = statSync(path);
    const id = fileName.slice(0, -".json".length);
    return {
      id,
      title: makeSessionTitle(display, id),
      updatedAt: stat.mtimeMs,
      messageCount: display.length,
      preview: (last?.content || "").slice(0, 80),
    };
  } catch {
    return null;
  }
}

function loadSessionMessages(workspacePath: string, session: string): DisplayMessage[] {
  try {
    const path = sessionPath(workspacePath, session);
    const messages = JSON.parse(readFileSync(path, "utf-8"));
    return Array.isArray(messages) ? toDisplayMessages(messages) : [];
  } catch {
    return [];
  }
}

function deleteSessionFile(workspacePath: string, session: string): void {
  if (!workspacePath) {
    return;
  }
  rmSync(sessionPath(workspacePath, session), { force: true });
}

function toDisplayMessages(messages: Array<Record<string, unknown>>): DisplayMessage[] {
  return messages
    .filter((message) => message.role === "user" || message.role === "assistant")
    .map((message) => ({
      role: message.role as "user" | "assistant",
      content: String(message.content || ""),
    }))
    .filter((message) => message.content.trim());
}

export function makeSessionTitle(messages: DisplayMessage[], fallback: string): string {
  const firstUser = messages.find((message) => message.role === "user");
  const content = firstUser?.content.trim();
  if (!content) {
    return fallback;
  }
  const jd = extractSection(content, "岗位 JD", ["简历", "当前项目", "项目路径"]);
  const project = extractProjectName(content);
  const techs = extractTechKeywords(content);
  const candidates = [
    extractJobTitle(jd),
    firstMeaningfulLine(jd),
    buildProjectTechTitle(project, techs),
    project,
    firstMeaningfulLine(content),
  ];
  for (const candidate of candidates) {
    const title = normalizeTitle(candidate);
    if (title) {
      return title.slice(0, 32);
    }
  }
  return fallback;
}

export function canExportReport(messages: DisplayMessage[]): boolean {
  return messages.some((message) => message.role === "user" && message.content.trim());
}

export function sanitizeReportFileName(title: string): string {
  return normalizeTitle(title)
    .replace(/[<>:"/\\|?*\x00-\x1F]/g, "-")
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-")
    .replace(/^-|-$/g, "")
    .slice(0, 60) || "interview-report";
}

export function generateMarkdownReport(input: ReportInput): string {
  const firstUser = input.messages.find((message) => message.role === "user")?.content || "";
  const jd = extractSection(firstUser, "岗位 JD", ["简历", "当前项目", "项目路径"]);
  const resume = extractSection(firstUser, "简历", ["当前项目", "项目路径"]);
  const project = extractProjectName(firstUser) || input.workspaceName || "未命名项目";
  const allText = input.messages.map((message) => message.content).join("\n");
  const techs = extractTechKeywords(allText);
  const userAnswers = input.messages.filter((message) => message.role === "user").slice(1);
  const assistantQuestions = input.messages.filter((message) => message.role === "assistant");

  return [
    `# ${input.title}`,
    "",
    "## 基本信息",
    "",
    `- 导出时间：${input.createdAt.toLocaleString()}`,
    `- 会话 ID：${input.sessionId}`,
    `- 当前项目：${project}`,
    `- 项目路径：${input.workspacePath}`,
    "",
    "## JD 摘要",
    "",
    formatBlock(jd || "未提供明确 JD。"),
    "",
    "## 项目摘要",
    "",
    `- 项目：${project}`,
    `- 识别到的技术点：${techs.length ? techs.join("、") : "待根据后续回答补充"}`,
    resume ? `- 简历摘要：${truncate(resume, 300)}` : "- 简历摘要：未提供或仅在附件中提交。",
    "",
    "## 考察技术点",
    "",
    formatBulletList(techs.length ? techs : ["项目结构", "技术选型", "实现权衡"]),
    "",
    "## 回答表现",
    "",
    userAnswers.length
      ? formatBulletList(userAnswers.slice(-5).map((message) => truncate(message.content, 180)))
      : "- 本次会话还没有正式回答轮次。",
    "",
    "## 薄弱点",
    "",
    formatBulletList(buildWeaknesses(userAnswers, techs)),
    "",
    "## 复习建议",
    "",
    formatBulletList(buildReviewSuggestions(techs)),
    "",
    "## 对话摘要",
    "",
    formatBulletList(
      assistantQuestions.slice(-5).map((message) => `面试官追问：${truncate(message.content, 180)}`),
    ),
    "",
  ].join("\n");
}

function extractSection(content: string, label: string, stopLabels: string[]): string {
  const start = content.match(new RegExp(`${escapeRegex(label)}：\\s*`, "m"));
  if (!start || start.index === undefined) {
    return "";
  }
  const from = start.index + start[0].length;
  const rest = content.slice(from);
  const stopPattern = new RegExp(`\\n(?:${stopLabels.map(escapeRegex).join("|")})：`, "m");
  const stop = rest.search(stopPattern);
  return (stop >= 0 ? rest.slice(0, stop) : rest).trim();
}

function extractJobTitle(jd: string): string {
  for (const line of jd.split(/\r?\n/).slice(0, 8)) {
    const cleaned = normalizeTitle(line.replace(/^[-*+\d.)\s]+/, ""));
    if (!cleaned) {
      continue;
    }
    const labeled = cleaned.match(/^(?:岗位名称|岗位|职位名称|职位|招聘岗位)[:：]\s*(.+)$/);
    if (labeled?.[1]) {
      return labeled[1];
    }
    if (/(工程师|开发|实习|算法|后端|前端|全栈|测试|运维|Agent|AI|LLM)/i.test(cleaned)) {
      return cleaned;
    }
  }
  return "";
}

function extractProjectName(content: string): string {
  const match = content.match(/当前项目：([^\r\n]+)/);
  return match?.[1]?.trim() || "";
}

function extractTechKeywords(content: string): string[] {
  const techs = [
    "Python", "TypeScript", "JavaScript", "Java", "Spring", "Vue", "React",
    "Node", "FastAPI", "Django", "Flask", "Redis", "MySQL", "PostgreSQL",
    "Docker", "Kubernetes", "LangChain", "RAG", "LLM", "OCR",
  ];
  const lower = content.toLowerCase();
  return techs.filter((tech) => lower.includes(tech.toLowerCase()));
}

function buildProjectTechTitle(project: string, techs: string[]): string {
  if (!project || !techs.length) {
    return "";
  }
  const full = `${project} · ${techs.slice(0, 2).join("/")}`;
  if (full.length <= 32) {
    return full;
  }
  return `${project} · ${techs[0]}`;
}

function firstMeaningfulLine(content: string): string {
  return content
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find((line) => line && line !== "我们开始一场技术面试。") || "";
}

function normalizeTitle(value: string | undefined): string {
  return (value || "").replace(/\s+/g, " ").trim();
}

function formatBlock(value: string): string {
  return truncate(value, 600)
    .split(/\r?\n/)
    .map((line) => `> ${line}`)
    .join("\n");
}

function formatBulletList(items: string[]): string {
  return items.map((item) => `- ${item}`).join("\n");
}

function buildWeaknesses(userAnswers: DisplayMessage[], techs: string[]): string[] {
  if (userAnswers.length < 2) {
    return ["回答轮次偏少，建议继续补充项目背景、关键实现和取舍依据。"];
  }
  return [
    "逐项复盘回答中没有展开的原理、边界条件和失败场景。",
    techs.length
      ? `优先补强 ${techs.slice(0, 3).join("、")} 的底层机制和项目落地细节。`
      : "补充项目技术栈、核心模块和关键难点的可验证细节。",
  ];
}

function buildReviewSuggestions(techs: string[]): string[] {
  const focus = techs.slice(0, 4);
  return [
    focus.length
      ? `围绕 ${focus.join("、")} 各准备一个“原理 - 项目用法 - 踩坑 - 优化”的回答。`
      : "先整理项目的核心技术栈，再为每个技术点准备原理和项目用法。",
    "把回答压缩成 2 分钟版本，再准备一个可继续深入的细节版本。",
    "针对薄弱点补一次追问练习，重点验证是否能讲清取舍和边界。",
  ];
}

function truncate(value: string, max: number): string {
  const text = value.trim();
  return text.length > max ? `${text.slice(0, max)}...` : text;
}

function formatReportTimestamp(date: Date): string {
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}`;
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function getNonce(): string {
  return randomBytes(16).toString("base64");
}

function makeSessionId(): string {
  return `vscode-${Date.now()}`;
}
