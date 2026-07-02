/**
 * Webview 面板管理（设计第 5.5 节两跳中转 + 第 5.3.3 节选中代码注入）。
 *
 * 这是 Extension Host 里的"中转站"：
 * - Webview → Host（chat/stop 消息）→ 转成 Request 发给 Python
 * - Python → Host（stream/tool_call/done/error 通知）→ postMessage 给 Webview
 *
 * Host 不放业务逻辑（设计第 5.5 节），只做翻译和转发。
 */

import { randomBytes } from "crypto";
import {
  OutputChannel,
  Uri,
  ViewColumn,
  Webview,
  WebviewView,
  window,
} from "vscode";
import { AgentClient } from "./agentClient";
import { buildChat, buildStop } from "./protocol";

/** 共享的调试输出通道（整个插件一个，所有面板的 Python 日志都写这里）。 */
let debugChannel: OutputChannel | null = null;
function getDebugChannel(): OutputChannel {
  if (!debugChannel) {
    debugChannel = window.createOutputChannel("Interview Agent", { log: true });
  }
  return debugChannel;
}

/** 从配置构造 AgentClient 需要的参数。 */
export interface PanelOptions {
  pythonPath: string;
  scriptPath: string;
  /** 被面试的项目根（工具翻代码的根）。 */
  workspace: string;
  /** agent 包所在根（PYTHONPATH 用），通常 = 仓库根。 */
  pythonPathRoot: string;
  apiKey: string;
  model: string;
  baseUrl?: string;
  resume?: string;
  /** 演示模式：用 FakeLLM，零费用（设计第 5E 节冒烟）。 */
  demoMode?: boolean;
  // 调优参数（Phase 7-D 可配化，可选）
  maxSteps?: number;
  maxHistoryTokens?: number;
  maxKeptFull?: number;
  /** 历史落盘目录（#3）：插件数据目录，透传给 Python。 */
  storageDir?: string;
}

export class InterviewPanel {
  private agent: AgentClient;
  private readonly sessionId: string;

  constructor(
    private readonly htmlBasePath: Uri,
    options: PanelOptions,
  ) {
    // 每个 Webview 一个独立会话 id（Python 侧据此隔离历史）
    this.sessionId = `vscode-${Date.now()}`;

    this.agent = new AgentClient({
      pythonPath: options.pythonPath,
      scriptPath: options.scriptPath,
      workspace: options.workspace,
      pythonPathRoot: options.pythonPathRoot,
      apiKey: options.apiKey,
      model: options.model,
      baseUrl: options.baseUrl,
      resume: options.resume,
      session: this.sessionId,
      demoMode: options.demoMode,
      maxSteps: options.maxSteps,
      maxHistoryTokens: options.maxHistoryTokens,
      maxKeptFull: options.maxKeptFull,
      storageDir: options.storageDir,
    });
  }

  /** 打开 Webview 面板，spawn Python，接通双向通信。 */
  open(): void {
    const panel = window.createWebviewPanel(
      "interviewAgent",
      "Interview Agent",
      // 在活动编辑器列打开，没有活动编辑器则用当前活动列
      window.activeTextEditor?.viewColumn ?? ViewColumn.Active,
      {
        enableScripts: true,
        retainContextWhenHidden: true,
        localResourceRoots: [this.htmlBasePath],
      },
    );

    panel.webview.html = this.buildHtml(panel.webview);

    this.wireMessages(panel.webview);
    this.wireAgent(panel.webview);

    panel.onDidDispose(() => {
      this.agent.dispose();
    });

    // 启动 Python 子进程（内部自动发 init）
    this.agent.start();
  }

  // ──────────────────────────────────────────────
  // Webview → Host：用户的 chat/stop
  // ──────────────────────────────────────────────

  private wireMessages(webview: Webview): void {
    webview.onDidReceiveMessage((msg: WebviewToHostMessage) => {
      if (msg.type === "chat") {
        const attached = this.readSelection();
        this.agent.send(
          buildChat({
            session: this.sessionId,
            text: msg.text,
            attached_code: attached,
          }),
        );
      } else if (msg.type === "stop") {
        this.agent.send(buildStop(this.sessionId));
      }
    });
  }

  /** 读当前编辑器选中的代码（设计第 5.3.3 节）。 */
  private readSelection(): { file: string; content: string } | undefined {
    const editor = window.activeTextEditor;
    if (!editor) {
      return undefined;
    }
    const selection = editor.selection;
    if (selection.isEmpty) {
      return undefined;
    }
    const text = editor.document.getText(selection);
    if (!text.trim()) {
      return undefined;
    }
    return {
      file: editor.document.fileName,
      content: text,
    };
  }

  // ──────────────────────────────────────────────
  // Python → Host → Webview：通知转发
  // ──────────────────────────────────────────────

  private wireAgent(webview: Webview): void {
    const logger = getDebugChannel();

    // 诊断日志：Python 的 stderr、spawn/exit 事件都写进 OutputChannel
    // 这是排查"发消息没反应"的关键——能看到 Python 到底起没起来、报什么错
    this.agent.onLog((message) => {
      logger.appendLine(message);
    });

    this.agent.onNotification((n) => {
      // 透传给前端：通知原样 postMessage（method + params 结构不变）
      void webview.postMessage({ method: n.method, params: n.params });
    });

    this.agent.onError((message) => {
      // 错误同时记日志（诊断）和推给前端（红色气泡）
      logger.appendLine(`[error] ${message}`);
      void webview.postMessage({
        method: "error",
        params: { session: this.sessionId, message },
      });
    });
  }

  // ──────────────────────────────────────────────
  // HTML 构造 + CSP（设计第 5.5 节 webview 安全）
  // ──────────────────────────────────────────────

  private buildHtml(webview: Webview): string {
    const nonce = getNonce();
    const stylesUri = webview.asWebviewUri(
      Uri.joinPath(this.htmlBasePath, "styles.css"),
    );
    const scriptUri = webview.asWebviewUri(
      Uri.joinPath(this.htmlBasePath, "main.js"),
    );

    // 读 index.html 模板，替换占位符
    // 注：fs 在 extension 上下文可用，这里用同步读取（启动期，可接受）
    // 为避免引入额外依赖，HTML 模板内联在此构造（含 CSP 占位符替换）
    return `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src ${webview.cspSource} 'nonce-${nonce}'; script-src 'nonce-${nonce}';" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Interview Agent</title>
  <link rel="stylesheet" nonce="${nonce}" href="${stylesUri}" />
</head>
<body>
  <div id="app">
    <div id="messages" class="messages"></div>
    <div class="composer">
      <textarea id="input" class="composer__input" placeholder="和面试官聊聊你的项目…（Enter 发送，Shift+Enter 换行）" rows="4"></textarea>
      <button id="send" class="composer__send" title="发送" aria-label="发送">↑</button>
      <button id="stop" class="composer__stop" title="停止" aria-label="停止" disabled>■</button>
    </div>
  </div>
  <script nonce="${nonce}" src="${scriptUri}"></script>
</body>
</html>`;
  }
}

// ──────────────────────────────────────────────
// 辅助
// ──────────────────────────────────────────────

/** Webview 发给 Host 的消息。 */
type WebviewToHostMessage =
  | { type: "chat"; text: string }
  | { type: "stop" };

/** 生成 CSP nonce（16 字节十六进制）。 */
function getNonce(): string {
  return randomBytes(16).toString("base64");
}

// 保留 WebviewView 类型引用，便于未来改成侧边栏视图（设计第 5.2 节）
export type { WebviewView };
