/**
 * 插件激活入口（设计第 5、6 节）。
 *
 * 注册 interview.start / interview.askAboutSelection 命令：
 * - interview.start：打开面试面板，spawn Python，接通双向通信
 * - interview.askAboutSelection：对当前选中代码发起提问（设计第 5.3.3 节）
 *
 * 配置读取：apiKey / model / baseUrl / resume / pythonPath（命名空间 interview.*）。
 */

import * as path from "path";
import { existsSync } from "fs";
import {
  commands,
  ConfigurationTarget,
  ExtensionContext,
  Uri,
  window,
  workspace,
} from "vscode";
import {
  InterviewViewProvider,
  PanelOptions,
  WebviewConfigUpdate,
} from "./webviewPanel";

/** interview.* 配置的强类型读取。 */
interface InterviewConfig {
  apiKey: string;
  model: string;
  baseUrl: string;
  resume: string;
  pythonPath: string;
  demoMode: boolean;
  maxSteps: number;
  maxHistoryTokens: number;
  maxKeptFull: number;
  agentRuntime: "native" | "langchain";
}

function readConfig(): InterviewConfig {
  const cfg = workspace.getConfiguration("interview");
  return {
    apiKey: cfg.get<string>("apiKey", ""),
    model: cfg.get<string>("model", "gpt-4o-mini"),
    baseUrl: cfg.get<string>("baseUrl", ""),
    resume: cfg.get<string>("resume", ""),
    pythonPath: cfg.get<string>("pythonPath", "python"),
    demoMode: cfg.get<boolean>("demoMode", false),
    maxSteps: cfg.get<number>("maxSteps", 8),
    maxHistoryTokens: cfg.get<number>("maxHistoryTokens", 20000),
    maxKeptFull: cfg.get<number>("maxKeptFull", 3),
    agentRuntime: cfg.get<"native" | "langchain">("agentRuntime", "native"),
  };
}

function resolveAgentRoot(context: ExtensionContext): string {
  const bundledRoot = Uri.joinPath(context.extensionUri, "bundled-agent").fsPath;
  const bundledMain = path.join(bundledRoot, "agent", "main.py");
  if (existsSync(bundledMain)) {
    return bundledRoot;
  }

  // 开发调试时 bundled-agent 可能还没生成，保留源码目录兜底。
  return Uri.joinPath(context.extensionUri, "..").fsPath;
}

function buildPanelOptions(context: ExtensionContext): PanelOptions {
  const cfg = readConfig();
  const agentRoot = resolveAgentRoot(context);
  const intervieweeProject = workspace.workspaceFolders?.[0];

  return {
    pythonPath: cfg.pythonPath,
    vscodePythonPath: workspace
      .getConfiguration("python")
      .get<string>("defaultInterpreterPath", ""),
    scriptPath: path.join(agentRoot, "agent", "main.py"),
    requirementsPath: path.join(agentRoot, "requirements-agent.txt"),
    requirementsOcrPath: path.join(agentRoot, "requirements-ocr.txt"),
    workspace: intervieweeProject?.uri.fsPath ?? "",
    workspaceName: intervieweeProject?.name ?? "",
    hasWorkspace: Boolean(intervieweeProject),
    pythonPathRoot: agentRoot,
    apiKey: cfg.apiKey,
    model: cfg.model,
    baseUrl: cfg.baseUrl || undefined,
    resume: cfg.resume || undefined,
    demoMode: cfg.demoMode,
    maxSteps: cfg.maxSteps,
    maxHistoryTokens: cfg.maxHistoryTokens,
    maxKeptFull: cfg.maxKeptFull,
    agentRuntime: cfg.agentRuntime,
  };
}

async function saveWebviewConfig(config: WebviewConfigUpdate): Promise<void> {
  const cfg = workspace.getConfiguration("interview");
  const target = ConfigurationTarget.Global;
  if (typeof config.model === "string") {
    await cfg.update("model", config.model.trim() || "gpt-4o-mini", target);
  }
  if (typeof config.baseUrl === "string") {
    await cfg.update("baseUrl", config.baseUrl.trim(), target);
  }
  if (typeof config.apiKey === "string" && config.apiKey.trim()) {
    await cfg.update("apiKey", config.apiKey.trim(), target);
  }
  if (typeof config.demoMode === "boolean") {
    await cfg.update("demoMode", config.demoMode, target);
  }
}

export function activate(context: ExtensionContext): void {
  console.log("[Interview Agent] activate");

  const htmlBasePath = Uri.joinPath(context.extensionUri, "media");
  const provider = new InterviewViewProvider(
    htmlBasePath,
    () => buildPanelOptions(context),
    saveWebviewConfig,
  );

  // ───────── interview.start ─────────
  const startCmd = commands.registerCommand("interview.start", () => {
    provider.focus();
  });

  // ───────── interview.askAboutSelection ─────────
  // 对选中代码提问：填入输入框后自动发送（复用 start 打开的面板逻辑）
  const askCmd = commands.registerCommand(
    "interview.askAboutSelection",
    () => {
      provider.prefillSelectionQuestion();
    },
  );
  const captureDroppedResumeTab = window.tabGroups.onDidChangeTabs((event) => {
    for (const tab of event.opened) {
      provider.captureOpenedResumeTab(tab);
    }
  });

  context.subscriptions.push(
    window.registerWebviewViewProvider("interview.chatView", provider, {
      webviewOptions: { retainContextWhenHidden: true },
    }),
    startCmd,
    askCmd,
    captureDroppedResumeTab,
  );
}

export function deactivate(): void {
  // Webview 关闭时 InterviewPanel 自己 dispose AgentClient，无需额外清理
}

// 配置变更类型导出（供未来热重载用）
export type { ConfigurationTarget };
