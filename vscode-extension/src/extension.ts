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
import { readApiKey, saveApiKey } from "./credentials";
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
  piMaxSteps: number;
  agentRuntime: "native" | "langchain" | "pi";
  enabledTools: string[];
  compactionEnabled: boolean;
  compactionTriggerTokens: number;
  compactionKeepMessages: number;
}

function readConfig(): InterviewConfig {
  const cfg = workspace.getConfiguration("interview");
  return {
    apiKey: "", // 密钥只在模型请求前异步从 SecretStorage 读取。
    model: cfg.get<string>("model", "gpt-4o-mini"),
    baseUrl: cfg.get<string>("baseUrl", ""),
    resume: cfg.get<string>("resume", ""),
    pythonPath: cfg.get<string>("pythonPath", "python"),
    demoMode: cfg.get<boolean>("demoMode", false),
    maxSteps: cfg.get<number>("maxSteps", 8),
    maxHistoryTokens: cfg.get<number>("maxHistoryTokens", 20000),
    maxKeptFull: cfg.get<number>("maxKeptFull", 3),
    piMaxSteps: cfg.get<number>("piMaxSteps", 32),
    agentRuntime: cfg.get<"native" | "langchain" | "pi">("agentRuntime", "native"),
    enabledTools: cfg.get<string[]>("enabledTools", [
      "list_directory",
      "search_code",
      "read_file",
      "lookup_questions",
    ]),
    compactionEnabled: cfg.get<boolean>("compactionEnabled", false),
    compactionTriggerTokens: cfg.get<number>("compactionTriggerTokens", 12000),
    compactionKeepMessages: cfg.get<number>("compactionKeepMessages", 6),
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
    piMaxSteps: cfg.piMaxSteps,
    agentRuntime: cfg.agentRuntime,
    enabledTools: cfg.enabledTools,
    compactionEnabled: cfg.compactionEnabled,
    compactionTriggerTokens: cfg.compactionTriggerTokens,
    compactionKeepMessages: cfg.compactionKeepMessages,
  };
}

/** 保存普通配置及端点绑定的密钥；不删除用户旧设置。 */
export async function saveWebviewConfig(
  context: ExtensionContext, config: WebviewConfigUpdate,
): Promise<void> {
  const cfg = workspace.getConfiguration("interview");
  const target = ConfigurationTarget.Global;
  // 先保存凭证，失败时不把普通配置切到新服务；不自动删除旧明文设置。
  if (typeof config.apiKey === "string" && config.apiKey.trim()) {
    await saveApiKey(context.secrets,
      config.baseUrl ?? cfg.get<string>("baseUrl", ""), config.apiKey);
  }
  if (typeof config.model === "string") {
    await cfg.update("model", config.model.trim() || "gpt-4o-mini", target);
  }
  if (typeof config.baseUrl === "string") {
    await cfg.update("baseUrl", config.baseUrl.trim(), target);
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
    (config) => saveWebviewConfig(context, config),
    (baseUrl) => readApiKey(context.secrets, baseUrl,
      workspace.getConfiguration("interview").get<string>("apiKey", "")),
  );

  // ───────── interview.start ─────────
  const startCmd = commands.registerCommand("interview.start", () => {
    provider.focus();
  });

  const keyCmd = commands.registerCommand("interview.setApiKey", async () => {
    const baseUrl = workspace.getConfiguration("interview").get<string>("baseUrl", "");
    const apiKey = await window.showInputBox({
      title: "设置 API Key（加密保存）", password: true, ignoreFocusOut: true,
      prompt: `用于当前服务：${baseUrl || "OpenAI 默认端点"}。切换服务后需重新设置。`,
    });
    if (!apiKey?.trim()) { return; }
    try {
      await saveWebviewConfig(context, { apiKey, baseUrl });
      provider.refreshConfigFromSettings();
      void window.showInformationMessage("API Key 已加密保存。旧设置中的明文值可在确认后自行删除。");
    } catch {
      void window.showErrorMessage("API Key 加密保存失败，请检查凭证服务后重试。");
    }
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
  const configChanged = workspace.onDidChangeConfiguration((event) => {
    if (event.affectsConfiguration("interview")) {
      provider.refreshConfigFromSettings();
    }
  });

  context.subscriptions.push(
    window.registerWebviewViewProvider("interview.chatView", provider, {
      webviewOptions: { retainContextWhenHidden: true },
    }),
    startCmd,
    keyCmd,
    askCmd,
    captureDroppedResumeTab,
    configChanged,
  );
}

export function deactivate(): void {
  // Webview 关闭时 InterviewPanel 自己 dispose AgentClient，无需额外清理
}

// 配置变更类型导出（供未来热重载用）
export type { ConfigurationTarget };
