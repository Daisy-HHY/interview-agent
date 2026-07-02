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
import {
  commands,
  ConfigurationTarget,
  ExtensionContext,
  Uri,
  window,
  workspace,
} from "vscode";
import { InterviewPanel } from "./webviewPanel";

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
  };
}

export function activate(context: ExtensionContext): void {
  const htmlBasePath = Uri.joinPath(context.extensionUri, "media");

  // 关键：agent/main.py 的位置基于插件自身路径推算，不依赖用户打开了哪个工作区。
  // 插件在 <repo>/vscode-extension/，往上两级是仓库根（含 agent/）。
  // 这样即使用户在 Extension Host 窗口切到别的项目，也能找到正确的内核脚本。
  // 作为"被面试"的项目代码（workspace）则用用户当前打开的文件夹。
  const repoRoot = Uri.joinPath(context.extensionUri, "..").fsPath;
  const agentScriptPath = path.join(repoRoot, "agent", "main.py");

  // ───────── interview.start ─────────
  const startCmd = commands.registerCommand("interview.start", async () => {
    const cfg = readConfig();
    // demoMode 下用 FakeLLM，不需要 apiKey；非 demoMode 必须有 key
    if (!cfg.demoMode && !cfg.apiKey) {
      const choice = await window.showErrorMessage(
        "还未配置 API Key。请在设置里填写 interview.apiKey，或开启 interview.demoMode 体验零费用演示。",
        "打开设置",
      );
      if (choice === "打开设置") {
        commands.executeCommand("workbench.action.openSettings", "interview");
      }
      return;
    }

    // "被面试"的项目 = 用户当前打开的文件夹（面试官翻这里的代码）
    const intervieweeProject =
      workspace.workspaceFolders?.[0]?.uri.fsPath ?? repoRoot;

    const panel = new InterviewPanel(htmlBasePath, {
      pythonPath: cfg.pythonPath,
      scriptPath: agentScriptPath,
      workspace: intervieweeProject,
      // PYTHONPATH 用仓库根（agent 包所在），不随被面试项目变
      pythonPathRoot: repoRoot,
      apiKey: cfg.apiKey || "demo",
      model: cfg.model,
      baseUrl: cfg.baseUrl || undefined,
      resume: cfg.resume || undefined,
      demoMode: cfg.demoMode,
      // 调优参数透传（Phase 7-D 可配化）
      maxSteps: cfg.maxSteps,
      maxHistoryTokens: cfg.maxHistoryTokens,
      maxKeptFull: cfg.maxKeptFull,
      // 落盘目录（#3）：插件全局数据目录，跨 VS Code 重启稳定、可预测
      storageDir: context.globalStorageUri.fsPath,
    });
    panel.open();
  });

  // ───────── interview.askAboutSelection ─────────
  // 对选中代码提问：填入输入框后自动发送（复用 start 打开的面板逻辑）
  const askCmd = commands.registerCommand(
    "interview.askAboutSelection",
    async () => {
      // MVP：等同于 start（选中代码由 Webview 读取时自动注入）
      // 完整版应在面板输入框预填"针对这段代码："，这里先走 start
      commands.executeCommand("interview.start");
    },
  );

  context.subscriptions.push(startCmd, askCmd);
}

export function deactivate(): void {
  // Webview 关闭时 InterviewPanel 自己 dispose AgentClient，无需额外清理
}

// 配置变更类型导出（供未来热重载用）
export type { ConfigurationTarget };
