import { afterEach, describe, expect, it, vi } from "vitest";
import { mkdtempSync, mkdirSync, readFileSync, readdirSync, rmSync, writeFileSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";

const mocks = vi.hoisted(() => ({ agents: [] as any[], commands: vi.fn(), updates: vi.fn(), confirm: vi.fn(async () => "导出报告") }));
vi.mock("vscode", () => ({
  commands: { executeCommand: mocks.commands },
  ConfigurationTarget: { Global: 1 },
  Uri: { file: (path: string) => ({ fsPath: path }) },
  workspace: { getConfiguration: () => ({ get: (_: string, value: unknown) => value,
    update: mocks.updates }) },
  window: { showInformationMessage: mocks.confirm, activeTextEditor: undefined, createOutputChannel: () => ({ appendLine() {}, show() {} }) },
}));
vi.mock("../src/pythonLocator", () => ({ locatePython: () => ({ pythonPath: "python", diagnostics: [] }) }));
vi.mock("../src/agentClient", () => ({ AgentClient: class {
  sent: any[] = [];
  notify: any;
  error: any;
  disposed = false;
  constructor(public options: any) { mocks.agents.push(this); }
  start() {}
  send(message: any) { this.sent.push(message); }
  dispose() { this.disposed = true; }
  onLog() {}
  onNotification(fn: any) { this.notify = fn; }
  onError(fn: any) { this.error = fn; }
} }));

import { InterviewViewProvider } from "../src/webviewPanel";
import { saveWebviewConfig } from "../src/extension";

const directories: string[] = [];

/** 直接执行 Host 消息处理，临时目录与实际项目会话完全隔离。 */
function panel(loadApiKey = async () => "fake-key", demoMode = false) {
  const dir = mkdtempSync(join(tmpdir(), "interview-panel-"));
  directories.push(dir);
  const options: any = { workspace: dir, workspaceName: "test", hasWorkspace: true,
    pythonPathRoot: dir, pythonPath: "python", scriptPath: "main.py", apiKey: "",
    model: "fake", baseUrl: "https://provider.test/v1", demoMode };
  let receive: (message: any) => Promise<void>;
  const messages: any[] = [];
  const webview: any = { postMessage: (msg: any) => { messages.push(msg); return Promise.resolve(true); },
    onDidReceiveMessage: (fn: any) => { receive = fn; } };
  const provider = new InterviewViewProvider({} as any, () => ({ ...options }), async (config) => {
    Object.assign(options, config);
  }, loadApiKey);
  (provider as any).wireMessages(webview);
  return { provider, dir, options, messages, send: (message: any) => receive(message) };
}

afterEach(() => {
  for (const dir of directories.splice(0)) { rmSync(dir, { recursive: true, force: true }); }
  mocks.agents.length = 0;
  vi.clearAllMocks();
});

describe("Host 关键交互", () => {
  it("等待密钥时停止，不会在异步返回后偷偷启动", async () => {
    let resolve!: (key: string) => void;
    const p = panel(() => new Promise((done) => { resolve = done; }));
    const chat = p.send({ type: "chat", text: "question" });
    expect(mocks.agents).toHaveLength(0);
    await p.send({ type: "stop" });
    resolve("synthetic-key");
    await chat;
    expect(mocks.agents).toHaveLength(0);
    expect(p.messages.some((m) => m.method === "cancelled")).toBe(true);
  });

  it("开始、停止、错误后继续及配置延迟切换", async () => {
    const p = panel();
    await p.send({ type: "chat", text: "first" });
    const first = mocks.agents[0];
    expect(first.options.apiKey).toBe("fake-key");
    const session = first.sent[0].params.session;
    await p.send({ type: "stop" });
    expect(first.sent[1].method).toBe("stop");
    first.notify({ method: "cancelled", params: { session } });
    await p.send({ type: "chat", text: "retry" });
    first.error("synthetic error");
    await p.send({ type: "chat", text: "again" });
    expect(first.sent.filter((m: any) => m.method === "chat")).toHaveLength(3);
    await p.send({ type: "updateConfig", config: { model: "changed" } });
    await vi.waitFor(() => expect(p.messages.some((m) => m.type === "configSaved")).toBe(true));
    expect(first.disposed).toBe(false);
    first.notify({ method: "done", params: { session } });
    expect(first.disposed).toBe(true);
    await p.send({ type: "chat", text: "new model" });
    expect(mocks.agents[1].options.model).toBe("changed");
    expect(mocks.agents[1].sent[0].params.session).not.toBe(session);
    expect(JSON.stringify(p.messages)).not.toContain("fake-key");
  });

  it("密钥失败显式终止，Demo 不访问凭证", async () => {
    const loader = vi.fn(async () => { throw new Error("SYNTHETIC_SECRET"); });
    const p = panel(loader);
    await p.send({ type: "chat", text: "q" });
    expect(mocks.agents).toHaveLength(0);
    expect(p.messages.some((m) => m.method === "error")).toBe(true);
    expect(JSON.stringify(p.messages)).not.toContain("SYNTHETIC_SECRET");
    const demo = panel(loader, true);
    loader.mockClear();
    await demo.send({ type: "chat", text: "q" });
    expect(loader).not.toHaveBeenCalled();
    expect(mocks.agents).toHaveLength(1);
  });

  it("恢复真实临时会话并导出可解析的本地报告", async () => {
    const p = panel();
    mkdirSync(join(p.dir, ".sessions"));
    writeFileSync(join(p.dir, ".sessions/saved.json"), JSON.stringify([
      { role: "system", content: "test" },
      { role: "user", content: "岗位 JD：\nJavaScript 开发\n简历：\n学习中" },
      { role: "assistant", content: "说说闭包？" },
      { role: "user", content: "我想补充一个例子" },
    ]));
    await p.send({ type: "resumeSession", session: "saved" });
    expect(p.messages.find((m) => m.type === "sessionLoaded").messages).toHaveLength(3);
    await p.send({ type: "chat", text: "continue" });
    expect(mocks.agents[0].sent[0].params.session).toBe("saved");
    mocks.agents[0].notify({ method: "done", params: { session: "saved" } });
    await p.send({ type: "exportReport" });
    await vi.waitFor(() => expect(p.messages.some((m) => m.type === "reportExported")).toBe(true));
    expect(mocks.confirm).toHaveBeenCalledOnce();
    const reports = join(p.dir, ".interview-agent/reports");
    const report = readFileSync(join(reports, readdirSync(reports)[0]), "utf8");
    expect(report).toContain("我想补充一个例子");
    expect(report).toContain("本地整理");
    mocks.confirm.mockResolvedValueOnce(undefined as any);
    mocks.commands.mockClear();
    await p.send({ type: "exportReport" });
    expect(mocks.commands).not.toHaveBeenCalled();
    mocks.commands.mockRejectedValueOnce(new Error("open failed"));
    await p.send({ type: "exportReport" });
    await vi.waitFor(() => expect(p.messages.some((m) => m.type === "reportError")).toBe(true));
  });

  it("配置保存使用加密存储，普通设置不写 API Key", async () => {
    const store = vi.fn(async () => {});
    await saveWebviewConfig({ secrets: { store } } as any,
      { apiKey: "synthetic-key", baseUrl: "https://provider.test/v1", model: "fake" });
    expect(store).toHaveBeenCalledOnce();
    expect(mocks.updates.mock.calls.some((call) => call[0] === "apiKey")).toBe(false);
  });
});
