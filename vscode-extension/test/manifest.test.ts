import { readFileSync } from "fs";
import { join } from "path";
import { describe, expect, it } from "vitest";

const manifest = JSON.parse(
  readFileSync(join(__dirname, "..", "package.json"), "utf-8"),
);
const extensionSource = readFileSync(join(__dirname, "..", "src", "extension.ts"), "utf-8");

describe("VS Code manifest", () => {
  it("把面试视图声明为 webview，否则 VS Code 会按 Tree View 查找数据提供程序", () => {
    const view = manifest.contributes.views["interview-agent"].find(
      (item: { id?: string }) => item.id === "interview.chatView",
    );

    expect(view).toBeTruthy();
    expect(view.type).toBe("webview");
  });

  it("view id 与激活事件保持一致", () => {
    expect(manifest.activationEvents).toContain("onView:interview.chatView");
  });

  it("声明 Agent runtime 切换配置，默认保持 native", () => {
    const runtime = manifest.contributes.configuration.properties["interview.agentRuntime"];
    expect(runtime.default).toBe("native");
    expect(runtime.enum).toEqual(["native", "langchain"]);
  });

  it("监听系统拖入文件被 VS Code 打开后的兜底 Tab 事件", () => {
    expect(extensionSource).toContain("window.tabGroups.onDidChangeTabs");
    expect(extensionSource).toContain("provider.captureOpenedResumeTab(tab)");
    expect(extensionSource).toContain("event.opened");
  });

  it("监听 interview 配置变化并刷新面板快照", () => {
    expect(extensionSource).toContain("workspace.onDidChangeConfiguration");
    expect(extensionSource).toContain('event.affectsConfiguration("interview")');
    expect(extensionSource).toContain("provider.refreshConfigFromSettings()");
  });
});
