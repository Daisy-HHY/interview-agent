import { describe, expect, it } from "vitest";
import { vi } from "vitest";

vi.mock("vscode", () => ({
  commands: {},
  Uri: { file: (path: string) => ({ fsPath: path }), joinPath: (...parts: Array<{ fsPath?: string } | string>) => parts.at(-1) },
  window: {},
}));

import {
  canExportReport,
  generateMarkdownReport,
  sanitizeReportFileName,
} from "../src/webviewPanel";

describe("面试报告导出", () => {
  it("根据会话消息生成包含薄弱点和复习建议的 Markdown", () => {
    const report = generateMarkdownReport({
      sessionId: "s1",
      title: "AI 后端面试",
      workspaceName: "interview-agent",
      workspacePath: "D:\\project\\interview-agent",
      createdAt: new Date("2026-08-29T10:00:00Z"),
      messages: [
        {
          role: "user",
          content:
            "我们开始一场技术面试。\n\n岗位 JD：\nAI Agent 后端实习生\n\n简历：\n熟悉 Python 和 Redis\n\n当前项目：interview-agent\n项目路径：D:\\project\\interview-agent",
        },
        { role: "assistant", content: "请解释 Redis 缓存穿透。" },
        { role: "user", content: "可以用布隆过滤器。" },
      ],
    });

    expect(report).toContain("# AI 后端面试");
    expect(report).toContain("## JD 摘要");
    expect(report).toContain("AI Agent 后端实习生");
    expect(report).toContain("## 薄弱点");
    expect(report).toContain("## 复习建议");
    expect(report).toContain("Redis");
  });

  it("清理报告文件名里的非法字符", () => {
    expect(sanitizeReportFileName("AI/后端:面试*报告?")).toBe("AI-后端-面试-报告");
  });

  it("空会话不能导出报告", () => {
    expect(canExportReport([])).toBe(false);
    expect(canExportReport([{ role: "assistant", content: "你好" }])).toBe(false);
    expect(canExportReport([{ role: "user", content: "回答" }])).toBe(true);
  });

  it("仅有 JD 不推断候选人掌握或薄弱，JavaScript 不匹配 Java", () => {
    const report = generateMarkdownReport({ sessionId: "s", title: "test", workspaceName: "p",
      workspacePath: "p", createdAt: new Date(), messages: [
        { role: "user", content: "岗位 JD：\nJavaScript 开发\n简历：\n尚未提供" },
      ] });
    expect(report).toContain("尚未考察");
    expect(report).not.toContain("Java、");
    expect(report).not.toContain("JavaScript、Java");
    expect(report).not.toContain("优先补强");
    expect(report).toContain("本地整理");
  });

  it("不同回答保留对应原文，不生成同一份能力结论", () => {
    const input = { sessionId: "s", title: "test", workspaceName: "p", workspacePath: "p",
      createdAt: new Date(), messages: [
        { role: "user" as const, content: "岗位 JD：\nRedis\n简历：\nPython" },
        { role: "assistant" as const, content: "如何解决缓存穿透？" },
        { role: "user" as const, content: "我还不清楚" },
        { role: "assistant" as const, content: "可以先思考布隆过滤器。" },
      ] };
    const report = generateMarkdownReport(input);
    expect(report).toContain("回答原文摘录");
    expect(report).toContain("我还不清楚");
    expect(report).toContain("布隆过滤器");
    expect(report).toContain("消息 #3");
    input.messages[2].content = "可以使用布隆过滤器并处理误判";
    const changed = generateMarkdownReport(input);
    expect(changed).not.toContain("我还不清楚");
    expect(changed).toContain("处理误判");
    expect(changed).toContain("证据不足");
  });

  it("旧会话缺少背景时不把后续问题当作 JD", () => {
    const report = generateMarkdownReport({ sessionId: "s", title: "test", workspaceName: "p",
      workspacePath: "p", createdAt: new Date(), messages: [
        { role: "user", content: "继续上次的问题" },
      ] });
    expect(report).toContain("背景来源缺失");
    expect(report).toContain("尚未考察");
  });
});
