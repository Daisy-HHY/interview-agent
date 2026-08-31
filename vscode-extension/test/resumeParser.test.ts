import { mkdtempSync, rmSync, writeFileSync } from "fs";
import { join } from "path";
import { tmpdir } from "os";
import { afterEach, describe, expect, it, vi } from "vitest";
import { window } from "vscode";
// eslint-disable-next-line @typescript-eslint/no-var-requires
const JSZip = require("jszip");

vi.mock("vscode", () => ({
  commands: {},
  Uri: { joinPath: (...parts: Array<{ fsPath?: string } | string>) => parts.at(-1) },
  window: {
    createOutputChannel: vi.fn(() => ({
      appendLine: vi.fn(),
      show: vi.fn(),
    })),
  },
}));

import {
  buildInstallCommand,
  buildOcrInstallCommand,
  getResumeFilePathFromTabInput,
  InterviewViewProvider,
  parseResumeOcrProgressLine,
  parseResumeFile,
  withResumeParseTimeout,
} from "../src/webviewPanel";

let tempRoot = "";

afterEach(() => {
  if (tempRoot) {
    rmSync(tempRoot, { recursive: true, force: true });
    tempRoot = "";
  }
});

function tempFile(name: string, content: string): string {
  tempRoot = mkdtempSync(join(tmpdir(), "interview-agent-"));
  const path = join(tempRoot, name);
  writeFileSync(path, content, "utf-8");
  return path;
}

async function tempDocx(name: string, content: string): Promise<string> {
  tempRoot = mkdtempSync(join(tmpdir(), "interview-agent-"));
  const zip = new JSZip();
  zip.file(
    "[Content_Types].xml",
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
      + '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
      + '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
      + '<Default Extension="xml" ContentType="application/xml"/>'
      + '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
      + "</Types>",
  );
  zip.folder("_rels")?.file(
    ".rels",
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
      + '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
      + '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
      + "</Relationships>",
  );
  zip.folder("word")?.file(
    "document.xml",
    '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
      + '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
      + `<w:body><w:p><w:r><w:t>${content}</w:t></w:r></w:p></w:body></w:document>`,
  );
  const filePath = join(tempRoot, name);
  writeFileSync(filePath, await zip.generateAsync({ type: "nodebuffer" }));
  return filePath;
}

describe("parseResumeFile", () => {
  it("从系统拖入后被 VS Code 打开的本地简历 Tab 中提取路径", () => {
    const path = getResumeFilePathFromTabInput({
      uri: { scheme: "file", fsPath: "D:\\project\\interview-agent\\resume.pdf" },
    });

    expect(path).toBe("D:\\project\\interview-agent\\resume.pdf");
  });

  it("接收图片简历，忽略非本地或不支持格式 Tab", () => {
    expect(getResumeFilePathFromTabInput({
      uri: { scheme: "file", fsPath: "D:\\project\\interview-agent\\image.png" },
    })).toBe("D:\\project\\interview-agent\\image.png");
    expect(getResumeFilePathFromTabInput({
      uri: { scheme: "untitled", fsPath: "resume.md" },
    })).toBe("");
    expect(getResumeFilePathFromTabInput({
      uri: { scheme: "file", fsPath: "D:\\project\\interview-agent\\resume.xlsx" },
    })).toBe("");
  });

  it("读取 txt 简历并返回文件名", async () => {
    const file = tempFile("resume.txt", "后端开发，熟悉 Redis");

    const result = await parseResumeFile(file);

    expect(result.fileName).toBe("resume.txt");
    expect(result.content).toContain("Redis");
    expect(result.truncated).toBe(false);
  });

  it("超长文本截断到 80000 字", async () => {
    const file = tempFile("resume.md", "a".repeat(80_010));

    const result = await parseResumeFile(file);

    expect(result.content).toHaveLength(80_000);
    expect(result.truncated).toBe(true);
  });

  it("空文本给出可读错误", async () => {
    const file = tempFile("empty.txt", "   \n");

    await expect(parseResumeFile(file)).rejects.toThrow("未从简历附件中提取到文字内容");
  });

  it("拒绝不支持的格式", async () => {
    const file = tempFile("resume.xlsx", "x");

    await expect(parseResumeFile(file)).rejects.toThrow(".pdf、.docx、.txt");
  });

  it("解析真实 docx 文本", async () => {
    const file = await tempDocx("resume.docx", "Resume Redis MySQL");

    const result = await parseResumeFile(file);

    expect(result.content).toContain("Resume Redis MySQL");
  });

  it("解析带文字层的 pdf 文本", async () => {
    const file = join(__dirname, "..", "node_modules", "pdf-parse", "test", "data", "01-valid.pdf");

    const result = await parseResumeFile(file);

    expect(result.content).toContain("Because traces are in SSA form");
  });

  it("PDF 文字层为空时触发 OCR fallback", async () => {
    const file = tempFile("resume.pdf", "%PDF-1.4\n");
    const statuses: string[] = [];

    const result = await parseResumeFile(file, {
      onStatus: (message) => statuses.push(message),
      ocr: async () => "OCR Redis MySQL",
      pdfText: async () => "",
    });

    expect(result.content).toBe("OCR Redis MySQL");
    expect(statuses).toContain("正在识别扫描版 PDF...");
  });

  it("PDF 主解析器卡住时触发文字层备用解析", async () => {
    const file = tempFile("resume.pdf", "%PDF-1.4\n");
    const statuses: string[] = [];
    let fallbackReason = "";

    const result = await parseResumeFile(file, {
      onStatus: (message) => statuses.push(message),
      pdfText: async () => new Promise<string>(() => {}),
      pdfTextFallback: async (_filePath, reason) => {
        fallbackReason = reason;
        return "PyMuPDF Redis MySQL";
      },
      pdfTextTimeoutMs: 5,
    });

    expect(result.content).toBe("PyMuPDF Redis MySQL");
    expect(fallbackReason).toContain("PDF 文字层解析超时");
    expect(statuses).toContain("正在使用备用 PDF 解析...");
  });

  it("PDF 有文字层时不触发 OCR fallback", async () => {
    const file = tempFile("resume.pdf", "%PDF-1.4\n");
    let ocrCalled = false;

    const result = await parseResumeFile(file, {
      ocr: async () => {
        ocrCalled = true;
        return "不应该执行";
      },
      pdfText: async () => "PDF text Redis",
    });

    expect(result.content).toBe("PDF text Redis");
    expect(ocrCalled).toBe(false);
  });

  it("扫描版 PDF 缺 OCR 回调时提示安装 OCR 依赖", async () => {
    const file = tempFile("resume.pdf", "%PDF-1.4\n");

    await expect(parseResumeFile(file, { pdfText: async () => "" })).rejects.toThrow(
      "扫描版 PDF 需要 OCR 依赖",
    );
  });

  it("图片简历直接触发 OCR fallback", async () => {
    const file = tempFile("resume.png", "not real image");
    const statuses: string[] = [];

    const result = await parseResumeFile(file, {
      onStatus: (message) => statuses.push(message),
      ocr: async () => "OCR Python Redis",
    });

    expect(result.content).toBe("OCR Python Redis");
    expect(statuses).toContain("正在识别图片简历...");
  });

  it("图片简历缺 OCR 回调时提示安装 OCR 依赖", async () => {
    const file = tempFile("resume.jpg", "not real image");

    await expect(parseResumeFile(file)).rejects.toThrow("图片简历需要 OCR 依赖");
  });

  it("生成 Windows Terminal 依赖安装命令", () => {
    const original = Object.getOwnPropertyDescriptor(process, "platform");
    Object.defineProperty(process, "platform", { value: "win32" });
    try {
      const command = buildInstallCommand("C:\\Python\\python.exe", "D:\\a b\\requirements-agent.txt");
      expect(command).toContain("& \"C:\\Python\\python.exe\" -m pip install -r");
      expect(command).toContain("\"D:\\a b\\requirements-agent.txt\"");
    } finally {
      if (original) {
        Object.defineProperty(process, "platform", original);
      }
    }
  });

  it("生成 Windows Terminal OCR 可选依赖安装命令", () => {
    const original = Object.getOwnPropertyDescriptor(process, "platform");
    Object.defineProperty(process, "platform", { value: "win32" });
    try {
      const command = buildOcrInstallCommand("C:\\Python\\python.exe", "D:\\a b\\requirements-ocr.txt");
      expect(command).toContain("-r");
      expect(command).toContain("requirements-ocr.txt");
    } finally {
      if (original) {
        Object.defineProperty(process, "platform", original);
      }
    }
  });

  it("解析 OCR 进度 JSON Lines，忽略普通 stderr", () => {
    expect(parseResumeOcrProgressLine("not json")).toBeNull();
    expect(parseResumeOcrProgressLine('{"kind":"other","message":"x"}')).toBeNull();

    const progress = parseResumeOcrProgressLine(
      '{"kind":"ocr_progress","stage":"recognize","message":"正在识别","currentPage":2,"totalPages":3,"elapsedMs":1234}',
    );

    expect(progress).toEqual({
      stage: "recognize",
      message: "正在识别",
      currentPage: 2,
      totalPages: 3,
      elapsedMs: 1234,
    });
  });

  it("简历解析卡住时会超时返回错误", async () => {
    const never = new Promise<string>(() => {});

    await expect(withResumeParseTimeout(never, 5)).rejects.toThrow("读取简历超时");
  });

  it("点击上传走 Host 文件选择器并成功回传 resumePicked", async () => {
    const file = tempFile("resume.txt", "后端开发，熟悉 Redis");
    const posted: any[] = [];
    let receiveMessage: ((message: any) => void) | undefined;
    vi.mocked(window).showOpenDialog = vi.fn().mockResolvedValue([
      { fsPath: file },
    ]);

    const webview = {
      onDidReceiveMessage: (callback: (message: any) => void) => {
        receiveMessage = callback;
      },
      postMessage: vi.fn((message: any) => {
        posted.push(message);
        return Promise.resolve(true);
      }),
    };
    const provider = new InterviewViewProvider(
      { fsPath: "" } as any,
      () => ({
        pythonPath: "python",
        scriptPath: "",
        workspace: tempRoot,
        workspaceName: "test",
        hasWorkspace: true,
        pythonPathRoot: "",
        requirementsPath: "",
        requirementsOcrPath: "",
        apiKey: "sk-test",
        model: "gpt-test",
      }),
      async () => {},
    );

    (provider as any).wireMessages(webview);
    receiveMessage?.({ type: "pickResume" });

    await vi.waitFor(() => {
      expect(posted.some((item) => item.type === "resumePicked")).toBe(true);
    });

    const picked = posted.find((item) => item.type === "resumePicked");
    expect(picked.resume.fileName).toBe("resume.txt");
    expect(picked.resume.content).toContain("Redis");
  });
});
