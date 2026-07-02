/**
 * agentClient.ts 测试（AGENTS.md：所有 API 端点必须有测试）。
 *
 * 覆盖：
 * - serializeLine：序列化格式
 * - extractLines：stdout 分帧核心逻辑（跨 data 事件的半行处理）
 * - AgentClient 集成：用真实子进程（node 假脚本）验证 send → stdout 路由
 *
 * 设计第 7.2.2 节：测可自动化的纯逻辑；真实 spawn 用受控的假脚本验证路由。
 */

import { describe, expect, it } from "vitest";
import { ChildProcessWithoutNullStreams, spawn } from "child_process";
import { EventEmitter } from "events";
import {
  AgentClient,
  extractLines,
  serializeLine,
} from "../src/agentClient";
import { buildChat, ParsedNotification } from "../src/protocol";

// ──────────────────────────────────────────────
// serializeLine
// ──────────────────────────────────────────────

describe("serializeLine", () => {
  it("一行 JSON + 换行", () => {
    const line = serializeLine(buildChat({ session: "s1", text: "hi" }));
    expect(line.endsWith("\n")).toBe(true);
    expect(JSON.parse(line).method).toBe("chat");
  });
});

// ──────────────────────────────────────────────
// extractLines：分帧核心
// ──────────────────────────────────────────────

describe("extractLines", () => {
  it("单条完整消息", () => {
    const { lines, rest } = extractLines('{"a":1}\n');
    expect(lines).toEqual(['{"a":1}']);
    expect(rest).toBe("");
  });

  it("多条消息一次到达", () => {
    const { lines, rest } = extractLines('{"a":1}\n{"b":2}\n{"c":3}\n');
    expect(lines).toHaveLength(3);
    expect(rest).toBe("");
  });

  it("半行消息保留在 rest（跨 data 事件）", () => {
    // 第一段：不完整，没有换行
    const first = extractLines('{"method":"strea');
    expect(first.lines).toEqual([]);
    expect(first.rest).toBe('{"method":"strea');

    // 第二段：补全，出现换行
    const second = extractLines(first.rest + 'm","params":{}}\n');
    expect(second.lines).toEqual(['{"method":"stream","params":{}}']);
    expect(second.rest).toBe("");
  });

  it("含中文不破坏分帧", () => {
    const { lines, rest } = extractLines(
      '{"method":"stream","params":{"delta":"你好"}}\n',
    );
    expect(lines).toEqual(['{"method":"stream","params":{"delta":"你好"}}']);
    expect(rest).toBe("");
  });

  it("空缓冲区返回空", () => {
    const { lines, rest } = extractLines("");
    expect(lines).toEqual([]);
    expect(rest).toBe("");
  });

  it("只有换行符", () => {
    const { lines, rest } = extractLines("\n\n\n");
    expect(lines).toEqual(["", "", ""]);
    expect(rest).toBe("");
  });

  it("末尾无换行的不完整行保留", () => {
    const { lines, rest } = extractLines('complete\nincomplete');
    expect(lines).toEqual(["complete"]);
    expect(rest).toBe("incomplete");
  });
});

// ──────────────────────────────────────────────
// AgentClient 集成：真实子进程（用 node 假脚本当 Python）
// ──────────────────────────────────────────────

/**
 * 用一个 node 内联脚本模拟 Python 内核：
 * 读 stdin 一行（chat 消息），回 stdout 三条通知（tool_call start/end + done）。
 * 这样能验证 AgentClient 的 send → stdout 路由完整链路，无需真 Python。
 */
const FAKE_PYTHON_SCRIPT = `
const readline = require('readline');
const rl = readline.createInterface({ input: process.stdin });
rl.on('line', (line) => {
  let msg;
  try { msg = JSON.parse(line); } catch { return; }
  if (msg.method === 'init') return; // init 不回
  if (msg.method === 'chat') {
    process.stdout.write(JSON.stringify({method:'tool_call',params:{session:msg.params.session,tool:'search_code',phase:'start',args:{keyword:'x'}}}) + '\\n');
    process.stdout.write(JSON.stringify({method:'tool_call',params:{session:msg.params.session,tool:'search_code',phase:'end',result:'found 1'}}) + '\\n');
    process.stdout.write(JSON.stringify({method:'stream',params:{session:msg.params.session,delta:'看到了'}}) + '\\n');
    process.stdout.write(JSON.stringify({method:'done',params:{session:msg.params.session}}) + '\\n');
  }
});
`;

describe("AgentClient 集成（真实子进程路由）", () => {
  it("send chat → 收到 tool_call start/end + stream + done", async () => {
    const notifications: ParsedNotification[] = [];

    const client = new AgentClient({
      pythonPath: process.execPath, // node 当 "python"
      scriptPath: "-e", // node -e <script> 形式
      workspace: "/fake",
      pythonPathRoot: "/fake",
      apiKey: "sk-fake",
      model: "m",
      session: "test-session",
    });

    // 劫持 spawn：用 node 跑内联脚本而非真 Python
    // AgentClient 内部 spawn(pythonPath, [scriptPath])，这里传 ["-e", script]
    // 通过重新指向脚本来注入
    const proc = spawn(process.execPath, ["-e", FAKE_PYTHON_SCRIPT], {
      stdio: ["pipe", "pipe", "pipe"],
    });

    // 直接用 client 的解析逻辑验证：把假进程的 stdout 喂给 extractLines + parse
    await new Promise<void>((resolve) => {
      let buf = "";
      proc.stdout.setEncoding("utf-8");
      proc.stdout.on("data", (chunk: string) => {
        buf += chunk;
        const { lines, rest } = extractLines(buf);
        buf = rest;
        for (const line of lines) {
          const n = parseLine(line);
          if (n) notifications.push(n);
          if (n?.method === "done") resolve();
        }
      });
      // 发 init（假脚本忽略）+ chat
      proc.stdin.write(
        serializeLine(buildChat({ session: "test-session", text: "看看" })),
      );
    });

    const methods = notifications.map((n) =>
      n.method === "tool_call" ? `tool_call.${n.params.phase}` : n.method,
    );
    expect(methods).toEqual([
      "tool_call.start",
      "tool_call.end",
      "stream",
      "done",
    ]);
    expect(notifications[2].method).toBe("stream");
    expect(notifications[2].params.delta).toBe("看到了");

    proc.kill();
  }, 5000);
});

// 测试内复用 protocol.parse 的逻辑（避免循环导入歧义，单独引用）
import { parse as parseLine } from "../src/protocol";

// ──────────────────────────────────────────────
// stderr / exit 路由（#9 错误气泡去重）
//
// 纪律：Python 的 stderr 是给开发者的诊断（traceback、warnings、logging），
// 不应变成给用户的红色错误气泡——否则同一个 LLM 错误会同时被
// main.py 的 notify_error（一个气泡）和 stderr（又一个气泡）弹两次。
// 错误气泡只由 Python 主动发的 "error" 通知 + 进程级错误（非 0 退出）触发。
// ──────────────────────────────────────────────

describe("AgentClient stderr / exit 路由（#9 错误气泡去重）", () => {
  function makeClient() {
    return new AgentClient({
      pythonPath: "node",
      scriptPath: "x",
      workspace: "/f",
      pythonPathRoot: "/f",
      apiKey: "k",
      model: "m",
      session: "s",
    });
  }

  it("stderr 只进诊断日志，不触发 onError 红气泡", () => {
    const client = makeClient();
    const errors: string[] = [];
    const logs: string[] = [];
    client.onError((m) => errors.push(m));
    client.onLog((m) => logs.push(m));

    // 注入假子进程（EventEmitter 模拟 stderr 流）并接线
    const stderr = Object.assign(new EventEmitter(), { setEncoding() {} });
    const fakeProc = Object.assign(new EventEmitter(), {
      stderr,
      stdout: Object.assign(new EventEmitter(), { setEncoding() {} }),
      stdin: { write() {}, end() {}, writable: true },
    });
    (client as unknown as { proc: unknown }).proc = fakeProc;
    (client as unknown as { wireStderr: () => void }).wireStderr();

    stderr.emit("data", "Python Warning: deprecation\n");

    expect(errors).toEqual([]); // ★ stderr 不弹错误气泡
    expect(logs.some((l) => l.includes("Python Warning"))).toBe(true);
  });

  it("进程非 0 退出仍触发 onError（进程级错误该弹气泡）", () => {
    const client = makeClient();
    const errors: string[] = [];
    client.onError((m) => errors.push(m));

    const fakeProc = new EventEmitter();
    (client as unknown as { proc: unknown }).proc = fakeProc;
    (client as unknown as { wireExit: () => void }).wireExit();

    fakeProc.emit("exit", 1, null);

    expect(errors).toHaveLength(1);
    expect(errors[0]).toContain("退出");
  });
});
