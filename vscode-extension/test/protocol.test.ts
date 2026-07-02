/**
 * protocol.ts 测试（AGENTS.md：所有 API 端点必须有测试）。
 *
 * 覆盖：
 * - serialize：Request 序列化成一行 JSON（字段名 snake_case 对齐 Python）
 * - parse：4 种 Notification 解析 + 容错（脏数据返回 null）
 * - 构造器：buildInit/buildChat/buildStop
 */

import { describe, expect, it } from "vitest";
import {
  buildChat,
  buildInit,
  buildStop,
  parse,
  serialize,
} from "../src/protocol";

// ──────────────────────────────────────────────
// serialize（TS → Python）
// ──────────────────────────────────────────────

describe("serialize", () => {
  it("序列化成一行 JSON，以换行结尾（分帧约定）", () => {
    const line = serialize(buildChat({ session: "s1", text: "hi" }));
    expect(line.endsWith("\n")).toBe(true);
    // 去掉换行后应是合法 JSON
    expect(() => JSON.parse(line.trim())).not.toThrow();
  });

  it("chat 消息字段用 snake_case（对齐 Python）", () => {
    const line = serialize(
      buildChat({
        session: "s1",
        text: "看看代码",
        attached_code: { file: "db.py", content: "connect()" },
      }),
    );
    const msg = JSON.parse(line);
    expect(msg.params.attached_code.file).toBe("db.py");
    expect(msg.params.session).toBe("s1");
  });

  it("init 消息含 workspace/api_key/model", () => {
    const line = serialize(
      buildInit({
        workspace: "/proj",
        api_key: "sk-x",
        model: "gpt-4o-mini",
      }),
    );
    const msg = JSON.parse(line);
    expect(msg.method).toBe("init");
    expect(msg.params.api_key).toBe("sk-x");
    expect(msg.params.workspace).toBe("/proj");
  });

  it("中文字符不被转义", () => {
    const line = serialize(buildChat({ session: "s1", text: "做了一个选课系统" }));
    // ensure_ascii 默认 false：中文原样出现
    expect(line).toContain("做了一个选课系统");
  });

  it("stop 消息结构正确", () => {
    const line = serialize(buildStop("s1"));
    const msg = JSON.parse(line);
    expect(msg.method).toBe("stop");
    expect(msg.params.session).toBe("s1");
  });
});

// ──────────────────────────────────────────────
// parse（Python → TS）
// ──────────────────────────────────────────────

describe("parse", () => {
  it("解析 stream 通知", () => {
    const n = parse('{"method":"stream","params":{"session":"s1","delta":"你好"}}');
    expect(n?.method).toBe("stream");
    expect(n?.params.delta).toBe("你好");
  });

  it("解析 tool_call start 通知（含 args）", () => {
    const n = parse(
      '{"method":"tool_call","params":{"session":"s1","tool":"search_code","phase":"start","args":{"keyword":"redis"}}}',
    );
    expect(n?.method).toBe("tool_call");
    expect(n?.params.tool).toBe("search_code");
    expect(n?.params.phase).toBe("start");
    expect(n?.params.args?.keyword).toBe("redis");
  });

  it("解析 tool_call end 通知（含 result）", () => {
    const n = parse(
      '{"method":"tool_call","params":{"session":"s1","tool":"search_code","phase":"end","result":"找到3处"}}',
    );
    expect(n?.params.phase).toBe("end");
    expect(n?.params.result).toBe("找到3处");
  });

  it("解析 done 通知", () => {
    const n = parse('{"method":"done","params":{"session":"s1"}}');
    expect(n?.method).toBe("done");
  });

  it("解析 error 通知", () => {
    const n = parse(
      '{"method":"error","params":{"session":"s1","message":"API 失效"}}',
    );
    expect(n?.method).toBe("error");
    expect(n?.params.message).toBe("API 失效");
  });

  it("容错：格式错误的 JSON 返回 null", () => {
    expect(parse("{not json}")).toBeNull();
    expect(parse("随机文字")).toBeNull();
    expect(parse('{"method":}')).toBeNull();
  });

  it("容错：空行返回 null", () => {
    expect(parse("")).toBeNull();
    expect(parse("   \n")).toBeNull();
  });

  it("容错：缺少 method 字段返回 null", () => {
    expect(parse('{"jsonrpc":"2.0","params":{}}')).toBeNull();
  });

  it("容错：未知 method 返回 null", () => {
    expect(parse('{"method":"mystery","params":{}}')).toBeNull();
  });

  it("容错：非对象（数组/数字）返回 null", () => {
    expect(parse("[1,2,3]")).toBeNull();
    expect(parse("42")).toBeNull();
  });

  it("容错：尾随换行/空白能解析", () => {
    const n = parse('{"method":"done","params":{"session":"s1"}}\n');
    expect(n?.method).toBe("done");
  });

  it("保留中文字符不丢失", () => {
    const n = parse(
      '{"method":"stream","params":{"session":"s1","delta":"你用了什么数据库？"}}',
    );
    expect(n?.params.delta).toBe("你用了什么数据库？");
  });
});

// ──────────────────────────────────────────────
// 构造器
// ──────────────────────────────────────────────

describe("buildInit", () => {
  it("含可选 base_url 和 resume", () => {
    const req = buildInit({
      workspace: "/p",
      api_key: "k",
      model: "m",
      base_url: "https://api.deepseek.com",
      resume: "张三 大三",
    });
    expect(req.method).toBe("init");
    expect(req.params.base_url).toBe("https://api.deepseek.com");
    expect(req.params.resume).toBe("张三 大三");
  });

  it("可选字段省略时不出现", () => {
    const req = buildInit({ workspace: "/p", api_key: "k", model: "m" });
    expect(req.params.base_url).toBeUndefined();
    expect(req.params.resume).toBeUndefined();
  });

  it("含 storage_dir（#3 落盘目录，对齐 Python storage_dir）", () => {
    const req = buildInit({
      workspace: "/p",
      api_key: "k",
      model: "m",
      storage_dir: "/data/sessions",
    });
    expect(req.params.storage_dir).toBe("/data/sessions");
  });
});
