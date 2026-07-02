/**
 * TS 侧协议层（对应 Python agent/protocol.py）。
 *
 * 设计第 1.4、1.5 节：Extension Host 和 Python 子进程通过 stdio 用
 * JSON-RPC 通信，一行一条消息。
 *
 * 两类消息：
 * - Request（TS → Python）：init / chat / stop
 * - Notification（Python → TS）：stream / tool_call / done / error
 *
 * 关键：Python 侧字段用 snake_case（api_key / base_url / attached_code），
 * 这里定义的类型和序列化函数都用 snake_case，与 Python 严格一致，
 * 避免在边界来回转换。
 */

// ──────────────────────────────────────────────
// Request（TS → Python）—— 字段名对齐 Python main.py
// ──────────────────────────────────────────────

/** init 消息：会话开始发一次，给 Python 工作区和配置。 */
export interface InitRequest {
  method: "init";
  params: {
    workspace: string;
    api_key: string;
    model: string;
    base_url?: string;
    resume?: string;
    session?: string;
    // 调优参数（Phase 7-D 可配化，可选；不传则 Python 用默认值）
    max_steps?: number;
    max_history_tokens?: number;
    max_kept_full?: number;
    /** 历史落盘目录（#3）：TS 传插件 globalStorageUri，让落盘稳定可预测。 */
    storage_dir?: string;
  };
}

/** chat 消息：用户说的话。 */
export interface ChatRequest {
  method: "chat";
  params: {
    session: string;
    text: string;
    /** 选中代码（设计第 5.3.3 节），可选。 */
    attached_code?: {
      file: string;
      content: string;
    };
  };
}

/** stop 消息：中断当前生成（MVP 占位）。 */
export interface StopRequest {
  method: "stop";
  params: {
    session: string;
  };
}

export type Request = InitRequest | ChatRequest | StopRequest;

// ──────────────────────────────────────────────
// Notification（Python → TS）—— 字段名对齐 Python protocol.py 的 notify_*
// ──────────────────────────────────────────────

/** 流式输出：LLM 吐一段文字就推一段。 */
export interface StreamNotification {
  method: "stream";
  params: {
    session: string;
    delta: string;
  };
}

/** 工具调用：让 UI 显示"正在搜代码"气泡。 */
export interface ToolCallNotification {
  method: "tool_call";
  params: {
    session: string;
    tool: string;
    /** "start" 或 "end" */
    phase: "start" | "end";
    /** 工具参数（start 时用）。 */
    args?: Record<string, unknown>;
    /** 工具结果（end 时用）。 */
    result?: string;
  };
}

/** 本轮 Agent 循环结束。 */
export interface DoneNotification {
  method: "done";
  params: {
    session: string;
  };
}

/** 错误通知：API 失效、网络断、工具报错。 */
export interface ErrorNotification {
  method: "error";
  params: {
    session: string;
    message: string;
  };
}

export type Notification =
  | StreamNotification
  | ToolCallNotification
  | DoneNotification
  | ErrorNotification;

/** 解析出的通知类型（带 method 字面量，方便 switch 收窄）。 */
export type ParsedNotification = Notification;

// ──────────────────────────────────────────────
// 序列化（TS → Python）：一行 JSON
// ──────────────────────────────────────────────

/**
 * 把 Request 序列化成一行 JSON 字符串（含换行符）。
 * 对应 Python protocol.parse_message 能解析的格式。
 */
export function serialize(msg: Request): string {
  // ensure_ascii 在 TS/JS 默认就是 false（保留中文原样）
  return JSON.stringify(msg) + "\n";
}

// ──────────────────────────────────────────────
// 解析（Python → TS）：容错，脏数据返回 null
// ──────────────────────────────────────────────

/**
 * 解析 Python stdout 的一行 JSON 通知。
 *
 * 容错策略（对应 Python parse_message）：格式错误返回 null，不抛异常，
 * 这样"喂脏数据"不会让 Extension Host 崩溃。
 */
export function parse(line: string): ParsedNotification | null {
  const trimmed = line.trim();
  if (!trimmed) {
    return null;
  }
  let msg: unknown;
  try {
    msg = JSON.parse(trimmed);
  } catch {
    return null;
  }
  if (!isNotificationShape(msg)) {
    return null;
  }
  return msg;
}

/** 类型守卫：验证一个值是不是合法的通知结构。 */
function isNotificationShape(value: unknown): value is ParsedNotification {
  if (typeof value !== "object" || value === null) {
    return false;
  }
  const method = (value as { method?: unknown }).method;
  return (
    method === "stream" ||
    method === "tool_call" ||
    method === "done" ||
    method === "error"
  );
}

// ──────────────────────────────────────────────
// 便捷构造器：让调用处语义清晰
// ──────────────────────────────────────────────

export function buildInit(params: InitRequest["params"]): InitRequest {
  return { method: "init", params };
}

export function buildChat(params: ChatRequest["params"]): ChatRequest {
  return { method: "chat", params };
}

export function buildStop(session: string): StopRequest {
  return { method: "stop", params: { session } };
}
