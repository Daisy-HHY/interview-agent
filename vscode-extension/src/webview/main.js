// @ts-check
(function () {
  const vscode = acquireVsCodeApi();

  const messagesEl = document.getElementById("messages");
  const inputEl = document.getElementById("input");
  const actionBtn = document.getElementById("action");
  const settingsBtn = document.getElementById("settings");
  const exportReportBtn = document.getElementById("exportReport");
  const historyToggleBtn = document.getElementById("historyToggle");
  const configStatusEl = document.getElementById("configStatus");
  const modelConfigSummaryEl = document.getElementById("modelConfigSummary");
  const modelTestStatusEl = document.getElementById("modelTestStatus");
  const runtimeDiagnosticsEl = document.getElementById("runtimeDiagnostics");
  const runtimeTimelineEl = document.getElementById("runtimeTimeline");
  const dependencyPanelEl = document.getElementById("dependencyPanel");
  const dependencyMessageEl = document.getElementById("dependencyMessage");
  const dependencyCommandEl = document.getElementById("dependencyCommand");
  const installDependenciesBtn = document.getElementById("installDependencies");
  const checkDependenciesBtn = document.getElementById("checkDependencies");
  const openModelSettingsBtn = document.getElementById("openModelSettings");
  const testModelConnectionBtn = document.getElementById("testModelConnection");
  const jdEl = document.getElementById("jd");
  const resumeSupplementEl = document.getElementById("resumeSupplement");
  const pickResumeBtn = document.getElementById("pickResume");
  const resumeFileInputEl = document.getElementById("resumeFileInput");
  const resumeFileEl = document.getElementById("resumeFile");
  const workspaceInfoEl = document.getElementById("workspaceInfo");
  const startInterviewBtn = document.getElementById("startInterview");
  const setupEl = document.getElementById("setup");
  const chatEl = document.getElementById("chat");
  const historyPanelEl = document.getElementById("historyPanel");
  const sessionListEl = document.getElementById("sessionList");
  const newSessionBtn = document.getElementById("newSession");
  const refreshSessionsBtn = document.getElementById("refreshSessions");
  const DROPPED_RESUME_MAX_BYTES = 10 * 1024 * 1024;

  let currentInterviewerBubble = null;
  let thinkingBubble = null;
  let thinkingToolLogs = [];
  let awaiting = false;
  let stopping = false;
  let interviewStarted = false;
  let canExportReport = false;
  let resumeAttachment = null;
  let resumeDropArmSentAt = 0;
  let resumeCaptureEnabled = null;
  let workspaceState = { hasWorkspace: false, workspaceName: "", workspacePath: "" };
  let currentConfig = null;
  let actualRuntime = null;
  let runtimeEvents = [];
  let runtimeDroppedEvents = 0;
  let runtimeTruncatedEvents = 0;
  let runtimeIgnoredEvents = 0;
  const MAX_RUNTIME_EVENTS = 80;
  const MAX_RUNTIME_EVENT_CHARS = 160;
  const MAX_RUNTIME_TIMELINE_CHARS = 12000;
  const KNOWN_RUNTIME_EVENTS = new Set([
    "agent_start",
    "agent_end",
    "turn_start",
    "turn_end",
    "tool_execution_start",
    "tool_execution_end",
    "message_update",
    "context_compaction",
  ]);

  function setAwaiting(value) {
    awaiting = value;
    updateActionButton();
  }

  function updateActionButton() {
    if (stopping) {
      actionBtn.textContent = "■";
      actionBtn.title = "正在停止";
      actionBtn.setAttribute("aria-label", "正在停止");
      actionBtn.disabled = true;
      actionBtn.classList.add("is-stop");
      return;
    }
    if (awaiting) {
      actionBtn.textContent = "■";
      actionBtn.title = "停止";
      actionBtn.setAttribute("aria-label", "停止");
      actionBtn.disabled = false;
      actionBtn.classList.add("is-stop");
      return;
    }
    actionBtn.textContent = "↑";
    actionBtn.title = "发送";
    actionBtn.setAttribute("aria-label", "发送");
    actionBtn.disabled = !inputEl.value.trim();
    actionBtn.classList.remove("is-stop");
  }

  function startInterview() {
    const jd = jdEl.value.trim();
    if (!jd) {
      setStatus("请先填写岗位 JD。");
      jdEl.focus();
      return;
    }
    if (!workspaceState.hasWorkspace) {
      setStatus("请先打开要面试的目标项目文件夹。");
      return;
    }

    const resumeSupplement = resumeSupplementEl.value.trim();
    const resumeParts = [];
    if (resumeAttachment) {
      resumeParts.push(
        `简历附件：${resumeAttachment.fileName}\n${resumeAttachment.content}`,
      );
    }
    if (resumeSupplement) {
      resumeParts.push(`简历补充：\n${resumeSupplement}`);
    }

    const text = [
      "我们开始一场技术面试。",
      "",
      `岗位 JD：\n${jd}`,
      resumeParts.length ? `\n简历：\n${resumeParts.join("\n\n")}` : "",
      `\n当前项目：${workspaceState.workspaceName || "未命名工作区"}`,
      `项目路径：${workspaceState.workspacePath}`,
      "请自动读取当前 VS Code 工作区下的项目情况，先了解项目结构和技术栈，再开始第一轮面试提问。",
    ].join("\n");

    interviewStarted = true;
    canExportReport = false;
    updateExportButton();
    showChatPage();
    sendChat(text, "已提交岗位 JD 和简历，开始面试。");
  }

  function onAction() {
    if (awaiting) {
      stopping = true;
      vscode.postMessage({ type: "stop" });
      updateActionButton();
      return;
    }
    send();
  }

  function send() {
    const text = inputEl.value.trim();
    if (!text || awaiting) {
      return;
    }
    sendChat(text, text);
  }

  function sendChat(text, displayText) {
    appendBubble("user", "我", displayText);
    showThinkingOrb();
    inputEl.value = "";
    setAwaiting(true);
    vscode.postMessage({ type: "chat", text });
  }

  window.addEventListener("message", (event) => {
    const msg = event.data;
    if (!msg) {
      return;
    }

    if (msg.type === "config") {
      applyConfig(msg.config);
      return;
    }
    if (msg.type === "resumePicked") {
      resumeAttachment = msg.resume;
      resumeFileEl.textContent = msg.resume.truncated
        ? `${msg.resume.fileName}（已截取前 80000 字）`
        : msg.resume.fileName;
      setStatus("简历已读取");
      return;
    }
    if (msg.type === "resumeStatus") {
      setStatus(msg.message || "");
      return;
    }
    if (msg.type === "resumeOcrProgress") {
      setStatus(formatOcrProgress(msg.progress));
      return;
    }
    if (msg.type === "resumeError") {
      appendBubble("error", "出错了", msg.message || "读取简历失败");
      setStatus("");
      return;
    }
    if (msg.type === "modelTestStatus") {
      setModelTestStatus(msg.message || "", "");
      return;
    }
    if (msg.type === "modelTestResult") {
      setModelTestStatus(
        msg.message || "模型连接测试完成",
        msg.ok ? "success" : "error",
      );
      return;
    }
    if (msg.type === "reportExported") {
      setStatus(msg.message || "报告已导出");
      canExportReport = true;
      updateExportButton();
      return;
    }
    if (msg.type === "reportError") {
      setStatus(msg.message || "导出报告失败");
      return;
    }
    if (msg.type === "dependencyStatus") {
      showDependencyStatus(msg);
      return;
    }
    if (msg.type === "sessions") {
      renderSessions(msg.sessions || [], msg.current || "");
      return;
    }
    if (msg.type === "sessionNew") {
      clearMessages();
      interviewStarted = false;
      canExportReport = false;
      updateExportButton();
      showSetupPage("已新建会话");
      jdEl.focus();
      return;
    }
    if (msg.type === "sessionLoaded") {
      clearMessages();
      canExportReport = (msg.messages || []).some((item) =>
        item.role === "user" && (item.content || "").trim(),
      );
      (msg.messages || []).forEach((item) => {
        appendBubble(
          item.role === "user" ? "user" : "interviewer",
          item.role === "user" ? "我" : "面试官",
          item.content || "",
        );
      });
      interviewStarted = true;
      showChatPage();
      setStatus("已继续历史会话");
      updateExportButton();
      return;
    }
    if (msg.type === "prefill") {
      inputEl.value = msg.text || "";
      inputEl.focus();
      updateActionButton();
      return;
    }
    if (typeof msg.method !== "string") {
      return;
    }

    switch (msg.method) {
      case "stream":
        onStream(msg.params);
        break;
      case "tool_call":
        onToolCall(msg.params);
        break;
      case "runtime_metric":
        renderRuntimeMetric(msg.params);
        break;
      case "agent_event":
        renderAgentEvent(msg.params);
        break;
      case "done":
        onDone();
        break;
      case "cancelled":
        onCancelled(msg.params);
        break;
      case "error":
        onError(msg.params);
        break;
    }
  });

  function applyConfig(config) {
    currentConfig = config;
    actualRuntime = null;
    workspaceState = {
      hasWorkspace: Boolean(config.hasWorkspace),
      workspaceName: config.workspaceName || "",
      workspacePath: config.workspacePath || "",
    };
    workspaceInfoEl.textContent = workspaceState.hasWorkspace
      ? `自动读取：${workspaceState.workspaceName || workspaceState.workspacePath}`
      : "未打开目标项目文件夹";
    workspaceInfoEl.title = workspaceState.workspacePath || "";
    settingsBtn.title = config.demoMode
      ? "打开设置：当前为 Demo Mode"
      : `打开设置：当前模型 ${config.model || "未配置模型"}`;
    const runtime = actualRuntime || config.agentRuntime || "native";
    modelConfigSummaryEl.textContent = config.demoMode
      ? `当前：Demo Mode（不调用真实模型） · runtime ${runtime}`
      : `当前：${config.model || "未配置模型"} · ${config.baseUrl || "OpenAI 默认端点"} · runtime ${runtime} · ${config.hasApiKey ? "已配置 API Key" : "未配置 API Key"}`;
  }

  function onStream(params) {
    if (stopping) {
      return;
    }
    const delta = params.delta || "";
    if (!delta) {
      return;
    }
    const hadTrace = finishThinkingTrace();
    if (hadTrace) {
      currentInterviewerBubble = null;
    }
    if (!currentInterviewerBubble) {
      currentInterviewerBubble = appendBubble("interviewer", "面试官", "");
    }
    const segment = getCurrentInterviewerSegment();
    segment.__raw = (segment.__raw || "") + delta;
    segment.innerHTML = renderMarkdown(segment.__raw);
    scrollToBottom();
  }

  function onToolCall(params) {
    const { tool, phase, args, result } = params;
    if (phase === "start") {
      if (!thinkingBubble) {
        showThinkingOrb();
      }
      thinkingToolLogs.push({
        tool,
        args: formatArgs(args),
        result: "",
        running: true,
      });
      updateThinkingDetails();
    } else {
      const last = [...thinkingToolLogs].reverse().find((item) =>
        item.tool === tool && item.running,
      );
      if (last) {
        last.result = result || "";
        last.running = false;
      }
      updateThinkingDetails();
    }
    scrollToBottom();
  }

  function getCurrentInterviewerSegment() {
    const body = currentInterviewerBubble.querySelector(".bubble__body");
    const last = body.lastElementChild;
    if (last?.classList.contains("interviewer-segment")) {
      return last;
    }
    const segment = document.createElement("div");
    segment.className = "interviewer-segment";
    segment.__raw = "";
    body.appendChild(segment);
    return segment;
  }

  function onDone() {
    stopping = false;
    finishThinkingTrace();
    if (currentInterviewerBubble) {
      currentInterviewerBubble = null;
    }
    setAwaiting(false);
    inputEl.focus();
    canExportReport = true;
    updateExportButton();
    vscode.postMessage({ type: "listSessions" });
  }

  function onCancelled(params) {
    finishThinkingTrace();
    const partial = params && params.partial ? String(params.partial) : "";
    const bubble = currentInterviewerBubble || appendBubble("interviewer", "面试官", "");
    currentInterviewerBubble = bubble;
    const segment = getCurrentInterviewerSegment();
    const raw = segment.__raw || partial;
    const suffix = raw
      ? `\n\n（已停止，生成 ${raw.length} 字）`
      : "（已停止）";
    segment.__raw = raw + suffix;
    segment.innerHTML = renderMarkdown(segment.__raw);
    currentInterviewerBubble = null;
    stopping = false;
    setAwaiting(false);
    inputEl.focus();
    canExportReport = true;
    updateExportButton();
    vscode.postMessage({ type: "listSessions" });
  }

  function onError(params) {
    stopping = false;
    finishThinkingTrace();
    appendBubble("error", "出错了", params.message || "未知错误");
    if (currentInterviewerBubble) {
      currentInterviewerBubble = null;
    }
    setAwaiting(false);
  }

  function appendBubble(kind, title, body) {
    const bubble = document.createElement("div");
    bubble.className = `bubble bubble--${kind}`;

    const roleEl = document.createElement("div");
    roleEl.className = "bubble__role bubble__title";
    roleEl.textContent = title;
    bubble.appendChild(roleEl);

    const bodyEl = document.createElement("div");
    bodyEl.className = "bubble__body";
    if (kind === "interviewer") {
      // 面试官输出按 Markdown 渲染；渲染器先整体转义再转换，无注入风险
      const segment = document.createElement("div");
      segment.className = "interviewer-segment";
      segment.__raw = body || "";
      segment.innerHTML = renderMarkdown(segment.__raw);
      bodyEl.appendChild(segment);
    } else {
      bodyEl.textContent = body || "";
    }
    bubble.appendChild(bodyEl);

    messagesEl.appendChild(bubble);
    updateExportButton();
    scrollToBottom();
    return bubble;
  }

  function showThinkingOrb(options = {}) {
    if (thinkingBubble) {
      return;
    }
    if (options.resetLogs !== false) {
      thinkingToolLogs = [];
    }
    thinkingBubble = document.createElement("div");
    thinkingBubble.className = "bubble bubble--thinking";

    const bodyEl = document.createElement("div");
    bodyEl.className = "thinking";
    const orb = document.createElement("div");
    orb.className = "thinking-orb";
    orb.setAttribute("aria-hidden", "true");
    const text = document.createElement("span");
    text.className = "thinking__text";
    text.textContent = "Waiting for model...";
    const details = document.createElement("details");
    details.className = "thinking-details";
    const summary = document.createElement("summary");
    summary.className = "thinking-details__summary";
    summary.title = "查看运行命令和工具调用";
    summary.setAttribute("aria-label", "查看运行命令和工具调用");
    const detailBody = document.createElement("pre");
    detailBody.className = "thinking-details__body";
    detailBody.textContent = "暂无工具调用";
    details.append(summary, detailBody);
    bodyEl.append(orb, text, details);
    thinkingBubble.appendChild(bodyEl);

    messagesEl.appendChild(thinkingBubble);
    scrollToBottom();
  }

  function removeThinkingOrb() {
    if (!thinkingBubble) {
      return;
    }
    thinkingBubble.remove();
    thinkingBubble = null;
  }

  function finishThinkingTrace() {
    if (!thinkingBubble) {
      return false;
    }
    if (!thinkingToolLogs.length) {
      removeThinkingOrb();
      return false;
    }
    const orb = thinkingBubble.querySelector(".thinking-orb");
    if (orb) {
      orb.remove();
    }
    const text = thinkingBubble.querySelector(".thinking__text");
    if (text) {
      text.textContent = `已完成 ${thinkingToolLogs.length} 项运行步骤`;
    }
    thinkingBubble.classList.add("is-complete");
    thinkingBubble = null;
    scrollToBottom();
    return true;
  }

  // ──────────────────────────────────────────────
  // Markdown 渲染（面试官输出）
  // 流程：整体 HTML 转义 → 行级分块（代码块/标题/列表/引用）→ 行内转换。
  // 只输出自己拼的标签，转义过的文本无法注入 HTML。
  // ──────────────────────────────────────────────

  function escapeHtml(text) {
    return String(text)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;");
  }

  function renderInline(text) {
    // text 已转义。先抽出行内代码，避免内部内容再被粗体/斜体规则处理
    const codeSpans = [];
    let s = text.replace(/`([^`\n]+)`/g, (_m, code) => {
      codeSpans.push(code);
      return `\u0000${codeSpans.length - 1}\u0000`;
    });
    s = s.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
    s = s.replace(/(^|[^*])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
    // 链接不生成 <a>（webview 内导航不可用），以「文本（URL）」呈现
    s = s.replace(/\[([^\]\n]+)\]\((https?:\/\/[^)\s]+)\)/g, "$1（$2）");
    return s.replace(/\u0000(\d+)\u0000/g, (_m, i) =>
      `<code class="md-code">${codeSpans[Number(i)]}</code>`,
    );
  }

  function renderMarkdown(raw) {
    const lines = escapeHtml(raw ?? "").split(/\r?\n/);
    const out = [];
    let listType = null;
    let quoteOpen = false;
    let para = [];

    const closePara = () => {
      if (para.length) {
        out.push(`<p>${para.map(renderInline).join("<br>")}</p>`);
        para = [];
      }
    };
    const closeList = () => {
      if (listType) {
        out.push(`</${listType}>`);
        listType = null;
      }
    };
    const closeQuote = () => {
      if (quoteOpen) {
        out.push("</blockquote>");
        quoteOpen = false;
      }
    };
    const closeAll = () => {
      closePara();
      closeList();
      closeQuote();
    };

    for (let i = 0; i < lines.length; i++) {
      const trimmed = lines[i].trim();

      if (trimmed.startsWith("```")) {
        closeAll();
        const buf = [];
        i += 1;
        while (i < lines.length && !lines[i].trim().startsWith("```")) {
          buf.push(lines[i]);
          i += 1;
        }
        out.push(`<pre class="md-pre"><code>${buf.join("\n")}</code></pre>`);
        continue;
      }
      if (!trimmed) {
        closeAll();
        continue;
      }
      const heading = trimmed.match(/^(#{1,6})\s+(.*)$/);
      if (heading) {
        closeAll();
        out.push(
          `<div class="md-h md-h${heading[1].length}">${renderInline(heading[2])}</div>`,
        );
        continue;
      }
      if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
        closeAll();
        out.push('<hr class="md-hr" />');
        continue;
      }
      if (trimmed.startsWith("&gt;")) {
        closePara();
        closeList();
        if (!quoteOpen) {
          out.push('<blockquote class="md-quote">');
          quoteOpen = true;
        }
        out.push(`<p>${renderInline(trimmed.replace(/^&gt;\s?/, ""))}</p>`);
        continue;
      }
      const ul = trimmed.match(/^[-*+]\s+(.*)$/);
      if (ul) {
        closePara();
        closeQuote();
        if (listType !== "ul") {
          closeList();
          out.push('<ul class="md-ul">');
          listType = "ul";
        }
        out.push(`<li>${renderInline(ul[1])}</li>`);
        continue;
      }
      const ol = trimmed.match(/^(\d+)[.)]\s+(.*)$/);
      if (ol) {
        closePara();
        closeQuote();
        if (listType !== "ol") {
          closeList();
          out.push('<ol class="md-ol">');
          listType = "ol";
        }
        out.push(`<li>${renderInline(ol[2])}</li>`);
        continue;
      }
      closeList();
      closeQuote();
      para.push(lines[i]);
    }
    closeAll();
    return out.join("");
  }

  function formatArgs(args) {
    if (!args) {
      return "";
    }
    try {
      return JSON.stringify(args);
    } catch {
      return String(args);
    }
  }

  function updateThinkingDetails() {
    if (!thinkingBubble) {
      return;
    }
    const body = thinkingBubble.querySelector(".thinking-details__body");
    if (body) {
      body.textContent = formatThinkingDetails();
    }
  }

  function formatThinkingDetails() {
    if (!thinkingToolLogs.length) {
      return "暂无工具调用";
    }
    return thinkingToolLogs.map((item, index) => {
      const parts = [`#${index + 1} ${item.running ? "运行中" : "已完成"}：${item.tool}`];
      if (item.args) {
        parts.push(`参数：${item.args}`);
      }
      if (item.result) {
        parts.push(`结果：\n${item.result}`);
      }
      return parts.join("\n");
    }).join("\n\n");
  }

  function renderRuntimeMetric(metric) {
    if (!runtimeDiagnosticsEl || !metric) {
      return;
    }
    actualRuntime = metric.runtime || actualRuntime;
    if (currentConfig) {
      const runtime = actualRuntime || currentConfig.agentRuntime || "native";
      modelConfigSummaryEl.textContent = currentConfig.demoMode
        ? `当前：Demo Mode（不调用真实模型） · runtime ${runtime}`
        : `当前：${currentConfig.model || "未配置模型"} · ${currentConfig.baseUrl || "OpenAI 默认端点"} · runtime ${runtime} · ${currentConfig.hasApiKey ? "已配置 API Key" : "未配置 API Key"}`;
    }
    const firstDelta = typeof metric.first_delta_ms === "number"
      ? `${metric.first_delta_ms}ms`
      : "-";
    const tools = Array.isArray(metric.tools) && metric.tools.length
      ? metric.tools.map((item) => {
        const elapsed = typeof item.elapsed_ms === "number" ? `${item.elapsed_ms}ms` : "-";
        const keys = Array.isArray(item.args_keys) && item.args_keys.length
          ? ` 参数:${item.args_keys.join(",")}`
          : "";
        const result = typeof item.result_chars === "number"
          ? ` 结果:${item.result_chars}字`
          : "";
        return `${item.tool} ${elapsed}${keys}${result}`;
      }).join("；")
      : "无工具调用";
    const budget = metric.budget || {};
    runtimeDiagnosticsEl.textContent = [
      `runtime ${metric.runtime || "unknown"} · ${metric.status || "-"}`,
      `首 token ${firstDelta} · 模型 ${metric.model_elapsed_ms || 0}ms · 总计 ${metric.total_elapsed_ms || 0}ms`,
      `token 粗估 ${metric.estimated_tokens || 0} · 错误 ${metric.error_kind || "-"}`,
      `压缩 ${metric.compaction?.state || "disabled"}`,
      `预算 ${budget.steps_used || 0}/${budget.hard_limit || "-"} · ${budget.reason || "-"}`,
      `工具 ${tools}`,
    ].join("\n");
  }

  function renderAgentEvent(event) {
    if (!runtimeTimelineEl || !event || !event.event) {
      return;
    }
    if (!KNOWN_RUNTIME_EVENTS.has(String(event.event))) {
      runtimeIgnoredEvents += 1;
      renderRuntimeTimeline();
      return;
    }
    const normalized = normalizeRuntimeEvent(event);
    const last = runtimeEvents[runtimeEvents.length - 1];
    if (last && last.event === "message_update" && normalized.event === "message_update") {
      last.delta_chars = Math.min(
        MAX_RUNTIME_EVENT_CHARS,
        (last.delta_chars || 0) + (normalized.delta_chars || 0),
      );
    } else {
      runtimeEvents.push(normalized);
    }
    runtimeEvents.sort((a, b) => (a.event_seq ?? Number.MAX_SAFE_INTEGER) - (b.event_seq ?? Number.MAX_SAFE_INTEGER));
    capRuntimeEvents();
    renderRuntimeTimeline();
  }

  function normalizeRuntimeEvent(event) {
    // 只复制稳定、脱敏字段，避免把未来事件对象的正文带进内存时间线。
    return {
      event: String(event.event).slice(0, MAX_RUNTIME_EVENT_CHARS),
      event_seq: Number.isInteger(event.event_seq) ? Math.max(0, event.event_seq) : null,
      elapsed_ms: Number.isInteger(event.elapsed_ms) ? Math.max(0, event.elapsed_ms) : null,
      step: Number.isInteger(event.step) ? Math.max(0, event.step) : null,
      tool: boundedRuntimeText(event.tool),
      state: boundedRuntimeText(event.state),
      delta_chars: Number.isInteger(event.delta_chars)
        ? Math.min(Math.max(event.delta_chars, 0), MAX_RUNTIME_EVENT_CHARS)
        : 0,
      result_chars: Number.isInteger(event.result_chars)
        ? Math.max(event.result_chars, 0)
        : null,
      error_kind: boundedRuntimeText(event.error_kind),
    };
  }

  function boundedRuntimeText(value) {
    // 限制异常长字段，避免错误信息或工具名撑大诊断面板。
    if (!value) {
      return "";
    }
    const text = String(value);
    if (text.length > MAX_RUNTIME_EVENT_CHARS) {
      runtimeTruncatedEvents += 1;
      return `${text.slice(0, MAX_RUNTIME_EVENT_CHARS - 1)}…`;
    }
    return text;
  }

  function capRuntimeEvents() {
    // 同时限制条数和渲染字符数，避免长会话无限增长。
    if (runtimeEvents.length > MAX_RUNTIME_EVENTS) {
      runtimeDroppedEvents += runtimeEvents.length - MAX_RUNTIME_EVENTS;
      runtimeEvents = runtimeEvents.slice(-MAX_RUNTIME_EVENTS);
    }
    while (runtimeEvents.length > 1 && runtimeEvents.map(formatAgentEvent).join("\n").length > MAX_RUNTIME_TIMELINE_CHARS) {
      runtimeEvents.shift();
      runtimeDroppedEvents += 1;
    }
  }

  function renderRuntimeTimeline() {
    // 时间线只显示调试摘要，丢弃/截断统计也保持脱敏。
    if (!runtimeTimelineEl) {
      return;
    }
    const lines = runtimeEvents.map(formatAgentEvent);
    const notes = [];
    if (runtimeDroppedEvents) {
      notes.push(`已丢弃 ${runtimeDroppedEvents} 条旧事件`);
    }
    if (runtimeTruncatedEvents) {
      notes.push(`已截断 ${runtimeTruncatedEvents} 条字段`);
    }
    if (runtimeIgnoredEvents) {
      notes.push(`已忽略 ${runtimeIgnoredEvents} 条未知事件`);
    }
    runtimeTimelineEl.textContent = [...lines, ...notes].join("\n") || "暂无运行事件";
  }

  function formatAgentEvent(event) {
    const seq = event.event_seq == null ? "-" : `#${event.event_seq}`;
    const elapsed = event.elapsed_ms == null ? "-" : `${event.elapsed_ms}ms`;
    const target = event.tool ? ` ${event.tool}` : "";
    const state = event.state ? ` ${event.state}` : "";
    const step = event.step == null ? "" : ` step:${event.step}`;
    const delta = event.event === "message_update" ? ` ${event.delta_chars}字` : "";
    const result = event.result_chars == null ? "" : ` 结果${event.result_chars}字`;
    const error = event.error_kind ? ` 错误:${event.error_kind}` : "";
    return `${seq} ${elapsed}${step} ${event.event}${target}${state}${delta}${result}${error}`
      .slice(0, MAX_RUNTIME_EVENT_CHARS * 2);
  }

  function setStatus(text) {
    configStatusEl.textContent = text;
  }

  function formatOcrProgress(progress) {
    const message = progress?.message || "正在 OCR 识别...";
    const elapsed = typeof progress?.elapsedMs === "number"
      ? `，已耗时 ${Math.max(0, Math.round(progress.elapsedMs / 1000))}s`
      : "";
    if (
      typeof progress?.currentPage === "number"
      && typeof progress?.totalPages === "number"
      && progress.totalPages > 0
    ) {
      const page = `第 ${progress.currentPage}/${progress.totalPages} 页`;
      return message.includes(page)
        ? `${message}${elapsed}`
        : `${message}（${page}${elapsed}）`;
    }
    return `${message}${elapsed}`;
  }

  function setModelTestStatus(text, kind) {
    modelTestStatusEl.textContent = text;
    modelTestStatusEl.classList.toggle("is-error", kind === "error");
    modelTestStatusEl.classList.toggle("is-success", kind === "success");
  }

  function showDependencyStatus(status) {
    dependencyPanelEl.classList.toggle("is-hidden", !status.message);
    if (status.message) {
      showSetupPage();
    }
    dependencyMessageEl.textContent = status.message || "";
    dependencyCommandEl.textContent = status.command || "";
    dependencyCommandEl.style.display = status.command ? "block" : "none";
    installDependenciesBtn.style.display = status.canInstall ? "inline-flex" : "none";
    installDependenciesBtn.dataset.installType = status.installType || "agent";
    installDependenciesBtn.textContent = status.buttonLabel || "安装 Agent 依赖";
  }

  function renderSessions(sessions, current) {
    sessionListEl.textContent = "";
    if (!sessions.length) {
      const empty = document.createElement("div");
      empty.className = "session-item__meta";
      empty.textContent = "暂无历史会话";
      sessionListEl.appendChild(empty);
      return;
    }
    sessions.forEach((session) => {
      const item = document.createElement("div");
      item.className = "session-item";

      const top = document.createElement("div");
      top.className = "session-item__top";

      const title = document.createElement("div");
      title.className = "session-item__title";
      title.textContent = session.id === current ? `${session.title}（当前）` : session.title;

      const buttons = document.createElement("div");
      buttons.className = "session-item__buttons";

      const resume = document.createElement("button");
      resume.className = "secondary-button";
      resume.type = "button";
      resume.textContent = "继续";
      resume.addEventListener("click", () => {
        vscode.postMessage({ type: "resumeSession", session: session.id });
      });

      const remove = document.createElement("button");
      remove.className = "secondary-button";
      remove.type = "button";
      remove.textContent = "删除";
      remove.addEventListener("click", () => {
        vscode.postMessage({ type: "deleteSession", session: session.id });
      });

      buttons.append(resume, remove);
      top.append(title, buttons);

      const preview = document.createElement("div");
      preview.className = "session-item__preview";
      preview.textContent = session.preview || "";

      const meta = document.createElement("div");
      meta.className = "session-item__meta";
      meta.textContent = `${new Date(session.updatedAt).toLocaleString()} · ${session.messageCount} 条`;

      item.append(top, preview, meta);
      sessionListEl.appendChild(item);
    });
  }

  function clearMessages() {
    messagesEl.textContent = "";
    currentInterviewerBubble = null;
    thinkingBubble = null;
    thinkingToolLogs = [];
    runtimeEvents = [];
    runtimeDroppedEvents = 0;
    runtimeTruncatedEvents = 0;
    runtimeIgnoredEvents = 0;
    if (runtimeTimelineEl) {
      runtimeTimelineEl.textContent = "暂无运行事件";
    }
    stopping = false;
    setAwaiting(false);
    canExportReport = false;
    updateExportButton();
  }

  function updateExportButton() {
    exportReportBtn.disabled = !canExportReport || awaiting;
  }

  function scrollToBottom() {
    messagesEl.scrollTop = messagesEl.scrollHeight;
  }

  function showSetupPage(status) {
    setResumeCaptureState(true);
    setupEl.classList.remove("is-collapsed");
    chatEl.classList.add("is-hidden");
    if (status) {
      setStatus(status);
    }
  }

  function showChatPage() {
    setResumeCaptureState(false);
    setupEl.classList.add("is-collapsed");
    chatEl.classList.remove("is-hidden");
    inputEl.focus();
    scrollToBottom();
  }

  historyToggleBtn.addEventListener("click", () => {
    historyPanelEl.classList.toggle("is-collapsed");
    vscode.postMessage({ type: "listSessions" });
  });
  settingsBtn.addEventListener("click", () => {
    vscode.postMessage({ type: "openSettings" });
  });
  openModelSettingsBtn.addEventListener("click", () => {
    vscode.postMessage({ type: "openSettings" });
  });
  exportReportBtn.addEventListener("click", () => {
    if (exportReportBtn.disabled) {
      return;
    }
    const ok = window.confirm(
      "导出的 Markdown 报告会包含 JD、简历和面试对话内容，并保存到当前工作区。确认导出？",
    );
    if (ok) {
      vscode.postMessage({ type: "exportReport" });
    }
  });
  installDependenciesBtn.addEventListener("click", () => {
    vscode.postMessage({
      type: installDependenciesBtn.dataset.installType === "ocr"
        ? "installOcrDependencies"
        : "installDependencies",
    });
  });
  checkDependenciesBtn.addEventListener("click", () => {
    vscode.postMessage({ type: "checkDependencies" });
  });
  testModelConnectionBtn.addEventListener("click", () => {
    setModelTestStatus("正在测试模型连接...", "");
    vscode.postMessage({ type: "testModelConnection" });
  });
  newSessionBtn.addEventListener("click", () => {
    vscode.postMessage({ type: "newSession" });
  });
  refreshSessionsBtn.addEventListener("click", () => {
    vscode.postMessage({ type: "listSessions" });
  });
  resumeFileInputEl.addEventListener("change", () => {
    const file = resumeFileInputEl.files?.[0];
    if (file) {
      readDroppedResume(file);
    }
    resumeFileInputEl.value = "";
  });
  pickResumeBtn.addEventListener("click", (event) => {
    if (event.target === resumeFileInputEl) {
      return;
    }
    event.preventDefault();
    vscode.postMessage({ type: "pickResume" });
  });
  pickResumeBtn.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      vscode.postMessage({ type: "pickResume" });
    }
  });
  [pickResumeBtn, resumeFileInputEl].forEach((dropTarget) => {
    dropTarget.addEventListener("dragenter", (event) => {
      onResumeDrag(event);
    });
    dropTarget.addEventListener("dragover", (event) => {
      onResumeDrag(event);
    });
    dropTarget.addEventListener("dragleave", () => {
      pickResumeBtn.classList.remove("is-dragover");
    });
  });
  resumeFileInputEl.addEventListener("drop", () => {
    pickResumeBtn.classList.remove("is-dragover");
  });
  pickResumeBtn.addEventListener("drop", (event) => {
    if (event.target === resumeFileInputEl && event.dataTransfer?.files?.length) {
      return;
    }
    handleResumeDrop(event);
  });
  document.addEventListener("dragenter", (event) => {
    if (!isResumeDropEvent(event)) {
      return;
    }
    onResumeDrag(event);
    pickResumeBtn.classList.add("is-dragover");
  }, true);
  document.addEventListener("dragover", (event) => {
    if (!isResumeDropEvent(event)) {
      return;
    }
    onResumeDrag(event);
    pickResumeBtn.classList.add("is-dragover");
  }, true);
  document.addEventListener("drop", (event) => {
    if (!isResumeDropEvent(event)) {
      return;
    }
    if (event.target === resumeFileInputEl && event.dataTransfer?.files?.length) {
      pickResumeBtn.classList.remove("is-dragover");
      return;
    }
    handleResumeDrop(event);
  }, true);
  startInterviewBtn.addEventListener("click", startInterview);
  actionBtn.addEventListener("click", onAction);

  inputEl.addEventListener("input", updateActionButton);
  inputEl.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });

  vscode.postMessage({ type: "ready" });
  vscode.postMessage({ type: "listSessions" });
  updateActionButton();
  updateExportButton();
  if (!interviewStarted) {
    setResumeCaptureState(true);
    jdEl.focus();
  }

  function isResumeDropEvent(event) {
    return !setupEl.classList.contains("is-collapsed");
  }

  function onResumeDrag(event) {
    armResumeFileDrop();
    if (!isSystemFileDrag(event)) {
      event.preventDefault();
    }
    pickResumeBtn.classList.add("is-dragover");
  }

  function armResumeFileDrop() {
    const now = Date.now();
    if (now - resumeDropArmSentAt < 1000) {
      return;
    }
    resumeDropArmSentAt = now;
    vscode.postMessage({ type: "armResumeFileDrop" });
  }

  function setResumeCaptureState(enabled) {
    if (resumeCaptureEnabled === enabled) {
      return;
    }
    resumeCaptureEnabled = enabled;
    vscode.postMessage({ type: "resumeCaptureState", enabled });
  }

  function isSystemFileDrag(event) {
    return Array.from(event.dataTransfer?.types || []).includes("Files");
  }

  function handleResumeDrop(event) {
    event.preventDefault();
    event.stopPropagation();
    pickResumeBtn.classList.remove("is-dragover");
    const payload = getDroppedResumePayload(event);
    if (!payload) {
      setStatus("没有识别到拖拽文件，请拖入单个简历文件，或点击上传区域选择文件。");
      return;
    }
    if (payload.type === "file") {
      readDroppedResume(payload.file);
      return;
    }
    setStatus("正在读取拖拽文件...");
    vscode.postMessage({ type: "pickResumePath", path: payload.path });
  }

  // VS Code 资源管理器拖入通常只有 URI 文本；系统文件拖入通常有 File 对象。
  function getDroppedResumePayload(event) {
    const file = event.dataTransfer?.files?.[0];
    if (file) {
      return { type: "file", file };
    }
    const items = Array.from(event.dataTransfer?.items || []);
    const fileItem = items.find((item) => item.kind === "file");
    const itemFile = fileItem?.getAsFile();
    if (itemFile) {
      return { type: "file", file: itemFile };
    }
    const path = getDroppedFilePath(event.dataTransfer);
    return path ? { type: "path", path } : null;
  }

  function getDroppedFilePath(dataTransfer) {
    const types = Array.from(dataTransfer?.types || []);
    const candidates = [
      "text/uri-list",
      "application/vnd.code.tree.resourceuris",
      "application/vnd.code.tree.explorer",
      "text/plain",
      ...types.filter((type) => type.startsWith("application/vnd.code.tree.")),
    ];
    for (const type of [...new Set(candidates)]) {
      const value = dataTransfer?.getData?.(type);
      const path = parseDroppedFilePath(value);
      if (path) {
        return path;
      }
    }
    return "";
  }

  function parseDroppedFilePath(value) {
    const entries = readDroppedEntries(value);
    for (const entry of entries) {
      const path = fileUriToFsPath(entry) || normalizeDroppedFsPath(entry);
      if (path && isSupportedResumePath(path)) {
        return path;
      }
    }
    return "";
  }

  function readDroppedEntries(value) {
    const text = String(value || "").trim();
    if (!text) {
      return [];
    }
    try {
      const parsed = JSON.parse(text);
      if (Array.isArray(parsed)) {
        return parsed.flatMap(readDroppedEntries);
      }
      if (parsed && typeof parsed === "object") {
        return readDroppedEntries(parsed.resourceUri || parsed.uri || parsed.fsPath || "");
      }
    } catch {
      // 非 JSON 的拖拽载荷继续按普通文本解析。
    }
    return text
      .split(/\r?\n/)
      .map((line) => line.trim())
      .filter((line) => line && !line.startsWith("#"));
  }

  function fileUriToFsPath(value) {
    if (!/^file:/i.test(value)) {
      return "";
    }
    try {
      const url = new URL(value);
      let path = decodeURIComponent(url.pathname || "");
      if (/^\/[a-zA-Z]:\//.test(path)) {
        path = path.slice(1);
      }
      return path.replace(/\//g, "\\");
    } catch {
      return "";
    }
  }

  function normalizeDroppedFsPath(value) {
    const path = String(value || "").trim().replace(/^"|"$/g, "");
    return /^[a-zA-Z]:[\\/]/.test(path) ? path : "";
  }

  function isSupportedResumePath(path) {
    return /\.(pdf|docx|txt|md|markdown|png|jpe?g|webp)$/i.test(path);
  }

  function readDroppedResume(file) {
    if (file.size > DROPPED_RESUME_MAX_BYTES) {
      setStatus("简历文件超过 10MB，请点击上传区域选择文件。");
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = String(reader.result || "");
      const dataBase64 = dataUrl.includes(",") ? dataUrl.split(",").pop() : dataUrl;
      vscode.postMessage({
        type: "pickResumeUpload",
        fileName: file.name,
        dataBase64,
      });
    };
    reader.onerror = () => {
      setStatus("拖拽文件读取失败，请点击上传区域选择文件。");
    };
    setStatus("正在读取拖拽文件...");
    reader.readAsDataURL(file);
  }
})();
