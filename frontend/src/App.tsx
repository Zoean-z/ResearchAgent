import { useEffect, useRef, useState } from "react";
import {
  BookOpenText,
  BrainCircuit,
  ArrowDownToLine,
  FileUp,
  Languages,
  MessageSquarePlus,
  Settings2,
  Trash2,
  X,
} from "lucide-react";

import { api } from "./lib/api";
import type {
  MemorySnapshot,
  Message,
  RuntimeStatus,
  Session,
  TaskRun,
  TimelineEvent,
  TraceStep,
} from "./lib/types";

type InspectorTab = "memory" | "timeline";
type DrawerKind = "memory" | "settings" | null;
type MixedInputMode = "query" | "ingest";
type MemoryKind = "all" | "paper_memory" | "relation_memory" | "open_question_memory";
type ProgressState = "done" | "active" | "pending";
type UiLanguage = "zh" | "en";

type LiveRunState = {
  runId: string;
  status: string;
  stepCount: number;
  currentAction: string | null;
  startedAt: string | null;
  finishedAt: string | null;
  finishReason: string | null;
};

type ComposerClassification = {
  kind: "empty" | "plain_query" | "pure_arxiv" | "mixed_with_arxiv";
  arxivToken: string | null;
};

const PURE_ARXIV_PATTERNS = [
  /^(https?:\/\/)?(www\.)?arxiv\.org\/abs\/[0-9]{4}\.[0-9]{4,5}(v\d+)?\/?$/i,
  /^(https?:\/\/)?(www\.)?arxiv\.org\/pdf\/[0-9]{4}\.[0-9]{4,5}(v\d+)?(\.pdf)?\/?$/i,
  /^(arxiv:)?[0-9]{4}\.[0-9]{4,5}(v\d+)?$/i,
];
const EMBEDDED_ARXIV_PATTERN =
  /(https?:\/\/(?:www\.)?arxiv\.org\/(?:abs|pdf)\/[0-9]{4}\.[0-9]{4,5}(?:v\d+)?(?:\.pdf)?|(?:arxiv:)?[0-9]{4}\.[0-9]{4,5}(?:v\d+)?)/i;

function classifyComposer(value: string): ComposerClassification {
  const trimmed = value.trim();
  if (!trimmed) {
    return { kind: "empty", arxivToken: null };
  }
  if (PURE_ARXIV_PATTERNS.some((pattern) => pattern.test(trimmed))) {
    return { kind: "pure_arxiv", arxivToken: trimmed };
  }
  const embedded = trimmed.match(EMBEDDED_ARXIV_PATTERN);
  if (embedded) {
    return { kind: "mixed_with_arxiv", arxivToken: embedded[0] };
  }
  return { kind: "plain_query", arxivToken: null };
}

function formatTime(value: string): string {
  return new Date(value).toLocaleString("zh-CN", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function memoryKindLabel(kind: MemoryKind): string {
  if (kind === "paper_memory") return "论文记忆";
  if (kind === "relation_memory") return "关系记忆";
  if (kind === "open_question_memory") return "开放问题";
  return "全部";
}

function classifyRunType(messageType: string | undefined): "query" | "ingest" {
  return messageType?.startsWith("ingest") ? "ingest" : "query";
}

function summarizeRun(run: TaskRun, messageType: string | undefined): string {
  const runType = classifyRunType(messageType);
  const prefix = runType === "ingest" ? "导入" : "问答";
  return `${prefix} · ${runStatusLabel(run.status)} · ${run.step_count} 步`;
}

function readStringList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string");
}

function runTypeLabel(messageType: string | undefined): string {
  return classifyRunType(messageType) === "ingest" ? "导入" : "问答";
}

function runStatusLabel(status: string | null | undefined): string {
  if (status === "finished") return "完成";
  if (status === "running") return "运行中";
  if (status === "pending") return "等待中";
  if (status === "failed") return "失败";
  if (status === "step_limit_reached") return "已到步数上限";
  return status ?? "未知";
}

function messageTypeLabel(messageType: string | undefined): string {
  if (messageType === "followup_query") return "追问";
  if (messageType === "ingest_arxiv") return "arXiv 导入";
  if (messageType === "ingest_pdf") return "PDF 导入";
  return messageType ?? "消息";
}

function eventTypeLabel(eventType: string): string {
  if (eventType === "run_created") return "运行已创建";
  if (eventType === "run_started") return "运行已开始";
  if (eventType === "step_completed") return "步骤已完成";
  if (eventType === "assistant_message_committed") return "助手消息已写入";
  if (eventType === "run_finished") return "运行已完成";
  if (eventType === "run_failed") return "运行失败";
  return eventType;
}

function runStatusTone(status: string | null | undefined): string {
  if (status === "finished") return "success";
  if (status === "failed" || status === "step_limit_reached") return "danger";
  if (status === "running") return "active";
  return "muted";
}

function truncateText(value: string, maxLength: number): string {
  if (value.length <= maxLength) {
    return value;
  }
  return `${value.slice(0, maxLength - 1)}...`;
}

function buildSyntheticEvent(runId: string, eventType: string, summary: string): TimelineEvent {
  const stamp = new Date().toISOString();
  return {
    id: `live-${runId}-${eventType}-${stamp}`,
    session_id: "",
    run_id: runId,
    event_type: eventType,
    summary,
    related_memory_ids: [],
    related_paper_ids: [],
    created_at: stamp,
  };
}

function readString(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

function readBoolean(value: unknown): boolean | null {
  return typeof value === "boolean" ? value : null;
}

function readNumber(value: unknown): number | null {
  return typeof value === "number" ? value : null;
}

function stringifyCompact(value: unknown): string {
  if (typeof value === "string") {
    return value;
  }
  if (typeof value === "number" || typeof value === "boolean") {
    return String(value);
  }
  if (Array.isArray(value)) {
    return value.map((item) => stringifyCompact(item)).join(", ");
  }
  if (value && typeof value === "object") {
    return JSON.stringify(value);
  }
  return "n/a";
}

function summarizeTraceResult(step: TraceStep): string[] {
  const payload = step.result_payload;
  const lines: string[] = [];

  if (step.action === "retrieve_session_memories" || step.action === "retrieve_global_memories") {
    lines.push(`memory_ids=${readStringList(payload.memory_ids).join(", ") || "none"}`);
    const coverage = readNumber(payload.coverage_score);
    if (coverage !== null) {
      lines.push(`coverage=${coverage.toFixed(2)}`);
    }
    lines.push(`matched_terms=${readStringList(payload.matched_query_terms).join(", ") || "none"}`);
  } else if (step.action === "rerank_context_candidates") {
    lines.push(`selected_memory_ids=${readStringList(payload.selected_memory_ids).join(", ") || "none"}`);
    lines.push(`selection_source=${stringifyCompact(payload.selection_source)}`);
    lines.push(`fallback=${stringifyCompact(payload.fallback_used)}`);
  } else if (step.action === "decide_reread_source") {
    lines.push(`should_reread=${stringifyCompact(payload.should_reread_source)}`);
    lines.push(`reason=${stringifyCompact(payload.reason)}`);
  } else if (step.action === "reread_source_passages") {
    lines.push(`chunk_ids=${readStringList(payload.chunk_ids).join(", ") || "none"}`);
    lines.push(`paper_ids=${readStringList(payload.paper_ids).join(", ") || "none"}`);
    lines.push(`selection_source=${stringifyCompact(payload.selection_source)}`);
  } else if (step.action === "compose_mock_answer") {
    lines.push(`answer_preview=${stringifyCompact(payload.answer_preview)}`);
    lines.push(`memory_citations=${Array.isArray(payload.memory_citations) ? payload.memory_citations.length : 0}`);
    lines.push(`source_chunks=${Array.isArray(payload.source_reread_chunks) ? payload.source_reread_chunks.length : 0}`);
  } else if (step.action === "direct_final_answer") {
    lines.push(`answer_preview=${stringifyCompact(payload.answer_preview)}`);
    lines.push(`retrieval_skipped=${stringifyCompact(step.input_payload.retrieval_skipped)}`);
  } else if (step.action === "inspect_ingest_request") {
    lines.push(`paper_id=${stringifyCompact(payload.paper_id)}`);
    lines.push(`artifact_id=${stringifyCompact(payload.artifact_id)}`);
    lines.push(`session_document_id=${stringifyCompact(payload.session_document_id)}`);
  } else if (step.action === "extract_local_pdf_text" || step.action === "extract_arxiv_pdf_text") {
    lines.push(`paper_id=${stringifyCompact(payload.paper_id)}`);
    lines.push(`artifact_id=${stringifyCompact(payload.artifact_id)}`);
    lines.push(`chunk_count=${stringifyCompact(payload.chunk_count)}`);
  } else if (step.action === "persist_pdf_chunks" || step.action === "persist_arxiv_chunks") {
    lines.push(`chunk_count=${stringifyCompact(payload.chunk_count)}`);
    lines.push(`session_document_id=${stringifyCompact(payload.session_document_id)}`);
  } else if (step.action === "compose_ingest_summary") {
    lines.push(`paper_id=${stringifyCompact(payload.paper_id)}`);
    lines.push(`summary=${stringifyCompact(payload.summary)}`);
  } else if (step.action === "extract_paper_memory") {
    lines.push(`paper_memory_id=${stringifyCompact(payload.paper_memory_id)}`);
    lines.push(`operation=${stringifyCompact(payload.paper_operation)}`);
  } else if (step.action === "derive_relation_memory") {
    lines.push(`relation_memory_id=${stringifyCompact(payload.relation_memory_id)}`);
    lines.push(`operation=${stringifyCompact(payload.relation_operation)}`);
  } else if (step.action === "capture_open_questions") {
    lines.push(`open_question_memory_id=${stringifyCompact(payload.open_question_memory_id)}`);
    lines.push(`operation=${stringifyCompact(payload.open_question_operation)}`);
  }

  return lines;
}

function describeStepAction(action: string): string {
  if (action === "direct_final_answer") return "直接生成回答";
  if (action === "retrieve_session_memories") return "检索会话记忆";
  if (action === "retrieve_global_memories") return "检索全局记忆";
  if (action === "rerank_context_candidates") return "重排记忆候选";
  if (action === "decide_reread_source") return "判断是否重读原文";
  if (action === "reread_source_passages") return "读取原文片段";
  if (action === "compose_mock_answer") return "生成回答";
  if (action === "inspect_ingest_request") return "检查导入请求";
  if (action === "extract_local_pdf_text") return "抽取本地 PDF 文本";
  if (action === "extract_arxiv_pdf_text") return "抽取 arXiv PDF 文本";
  if (action === "persist_pdf_chunks") return "保存 PDF 分块";
  if (action === "persist_arxiv_chunks") return "保存 arXiv 分块";
  if (action === "compose_ingest_summary") return "生成导入摘要";
  if (action === "extract_paper_memory") return "写入论文记忆";
  if (action === "derive_relation_memory") return "写入关系记忆";
  if (action === "capture_open_questions") return "写入开放问题记忆";
  return action;
}

function deriveRunProgress(
  messageType: string | undefined,
  traceSteps: TraceStep[],
  runStatus: string | null,
): Array<{ action: string; label: string; state: ProgressState }> {
  const querySteps = [
    ["direct_final_answer", "直接回答"],
    ["retrieve_session_memories", "会话记忆"],
    ["retrieve_global_memories", "全局记忆"],
    ["rerank_context_candidates", "重排"],
    ["decide_reread_source", "重读判断"],
    ["reread_source_passages", "读取原文"],
    ["compose_mock_answer", "回答"],
  ] as const;
  const ingestSteps = [
    ["inspect_ingest_request", "检查请求"],
    ["extract_local_pdf_text", "抽取 PDF"],
    ["extract_arxiv_pdf_text", "抽取 arXiv"],
    ["persist_pdf_chunks", "保存分块"],
    ["persist_arxiv_chunks", "保存分块"],
    ["compose_ingest_summary", "生成摘要"],
    ["extract_paper_memory", "论文记忆"],
    ["derive_relation_memory", "关系记忆"],
    ["capture_open_questions", "开放问题"],
  ] as const;

  const observed = new Set(traceSteps.map((step) => step.action));
  const source = classifyRunType(messageType) === "ingest" ? ingestSteps : querySteps;
  const ordered = new Map<string, string>();
  for (const [action, label] of source) {
    if (observed.has(action) && !ordered.has(action)) {
      ordered.set(action, label);
    }
  }

  const visibleProgress: Array<{ action: string; label: string; state: ProgressState }> = Array.from(ordered.entries()).map(
    ([action, label]) => ({ action, label, state: "done" as const }),
  );
  if (runStatus !== "running") {
    return visibleProgress;
  }

  const nextStep = source.find(([action]) => !observed.has(action));
  if (nextStep) {
    visibleProgress.push({ action: nextStep[0], label: nextStep[1], state: "active" as const });
  }
  return visibleProgress;
}

function deriveEvidenceBlocks(
  messageType: string | undefined,
  traceSteps: TraceStep[],
): Array<{ title: string; lines: string[] }> {
  const findStep = (action: string) => [...traceSteps].reverse().find((step) => step.action === action);
  const blocks: Array<{ title: string; lines: string[] }> = [];

  if (classifyRunType(messageType) === "query") {
    const rerankStep = findStep("rerank_context_candidates");
    const rereadStep = findStep("reread_source_passages");
    const answerStep = findStep("compose_mock_answer");

    if (rerankStep) {
      blocks.push({
        title: "记忆证据",
        lines: [
          `selected=${readStringList(rerankStep.result_payload.selected_memory_ids).join(", ") || "none"}`,
          `source=${stringifyCompact(rerankStep.result_payload.selection_source)}`,
          `fallback=${stringifyCompact(rerankStep.result_payload.fallback_used)}`,
        ],
      });
    }
    if (rereadStep) {
      blocks.push({
        title: "原文证据",
        lines: [
          `chunks=${readStringList(rereadStep.result_payload.chunk_ids).join(", ") || "none"}`,
          `papers=${readStringList(rereadStep.result_payload.paper_ids).join(", ") || "none"}`,
          `strategy=${stringifyCompact(rereadStep.result_payload.selection_source)}`,
        ],
      });
    }
    if (answerStep) {
      blocks.push({
        title: "回答载荷",
        lines: [
          `preview=${truncateText(stringifyCompact(answerStep.result_payload.answer_preview), 140)}`,
          `memory_citations=${stringifyCompact(Array.isArray(answerStep.result_payload.memory_citations) ? answerStep.result_payload.memory_citations.length : 0)}`,
          `source_chunks=${stringifyCompact(Array.isArray(answerStep.result_payload.source_reread_chunks) ? answerStep.result_payload.source_reread_chunks.length : 0)}`,
        ],
      });
    }
    return blocks;
  }

  const extractStep = findStep("extract_local_pdf_text") ?? findStep("extract_arxiv_pdf_text");
  const summaryStep = findStep("compose_ingest_summary");
  const paperMemoryStep = findStep("extract_paper_memory");
  const relationStep = findStep("derive_relation_memory");
  const openQuestionStep = findStep("capture_open_questions");

  if (extractStep) {
    blocks.push({
      title: "原文已入库",
      lines: [
        `paper_id=${stringifyCompact(extractStep.result_payload.paper_id)}`,
        `artifact_id=${stringifyCompact(extractStep.result_payload.artifact_id)}`,
        `chunks=${stringifyCompact(extractStep.result_payload.chunk_count)}`,
      ],
    });
  }
  if (summaryStep) {
    blocks.push({
      title: "摘要证据",
      lines: [
        `paper_id=${stringifyCompact(summaryStep.result_payload.paper_id)}`,
        `summary=${truncateText(stringifyCompact(summaryStep.result_payload.summary), 160)}`,
      ],
    });
  }
  if (paperMemoryStep || relationStep || openQuestionStep) {
    blocks.push({
      title: "记忆写入",
      lines: [
        `paper=${paperMemoryStep ? stringifyCompact(paperMemoryStep.result_payload.paper_memory_id) : "none"}`,
        `relation=${relationStep ? stringifyCompact(relationStep.result_payload.relation_memory_id) : "none"}`,
        `open_question=${openQuestionStep ? stringifyCompact(openQuestionStep.result_payload.open_question_memory_id) : "none"}`,
      ],
    });
  }

  return blocks;
}

function deriveInlineThoughts(traceSteps: TraceStep[]): Array<{ title: string; detail: string }> {
  return traceSteps.map((step) => {
    const planner = step.input_payload.planner_decision as Record<string, unknown> | undefined;
    const selectedTool = planner ? readString(planner.selected_tool) : null;
    const rationale = planner ? readString(planner.rationale) : null;
    const fallbackUsed = planner ? readBoolean(planner.fallback_used) : null;
    const resultLines = summarizeTraceResult(step);
    const segments = [
      selectedTool ? `tool=${selectedTool}` : null,
      rationale ? truncateText(rationale, 120) : null,
      fallbackUsed !== null ? `fallback=${fallbackUsed ? "yes" : "no"}` : null,
      resultLines[0] ?? null,
    ].filter((item): item is string => Boolean(item));

    return {
      title: describeStepAction(step.action),
      detail: segments.join(" | "),
    };
  });
}

export default function App() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [runs, setRuns] = useState<TaskRun[]>([]);
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const [memorySnapshot, setMemorySnapshot] = useState<MemorySnapshot>({
    paper_memories: [],
    relation_memories: [],
    open_question_memories: [],
  });
  const [traceSteps, setTraceSteps] = useState<TraceStep[]>([]);
  const [runEvents, setRunEvents] = useState<TimelineEvent[]>([]);
  const [runtimeStatus, setRuntimeStatus] = useState<RuntimeStatus | null>(null);
  const [liveRunState, setLiveRunState] = useState<LiveRunState | null>(null);
  const [drawer, setDrawer] = useState<DrawerKind>(null);
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("memory");
  const [memoryFilter, setMemoryFilter] = useState<MemoryKind>("all");
  const [composerValue, setComposerValue] = useState("");
  const [mixedInputMode, setMixedInputMode] = useState<MixedInputMode>("query");
  const [uiLanguage, setUiLanguage] = useState<UiLanguage>("zh");
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [showJumpToBottom, setShowJumpToBottom] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const messageStreamRef = useRef<HTMLDivElement | null>(null);

  const composerState = classifyComposer(composerValue);
  const messageMap = new Map(messages.map((message) => [message.id, message]));
  const runByMessageId = new Map(runs.map((run) => [run.message_id, run]));
  const selectedRun = runs.find((run) => run.id === selectedRunId) ?? null;
  const selectedRunMessage = selectedRun ? messageMap.get(selectedRun.message_id) : undefined;
  const selectedRunStatus = selectedRun && liveRunState?.runId === selectedRun.id ? liveRunState.status : selectedRun?.status ?? null;
  const selectedRunStepCount =
    selectedRun && liveRunState?.runId === selectedRun.id ? liveRunState.stepCount : (selectedRun?.step_count ?? 0);
  const selectedRunCurrentAction =
    selectedRun && liveRunState?.runId === selectedRun.id ? liveRunState.currentAction : null;
  const selectedRunProgress = deriveRunProgress(selectedRunMessage?.type, traceSteps, selectedRunStatus);
  const selectedRunEvidence = deriveEvidenceBlocks(selectedRunMessage?.type, traceSteps);
  const selectedRunThoughts = deriveInlineThoughts(traceSteps);
  const activeTimeline = runEvents.length > 0 ? runEvents : timeline;

  const drawerMemories: Array<{
    kind: Exclude<MemoryKind, "all">;
    id: string;
    title: string;
    detail: string;
    meta: string;
  }> = [];

  for (const memory of memorySnapshot.paper_memories) {
    drawerMemories.push({
      kind: "paper_memory",
      id: memory.id,
      title: memory.problem || memory.method || memory.novelty_claim || "论文记忆",
      detail: [memory.method, memory.key_results[0], memory.limitations[0]].filter(Boolean).join(" · "),
      meta: `论文=${memory.paper_id} · 置信度=${memory.confidence.toFixed(2)}`,
    });
  }
  for (const memory of memorySnapshot.relation_memories) {
    drawerMemories.push({
      kind: "relation_memory",
      id: memory.id,
      title: memory.summary,
      detail: memory.evidence[0] ?? "暂无证据文本",
      meta: `${memory.relation_type} · ${memory.source_paper} -> ${memory.target_paper}`,
    });
  }
  for (const memory of memorySnapshot.open_question_memories) {
    drawerMemories.push({
      kind: "open_question_memory",
      id: memory.id,
      title: memory.unresolved_question,
      detail: memory.why_open[0] ?? memory.possible_followup[0] ?? "暂无后续说明",
      meta: `论文=${memory.related_papers.join(", ") || "n/a"} · 置信度=${memory.confidence.toFixed(2)}`,
    });
  }

  const filteredDrawerMemories = drawerMemories.filter((item) => memoryFilter === "all" || item.kind === memoryFilter);

  useEffect(() => {
    void bootstrap();
  }, []);

  useEffect(() => {
    if (composerState.kind !== "mixed_with_arxiv") {
      setMixedInputMode("query");
    }
  }, [composerState.kind]);

  useEffect(() => {
    const stream = messageStreamRef.current;
    if (!stream || showJumpToBottom) {
      return;
    }
    stream.scrollTo({ top: stream.scrollHeight, behavior: "smooth" });
  }, [messages.length, traceSteps.length, runEvents.length, showJumpToBottom]);

  function handleMessageStreamScroll() {
    const stream = messageStreamRef.current;
    if (!stream) {
      return;
    }
    const distanceFromBottom = stream.scrollHeight - stream.scrollTop - stream.clientHeight;
    setShowJumpToBottom(distanceFromBottom > 160);
  }

  function scrollMessagesToBottom() {
    const stream = messageStreamRef.current;
    if (!stream) {
      return;
    }
    stream.scrollTo({ top: stream.scrollHeight, behavior: "smooth" });
    setShowJumpToBottom(false);
  }

  async function bootstrap() {
    setIsBusy(true);
    setError(null);
    try {
      const [sessionPayload, statusPayload] = await Promise.all([api.listSessions(), api.getRuntimeStatus()]);
      let nextSessions = sessionPayload.items;
      if (nextSessions.length === 0) {
        const created = await api.createSession("研究会话 01");
        nextSessions = [created];
      }
      setSessions(nextSessions);
      setRuntimeStatus(statusPayload);
      const initialSessionId = nextSessions[0]?.id ?? null;
      setSelectedSessionId(initialSessionId);
      if (initialSessionId) {
        await refreshSession(initialSessionId, null);
      }
    } catch (loadError) {
      setError((loadError as Error).message);
    } finally {
      setIsBusy(false);
    }
  }

  async function refreshSession(sessionId: string, runId: string | null) {
    const [messagesPayload, runsPayload, timelinePayload, memoryPayload] = await Promise.all([
      api.listMessages(sessionId),
      api.listRuns(sessionId),
      api.listTimeline(sessionId),
      api.getMemorySnapshot(sessionId),
    ]);
    setMessages(messagesPayload.items);
    setRuns(runsPayload.items);
    setTimeline(timelinePayload.items);
    setMemorySnapshot(memoryPayload);

    const nextRunId = runId ?? runsPayload.items[0]?.id ?? null;
    if (nextRunId) {
      await inspectRun(sessionId, nextRunId);
    } else {
      setTraceSteps([]);
      setRunEvents([]);
      setSelectedRunId(null);
      setLiveRunState(null);
    }
  }

  async function inspectRun(sessionId: string, runId: string) {
    const [tracePayload, eventPayload] = await Promise.all([
      api.getTrace(sessionId, runId),
      api.getRunEvents(sessionId, runId),
    ]);
    setTraceSteps(tracePayload.steps);
    setRunEvents(eventPayload.items);
    setSelectedRunId(runId);
  }

  async function handleCreateSession() {
    setIsBusy(true);
    setError(null);
    try {
      const nextIndex = sessions.length + 1;
      const created = await api.createSession(`研究会话 ${String(nextIndex).padStart(2, "0")}`);
      setSessions([created, ...sessions]);
      setSelectedSessionId(created.id);
      setMessages([]);
      setRuns([]);
      setTimeline([]);
      setMemorySnapshot({
        paper_memories: [],
        relation_memories: [],
        open_question_memories: [],
      });
      setTraceSteps([]);
      setRunEvents([]);
      setSelectedRunId(null);
      setLiveRunState(null);
    } catch (createError) {
      setError((createError as Error).message);
    } finally {
      setIsBusy(false);
    }
  }

  async function handleDeleteSession() {
    if (!selectedSessionId || isBusy) {
      return;
    }
    const confirmed = window.confirm("删除当前 session 以及关联消息、runs 和 memory？");
    if (!confirmed) {
      return;
    }

    setIsBusy(true);
    setError(null);
    try {
      await api.deleteSession(selectedSessionId);
      const nextSessions = sessions.filter((session) => session.id !== selectedSessionId);
      setSessions(nextSessions);

      if (nextSessions.length === 0) {
        const created = await api.createSession("研究会话 01");
        setSessions([created]);
        setSelectedSessionId(created.id);
        await refreshSession(created.id, null);
      } else {
        const fallbackSession = nextSessions[0];
        setSelectedSessionId(fallbackSession.id);
        await refreshSession(fallbackSession.id, null);
      }
    } catch (deleteError) {
      setError((deleteError as Error).message);
    } finally {
      setIsBusy(false);
    }
  }

  async function handleSelectSession(sessionId: string) {
    setSelectedSessionId(sessionId);
    setIsBusy(true);
    setError(null);
    try {
      await refreshSession(sessionId, null);
    } catch (refreshError) {
      setError((refreshError as Error).message);
    } finally {
      setIsBusy(false);
    }
  }

  async function handleSelectRun(runId: string) {
    if (!selectedSessionId || isBusy) {
      return;
    }
    setIsBusy(true);
    setError(null);
    try {
      await inspectRun(selectedSessionId, runId);
      setInspectorTab("timeline");
    } catch (inspectError) {
      setError((inspectError as Error).message);
    } finally {
      setIsBusy(false);
    }
  }

  async function handleSubmitComposer() {
    if (!selectedSessionId || composerState.kind === "empty" || isBusy) {
      return;
    }
    setIsBusy(true);
    setError(null);
    try {
      const submission =
        composerState.kind === "pure_arxiv" ||
        (composerState.kind === "mixed_with_arxiv" && mixedInputMode === "ingest")
          ? await api.submitArxiv(selectedSessionId, composerState.arxivToken ?? composerValue.trim())
          : await api.submitText(selectedSessionId, composerValue.trim());

      setComposerValue("");
      await executeAcceptedRun(selectedSessionId, submission.run_id, submission.message_type);
    } catch (submitError) {
      setError((submitError as Error).message);
    } finally {
      setIsBusy(false);
    }
  }

  async function executeAcceptedRun(sessionId: string, runId: string, messageType: string) {
    await refreshSession(sessionId, runId);
    if (messageType === "followup_query") {
      await executeStreamedQueryRun(sessionId, runId);
    } else {
      await executeStreamedIngestRun(sessionId, runId);
    }
  }

  async function handleUploadPdf(file: File | null) {
    if (!file || !selectedSessionId || isBusy) {
      return;
    }
    setIsBusy(true);
    setError(null);
    try {
      const submission = await api.uploadPdf(selectedSessionId, file);
      await executeAcceptedRun(selectedSessionId, submission.run_id, submission.message_type);
    } catch (uploadError) {
      setError((uploadError as Error).message);
    } finally {
      setIsBusy(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  }

  async function handleDeleteMemory(memoryKind: Exclude<MemoryKind, "all">, memoryId: string) {
    if (!selectedSessionId || isBusy) {
      return;
    }
    setIsBusy(true);
    setError(null);
    try {
      await api.deleteMemory(selectedSessionId, memoryKind, memoryId);
      await refreshSession(selectedSessionId, selectedRunId);
    } catch (deleteError) {
      setError((deleteError as Error).message);
    } finally {
      setIsBusy(false);
    }
  }

  async function executeStreamedQueryRun(sessionId: string, runId: string) {
    await executeStreamedRun(sessionId, runId, {
      runLabel: "Query",
      startRun: () => api.startQueryRun(sessionId, runId),
      onCompleted: async () => {
        await refreshSession(sessionId, runId);
      },
    });
  }

  async function executeStreamedIngestRun(sessionId: string, runId: string) {
    await executeStreamedRun(sessionId, runId, {
      runLabel: "Ingest",
      startRun: () => api.startIngestRun(sessionId, runId),
      onCompleted: async () => {
        await refreshSession(sessionId, runId);
        setDrawer(null);
        setInspectorTab("memory");
      },
    });
  }

  async function executeStreamedRun(
    sessionId: string,
    runId: string,
    options: {
      runLabel: "Query" | "Ingest";
      startRun: () => Promise<TaskRun>;
      onCompleted: () => Promise<void>;
    },
  ) {
    setInspectorTab("timeline");
    setSelectedRunId(runId);
    setTraceSteps([]);
    setRunEvents([]);
    setLiveRunState({
      runId,
      status: "pending",
      stepCount: 0,
      currentAction: null,
      startedAt: null,
      finishedAt: null,
      finishReason: null,
    });

    await new Promise<void>((resolve, reject) => {
      const streamUrl = `${window.location.origin}/api/sessions/${sessionId}/runs/${runId}/stream`;
      const source = new EventSource(streamUrl);
      let settled = false;
      let started = false;
      const seenStepIds = new Set<string>();

      const close = () => {
        source.close();
      };

      const finish = (callback: () => void) => {
        if (settled) {
          return;
        }
        settled = true;
        close();
        callback();
      };

      const appendTraceStep = (step: TraceStep) => {
        setTraceSteps((current) => {
          const next = current.filter((item) => item.id !== step.id);
          next.push(step);
          return next;
        });
      };

      const startRunOnce = () => {
        if (started || settled) {
          return;
        }
        started = true;
        void options.startRun().catch((error: Error) => {
          finish(() => reject(error));
        });
      };

      source.onopen = startRunOnce;

      source.addEventListener("run_started", (event) => {
        const payload = JSON.parse((event as MessageEvent).data) as {
          payload?: {
            task_run?: {
              status?: string;
              step_count?: number;
              started_at?: string;
            };
          };
        };
        const taskRun = payload.payload?.task_run;
        setRunEvents((current) => [
          ...current,
          buildSyntheticEvent(runId, "run_started", options.runLabel === "Query" ? "问答已开始" : "导入已开始"),
        ]);
        setRuns((current) =>
          current.map((run) =>
            run.id === runId
              ? { ...run, status: taskRun?.status ?? "running", step_count: taskRun?.step_count ?? run.step_count }
              : run,
          ),
        );
        setLiveRunState((current) => ({
          runId,
          status: taskRun?.status ?? "running",
          stepCount: taskRun?.step_count ?? (current?.runId === runId ? current.stepCount : 0),
          currentAction: null,
          startedAt: taskRun?.started_at ?? current?.startedAt ?? new Date().toISOString(),
          finishedAt: null,
          finishReason: null,
        }));
      });

      source.addEventListener("step_completed", (event) => {
        const payload = JSON.parse((event as MessageEvent).data) as {
          payload?: { trace_step?: TraceStep };
        };
        const step = payload.payload?.trace_step;
        if (!step) {
          return;
        }
        seenStepIds.add(step.id);
        appendTraceStep(step);
        setRunEvents((current) => [...current, buildSyntheticEvent(runId, "step_completed", describeStepAction(step.action))]);
        setRuns((current) =>
          current.map((run) => (run.id === runId ? { ...run, status: "running", step_count: seenStepIds.size } : run)),
        );
        setLiveRunState((current) => ({
          runId,
          status: "running",
          stepCount: seenStepIds.size,
          currentAction: step.action,
          startedAt: current?.startedAt ?? new Date().toISOString(),
          finishedAt: null,
          finishReason: null,
        }));
      });

      source.addEventListener("assistant_message_committed", (event) => {
        const payload = JSON.parse((event as MessageEvent).data) as {
          payload?: { message?: Message };
        };
        const message = payload.payload?.message;
        if (!message) {
          return;
        }
        setMessages((current) => {
          if (current.some((item) => item.id === message.id)) {
            return current;
          }
          return [...current, message];
        });
        setRunEvents((current) => [...current, buildSyntheticEvent(runId, "assistant_message_committed", "助手消息已写入")]);
      });

      source.addEventListener("run_finished", (event) => {
        const payload = JSON.parse((event as MessageEvent).data) as {
          payload?: {
            task_run?: {
              status?: string;
              step_count?: number;
              finished_at?: string | null;
              finish_reason?: string | null;
            };
          };
        };
        const taskRun = payload.payload?.task_run;
        const stepCount = taskRun?.step_count ?? seenStepIds.size;
        setRunEvents((current) => [
          ...current,
          buildSyntheticEvent(runId, "run_finished", options.runLabel === "Query" ? "问答已完成" : "导入已完成"),
        ]);
        setRuns((current) =>
          current.map((run) =>
            run.id === runId
              ? { ...run, status: taskRun?.status ?? "finished", step_count: stepCount }
              : run,
          ),
        );
        setLiveRunState((current) => ({
          runId,
          status: taskRun?.status ?? "finished",
          stepCount,
          currentAction: current?.currentAction ?? null,
          startedAt: current?.startedAt ?? new Date().toISOString(),
          finishedAt: taskRun?.finished_at ?? new Date().toISOString(),
          finishReason: taskRun?.finish_reason ?? "completed",
        }));
        finish(() => {
          void options.onCompleted().then(() => {
            resolve();
          }).catch((error: Error) => reject(error));
        });
      });

      source.addEventListener("run_failed", (event) => {
        const payload = JSON.parse((event as MessageEvent).data) as { payload?: { reason?: string } };
        const reason = payload.payload?.reason ?? (options.runLabel === "Query" ? "问答运行失败" : "导入运行失败");
        setRunEvents((current) => [...current, buildSyntheticEvent(runId, "run_failed", reason)]);
        setRuns((current) =>
          current.map((run) => (run.id === runId ? { ...run, status: "failed", step_count: seenStepIds.size } : run)),
        );
        setLiveRunState((current) => ({
          runId,
          status: "failed",
          stepCount: seenStepIds.size,
          currentAction: current?.currentAction ?? null,
          startedAt: current?.startedAt ?? new Date().toISOString(),
          finishedAt: new Date().toISOString(),
          finishReason: reason,
        }));
        finish(() => {
          setError(reason);
          void refreshSession(sessionId, runId).finally(() => reject(new Error(reason)));
        });
      });

      source.onerror = () => {
        if (settled) {
          return;
        }
        finish(() => reject(new Error(options.runLabel === "Query" ? "问答状态流已断开" : "导入状态流已断开")));
      };

      window.setTimeout(startRunOnce, 1500);
    });
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar__brand">
          <span className="sidebar__eyebrow">Research Agent</span>
          <h1>研究工作台</h1>
        </div>

        <div className="sidebar__actions">
          <button className="action-button action-button--primary" type="button" onClick={handleCreateSession} disabled={isBusy}>
            <MessageSquarePlus size={16} />
            <span>新建会话</span>
          </button>
          <button className="icon-button" type="button" onClick={handleDeleteSession} disabled={isBusy || !selectedSessionId} title="删除会话">
            <Trash2 size={16} />
          </button>
        </div>

        <div className="session-list">
          {sessions.map((session) => (
            <button
              key={session.id}
              className={`session-row${session.id === selectedSessionId ? " session-row--active" : ""}`}
              type="button"
              onClick={() => void handleSelectSession(session.id)}
            >
              <span className="session-row__title">{session.title}</span>
              <span className="session-row__meta">{formatTime(session.updated_at)}</span>
            </button>
          ))}
        </div>

      </aside>

      <main className="workspace">
        <header className="workspace__header">
          <div>
            <span className="workspace__eyebrow">当前会话</span>
            <h2>{sessions.find((session) => session.id === selectedSessionId)?.title ?? "加载中"}</h2>
          </div>

          <div className="workspace__toolbar">
            <button className="icon-button" type="button" onClick={() => setDrawer("memory")} title="记忆库">
              <BrainCircuit size={18} />
            </button>
            <button className="icon-button" type="button" onClick={() => setDrawer("settings")} title="设置">
              <Settings2 size={18} />
            </button>
          </div>
        </header>

        <div className="workspace__body">
          <section className="chat-panel">
            <div ref={messageStreamRef} className="message-stream" onScroll={handleMessageStreamScroll}>
              {messages.map((message) => {
                const linkedRun = runByMessageId.get(message.id);
                const isSelectedInlineRun = linkedRun?.id === selectedRunId;

                return (
                  <div key={message.id} className="message-thread">
                    <article className={`message-bubble message-bubble--${message.role}`}>
                      <header>
                        <span>{message.role === "assistant" ? "助手" : "用户"}</span>
                        <span>{messageTypeLabel(message.type)}</span>
                      </header>
                      <p>{message.content}</p>
                    </article>

                    {linkedRun ? (
                      <details className={`inline-run-card${isSelectedInlineRun ? " inline-run-card--active" : ""}`} open={isSelectedInlineRun}>
                        <summary className="inline-run-card__summary">
                          <div className="inline-run-card__summary-main">
                            <span className="inline-run-card__label">{runTypeLabel(message.type)}运行</span>
                            <span className={`surface-badge surface-badge--${runStatusTone(linkedRun.id === liveRunState?.runId ? liveRunState.status : linkedRun.status)}`}>
                              {runStatusLabel(linkedRun.id === liveRunState?.runId ? liveRunState.status : linkedRun.status)}
                            </span>
                            <span className="inline-run-card__meta">
                              {linkedRun.id === selectedRunId ? selectedRunStepCount : linkedRun.step_count} 步
                            </span>
                          </div>
                          <button
                            className="inline-run-card__inspect"
                            type="button"
                            onClick={(event) => {
                              event.preventDefault();
                              void handleSelectRun(linkedRun.id);
                            }}
                          >
                            {isSelectedInlineRun ? "当前" : "查看"}
                          </button>
                        </summary>

                        {isSelectedInlineRun ? (
                          <div className="inline-run-card__body">
                            {selectedRunCurrentAction ? (
                              <div className="inline-run-card__section">
                                <span className="inline-run-card__section-title">当前步骤</span>
                                <p>{describeStepAction(selectedRunCurrentAction)}</p>
                              </div>
                            ) : null}

                            {selectedRunProgress.length > 0 ? (
                              <div className="inline-run-card__section">
                                <span className="inline-run-card__section-title">进度</span>
                                <div className="progress-rail progress-rail--inline">
                                  {selectedRunProgress.map((item) => (
                                    <div key={item.action} className={`progress-step progress-step--${item.state}`}>
                                      <span className="progress-step__dot" />
                                      <span>{item.label}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            ) : null}

                            {selectedRunThoughts.length > 0 ? (
                              <div className="inline-run-card__section">
                                <span className="inline-run-card__section-title">工具与决策</span>
                                <div className="inline-run-list">
                                  {selectedRunThoughts.map((item) => (
                                    <div key={`${linkedRun.id}-${item.title}-${item.detail}`} className="inline-run-list__item">
                                      <strong>{item.title}</strong>
                                      <span>{item.detail || "没有更多细节"}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            ) : null}

                            {selectedRunEvidence.length > 0 ? (
                              <div className="inline-run-card__section">
                                <span className="inline-run-card__section-title">证据</span>
                                <div className="inline-run-list">
                                  {selectedRunEvidence.map((block) => (
                                    <div key={`${linkedRun.id}-${block.title}`} className="inline-run-list__item">
                                      <strong>{block.title}</strong>
                                      <span>{block.lines.join(" | ")}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            ) : null}

                            {activeTimeline.length > 0 ? (
                              <div className="inline-run-card__section">
                                <span className="inline-run-card__section-title">实时事件</span>
                                <div className="inline-run-events">
                                  {activeTimeline.slice(-4).map((event) => (
                                    <div key={event.id} className="inline-run-events__item">
                                      <span>{eventTypeLabel(event.event_type)}</span>
                                      <span>{event.summary}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            ) : null}
                          </div>
                        ) : (
                          <div className="inline-run-card__body inline-run-card__body--compact">
                            <span>这条消息已创建运行。</span>
                            <span>{truncateText(summarizeRun(linkedRun, message.type), 72)}</span>
                          </div>
                        )}
                      </details>
                    ) : null}
                  </div>
                );
              })}
              {messages.length === 0 ? (
                <div className="empty-state">
                  <BookOpenText size={20} />
                  <p>输入问题、粘贴 arXiv 链接，或者上传 PDF。</p>
                </div>
              ) : null}
            </div>

            {showJumpToBottom ? (
              <button className="jump-to-bottom" type="button" onClick={scrollMessagesToBottom} title="跳到最新消息">
                <ArrowDownToLine size={16} />
                <span>最新</span>
              </button>
            ) : null}

            <div className="composer">
              <div className="composer__topline">
                {composerState.kind === "pure_arxiv" ? (
                  <span className="mode-chip mode-chip--ingest">检测到纯 arXiv 输入，将创建导入运行</span>
                ) : null}
                {composerState.kind === "mixed_with_arxiv" ? (
                  <div className="mode-switch">
                    <span className="mode-chip mode-chip--mixed">检测到文本中包含 arXiv 链接</span>
                    <button
                      className={`segmented-button${mixedInputMode === "query" ? " segmented-button--active" : ""}`}
                      type="button"
                      onClick={() => setMixedInputMode("query")}
                    >
                      作为问题发送
                    </button>
                    <button
                      className={`segmented-button${mixedInputMode === "ingest" ? " segmented-button--active" : ""}`}
                      type="button"
                      onClick={() => setMixedInputMode("ingest")}
                    >
                      导入论文
                    </button>
                  </div>
                ) : null}
              </div>

              <div className="composer__row">
                <textarea
                  value={composerValue}
                  onChange={(event) => setComposerValue(event.target.value)}
                  placeholder="提问，或粘贴 arXiv 链接。PDF 用右侧上传按钮。"
                  rows={4}
                />
                <div className="composer__actions">
                  <button className="icon-button" type="button" title="上传 PDF" onClick={() => fileInputRef.current?.click()} disabled={isBusy}>
                    <FileUp size={18} />
                  </button>
                  <input
                    ref={fileInputRef}
                    className="hidden-input"
                    type="file"
                    accept="application/pdf"
                    onChange={(event) => void handleUploadPdf(event.target.files?.[0] ?? null)}
                  />
                  <button className="action-button action-button--accent" type="button" onClick={() => void handleSubmitComposer()} disabled={isBusy || composerState.kind === "empty"}>
                    <span>{isBusy ? "运行中" : "发送"}</span>
                  </button>
                </div>
              </div>
            </div>
          </section>

          <aside className="inspector">
            {drawer ? (
              <header className="inspector__header">
                <strong>{drawer === "memory" ? "记忆库" : "设置"}</strong>
                <button className="icon-button" type="button" onClick={() => setDrawer(null)} title="关闭">
                  <X size={16} />
                </button>
              </header>
            ) : (
              <div className="inspector__tabs">
                <button className={inspectorTab === "memory" ? "tab-button tab-button--active" : "tab-button"} type="button" onClick={() => setInspectorTab("memory")}>
                  记忆
                </button>
                <button className={inspectorTab === "timeline" ? "tab-button tab-button--active" : "tab-button"} type="button" onClick={() => setInspectorTab("timeline")}>
                  时间线
                </button>
              </div>
            )}

            <div className="inspector__content">
              {drawer === "memory" ? (
                <>
                  <div className="filter-row">
                    {(["all", "paper_memory", "relation_memory", "open_question_memory"] as MemoryKind[]).map((kind) => (
                      <button
                        key={kind}
                        className={memoryFilter === kind ? "segmented-button segmented-button--active" : "segmented-button"}
                        type="button"
                        onClick={() => setMemoryFilter(kind)}
                      >
                        {memoryKindLabel(kind)}
                      </button>
                    ))}
                  </div>
                  <div className="drawer__list">
                    {filteredDrawerMemories.map((memory) => (
                      <article key={memory.id} className="memory-row">
                        <div>
                          <header>
                            <span>{memoryKindLabel(memory.kind)}</span>
                            <span>{memory.id.slice(0, 8)}</span>
                          </header>
                          <h4>{memory.title}</h4>
                          <p>{memory.detail}</p>
                          <small>{memory.meta}</small>
                        </div>
                        <button className="icon-button" type="button" title="删除记忆" onClick={() => void handleDeleteMemory(memory.kind, memory.id)}>
                          <Trash2 size={16} />
                        </button>
                      </article>
                    ))}
                    {filteredDrawerMemories.length === 0 ? <p className="muted-copy">当前过滤条件下没有记忆。</p> : null}
                  </div>
                </>
              ) : null}

              {drawer === "settings" ? (
                <div className="settings-grid">
                  <section className="summary-block">
                    <span>界面与输出语言</span>
                    <strong>{uiLanguage === "zh" ? "中文" : "English"}</strong>
                    <div className="language-switch">
                      <Languages size={15} />
                      <button
                        className={uiLanguage === "zh" ? "segmented-button segmented-button--active" : "segmented-button"}
                        type="button"
                        onClick={() => {
                          setUiLanguage("zh");
                        }}
                      >
                        中文
                      </button>
                      <button
                        className={uiLanguage === "en" ? "segmented-button segmented-button--active" : "segmented-button"}
                        type="button"
                        onClick={() => {
                          setUiLanguage("en");
                        }}
                      >
                        English
                      </button>
                    </div>
                  </section>
                  <section className="summary-block">
                    <span>存储</span>
                    <strong>{runtimeStatus?.storage_backend ?? "sqlite"}</strong>
                    <small>{runtimeStatus?.sqlite_path ?? "默认路径"}</small>
                  </section>
                  <section className="summary-block">
                    <span>问答代理</span>
                    <strong>{runtimeStatus?.query_agent_backend ?? "turn_adapter"}</strong>
                    <small>{runtimeStatus?.query_agent_provider}:{runtimeStatus?.query_agent_model}</small>
                  </section>
                  <section className="summary-block">
                    <span>导入抽取</span>
                    <strong>{runtimeStatus?.ingest_extraction_backend ?? "heuristic"}</strong>
                    <small>{runtimeStatus?.ingest_extraction_provider}:{runtimeStatus?.ingest_extraction_model}</small>
                  </section>
                  <section className="summary-block">
                    <span>OpenViking</span>
                    <strong>{runtimeStatus?.openviking_backend ?? "noop"}</strong>
                    <small>{runtimeStatus?.openviking_data_path ?? runtimeStatus?.openviking_url ?? "未配置"}</small>
                  </section>
                </div>
              ) : null}

              {!drawer && inspectorTab === "memory" ? (
                <div className="stack-list">
                  <section className="summary-block">
                    <span>论文记忆</span>
                    <strong>{memorySnapshot.paper_memories.length}</strong>
                  </section>
                  <section className="summary-block">
                    <span>关系记忆</span>
                    <strong>{memorySnapshot.relation_memories.length}</strong>
                  </section>
                  <section className="summary-block">
                    <span>开放问题</span>
                    <strong>{memorySnapshot.open_question_memories.length}</strong>
                  </section>
                  {memorySnapshot.paper_memories.slice(0, 2).map((memory) => (
                    <article key={memory.id} className="detail-row">
                      <header>
                        <span>论文记忆</span>
                        <span>{memory.confidence.toFixed(2)}</span>
                      </header>
                      <p>{memory.problem || memory.method || memory.novelty_claim || "暂无抽象摘要。"}</p>
                    </article>
                  ))}
                </div>
              ) : null}
              {!drawer && inspectorTab === "timeline" ? (
                <div className="stack-list">
                  {activeTimeline.map((event) => (
                    <article key={event.id} className="detail-row">
                      <header>
                        <span>{eventTypeLabel(event.event_type)}</span>
                        <span>{formatTime(event.created_at)}</span>
                      </header>
                      <p>{event.summary}</p>
                    </article>
                  ))}
                </div>
              ) : null}
            </div>
          </aside>
        </div>
      </main>

      {error ? <div className="toast toast--error">{error}</div> : null}
    </div>
  );
}

