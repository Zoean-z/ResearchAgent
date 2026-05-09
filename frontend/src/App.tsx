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
} from "lucide-react";

import { api } from "./lib/api";
import { ModalOverlay } from "./components/ModalOverlay";
import { MemoryBundlesView } from "./components/MemoryBundlesView";
import type {
  MemoryBundleGroup,
  MemoryBundleItem,
  MemoryBundles,
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
type MemorySortMode = "updated_at" | "created_at" | "title";
type MemoryScope = "session" | "global";
type UiLanguage = "zh" | "en";
type UiText = ReturnType<typeof getUiText>;

type LiveRunState = {
  runId: string;
  status: string;
  stepCount: number;
  currentAction: string | null;
  startedAt: string | null;
  finishedAt: string | null;
  finishReason: string | null;
};

type LiveStep = {
  id: string;
  action: string;
  label: string;
  detail: string;
  timestamp: string;
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
const UI_LANGUAGE_STORAGE_KEY = "research-agent-ui-language";
const CLIENT_API_KEY_STORAGE_KEY = "research-agent-client-api-key";

function getUiText(language: UiLanguage) {
  const isZh = language === "zh";
  return {
    language,
    brandTitle: isZh ? "研究工作台" : "Research Workbench",
    currentSession: isZh ? "当前会话" : "Current Session",
    loading: isZh ? "加载中" : "Loading",
    newSession: isZh ? "新建会话" : "New Session",
    deleteSession: isZh ? "删除会话" : "Delete Session",
    memoryTab: isZh ? "记忆" : "Memory",
    timelineTab: isZh ? "时间线" : "Timeline",
    memoryLibrary: isZh ? "记忆库" : "Memory Library",
    currentSessionMemories: isZh ? "当前对话" : "Current Session",
    globalMemories: isZh ? "全局记忆" : "Global Memories",
    memorySearch: isZh ? "搜索记忆" : "Search memories",
    memoryPaperFilter: isZh ? "论文筛选" : "Paper Filter",
    memoryTypeFilter: isZh ? "类型筛选" : "Type Filter",
    memorySort: isZh ? "排序" : "Sort",
    sortRecentUpdated: isZh ? "最近更新" : "Recently Updated",
    sortRecentCreated: isZh ? "最近创建" : "Recently Created",
    sortPaperTitle: isZh ? "论文标题" : "Paper Title",
    noMemory: isZh ? "暂无记忆" : "No memories yet.",
    noPaperSource: isZh ? "无论文来源 / No paper source" : "No paper source",
    createdAt: isZh ? "创建于" : "Created",
    updatedAt: isZh ? "更新于" : "Updated",
    memoryCountLabel: isZh ? "记忆数量" : "Memory Count",
    sourceChunksLabel: isZh ? "来源片段" : "Source Chunks",
    paperMemoryType: isZh ? "paper_memory" : "paper_memory",
    openQuestionMemoryType: isZh ? "open_question_memory" : "open_question_memory",
    relationMemoryType: isZh ? "relation_memory" : "relation_memory",
    sourceChunkType: isZh ? "source_chunk" : "source_chunk",
    expand: isZh ? "展开" : "Expand",
    collapse: isZh ? "收起" : "Collapse",
    showMore: isZh ? "展开内容" : "Show more",
    showLess: isZh ? "收起内容" : "Show less",
    relationFrom: isZh ? "来源论文" : "Source paper",
    relationTo: isZh ? "目标论文" : "Target paper",
    relationDirectionSource: isZh ? "source" : "source",
    relationDirectionTarget: isZh ? "target" : "target",
    settings: isZh ? "设置" : "Settings",
    memoryIconTitle: isZh ? "记忆库" : "Memory",
    settingsIconTitle: isZh ? "设置" : "Settings",
    memoryFilterAll: isZh ? "全部" : "All",
    memoryFilterPaper: isZh ? "论文记忆" : "Paper Memory",
    memoryFilterRelation: isZh ? "关系记忆" : "Relation Memory",
    memoryFilterOpenQuestion: isZh ? "开放问题" : "Open Question",
    memoryFilterSourceChunk: isZh ? "来源片段" : "Source Chunk",
    paperMemory: isZh ? "论文记忆" : "Paper Memory",
    relationMemory: isZh ? "关系记忆" : "Relation Memory",
    openQuestionMemory: isZh ? "开放问题" : "Open Question",
    noEvidenceText: isZh ? "暂无证据文本。" : "No evidence text yet.",
    noFollowupDetails: isZh ? "暂无后续说明。" : "No follow-up details yet.",
    emptyState: isZh ? "输入问题、粘贴 arXiv 链接，或者上传 PDF。" : "Ask a question, paste an arXiv link, or upload a PDF.",
    latest: isZh ? "最新" : "Latest",
    detectedPureArxiv: isZh ? "检测到纯 arXiv 输入，将创建导入运行" : "Pure arXiv input detected. An ingest run will be created.",
    detectedMixedArxiv: isZh ? "检测到文本中包含 arXiv 链接" : "An arXiv link was detected in the text.",
    asQuestion: isZh ? "作为问题发送" : "Send as Question",
    importPaper: isZh ? "导入论文" : "Import Paper",
    uploadPdfTitle: isZh ? "上传 PDF" : "Upload PDF",
    send: isZh ? "发送" : "Send",
    running: isZh ? "运行中" : "Running",
    close: isZh ? "关闭" : "Close",
    deleteMemory: isZh ? "删除记忆" : "Delete Memory",
    deleteMemoryConfirm: isZh ? "确定要删除这条记忆吗？此操作无法撤销。" : "Delete this memory? This cannot be undone.",
    current: isZh ? "当前" : "Current",
    view: isZh ? "查看" : "View",
    currentFilterEmpty: isZh ? "当前过滤条件下没有记忆。" : "No memories match the current filter.",
    languageSetting: isZh ? "语言 / Language" : "Language / 语言",
    apiKeySetting: isZh ? "API Key" : "API Key",
    apiKeyPlaceholder: isZh ? "输入用于请求的 API Key" : "Enter the API key to send with requests",
    apiKeyHelp: isZh
      ? "该 Key 会保存在本地浏览器，并随前端请求一起发送。当前后端如果未读取该 header，仍会继续使用服务端环境变量。"
      : "This key is stored in localStorage and sent with frontend requests. If the backend does not read this header yet, it will still use server-side environment variables.",
    clearApiKey: isZh ? "清除" : "Clear",
    storage: isZh ? "存储" : "Storage",
    queryAgent: isZh ? "问答代理" : "Query Agent",
    ingestExtraction: isZh ? "导入抽取" : "Ingest Extraction",
    openviking: isZh ? "OpenViking" : "OpenViking",
    notConfigured: isZh ? "未配置" : "Not configured",
    defaultPath: isZh ? "默认路径" : "Default path",
    interfaceLanguage: isZh ? "界面与输出语言" : "Interface Language",
    chinese: isZh ? "中文" : "Chinese",
    english: isZh ? "English" : "English",
    titleMemoryEvidence: isZh ? "记忆证据" : "Memory Evidence",
    sourceEvidence: isZh ? "原文证据" : "Source Evidence",
    answerPayload: isZh ? "回答载荷" : "Answer Payload",
    originalLoaded: isZh ? "原文已入库" : "Source Stored",
    summaryEvidence: isZh ? "摘要证据" : "Summary Evidence",
    memoryWritten: isZh ? "记忆写入" : "Memory Written",
    assistantLabel: isZh ? "助手" : "Assistant",
    userLabel: isZh ? "用户" : "User",
    currentStep: isZh ? "当前步骤" : "Current Step",
    progress: isZh ? "进度" : "Progress",
    toolsAndDecisions: isZh ? "工具与决策" : "Tools & Decisions",
    evidence: isZh ? "证据" : "Evidence",
    liveEvents: isZh ? "实时事件" : "Live Events",
    noFurtherDetails: isZh ? "没有更多细节。" : "No further details.",
    thisMessageCreatedRun: isZh ? "这条消息已创建运行。" : "This message created a run.",
    paperSummaryLabel: isZh ? "论文记忆" : "Paper Memory",
    relationSummaryLabel: isZh ? "关系记忆" : "Relation Memory",
    openQuestionSummaryLabel: isZh ? "开放问题" : "Open Question",
    queryRun: isZh ? "问答" : "Query",
    ingestRun: isZh ? "导入" : "Ingest",
    completed: isZh ? "完成" : "Completed",
    runningStatus: isZh ? "运行中" : "Running",
    pending: isZh ? "等待中" : "Pending",
    failed: isZh ? "失败" : "Failed",
    stepLimitReached: isZh ? "已到步数上限" : "Step limit reached",
    unknown: isZh ? "未知" : "Unknown",
    followupQuery: isZh ? "追问" : "Follow-up",
    ingestArxiv: isZh ? "arXiv 导入" : "arXiv Ingest",
    ingestPdf: isZh ? "PDF 导入" : "PDF Ingest",
    runCreated: isZh ? "运行已创建" : "Run Created",
    runStarted: isZh ? "运行已开始" : "Run Started",
    stepCompleted: isZh ? "步骤已完成" : "Step Completed",
    assistantCommitted: isZh ? "助手消息已写入" : "Assistant Message Committed",
    runFinished: isZh ? "运行已完成" : "Run Finished",
    runFailed: isZh ? "运行失败" : "Run Failed",
    failureReason: isZh ? "失败原因" : "Failure Reason",
    thinking: isZh ? "正在思考中" : "Thinking",
    directFinalAnswer: isZh ? "直接生成回答" : "Direct Final Answer",
    searchArxiv: isZh ? "搜索 arXiv" : "Search arXiv",
    importArxivPaper: isZh ? "导入 arXiv 论文" : "Import arXiv Paper",
    retrieveSessionMemories: isZh ? "检索会话记忆" : "Retrieve Session Memories",
    retrieveGlobalMemories: isZh ? "检索全局记忆" : "Retrieve Global Memories",
    rerankContextCandidates: isZh ? "重排记忆候选" : "Rerank Context Candidates",
    decideRereadSource: isZh ? "判断是否重读原文" : "Decide Whether to Reread Source",
    rereadSourcePassages: isZh ? "读取原文片段" : "Reread Source Passages",
    composeMockAnswer: isZh ? "生成回答" : "Compose Answer",
    inspectIngestRequest: isZh ? "检查导入请求" : "Inspect Ingest Request",
    extractLocalPdfText: isZh ? "抽取本地 PDF 文本" : "Extract Local PDF Text",
    extractArxivPdfText: isZh ? "抽取 arXiv PDF 文本" : "Extract arXiv PDF Text",
    persistPdfChunks: isZh ? "保存 PDF 分块" : "Persist PDF Chunks",
    persistArxivChunks: isZh ? "保存 arXiv 分块" : "Persist arXiv Chunks",
    composeIngestSummary: isZh ? "生成导入摘要" : "Compose Ingest Summary",
    extractPaperMemory: isZh ? "写入论文记忆" : "Extract Paper Memory",
    deriveRelationMemory: isZh ? "写入关系记忆" : "Derive Relation Memory",
    captureOpenQuestions: isZh ? "写入开放问题记忆" : "Capture Open Questions",
    memoryIds: isZh ? "记忆ID" : "memory_ids",
    coverage: isZh ? "覆盖率" : "coverage",
    matchedTerms: isZh ? "匹配词" : "matched_terms",
    selectedMemoryIds: isZh ? "选中记忆" : "selected_memory_ids",
    selectionSource: isZh ? "选择来源" : "selection_source",
    fallback: isZh ? "回退" : "fallback",
    shouldReread: isZh ? "是否重读" : "should_reread",
    reason: isZh ? "原因" : "reason",
    chunkIds: isZh ? "片段ID" : "chunk_ids",
    paperIds: isZh ? "论文ID" : "paper_ids",
    answerPreview: isZh ? "回答预览" : "answer_preview",
    memoryCitations: isZh ? "记忆引用" : "memory_citations",
    sourceChunks: isZh ? "来源片段" : "source_chunks",
    retrievalSkipped: isZh ? "跳过检索" : "retrieval_skipped",
    paperPrefix: isZh ? "论文" : "Paper",
    confidencePrefix: isZh ? "置信度" : "Confidence",
    relatedPapersPrefix: isZh ? "相关论文" : "Related papers",
    stepsUnit: isZh ? "步" : "steps",
    deleteCurrentSessionConfirm: isZh
      ? "删除当前 session 以及关联消息、runs 和 memory？"
      : "Delete the current session and its related messages, runs, and memories?",
    sessionTitle(index: number) {
      return isZh ? `研究会话 ${String(index).padStart(2, "0")}` : `Research Session ${String(index).padStart(2, "0")}`;
    },
    queryStartedSummary: isZh ? "问答已开始" : "Query started",
    ingestStartedSummary: isZh ? "导入已开始" : "Ingest started",
    queryFinishedSummary: isZh ? "问答已完成" : "Query finished",
    ingestFinishedSummary: isZh ? "导入已完成" : "Ingest finished",
    queryFailedSummary: isZh ? "问答运行失败" : "Query run failed",
    ingestFailedSummary: isZh ? "导入运行失败" : "Ingest run failed",
    queryStreamDisconnected: isZh ? "问答状态流已断开" : "Query status stream disconnected",
    ingestStreamDisconnected: isZh ? "导入状态流已断开" : "Ingest status stream disconnected",
    assistantMessageWritten: isZh ? "助手消息已写入" : "Assistant message committed",
    paperId: isZh ? "paper_id" : "paper_id",
    artifactId: isZh ? "artifact_id" : "artifact_id",
    sessionDocumentId: isZh ? "session_document_id" : "session_document_id",
    chunkCount: isZh ? "chunk_count" : "chunk_count",
    operation: isZh ? "operation" : "operation",
    summary: isZh ? "summary" : "summary",
    paperMemoryId: isZh ? "paper_memory_id" : "paper_memory_id",
    relationMemoryId: isZh ? "relation_memory_id" : "relation_memory_id",
    openQuestionMemoryId: isZh ? "open_question_memory_id" : "open_question_memory_id",
    memoryCount: isZh ? "论文记忆" : "Paper memories",
    relationCount: isZh ? "关系记忆" : "Relation memories",
    openQuestionCount: isZh ? "开放问题" : "Open questions",
    noEvidence: isZh ? "暂无抽象摘要。" : "No abstract summary yet.",
    inputPrompt: isZh
      ? "提问，或粘贴 arXiv 链接。PDF 用右侧上传按钮。"
      : "Ask a question or paste an arXiv link. Use the upload button for PDFs.",
    queryModeChip: isZh ? "检测到纯 arXiv 输入，将创建导入运行" : "Pure arXiv input detected. An ingest run will be created.",
    mixedModeChip: isZh ? "检测到文本中包含 arXiv 链接" : "An arXiv link was detected in the text.",
    memoryKindLabel(kind: MemoryKind) {
      if (kind === "paper_memory") return isZh ? "论文记忆" : "Paper Memory";
      if (kind === "relation_memory") return isZh ? "关系记忆" : "Relation Memory";
      if (kind === "open_question_memory") return isZh ? "开放问题" : "Open Question";
      return isZh ? "全部" : "All";
    },
    runStatusLabel(status: string | null | undefined) {
      if (status === "finished") return isZh ? "完成" : "Completed";
      if (status === "running") return isZh ? "运行中" : "Running";
      if (status === "pending") return isZh ? "等待中" : "Pending";
      if (status === "failed") return isZh ? "失败" : "Failed";
      if (status === "step_limit_reached") return isZh ? "已到步数上限" : "Step limit reached";
      return status ?? (isZh ? "未知" : "Unknown");
    },
    messageTypeLabel(messageType: string | undefined) {
      if (messageType === "followup_query") return isZh ? "追问" : "Follow-up";
      if (messageType === "ingest_arxiv") return isZh ? "arXiv 导入" : "arXiv Ingest";
      if (messageType === "ingest_pdf") return isZh ? "PDF 导入" : "PDF Ingest";
      return messageType ?? (isZh ? "消息" : "Message");
    },
    describeStepAction(action: string) {
      if (action === "direct_final_answer") return isZh ? "直接生成回答" : "Direct Final Answer";
      if (action === "search_arxiv") return isZh ? "搜索 arXiv" : "Search arXiv";
      if (action === "import_arxiv_paper") return isZh ? "导入 arXiv 论文" : "Import arXiv Paper";
      if (action === "retrieve_session_memories") return isZh ? "检索会话记忆" : "Retrieve Session Memories";
      if (action === "retrieve_global_memories") return isZh ? "检索全局记忆" : "Retrieve Global Memories";
      if (action === "rerank_context_candidates") return isZh ? "重排记忆候选" : "Rerank Context Candidates";
      if (action === "decide_reread_source") return isZh ? "判断是否重读原文" : "Decide Whether to Reread Source";
      if (action === "reread_source_passages") return isZh ? "读取原文片段" : "Reread Source Passages";
      if (action === "compose_mock_answer") return isZh ? "生成回答" : "Compose Answer";
      if (action === "inspect_ingest_request") return isZh ? "检查导入请求" : "Inspect Ingest Request";
      if (action === "extract_local_pdf_text") return isZh ? "抽取本地 PDF 文本" : "Extract Local PDF Text";
      if (action === "extract_arxiv_pdf_text") return isZh ? "抽取 arXiv PDF 文本" : "Extract arXiv PDF Text";
      if (action === "persist_pdf_chunks") return isZh ? "保存 PDF 分块" : "Persist PDF Chunks";
      if (action === "persist_arxiv_chunks") return isZh ? "保存 arXiv 分块" : "Persist ArXiv Chunks";
      if (action === "compose_ingest_summary") return isZh ? "生成导入摘要" : "Compose Ingest Summary";
      if (action === "extract_paper_memory") return isZh ? "写入论文记忆" : "Extract Paper Memory";
      if (action === "derive_relation_memory") return isZh ? "写入关系记忆" : "Derive Relation Memory";
      if (action === "capture_open_questions") return isZh ? "写入开放问题记忆" : "Capture Open Questions";
      return action;
    },
    eventTypeLabel(eventType: string) {
      if (eventType === "run_created") return isZh ? "运行已创建" : "Run Created";
      if (eventType === "run_started") return isZh ? "运行已开始" : "Run Started";
      if (eventType === "step_completed") return isZh ? "步骤已完成" : "Step Completed";
      if (eventType === "assistant_message_committed") return isZh ? "助手消息已写入" : "Assistant Message Committed";
      if (eventType === "run_finished") return isZh ? "运行已完成" : "Run Finished";
      if (eventType === "run_failed") return isZh ? "运行失败" : "Run Failed";
      return eventType;
    },
    runTypeLabel(messageType: string | undefined) {
      return classifyRunType(messageType) === "ingest" ? (isZh ? "导入" : "Ingest") : (isZh ? "问答" : "Query");
    },
  };
}

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

function formatTime(value: string, language: UiLanguage): string {
  return new Date(value).toLocaleString(language === "zh" ? "zh-CN" : "en-US", {
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function classifyRunType(messageType: string | undefined): "query" | "ingest" {
  return messageType?.startsWith("ingest") ? "ingest" : "query";
}

function readStringList(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return [];
  }
  return value.filter((item): item is string => typeof item === "string");
}

function summarizeRun(run: TaskRun, messageType: string | undefined, ui: UiText): string {
  const runType = classifyRunType(messageType);
  const prefix = runType === "ingest" ? ui.ingestRun : ui.queryRun;
  return `${prefix} · ${runStatusLabel(run.status, ui)} · ${run.step_count} ${ui.stepsUnit}`;
}

function runStatusLabel(status: string | null | undefined, ui: UiText): string {
  if (status === "finished") return ui.completed;
  if (status === "running") return ui.runningStatus;
  if (status === "pending") return ui.pending;
  if (status === "failed") return ui.failed;
  if (status === "step_limit_reached") return ui.stepLimitReached;
  return status ?? ui.unknown;
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

function summarizeTraceResult(step: TraceStep, ui: UiText): string[] {
  const payload = step.result_payload;
  const lines: string[] = [];

  if (step.action === "retrieve_session_memories" || step.action === "retrieve_global_memories") {
    lines.push(`${ui.memoryIds}=${readStringList(payload.memory_ids).join(", ") || "none"}`);
    const coverage = readNumber(payload.coverage_score);
    if (coverage !== null) {
      lines.push(`${ui.coverage}=${coverage.toFixed(2)}`);
    }
    lines.push(`${ui.matchedTerms}=${readStringList(payload.matched_query_terms).join(", ") || "none"}`);
  } else if (step.action === "rerank_context_candidates") {
    lines.push(`${ui.selectedMemoryIds}=${readStringList(payload.selected_memory_ids).join(", ") || "none"}`);
    lines.push(`${ui.selectionSource}=${stringifyCompact(payload.selection_source)}`);
    lines.push(`${ui.fallback}=${stringifyCompact(payload.fallback_used)}`);
  } else if (step.action === "decide_reread_source") {
    lines.push(`${ui.shouldReread}=${stringifyCompact(payload.should_reread_source)}`);
    lines.push(`${ui.reason}=${stringifyCompact(payload.reason)}`);
  } else if (step.action === "reread_source_passages") {
    lines.push(`${ui.chunkIds}=${readStringList(payload.chunk_ids).join(", ") || "none"}`);
    lines.push(`${ui.paperIds}=${readStringList(payload.paper_ids).join(", ") || "none"}`);
    lines.push(`${ui.selectionSource}=${stringifyCompact(payload.selection_source)}`);
  } else if (step.action === "compose_mock_answer") {
    lines.push(`${ui.answerPreview}=${stringifyCompact(payload.answer_preview)}`);
    lines.push(`${ui.memoryCitations}=${Array.isArray(payload.memory_citations) ? payload.memory_citations.length : 0}`);
    lines.push(`${ui.sourceChunks}=${Array.isArray(payload.source_reread_chunks) ? payload.source_reread_chunks.length : 0}`);
  } else if (step.action === "search_arxiv") {
    lines.push(`query=${stringifyCompact(payload.query)}`);
    lines.push(`count=${stringifyCompact(payload.count)}`);
    lines.push(`success=${stringifyCompact(payload.success)}`);
  } else if (step.action === "import_arxiv_paper") {
    lines.push(`arxiv_id=${stringifyCompact(payload.arxiv_id)}`);
    lines.push(`${ui.paperId}=${stringifyCompact(payload.paper_id)}`);
    lines.push(`${ui.chunkCount}=${stringifyCompact(payload.chunk_count)}`);
  } else if (step.action === "direct_final_answer") {
    lines.push(`${ui.answerPreview}=${stringifyCompact(payload.answer_preview)}`);
    lines.push(`${ui.retrievalSkipped}=${stringifyCompact(step.input_payload.retrieval_skipped)}`);
  } else if (step.action === "inspect_ingest_request") {
    lines.push(`${ui.paperId}=${stringifyCompact(payload.paper_id)}`);
    lines.push(`${ui.artifactId}=${stringifyCompact(payload.artifact_id)}`);
    lines.push(`${ui.sessionDocumentId}=${stringifyCompact(payload.session_document_id)}`);
  } else if (step.action === "extract_local_pdf_text" || step.action === "extract_arxiv_pdf_text") {
    lines.push(`${ui.paperId}=${stringifyCompact(payload.paper_id)}`);
    lines.push(`${ui.artifactId}=${stringifyCompact(payload.artifact_id)}`);
    lines.push(`${ui.chunkCount}=${stringifyCompact(payload.chunk_count)}`);
  } else if (step.action === "persist_pdf_chunks" || step.action === "persist_arxiv_chunks") {
    lines.push(`${ui.chunkCount}=${stringifyCompact(payload.chunk_count)}`);
    lines.push(`${ui.sessionDocumentId}=${stringifyCompact(payload.session_document_id)}`);
  } else if (step.action === "compose_ingest_summary") {
    lines.push(`${ui.paperId}=${stringifyCompact(payload.paper_id)}`);
    lines.push(`${ui.summary}=${stringifyCompact(payload.summary)}`);
  } else if (step.action === "extract_paper_memory") {
    lines.push(`${ui.paperMemoryId}=${stringifyCompact(payload.paper_memory_id)}`);
    lines.push(`${ui.operation}=${stringifyCompact(payload.paper_operation)}`);
  } else if (step.action === "derive_relation_memory") {
    lines.push(`${ui.relationMemoryId}=${stringifyCompact(payload.relation_memory_id)}`);
    lines.push(`${ui.operation}=${stringifyCompact(payload.relation_operation)}`);
  } else if (step.action === "capture_open_questions") {
    lines.push(`${ui.openQuestionMemoryId}=${stringifyCompact(payload.open_question_memory_id)}`);
    lines.push(`${ui.operation}=${stringifyCompact(payload.open_question_operation)}`);
  }

  return lines;
}

function describeStepAction(action: string, ui: UiText): string {
  if (action === "direct_final_answer") return ui.directFinalAnswer;
  if (action === "search_arxiv") return ui.searchArxiv;
  if (action === "import_arxiv_paper") return ui.importArxivPaper;
  if (action === "retrieve_session_memories") return ui.retrieveSessionMemories;
  if (action === "retrieve_global_memories") return ui.retrieveGlobalMemories;
  if (action === "rerank_context_candidates") return ui.rerankContextCandidates;
  if (action === "decide_reread_source") return ui.decideRereadSource;
  if (action === "reread_source_passages") return ui.rereadSourcePassages;
  if (action === "compose_mock_answer") return ui.composeMockAnswer;
  if (action === "inspect_ingest_request") return ui.inspectIngestRequest;
  if (action === "extract_local_pdf_text") return ui.extractLocalPdfText;
  if (action === "extract_arxiv_pdf_text") return ui.extractArxivPdfText;
  if (action === "persist_pdf_chunks") return ui.persistPdfChunks;
  if (action === "persist_arxiv_chunks") return ui.persistArxivChunks;
  if (action === "compose_ingest_summary") return ui.composeIngestSummary;
  if (action === "extract_paper_memory") return ui.extractPaperMemory;
  if (action === "derive_relation_memory") return ui.deriveRelationMemory;
  if (action === "capture_open_questions") return ui.captureOpenQuestions;
  return action;
}

function extractPlannerDetail(step: TraceStep): string {
  const planner = step.input_payload.planner_decision as Record<string, unknown> | undefined;
  if (!planner) return "";
  const parts: string[] = [];
  const tool = readString(planner.selected_tool);
  if (tool) parts.push(`tool=${tool}`);
  const rationale = readString(planner.rationale);
  if (rationale) parts.push(truncateText(rationale, 120));
  const fallback = readBoolean(planner.fallback_used);
  if (fallback !== null) parts.push(`fallback=${fallback ? "yes" : "no"}`);
  return parts.join(" | ");
}

function deriveEvidenceBlocks(
  messageType: string | undefined,
  traceSteps: TraceStep[],
  ui: UiText,
): Array<{ title: string; lines: string[] }> {
  const findStep = (action: string) => [...traceSteps].reverse().find((step) => step.action === action);
  const blocks: Array<{ title: string; lines: string[] }> = [];

  if (classifyRunType(messageType) === "query") {
    const rerankStep = findStep("rerank_context_candidates");
    const rereadStep = findStep("reread_source_passages");
    const answerStep = findStep("compose_mock_answer");

    if (rerankStep) {
      blocks.push({
        title: ui.titleMemoryEvidence,
        lines: [
          `selected=${readStringList(rerankStep.result_payload.selected_memory_ids).join(", ") || "none"}`,
          `source=${stringifyCompact(rerankStep.result_payload.selection_source)}`,
          `fallback=${stringifyCompact(rerankStep.result_payload.fallback_used)}`,
        ],
      });
    }
    if (rereadStep) {
      blocks.push({
        title: ui.sourceEvidence,
        lines: [
          `chunks=${readStringList(rereadStep.result_payload.chunk_ids).join(", ") || "none"}`,
          `papers=${readStringList(rereadStep.result_payload.paper_ids).join(", ") || "none"}`,
          `strategy=${stringifyCompact(rereadStep.result_payload.selection_source)}`,
        ],
      });
    }
    if (answerStep) {
      blocks.push({
        title: ui.answerPayload,
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
      title: ui.originalLoaded,
      lines: [
        `paper_id=${stringifyCompact(extractStep.result_payload.paper_id)}`,
        `artifact_id=${stringifyCompact(extractStep.result_payload.artifact_id)}`,
        `chunks=${stringifyCompact(extractStep.result_payload.chunk_count)}`,
      ],
    });
  }
  if (summaryStep) {
    blocks.push({
      title: ui.summaryEvidence,
      lines: [
        `paper_id=${stringifyCompact(summaryStep.result_payload.paper_id)}`,
        `summary=${truncateText(stringifyCompact(summaryStep.result_payload.summary), 160)}`,
      ],
    });
  }
  if (paperMemoryStep || relationStep || openQuestionStep) {
    blocks.push({
      title: ui.memoryWritten,
      lines: [
        `paper=${paperMemoryStep ? stringifyCompact(paperMemoryStep.result_payload.paper_memory_id) : "none"}`,
        `relation=${relationStep ? stringifyCompact(relationStep.result_payload.relation_memory_id) : "none"}`,
        `open_question=${openQuestionStep ? stringifyCompact(openQuestionStep.result_payload.open_question_memory_id) : "none"}`,
      ],
    });
  }

  return blocks;
}

function deriveInlineThoughts(traceSteps: TraceStep[], ui: UiText): Array<{ title: string; detail: string }> {
  return traceSteps.map((step) => {
    const planner = step.input_payload.planner_decision as Record<string, unknown> | undefined;
    const selectedTool = planner ? readString(planner.selected_tool) : null;
    const rationale = planner ? readString(planner.rationale) : null;
    const fallbackUsed = planner ? readBoolean(planner.fallback_used) : null;
    const resultLines = summarizeTraceResult(step, ui);
    const segments = [
      selectedTool ? `tool=${selectedTool}` : null,
      rationale ? truncateText(rationale, 120) : null,
      fallbackUsed !== null ? `fallback=${fallbackUsed ? "yes" : "no"}` : null,
      resultLines[0] ?? null,
    ].filter((item): item is string => Boolean(item));

    return {
      title: describeStepAction(step.action, ui),
      detail: segments.join(" | "),
    };
  });
}

function normalizeMemoryQuery(query: string): string {
  return query.trim().toLowerCase();
}

function memoryItemMatchesQuery(item: MemoryBundleItem, query: string): boolean {
  if (!query) {
    return true;
  }
  const haystack = [
    item.content,
    item.memory_type,
    item.paper_id,
    item.source_paper,
    item.target_paper,
    item.relation_type,
    item.related_papers.join(" "),
    item.source_chunk_ids.join(" "),
  ]
    .filter(Boolean)
    .join(" ")
    .toLowerCase();
  return haystack.includes(query);
}

function paperGroupMatchesQuery(group: MemoryBundleGroup, query: string): boolean {
  if (!query) {
    return true;
  }
  const haystack = [group.paper.title, group.paper.file_name, group.paper.paper_id].filter(Boolean).join(" ").toLowerCase();
  if (haystack.includes(query)) {
    return true;
  }
  return (
    group.paper_memories.some((item) => memoryItemMatchesQuery(item, query)) ||
    group.open_question_memories.some((item) => memoryItemMatchesQuery(item, query)) ||
    group.relation_memories.some((item) => memoryItemMatchesQuery(item, query))
  );
}

function sortMemoryGroups(groups: MemoryBundleGroup[], sortMode: MemorySortMode): MemoryBundleGroup[] {
  const sorted = [...groups];
  sorted.sort((left, right) => {
    if (sortMode === "title") {
      return left.paper.title.localeCompare(right.paper.title);
    }
    const leftTime = new Date(sortMode === "created_at" ? (left.paper.created_at ?? left.paper.updated_at) : left.paper.updated_at).getTime();
    const rightTime = new Date(sortMode === "created_at" ? (right.paper.created_at ?? right.paper.updated_at) : right.paper.updated_at).getTime();
    if (leftTime !== rightTime) {
      return rightTime - leftTime;
    }
    return left.paper.title.localeCompare(right.paper.title);
  });
  return sorted;
}

function isItemVisibleByType(item: MemoryBundleItem, filter: MemoryKind): boolean {
  if (filter === "all") {
    return true;
  }
  return item.memory_type === filter;
}

export default function App() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [selectedSessionId, setSelectedSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [runs, setRuns] = useState<TaskRun[]>([]);
  const [timeline, setTimeline] = useState<TimelineEvent[]>([]);
  const [memoryBundles, setMemoryBundles] = useState<MemoryBundles | null>(null);
  const [globalMemoryBundles, setGlobalMemoryBundles] = useState<MemoryBundles | null>(null);
  const [memoryBundleLoading, setMemoryBundleLoading] = useState(false);
  const [globalMemoryBundleLoading, setGlobalMemoryBundleLoading] = useState(false);
  const [memoryBundleError, setMemoryBundleError] = useState<string | null>(null);
  const [globalMemoryBundleError, setGlobalMemoryBundleError] = useState<string | null>(null);
  const [memoryScope, setMemoryScope] = useState<MemoryScope>("session");
  const [traceSteps, setTraceSteps] = useState<TraceStep[]>([]);
  const [runEvents, setRunEvents] = useState<TimelineEvent[]>([]);
  const [runtimeStatus, setRuntimeStatus] = useState<RuntimeStatus | null>(null);
  const [liveRunState, setLiveRunState] = useState<LiveRunState | null>(null);
  const [liveSteps, setLiveSteps] = useState<LiveStep[]>([]);
  const [drawer, setDrawer] = useState<DrawerKind>(null);
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>("memory");
  const [memoryFilter, setMemoryFilter] = useState<MemoryKind>("all");
  const [memoryPaperFilter, setMemoryPaperFilter] = useState<string>("all");
  const [memorySearch, setMemorySearch] = useState("");
  const [memorySort, setMemorySort] = useState<MemorySortMode>("updated_at");
  const [expandedMemoryGroupIds, setExpandedMemoryGroupIds] = useState<string[]>([]);
  const [expandedMemoryItemIds, setExpandedMemoryItemIds] = useState<string[]>([]);
  const [composerValue, setComposerValue] = useState("");
  const [mixedInputMode, setMixedInputMode] = useState<MixedInputMode>("query");
  const [uiLanguage, setUiLanguage] = useState<UiLanguage>(() => {
    if (typeof window === "undefined") {
      return "zh";
    }
    const stored = window.localStorage.getItem(UI_LANGUAGE_STORAGE_KEY);
    return stored === "en" ? "en" : "zh";
  });
  const [clientApiKey, setClientApiKey] = useState(() => {
    if (typeof window === "undefined") {
      return "";
    }
    return window.localStorage.getItem(CLIENT_API_KEY_STORAGE_KEY) ?? "";
  });
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [isBusy, setIsBusy] = useState(false);
  const [showJumpToBottom, setShowJumpToBottom] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const messageStreamRef = useRef<HTMLDivElement | null>(null);
  const ui = getUiText(uiLanguage);

  const composerState = classifyComposer(composerValue);
  const messageMap = new Map(messages.map((message) => [message.id, message]));
  const runByMessageId = new Map(runs.map((run) => [run.message_id, run]));
  const selectedRun = runs.find((run) => run.id === selectedRunId) ?? null;
  const selectedRunMessage = selectedRun ? messageMap.get(selectedRun.message_id) : undefined;
  const selectedRunStepCount =
    selectedRun && liveRunState?.runId === selectedRun.id ? liveRunState.stepCount : (selectedRun?.step_count ?? 0);
  const selectedRunEvidence = deriveEvidenceBlocks(selectedRunMessage?.type, traceSteps, ui);
  const selectedRunThoughts = deriveInlineThoughts(traceSteps, ui);
  const activeTimeline = runEvents.length > 0 ? runEvents : timeline;

  const activeMemoryBundles = memoryScope === "global" ? globalMemoryBundles : memoryBundles;
  const activeMemoryLoading = memoryScope === "global" ? globalMemoryBundleLoading : memoryBundleLoading;
  const activeMemoryError = memoryScope === "global" ? globalMemoryBundleError : memoryBundleError;
  const normalizedMemorySearch = normalizeMemoryQuery(memorySearch);
  const visiblePaperGroups = activeMemoryBundles
    ? sortMemoryGroups(
        activeMemoryBundles.papers
          .filter((group) => memoryPaperFilter === "all" || group.paper.paper_id === memoryPaperFilter)
          .map((group) => {
            const paperMemories = group.paper_memories.filter(
              (item) => isItemVisibleByType(item, memoryFilter) && memoryItemMatchesQuery(item, normalizedMemorySearch),
            );
            const openQuestionMemories = group.open_question_memories.filter(
              (item) => isItemVisibleByType(item, memoryFilter) && memoryItemMatchesQuery(item, normalizedMemorySearch),
            );
            const relationMemories = group.relation_memories.filter(
              (item) => isItemVisibleByType(item, memoryFilter) && memoryItemMatchesQuery(item, normalizedMemorySearch),
            );
            const matchesMetadata = paperGroupMatchesQuery(group, normalizedMemorySearch);
            const hasVisibleMemories =
              paperMemories.length > 0 || openQuestionMemories.length > 0 || relationMemories.length > 0;
            const showGroup = matchesMetadata || hasVisibleMemories;
            return {
              ...group,
              paper_memories: paperMemories,
              open_question_memories: openQuestionMemories,
              relation_memories: relationMemories,
              source_chunks: [],
              showGroup,
            };
          })
          .filter((group) => group.showGroup),
        memorySort,
      )
    : [];
  const visibleUnscopedMemories = (activeMemoryBundles?.unscoped_memories ?? []).filter(
    (item) => isItemVisibleByType(item, memoryFilter) && memoryItemMatchesQuery(item, normalizedMemorySearch),
  );

  useEffect(() => {
    void bootstrap();
  }, []);

  useEffect(() => {
    const demoBridge = (globalThis as { __RESEARCH_AGENT_DEMO__?: { subscribe?: (listener: () => void) => () => void } })
      .__RESEARCH_AGENT_DEMO__;
    if (!selectedSessionId || !demoBridge?.subscribe) {
      return;
    }
    return demoBridge.subscribe(() => {
      void refreshSession(selectedSessionId, null);
    });
  }, [selectedSessionId]);

  useEffect(() => {
    if (composerState.kind !== "mixed_with_arxiv") {
      setMixedInputMode("query");
    }
  }, [composerState.kind]);

  useEffect(() => {
    window.localStorage.setItem(UI_LANGUAGE_STORAGE_KEY, uiLanguage);
  }, [uiLanguage]);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }
    const normalized = clientApiKey.trim();
    if (normalized) {
      window.localStorage.setItem(CLIENT_API_KEY_STORAGE_KEY, normalized);
      return;
    }
    window.localStorage.removeItem(CLIENT_API_KEY_STORAGE_KEY);
  }, [clientApiKey]);

  useEffect(() => {
    if (!activeMemoryBundles) {
      setExpandedMemoryGroupIds([]);
      setExpandedMemoryItemIds([]);
      return;
    }
    const defaults = activeMemoryBundles.papers.slice(0, 3).map((group) => group.paper.paper_id);
    setExpandedMemoryGroupIds(defaults);
    setExpandedMemoryItemIds([]);
  }, [selectedSessionId, memoryScope, activeMemoryBundles]);

  useEffect(() => {
    setMemoryPaperFilter("all");
    setMemoryFilter("all");
    setMemorySearch("");
    setExpandedMemoryItemIds([]);
  }, [selectedSessionId, memoryScope]);

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

  function toggleMemoryGroup(groupId: string) {
    setExpandedMemoryGroupIds((current) =>
      current.includes(groupId) ? current.filter((item) => item !== groupId) : [...current, groupId],
    );
  }

  function toggleMemoryItem(itemId: string) {
    setExpandedMemoryItemIds((current) =>
      current.includes(itemId) ? current.filter((item) => item !== itemId) : [...current, itemId],
    );
  }

  async function handleDeleteMemory(memoryKind: Exclude<MemoryKind, "all">, memoryId: string) {
    if (!selectedSessionId || isBusy) {
      return;
    }
    const confirmed = window.confirm(ui.deleteMemoryConfirm);
    if (!confirmed) {
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

  async function bootstrap() {
    setIsBusy(true);
    setError(null);
    try {
      const [sessionPayload, statusPayload] = await Promise.all([api.listSessions(), api.getRuntimeStatus()]);
      let nextSessions = sessionPayload.items;
      if (nextSessions.length === 0) {
        const created = await api.createSession(ui.sessionTitle(1));
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
    const [messagesPayload, runsPayload, timelinePayload] = await Promise.all([
      api.listMessages(sessionId),
      api.listRuns(sessionId),
      api.listTimeline(sessionId),
    ]);
    setMessages(messagesPayload.items);
    setRuns(runsPayload.items);
    setTimeline(timelinePayload.items);
    setMemoryBundleLoading(true);
    setMemoryBundleError(null);
    setGlobalMemoryBundleLoading(true);
    setGlobalMemoryBundleError(null);
    try {
      const bundlePayload = await api.getMemoryBundles(sessionId);
      setMemoryBundles(bundlePayload);
    } catch (loadError) {
      setMemoryBundles(null);
      setMemoryBundleError((loadError as Error).message);
    } finally {
      setMemoryBundleLoading(false);
    }

    try {
      const globalBundlePayload = await api.getGlobalMemoryBundles();
      setGlobalMemoryBundles(globalBundlePayload);
    } catch (loadError) {
      setGlobalMemoryBundles(null);
      setGlobalMemoryBundleError((loadError as Error).message);
    } finally {
      setGlobalMemoryBundleLoading(false);
    }

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
      const created = await api.createSession(ui.sessionTitle(nextIndex));
      setSessions([created, ...sessions]);
      setSelectedSessionId(created.id);
      setMessages([]);
      setRuns([]);
      setTimeline([]);
      setMemoryBundles(null);
      setGlobalMemoryBundles(null);
      setMemoryBundleLoading(false);
      setGlobalMemoryBundleLoading(false);
      setMemoryBundleError(null);
      setGlobalMemoryBundleError(null);
      setExpandedMemoryGroupIds([]);
      setExpandedMemoryItemIds([]);
      setTraceSteps([]);
      setRunEvents([]);
      setLiveSteps([]);
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
    const confirmed = window.confirm(
      ui.deleteCurrentSessionConfirm,
    );
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
        const created = await api.createSession(ui.sessionTitle(1));
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
    setLiveSteps([]);
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
          buildSyntheticEvent(runId, "run_started", options.runLabel === "Query" ? ui.queryStartedSummary : ui.ingestStartedSummary),
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
        setLiveSteps((current) => {
          if (current.some((item) => item.id === step.id)) return current;
          return [
            ...current,
            {
              id: step.id,
              action: step.action,
              label: ui.describeStepAction(step.action),
              detail: extractPlannerDetail(step),
              timestamp: new Date().toISOString(),
            },
          ];
        });
        setRunEvents((current) => [...current, buildSyntheticEvent(runId, "step_completed", ui.describeStepAction(step.action))]);
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
        setRunEvents((current) => [...current, buildSyntheticEvent(runId, "assistant_message_committed", ui.assistantMessageWritten)]);
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
          buildSyntheticEvent(runId, "run_finished", options.runLabel === "Query" ? ui.queryFinishedSummary : ui.ingestFinishedSummary),
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
        const reason = payload.payload?.reason ?? (options.runLabel === "Query" ? ui.queryFailedSummary : ui.ingestFailedSummary);
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
        finish(() => reject(new Error(options.runLabel === "Query" ? ui.queryStreamDisconnected : ui.ingestStreamDisconnected)));
      };

      window.setTimeout(startRunOnce, 1500);
    });
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="sidebar__brand">
          <span className="sidebar__eyebrow">Research Agent</span>
          <h1>{ui.brandTitle}</h1>
        </div>

        <div className="sidebar__actions">
          <button className="action-button action-button--primary" type="button" onClick={handleCreateSession} disabled={isBusy}>
            <MessageSquarePlus size={16} />
            <span>{ui.newSession}</span>
          </button>
          <button className="icon-button" type="button" onClick={handleDeleteSession} disabled={isBusy || !selectedSessionId} title={ui.deleteSession}>
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
              <span className="session-row__meta">{formatTime(session.updated_at, uiLanguage)}</span>
            </button>
          ))}
        </div>

      </aside>

      <main className="workspace">
        <header className="workspace__header">
          <div>
            <span className="workspace__eyebrow">{ui.currentSession}</span>
            <h2>{sessions.find((session) => session.id === selectedSessionId)?.title ?? ui.loading}</h2>
          </div>

          <div className="workspace__toolbar">
            <button className="icon-button" type="button" onClick={() => setDrawer("memory")} title={ui.memoryIconTitle}>
              <BrainCircuit size={18} />
            </button>
            <button className="icon-button" type="button" onClick={() => setDrawer("settings")} title={ui.settingsIconTitle}>
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
                        <span>{message.role === "assistant" ? ui.assistantLabel : ui.userLabel}</span>
                        <span>{ui.messageTypeLabel(message.type)}</span>
                      </header>
                      <p>{message.content}</p>
                    </article>

                    {linkedRun ? (
                      <details className={`inline-run-card${isSelectedInlineRun ? " inline-run-card--active" : ""}`} open={isSelectedInlineRun}>
                        <summary className="inline-run-card__summary">
                          <div className="inline-run-card__summary-main">
                            <span className="inline-run-card__label">{ui.runTypeLabel(message.type)}</span>
                            <span className={`surface-badge surface-badge--${runStatusTone(linkedRun.id === liveRunState?.runId ? liveRunState.status : linkedRun.status)}`}>
                              {ui.runStatusLabel(linkedRun.id === liveRunState?.runId ? liveRunState.status : linkedRun.status)}
                            </span>
                            <span className="inline-run-card__meta">
                              {linkedRun.id === selectedRunId ? selectedRunStepCount : linkedRun.step_count} {ui.stepsUnit}
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
                            {isSelectedInlineRun ? ui.current : ui.view}
                          </button>
                        </summary>

                        {isSelectedInlineRun ? (
                          <div className="inline-run-card__body">
                            {(linkedRun.id === liveRunState?.runId && liveSteps.length > 0) ? (
                              <div className="inline-run-card__section">
                                <span className="inline-run-card__section-title">{ui.toolsAndDecisions}</span>
                                <div className="live-steps">
                                  {liveSteps.map((step, index) => {
                                    const isLast = index === liveSteps.length - 1;
                                    const isRunning = liveRunState?.status === "running" && isLast;
                                    return (
                                      <div key={step.id} className={`live-step${isRunning ? " live-step--active" : " live-step--done"}`}>
                                        <span className="live-step__dot" />
                                        <div className="live-step__content">
                                          <strong>{step.label}</strong>
                                          {step.detail ? <span>{step.detail}</span> : null}
                                        </div>
                                      </div>
                                    );
                                  })}
                                  {liveRunState?.status === "running" ? (
                                    <div className="live-step live-step--pending">
                                      <span className="live-step__dot" />
                                      <div className="live-step__content">
                                        <span className="live-step__waiting">{ui.running}...</span>
                                      </div>
                                    </div>
                                  ) : null}
                                </div>
                              </div>
                            ) : selectedRunThoughts.length > 0 ? (
                              <div className="inline-run-card__section">
                                <span className="inline-run-card__section-title">{ui.toolsAndDecisions}</span>
                                <div className="inline-run-list">
                                  {selectedRunThoughts.map((item) => (
                                    <div key={`${linkedRun.id}-${item.title}-${item.detail}`} className="inline-run-list__item">
                                      <strong>{item.title}</strong>
                                      <span>{item.detail || ui.noFurtherDetails}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            ) : null}

                            {(linkedRun.id === liveRunState?.runId ? liveRunState.status : linkedRun.status) === "failed" ? (
                              <div className="inline-run-card__section">
                                <span className="inline-run-card__section-title">{ui.failureReason}</span>
                                <div className="inline-run-failure-reason">
                                  {linkedRun.id === liveRunState?.runId
                                    ? (liveRunState.finishReason ?? ui.queryFailedSummary)
                                    : (runEvents.find((evt) => evt.run_id === linkedRun.id && evt.event_type === "run_failed")?.summary ?? ui.queryFailedSummary)}
                                </div>
                              </div>
                            ) : null}

                            {selectedRunEvidence.length > 0 ? (
                              <div className="inline-run-card__section">
                                <span className="inline-run-card__section-title">{ui.evidence}</span>
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
                                <span className="inline-run-card__section-title">{ui.liveEvents}</span>
                                <div className="inline-run-events">
                                  {activeTimeline.slice(-4).map((event) => (
                                    <div key={event.id} className="inline-run-events__item">
                                      <span>{ui.eventTypeLabel(event.event_type)}</span>
                                      <span>{event.summary}</span>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            ) : null}
                          </div>
                        ) : (
                          <div className="inline-run-card__body inline-run-card__body--compact">
                            <span>{ui.thisMessageCreatedRun}</span>
                            <span>{truncateText(summarizeRun(linkedRun, message.type, ui), 72)}</span>
                          </div>
                        )}
                      </details>
                    ) : null}
                  </div>
                );
              })}
              {isBusy ? (
                <div className="thinking-indicator">
                  <span className="thinking-indicator__dot" />
                  <span className="thinking-indicator__dot" />
                  <span className="thinking-indicator__dot" />
                  <span className="thinking-indicator__text">{liveRunState?.currentAction ? `${ui.thinking} — ${liveRunState.currentAction}` : ui.thinking}</span>
                </div>
              ) : null}
              {messages.length === 0 ? (
                <div className="empty-state">
                  <BookOpenText size={20} />
                  <p>{ui.emptyState}</p>
                </div>
              ) : null}
            </div>

            {showJumpToBottom ? (
              <button className="jump-to-bottom" type="button" onClick={scrollMessagesToBottom} title={ui.latest}>
                <ArrowDownToLine size={16} />
                <span>{ui.latest}</span>
              </button>
            ) : null}

            <div className="composer">
              <div className="composer__topline">
                {composerState.kind === "pure_arxiv" ? (
                  <span className="mode-chip mode-chip--ingest">{ui.detectedPureArxiv}</span>
                ) : null}
                {composerState.kind === "mixed_with_arxiv" ? (
                  <div className="mode-switch">
                    <span className="mode-chip mode-chip--mixed">{ui.detectedMixedArxiv}</span>
                    <button
                      className={`segmented-button${mixedInputMode === "query" ? " segmented-button--active" : ""}`}
                      type="button"
                      onClick={() => setMixedInputMode("query")}
                    >
                      {ui.asQuestion}
                    </button>
                    <button
                      className={`segmented-button${mixedInputMode === "ingest" ? " segmented-button--active" : ""}`}
                      type="button"
                      onClick={() => setMixedInputMode("ingest")}
                    >
                      {ui.importPaper}
                    </button>
                  </div>
                ) : null}
              </div>

              <div className="composer__row">
                <textarea
                  value={composerValue}
                  onChange={(event) => setComposerValue(event.target.value)}
                  placeholder={ui.inputPrompt}
                  rows={4}
                />
                <div className="composer__actions">
                  <button className="icon-button" type="button" title={ui.uploadPdfTitle} onClick={() => fileInputRef.current?.click()} disabled={isBusy}>
                    <FileUp size={18} />
                  </button>
                  <input
                    ref={fileInputRef}
                    className="hidden-input"
                    type="file"
                    accept="application/pdf"
                    onChange={(event) => void handleUploadPdf(event.target.files?.[0] ?? null)}
                  />
                  <button className={`action-button action-button--accent${isBusy ? " action-button--pulsing" : ""}`} type="button" onClick={() => void handleSubmitComposer()} disabled={isBusy || composerState.kind === "empty"}>
                    <span>{isBusy ? ui.running : ui.send}</span>
                  </button>
                </div>
              </div>
            </div>
          </section>

          <aside className="inspector">
            <div className="inspector__tabs">
              <button className={inspectorTab === "memory" ? "tab-button tab-button--active" : "tab-button"} type="button" onClick={() => setInspectorTab("memory")}>
                {ui.memoryTab}
              </button>
              <button className={inspectorTab === "timeline" ? "tab-button tab-button--active" : "tab-button"} type="button" onClick={() => setInspectorTab("timeline")}>
                {ui.timelineTab}
              </button>
            </div>

            <div className="inspector__content">
              {inspectorTab === "memory" ? (
                <div className="stack-list">
                  {memoryBundleLoading ? <p className="muted-copy">{ui.loading}</p> : null}
                  {memoryBundleError ? <p className="muted-copy">{memoryBundleError}</p> : null}
                  {!memoryBundleLoading && !memoryBundleError && memoryBundles ? (
                    <>
                      <section className="summary-block">
                        <span>{ui.memoryCountLabel}</span>
                        <strong>{memoryBundles.papers.length}</strong>
                      </section>
                      <section className="summary-block">
                        <span>{ui.noPaperSource}</span>
                        <strong>{memoryBundles.unscoped_memories.length}</strong>
                      </section>
                      {memoryBundles.papers.slice(0, 2).map((group) => (
                        <article key={group.paper.paper_id} className="detail-row">
                          <header>
                            <span>{group.paper.title}</span>
                            <span>{group.paper.memory_count}</span>
                          </header>
                          <p>{group.paper.file_name ?? group.paper.paper_id}</p>
                        </article>
                      ))}
                    </>
                  ) : null}
                  {!memoryBundleLoading && !memoryBundleError && !memoryBundles ? <p className="muted-copy">{ui.noMemory}</p> : null}
                </div>
              ) : null}
              {inspectorTab === "timeline" ? (
                <div className="stack-list">
                  {activeTimeline.map((event) => (
                    <article key={event.id} className="detail-row">
                      <header>
                        <span>{ui.eventTypeLabel(event.event_type)}</span>
                        <span>{formatTime(event.created_at, uiLanguage)}</span>
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

      {drawer === "memory" ? (
        <ModalOverlay title={ui.memoryLibrary} onClose={() => setDrawer(null)} closeLabel={ui.close}>
          <div className="modal-scope-tabs">
            <button
              className={memoryScope === "session" ? "tab-button tab-button--active" : "tab-button"}
              type="button"
              onClick={() => setMemoryScope("session")}
            >
              {ui.currentSessionMemories}
            </button>
            <button
              className={memoryScope === "global" ? "tab-button tab-button--active" : "tab-button"}
              type="button"
              onClick={() => setMemoryScope("global")}
            >
              {ui.globalMemories}
            </button>
          </div>
          <MemoryBundlesView
            ui={ui}
            loading={activeMemoryLoading}
            error={activeMemoryError}
            search={memorySearch}
            onSearchChange={setMemorySearch}
            paperFilter={memoryPaperFilter}
            onPaperFilterChange={setMemoryPaperFilter}
            memoryFilter={memoryFilter}
            onMemoryFilterChange={setMemoryFilter}
            sortMode={memorySort}
            onSortModeChange={setMemorySort}
            paperGroups={activeMemoryBundles?.papers ?? []}
            visiblePaperGroups={visiblePaperGroups}
            visibleUnscopedMemories={visibleUnscopedMemories}
            expandedGroupIds={expandedMemoryGroupIds}
            onToggleGroup={toggleMemoryGroup}
            expandedItemIds={expandedMemoryItemIds}
            onToggleItem={toggleMemoryItem}
            onDeleteMemory={(item) =>
              void handleDeleteMemory(item.memory_type as Exclude<MemoryKind, "all">, item.id)
            }
          />
        </ModalOverlay>
      ) : null}

      {drawer === "settings" ? (
        <ModalOverlay title={ui.settings} onClose={() => setDrawer(null)} closeLabel={ui.close}>
          <div className="settings-grid">
            <section className="summary-block">
              <span>{ui.languageSetting}</span>
              <strong>{uiLanguage === "zh" ? ui.chinese : ui.english}</strong>
              <div className="language-switch">
                <Languages size={15} />
                <button
                  className={uiLanguage === "zh" ? "segmented-button segmented-button--active" : "segmented-button"}
                  type="button"
                  onClick={() => {
                    setUiLanguage("zh");
                  }}
                >
                  {ui.chinese}
                </button>
                <button
                  className={uiLanguage === "en" ? "segmented-button segmented-button--active" : "segmented-button"}
                  type="button"
                  onClick={() => {
                    setUiLanguage("en");
                  }}
                >
                  {ui.english}
                </button>
              </div>
            </section>
            <section className="summary-block">
              <span>{ui.apiKeySetting}</span>
              <strong>{clientApiKey.trim() ? "Configured" : ui.notConfigured}</strong>
              <div className="settings-field">
                <input
                  type="password"
                  value={clientApiKey}
                  placeholder={ui.apiKeyPlaceholder}
                  onChange={(event) => setClientApiKey(event.target.value)}
                  autoComplete="off"
                  spellCheck={false}
                />
                <button
                  className="segmented-button"
                  type="button"
                  onClick={() => setClientApiKey("")}
                  disabled={!clientApiKey.trim()}
                >
                  {ui.clearApiKey}
                </button>
              </div>
              <small className="settings-help">{ui.apiKeyHelp}</small>
            </section>
            <section className="summary-block">
              <span>{ui.storage}</span>
              <strong>{runtimeStatus?.storage_backend ?? "sqlite"}</strong>
              <small>{runtimeStatus?.sqlite_path ?? ui.defaultPath}</small>
            </section>
            <section className="summary-block">
              <span>{ui.queryAgent}</span>
              <strong>{runtimeStatus?.query_agent_backend ?? "turn_adapter"}</strong>
              <small>{runtimeStatus?.query_agent_provider}:{runtimeStatus?.query_agent_model}</small>
            </section>
            <section className="summary-block">
              <span>{ui.ingestExtraction}</span>
              <strong>{runtimeStatus?.ingest_extraction_backend ?? "heuristic"}</strong>
              <small>{runtimeStatus?.ingest_extraction_provider}:{runtimeStatus?.ingest_extraction_model}</small>
            </section>
            <section className="summary-block">
              <span>{ui.openviking}</span>
              <strong>{runtimeStatus?.openviking_backend ?? "noop"}</strong>
              <small>{runtimeStatus?.openviking_data_path ?? runtimeStatus?.openviking_url ?? ui.notConfigured}</small>
            </section>
          </div>
        </ModalOverlay>
      ) : null}

      {error ? <div className="toast toast--error">{error}</div> : null}
    </div>
  );
}
