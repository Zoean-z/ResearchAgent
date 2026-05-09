import type {
  MemoryBundles,
  Message,
  RuntimeStatus,
  Session,
  TaskRun,
  TimelineEvent,
  TraceNarrative,
  TraceStep,
} from "./types";

const SESSION_ID = "session-demo-memory-routed-agent";

type DemoRunDefinition = {
  id: string;
  messageId: string;
  userVisibleAt: number;
  assistantVisibleAt: number;
  stepCount: number;
  startedAt: string;
  finishedAt: string;
};

type DemoTrace = {
  steps: TraceStep[];
  narratives: TraceNarrative[];
};

const sessions: Session[] = [
  {
    id: SESSION_ID,
    title: "Memory-Routed CS Demo",
    created_at: "2026-05-08T10:00:00Z",
    updated_at: "2026-05-08T10:12:30Z",
    status: "active",
  },
];

const allMessages: Message[] = [
  {
    id: "msg-demo-user-1",
    session_id: SESSION_ID,
    role: "user",
    type: "followup_query",
    content:
      "Search a few computer science papers on arXiv about memory-routed research agents, import one, and summarize its main contribution.",
    created_at: "2026-05-08T10:00:10Z",
    status: "accepted",
  },
  {
    id: "msg-demo-assistant-1",
    session_id: SESSION_ID,
    role: "assistant",
    type: "followup_query",
    content:
      "I searched arXiv first, selected arXiv:2401.12345, and imported it through the real ingest path. The paper's main contribution is that it replaces a fixed retrieve-then-generate workflow with model-routed decisions over session memory, global memory, and source rereads, so later answers are shaped by durable research memory instead of always starting from the PDF.",
    created_at: "2026-05-08T10:01:24Z",
    status: "completed",
  },
  {
    id: "msg-demo-user-2",
    session_id: SESSION_ID,
    role: "user",
    type: "followup_query",
    content: "If I keep asking follow-up questions, how does long-term memory change the next answer?",
    created_at: "2026-05-08T10:03:05Z",
    status: "accepted",
  },
  {
    id: "msg-demo-assistant-2",
    session_id: SESSION_ID,
    role: "assistant",
    type: "followup_query",
    content:
      "The next answer starts from the session memory written during import, then checks related global memory, and only rereads source chunks when those memories are not enough. That means the agent's later decisions are visibly memory-first instead of PDF-first.",
    created_at: "2026-05-08T10:03:42Z",
    status: "completed",
  },
  {
    id: "msg-demo-user-3",
    session_id: SESSION_ID,
    role: "user",
    type: "followup_query",
    content: "Now compare this agent with a standard RAG workflow in one short paragraph.",
    created_at: "2026-05-08T10:06:10Z",
    status: "accepted",
  },
  {
    id: "msg-demo-assistant-3",
    session_id: SESSION_ID,
    role: "assistant",
    type: "followup_query",
    content:
      "A standard RAG system usually forces every question through the same retrieval pipeline. This agent does not. It can answer directly, search session memory, search global memory, or reread source passages depending on what the model thinks is necessary, and the trace shows which choice was made and why.",
    created_at: "2026-05-08T10:06:34Z",
    status: "completed",
  },
  {
    id: "msg-demo-user-4",
    session_id: SESSION_ID,
    role: "user",
    type: "followup_query",
    content: "What exactly gets stored in long-term memory after import?",
    created_at: "2026-05-08T10:09:15Z",
    status: "accepted",
  },
  {
    id: "msg-demo-assistant-4",
    session_id: SESSION_ID,
    role: "assistant",
    type: "followup_query",
    content:
      "The long-term layer stores structured paper memory, open questions, and the source chunks that support them. In this demo you can also see a related global paper memory, which lets future sessions reuse a stable lesson without re-importing the same paper each time.",
    created_at: "2026-05-08T10:09:44Z",
    status: "completed",
  },
];

const runDefinitions: DemoRunDefinition[] = [
  {
    id: "run-demo-search-import-answer",
    messageId: "msg-demo-user-1",
    userVisibleAt: 1,
    assistantVisibleAt: 2,
    stepCount: 5,
    startedAt: "2026-05-08T10:00:11Z",
    finishedAt: "2026-05-08T10:01:23Z",
  },
  {
    id: "run-demo-memory-followup",
    messageId: "msg-demo-user-2",
    userVisibleAt: 3,
    assistantVisibleAt: 4,
    stepCount: 4,
    startedAt: "2026-05-08T10:03:06Z",
    finishedAt: "2026-05-08T10:03:41Z",
  },
  {
    id: "run-demo-rag-comparison",
    messageId: "msg-demo-user-3",
    userVisibleAt: 5,
    assistantVisibleAt: 6,
    stepCount: 4,
    startedAt: "2026-05-08T10:06:11Z",
    finishedAt: "2026-05-08T10:06:33Z",
  },
  {
    id: "run-demo-memory-snapshot",
    messageId: "msg-demo-user-4",
    userVisibleAt: 7,
    assistantVisibleAt: 8,
    stepCount: 3,
    startedAt: "2026-05-08T10:09:16Z",
    finishedAt: "2026-05-08T10:09:43Z",
  },
];

const fullTraces: Record<string, DemoTrace> = {
  "run-demo-search-import-answer": {
    steps: [
      buildStep("trace-search-1", "run-demo-search-import-answer", "search_arxiv", {
        planner_decision: {
          selected_tool: "search_arxiv",
          rationale: "Search candidate papers before importing one into the current session.",
          fallback_used: false,
        },
        query: "computer science memory-routed research agents",
        category: "cs",
      }, {
        success: true,
        query: "all:computer science memory-routed research agents AND cat:cs",
        count: 3,
        papers: [
          { arxiv_id: "2401.12345", title: "Memory-Routed Research Agents" },
          { arxiv_id: "2402.23456", title: "Adaptive Retrieval for Paper Agents" },
          { arxiv_id: "2403.34567", title: "Session Memory for Scientific QA" },
        ],
      }, "2026-05-08T10:00:12Z", "2026-05-08T10:00:26Z"),
      buildStep("trace-search-2", "run-demo-search-import-answer", "import_arxiv_paper", {
        planner_decision: {
          selected_tool: "import_arxiv_paper",
          rationale: "Import the most relevant candidate through the existing arXiv ingest run.",
          fallback_used: false,
        },
        arxiv_id_or_url: "2401.12345",
      }, {
        success: true,
        arxiv_id: "2401.12345",
        paper_id: "paper:arxiv:2401.12345",
        artifact_id: "artifact-arxiv-2401",
        chunk_count: 42,
        ingest_summary: "Import completed and the paper memory plus open questions were written.",
      }, "2026-05-08T10:00:27Z", "2026-05-08T10:00:57Z"),
      buildStep("trace-search-3", "run-demo-search-import-answer", "retrieve_session_memories", {
        planner_decision: {
          selected_tool: "retrieve_session_memories",
          rationale: "Use the newly written session memories before rereading the source.",
          fallback_used: false,
        },
      }, {
        memory_ids: ["memory-paper-2401", "memory-open-2401"],
        coverage_score: 0.91,
        matched_query_terms: ["memory-routed", "agent", "decision"],
      }, "2026-05-08T10:00:58Z", "2026-05-08T10:01:08Z"),
      buildStep("trace-search-4", "run-demo-search-import-answer", "rerank_context_candidates", {
        planner_decision: {
          selected_tool: "rerank_context_candidates",
          rationale: "Promote the imported paper memory and keep the open question as a secondary signal.",
          fallback_used: false,
        },
      }, {
        selected_memory_ids: ["memory-paper-2401", "memory-open-2401"],
        selection_source: "session_memory_first",
        fallback_used: false,
      }, "2026-05-08T10:01:09Z", "2026-05-08T10:01:15Z"),
      buildStep("trace-search-5", "run-demo-search-import-answer", "compose_mock_answer", {
        planner_decision: {
          selected_tool: "compose_mock_answer",
          rationale: "The paper memory is already sufficient to summarize the main contribution.",
          fallback_used: false,
        },
      }, {
        answer_preview:
          "The agent searched arXiv, imported one paper, and used newly written memory to summarize the contribution.",
        memory_citations: [{ memory_id: "memory-paper-2401" }, { memory_id: "memory-open-2401" }],
        source_reread_chunks: [],
      }, "2026-05-08T10:01:16Z", "2026-05-08T10:01:23Z"),
    ],
    narratives: [
      {
        trace_step_id: "trace-search-1",
        reason_text: "The agent broadens the candidate set before committing to an import.",
        impact_text: "Import selection is grounded in search output instead of a hardcoded paper id.",
      },
      {
        trace_step_id: "trace-search-2",
        reason_text: "Import reuses the existing ingest chain rather than inventing a separate PDF path.",
        impact_text: "The imported paper becomes durable session memory that can influence later answers.",
      },
    ],
  },
  "run-demo-memory-followup": {
    steps: [
      buildStep("trace-follow-1", "run-demo-memory-followup", "retrieve_session_memories", {
        planner_decision: {
          selected_tool: "retrieve_session_memories",
          rationale: "Follow-up questions should start from current-session paper memory.",
          fallback_used: false,
        },
      }, {
        memory_ids: ["memory-paper-2401", "memory-open-2401"],
        coverage_score: 0.95,
        matched_query_terms: ["long-term memory", "next answer", "follow-up"],
      }, "2026-05-08T10:03:07Z", "2026-05-08T10:03:16Z"),
      buildStep("trace-follow-2", "run-demo-memory-followup", "retrieve_global_memories", {
        planner_decision: {
          selected_tool: "retrieve_global_memories",
          rationale: "Add one reusable global lesson before deciding whether the source is needed again.",
          fallback_used: false,
        },
      }, {
        memory_ids: ["memory-global-1"],
        coverage_score: 0.62,
        matched_query_terms: ["memory-first", "research qa"],
      }, "2026-05-08T10:03:17Z", "2026-05-08T10:03:23Z"),
      buildStep("trace-follow-3", "run-demo-memory-followup", "rerank_context_candidates", {
        planner_decision: {
          selected_tool: "rerank_context_candidates",
          rationale: "This is a mechanism question, so memory should be enough.",
          fallback_used: false,
        },
      }, {
        selected_memory_ids: ["memory-paper-2401", "memory-global-1"],
        selection_source: "session_then_global",
        fallback_used: false,
      }, "2026-05-08T10:03:24Z", "2026-05-08T10:03:28Z"),
      buildStep("trace-follow-4", "run-demo-memory-followup", "direct_final_answer", {
        retrieval_skipped: true,
        planner_decision: {
          selected_tool: "direct_final_answer",
          rationale: "Session and global memory already cover the answer.",
          fallback_used: false,
        },
      }, {
        answer_preview: "Later answers hit paper memory and open questions before any source reread.",
      }, "2026-05-08T10:03:29Z", "2026-05-08T10:03:41Z"),
    ],
    narratives: [
      {
        trace_step_id: "trace-follow-4",
        reason_text: "The question asks how the system behaves, not for a detailed paper citation.",
        impact_text: "The demo makes the memory-first path visible without rereading the PDF.",
      },
    ],
  },
  "run-demo-rag-comparison": {
    steps: [
      buildStep("trace-rag-1", "run-demo-rag-comparison", "retrieve_session_memories", {
        planner_decision: {
          selected_tool: "retrieve_session_memories",
          rationale: "Reuse the imported paper memory for the comparison.",
          fallback_used: false,
        },
      }, {
        memory_ids: ["memory-paper-2401"],
        coverage_score: 0.88,
        matched_query_terms: ["rag", "workflow", "compare"],
      }, "2026-05-08T10:06:12Z", "2026-05-08T10:06:18Z"),
      buildStep("trace-rag-2", "run-demo-rag-comparison", "retrieve_global_memories", {
        planner_decision: {
          selected_tool: "retrieve_global_memories",
          rationale: "Add one reusable global principle about memory-first routing.",
          fallback_used: false,
        },
      }, {
        memory_ids: ["memory-global-1"],
        coverage_score: 0.57,
        matched_query_terms: ["memory-first", "retrieval"],
      }, "2026-05-08T10:06:19Z", "2026-05-08T10:06:23Z"),
      buildStep("trace-rag-3", "run-demo-rag-comparison", "rerank_context_candidates", {
        planner_decision: {
          selected_tool: "rerank_context_candidates",
          rationale: "The comparison only needs durable memory, not source rereads.",
          fallback_used: false,
        },
      }, {
        selected_memory_ids: ["memory-paper-2401", "memory-global-1"],
        selection_source: "comparison_memory_pack",
        fallback_used: false,
      }, "2026-05-08T10:06:24Z", "2026-05-08T10:06:27Z"),
      buildStep("trace-rag-4", "run-demo-rag-comparison", "direct_final_answer", {
        retrieval_skipped: true,
        planner_decision: {
          selected_tool: "direct_final_answer",
          rationale: "The answer is conceptual and can be formed from memory evidence alone.",
          fallback_used: false,
        },
      }, {
        answer_preview: "Standard RAG forces retrieval; this agent chooses whether retrieval is needed at all.",
      }, "2026-05-08T10:06:28Z", "2026-05-08T10:06:33Z"),
    ],
    narratives: [
      {
        trace_step_id: "trace-rag-4",
        reason_text: "The comparison is about orchestration policy rather than missing source facts.",
        impact_text: "The answer can stay short and direct while still reflecting memory-driven decisions.",
      },
    ],
  },
  "run-demo-memory-snapshot": {
    steps: [
      buildStep("trace-snapshot-1", "run-demo-memory-snapshot", "retrieve_session_memories", {
        planner_decision: {
          selected_tool: "retrieve_session_memories",
          rationale: "List what the current session already stored after import.",
          fallback_used: false,
        },
      }, {
        memory_ids: ["memory-paper-2401", "memory-open-2401"],
        coverage_score: 0.93,
        matched_query_terms: ["stored", "long-term memory", "import"],
      }, "2026-05-08T10:09:17Z", "2026-05-08T10:09:24Z"),
      buildStep("trace-snapshot-2", "run-demo-memory-snapshot", "retrieve_global_memories", {
        planner_decision: {
          selected_tool: "retrieve_global_memories",
          rationale: "Mention the reusable global memory layer as well.",
          fallback_used: false,
        },
      }, {
        memory_ids: ["memory-global-1"],
        coverage_score: 0.44,
        matched_query_terms: ["global memory", "reusable lesson"],
      }, "2026-05-08T10:09:25Z", "2026-05-08T10:09:31Z"),
      buildStep("trace-snapshot-3", "run-demo-memory-snapshot", "direct_final_answer", {
        retrieval_skipped: true,
        planner_decision: {
          selected_tool: "direct_final_answer",
          rationale: "No source reread is required because the memory inventory is already structured.",
          fallback_used: false,
        },
      }, {
        answer_preview: "The system stores paper memory, open questions, and supporting source chunks.",
      }, "2026-05-08T10:09:32Z", "2026-05-08T10:09:43Z"),
    ],
    narratives: [
      {
        trace_step_id: "trace-snapshot-3",
        reason_text: "The memory drawer already exposes the stored structures directly.",
        impact_text: "The reply explains the persistent layer without re-opening the source document.",
      },
    ],
  },
};

const allTimelineEvents: Array<TimelineEvent & { visibleAt: number }> = [
  buildEvent("evt-1", "run-demo-search-import-answer", "run_started", "Query started: search candidate papers before import.", 1, [], []),
  buildEvent("evt-2", "run-demo-search-import-answer", "step_completed", "search_arxiv returned 3 candidate papers.", 2, [], []),
  buildEvent("evt-3", "run-demo-search-import-answer", "step_completed", "import_arxiv_paper imported arXiv:2401.12345 through the ingest run.", 2, [], ["paper:arxiv:2401.12345"]),
  buildEvent("evt-4", "run-demo-search-import-answer", "assistant_message_committed", "The imported paper is now available as session memory.", 2, ["memory-paper-2401", "memory-open-2401"], ["paper:arxiv:2401.12345"]),
  buildEvent("evt-5", "run-demo-memory-followup", "run_started", "Follow-up started: session memory is checked before source rereads.", 3, [], ["paper:arxiv:2401.12345"]),
  buildEvent("evt-6", "run-demo-memory-followup", "run_finished", "The answer was completed directly from session and global memory.", 4, ["memory-paper-2401", "memory-global-1"], ["paper:arxiv:2401.12345"]),
  buildEvent("evt-7", "run-demo-rag-comparison", "run_started", "Comparison run started: use durable memory to contrast this agent with RAG.", 5, [], ["paper:arxiv:2401.12345"]),
  buildEvent("evt-8", "run-demo-rag-comparison", "run_finished", "The comparison answer was produced without rereading the source.", 6, ["memory-paper-2401", "memory-global-1"], ["paper:arxiv:2401.12345"]),
  buildEvent("evt-9", "run-demo-memory-snapshot", "run_started", "Memory snapshot run started: explain what long-term memory stores.", 7, [], ["paper:arxiv:2401.12345"]),
  buildEvent("evt-10", "run-demo-memory-snapshot", "run_finished", "The memory inventory answer was completed from stored structures.", 8, ["memory-paper-2401", "memory-open-2401", "memory-global-1"], ["paper:arxiv:2401.12345"]),
];

const sessionMemoryBundles: MemoryBundles = {
  papers: [
    {
      paper: {
        paper_id: "paper:arxiv:2401.12345",
        title: "Memory-Routed Research Agents",
        file_name: "2401.12345.pdf",
        created_at: "2026-05-08T10:00:57Z",
        updated_at: "2026-05-08T10:01:12Z",
        memory_count: 2,
      },
      paper_memories: [
        {
          id: "memory-paper-2401",
          memory_type: "paper_memory",
          content:
            "The paper proposes a memory-routed research agent that decides whether to answer from structured memory, search additional memory, or reread source passages instead of following a fixed retrieve-then-generate pipeline.",
          created_at: "2026-05-08T10:01:02Z",
          updated_at: "2026-05-08T10:01:12Z",
          paper_id: "paper:arxiv:2401.12345",
          source_paper: null,
          target_paper: null,
          relation_direction: null,
          relation_type: null,
          related_papers: [],
          source_chunk_ids: ["chunk-2401-4", "chunk-2401-9"],
          evidence_count: 2,
        },
      ],
      open_question_memories: [
        {
          id: "memory-open-2401",
          memory_type: "open_question_memory",
          content:
            "Open question: how stable are the memory-routing decisions when the session accumulates many partially overlapping papers and the model must avoid stale summaries?",
          created_at: "2026-05-08T10:01:05Z",
          updated_at: "2026-05-08T10:01:12Z",
          paper_id: "paper:arxiv:2401.12345",
          source_paper: null,
          target_paper: null,
          relation_direction: null,
          relation_type: null,
          related_papers: ["paper:arxiv:2401.12345"],
          source_chunk_ids: ["chunk-2401-31"],
          evidence_count: 1,
        },
      ],
      relation_memories: [],
      source_chunks: [
        {
          chunk_id: "chunk-2401-4",
          paper_id: "paper:arxiv:2401.12345",
          page: 2,
          section: "Introduction",
          excerpt:
            "We replace fixed retrieval ladders with model-routed decisions over session memory, global memory, and source reread.",
        },
        {
          chunk_id: "chunk-2401-31",
          paper_id: "paper:arxiv:2401.12345",
          page: 8,
          section: "Limitations",
          excerpt:
            "Routing quality may degrade as the session accumulates overlapping summaries and unresolved questions.",
        },
      ],
      source_chunk_count: 2,
      empty_fields: [],
    },
  ],
  unscoped_memories: [],
};

const globalMemoryBundles: MemoryBundles = {
  papers: [
    {
      paper: {
        paper_id: "paper:arxiv:2310.56789",
        title: "Persistent Memory for Scientific QA",
        file_name: "2310.56789.pdf",
        created_at: "2026-04-30T08:10:00Z",
        updated_at: "2026-05-01T09:20:00Z",
        memory_count: 1,
      },
      paper_memories: [
        {
          id: "memory-global-1",
          memory_type: "paper_memory",
          content:
            "Global lesson: scientific agents answer more consistently when session memory is checked before any fresh passage reread.",
          created_at: "2026-05-01T09:20:00Z",
          updated_at: "2026-05-01T09:20:00Z",
          paper_id: "paper:arxiv:2310.56789",
          source_paper: null,
          target_paper: null,
          relation_direction: null,
          relation_type: null,
          related_papers: [],
          source_chunk_ids: ["chunk-2310-7"],
          evidence_count: 1,
        },
      ],
      open_question_memories: [],
      relation_memories: [],
      source_chunks: [
        {
          chunk_id: "chunk-2310-7",
          paper_id: "paper:arxiv:2310.56789",
          page: 3,
          section: "Approach",
          excerpt: "Memory-first routing reduces unnecessary rereads and makes later answers auditable.",
        },
      ],
      source_chunk_count: 1,
      empty_fields: [],
    },
  ],
  unscoped_memories: [],
};

const runtimeStatus: RuntimeStatus = {
  app_name: "OpenViking Memory-Routed Paper Agent",
  storage_backend: "sqlite",
  sqlite_path: "./data/sqlite/research_agent.sqlite3",
  query_agent_backend: "turn_adapter",
  query_agent_provider: "deepseek",
  query_agent_model: "deepseek-v4-flash",
  ingest_extraction_backend: "model_adapter",
  ingest_extraction_provider: "deepseek",
  ingest_extraction_model: "deepseek-v4-flash",
  openviking_backend: "embedded",
  openviking_data_path: "./data/openviking",
  openviking_url: null,
};

const playbackScheduleMs = [700, 2300, 4200, 5800, 7600, 9300, 11100, 12900];
const listeners = new Set<() => void>();
let visibleMessageCount = 0;
let playbackStarted = false;

function buildStep(
  id: string,
  runId: string,
  action: string,
  inputPayload: Record<string, unknown>,
  resultPayload: Record<string, unknown>,
  startedAt: string,
  finishedAt: string,
): TraceStep {
  return {
    id,
    run_id: runId,
    action,
    input_payload: inputPayload,
    result_payload: resultPayload,
    status: "completed",
    started_at: startedAt,
    finished_at: finishedAt,
  };
}

function buildEvent(
  id: string,
  runId: string,
  eventType: string,
  summary: string,
  visibleAt: number,
  relatedMemoryIds: string[],
  relatedPaperIds: string[],
): TimelineEvent & { visibleAt: number } {
  return {
    id,
    session_id: SESSION_ID,
    run_id: runId,
    event_type: eventType,
    summary,
    related_memory_ids: relatedMemoryIds,
    related_paper_ids: relatedPaperIds,
    created_at: new Date(visibleAt * 1000).toISOString(),
    visibleAt,
  };
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: {
      "Content-Type": "application/json",
      "Cache-Control": "no-store",
    },
  });
}

function readOnlyResponse(): Response {
  return jsonResponse({ detail: "Demo is read-only. Use Docker deployment for the live app." }, 405);
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

function notify(): void {
  for (const listener of listeners) {
    listener();
  }
}

function startPlayback(): void {
  if (playbackStarted || typeof window === "undefined") {
    return;
  }
  playbackStarted = true;
  playbackScheduleMs.forEach((delay, index) => {
    window.setTimeout(() => {
      visibleMessageCount = Math.max(visibleMessageCount, index + 1);
      notify();
    }, delay);
  });
}

function getVisibleMessages(): Message[] {
  return allMessages.slice(0, visibleMessageCount);
}

function getVisibleRuns(): TaskRun[] {
  return [...runDefinitions]
    .filter((run) => visibleMessageCount >= run.userVisibleAt)
    .map((run) => {
      const finished = visibleMessageCount >= run.assistantVisibleAt;
      const progress = Math.max(1, visibleMessageCount - run.userVisibleAt + 1);
      return {
        id: run.id,
        session_id: SESSION_ID,
        message_id: run.messageId,
        status: finished ? "finished" : "running",
        step_count: finished ? run.stepCount : Math.min(run.stepCount - 1, progress),
        started_at: run.startedAt,
        finished_at: finished ? run.finishedAt : null,
        finish_reason: finished ? "completed" : null,
      };
    })
    .sort((left, right) => right.started_at.localeCompare(left.started_at));
}

function getVisibleTimeline(): TimelineEvent[] {
  return allTimelineEvents
    .filter((event) => visibleMessageCount >= event.visibleAt)
    .map(({ visibleAt: _visibleAt, ...event }) => event)
    .sort((left, right) => left.created_at.localeCompare(right.created_at));
}

function getVisibleTrace(runId: string): DemoTrace {
  const definition = runDefinitions.find((run) => run.id === runId);
  const trace = fullTraces[runId];
  if (!definition || !trace) {
    return { steps: [], narratives: [] };
  }
  if (visibleMessageCount >= definition.assistantVisibleAt) {
    return trace;
  }
  if (visibleMessageCount < definition.userVisibleAt) {
    return { steps: [], narratives: [] };
  }
  const visibleStepCount = Math.max(1, Math.min(trace.steps.length - 1, visibleMessageCount - definition.userVisibleAt + 1));
  return {
    steps: trace.steps.slice(0, visibleStepCount),
    narratives: trace.narratives,
  };
}

function match(pathname: string, pattern: RegExp): RegExpMatchArray | null {
  return pathname.match(pattern);
}

function handleApiRequest(method: string, pathname: string): Response | null {
  if (method !== "GET") {
    return readOnlyResponse();
  }

  if (pathname === "/api/sessions") {
    return jsonResponse({ items: sessions });
  }
  if (pathname === `/api/sessions/${SESSION_ID}/messages`) {
    return jsonResponse({ items: getVisibleMessages() });
  }
  if (pathname === `/api/sessions/${SESSION_ID}/runs`) {
    return jsonResponse({ items: getVisibleRuns() });
  }
  if (pathname === `/api/sessions/${SESSION_ID}/timeline`) {
    return jsonResponse({ items: getVisibleTimeline() });
  }
  if (pathname === `/api/sessions/${SESSION_ID}/memory-bundles`) {
    return jsonResponse(sessionMemoryBundles);
  }
  if (pathname === "/api/memories/global-bundles") {
    return jsonResponse(globalMemoryBundles);
  }
  if (pathname === "/api/system/runtime") {
    return jsonResponse(runtimeStatus);
  }

  const traceMatch = match(pathname, new RegExp(`^/api/sessions/${SESSION_ID}/runs/([^/]+)/trace$`));
  if (traceMatch) {
    return jsonResponse(getVisibleTrace(traceMatch[1]));
  }

  const eventsMatch = match(pathname, new RegExp(`^/api/sessions/${SESSION_ID}/runs/([^/]+)/events$`));
  if (eventsMatch) {
    return jsonResponse({
      items: getVisibleTimeline().filter((event) => event.run_id === eventsMatch[1]),
    });
  }

  if (pathname === `/api/sessions/${SESSION_ID}/memory-snapshot`) {
    return jsonResponse({
      paper_memories: [],
      relation_memories: [],
      open_question_memories: [],
    });
  }

  return jsonResponse({ detail: "Demo route not found." }, 404);
}

export function installDemoApi(): void {
  if (typeof window === "undefined") {
    return;
  }

  visibleMessageCount = 0;
  (globalThis as { __RESEARCH_AGENT_DEMO__?: { subscribe: typeof subscribe } }).__RESEARCH_AGENT_DEMO__ = {
    subscribe,
  };

  const originalFetch = window.fetch.bind(window);
  window.fetch = async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const request = input instanceof Request ? input : null;
    const method = init?.method ?? request?.method ?? "GET";
    const rawUrl = typeof input === "string" ? input : input instanceof URL ? input.toString() : input.url;
    const url = new URL(rawUrl, window.location.origin);

    if (!url.pathname.startsWith("/api/")) {
      return originalFetch(input, init);
    }

    const response = handleApiRequest(method.toUpperCase(), url.pathname);
    if (response) {
      return response;
    }
    return jsonResponse({ detail: "Demo route not found." }, 404);
  };

  startPlayback();
}
