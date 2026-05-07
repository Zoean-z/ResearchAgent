export type Session = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  status: string;
};

export type Message = {
  id: string;
  session_id: string;
  role: string;
  type: string;
  content: string;
  created_at: string;
  status: string;
};

export type TimelineEvent = {
  id: string;
  session_id: string;
  run_id: string | null;
  event_type: string;
  summary: string;
  related_memory_ids: string[];
  related_paper_ids: string[];
  created_at: string;
};

export type TraceStep = {
  id: string;
  run_id: string;
  action: string;
  input_payload: Record<string, unknown>;
  result_payload: Record<string, unknown>;
  status: string;
  started_at: string;
  finished_at: string | null;
};

export type TraceNarrative = {
  trace_step_id: string;
  reason_text: string;
  impact_text: string;
};

export type PaperMemory = {
  id: string;
  paper_id: string;
  problem: string | null;
  method: string | null;
  key_results: string[];
  limitations: string[];
  novelty_claim: string | null;
  confidence: number;
  updated_at: string;
};

export type RelationMemory = {
  id: string;
  source_paper: string;
  target_paper: string;
  relation_type: string;
  summary: string;
  evidence: string[];
  confidence: number;
  updated_at: string;
};

export type OpenQuestionMemory = {
  id: string;
  unresolved_question: string;
  related_papers: string[];
  why_open: string[];
  possible_followup: string[];
  confidence: number;
  updated_at: string;
};

export type MemorySnapshot = {
  paper_memories: PaperMemory[];
  relation_memories: RelationMemory[];
  open_question_memories: OpenQuestionMemory[];
};

export type MemoryBundleSourceChunk = {
  chunk_id: string;
  paper_id: string;
  page: number | null;
  section: string | null;
  excerpt: string;
};

export type MemoryBundleItem = {
  id: string;
  memory_type: string;
  content: string;
  created_at: string | null;
  updated_at: string;
  paper_id: string | null;
  source_paper: string | null;
  target_paper: string | null;
  relation_direction: "source" | "target" | null;
  relation_type: string | null;
  related_papers: string[];
  source_chunk_ids: string[];
  evidence_count: number;
};

export type MemoryBundlePaperInfo = {
  paper_id: string;
  title: string;
  file_name: string | null;
  created_at: string | null;
  updated_at: string;
  memory_count: number;
};

export type MemoryBundleGroup = {
  paper: MemoryBundlePaperInfo;
  paper_memories: MemoryBundleItem[];
  open_question_memories: MemoryBundleItem[];
  relation_memories: MemoryBundleItem[];
  source_chunks: MemoryBundleSourceChunk[];
  source_chunk_count: number;
  empty_fields: string[];
};

export type MemoryBundles = {
  papers: MemoryBundleGroup[];
  unscoped_memories: MemoryBundleItem[];
};

export type RuntimeStatus = {
  app_name: string;
  storage_backend: string;
  sqlite_path: string | null;
  query_agent_backend: string;
  query_agent_provider: string;
  query_agent_model: string;
  ingest_extraction_backend: string;
  ingest_extraction_provider: string;
  ingest_extraction_model: string;
  openviking_backend: string;
  openviking_data_path: string | null;
  openviking_url: string | null;
};

export type TaskRun = {
  id: string;
  session_id: string;
  message_id: string;
  status: string;
  step_count: number;
  started_at: string;
  finished_at: string | null;
  finish_reason: string | null;
};

export type MessageSubmission = {
  accepted: boolean;
  session_id: string;
  message_id: string;
  run_id: string;
  message_type: string;
  status: string;
};

export type QueryExecution = {
  task_run: {
    id: string;
    status: string;
    step_count: number;
    finish_reason: string | null;
  };
  answer: string;
  should_reread_source: boolean;
  reread_reason: string;
  used_memory_citations: Array<{
    memory_id: string;
    memory_type: string;
    summary: string;
    selection_reason: string;
  }>;
};

export type IngestExecution = {
  task_run: {
    id: string;
    status: string;
    step_count: number;
    finish_reason: string | null;
  };
  summary: string;
  source_type: string;
  paper_summary: {
    what_it_is_about: string;
    problem_solved: string;
    new_ideas: string[];
    limitations: string[];
    suggestions_or_questions: string[];
    confidence: number;
  };
};
