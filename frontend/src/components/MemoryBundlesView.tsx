import type { ReactNode } from "react";
import { ChevronDown, ChevronRight, ListFilter, Search, Trash2 } from "lucide-react";

import type { MemoryBundleGroup, MemoryBundleItem } from "../lib/types";

type MemoryKind = "all" | "paper_memory" | "relation_memory" | "open_question_memory";
type MemorySortMode = "updated_at" | "created_at" | "title";
type UiLanguage = "zh" | "en";

type MemoryBundlesUi = {
  language: UiLanguage;
  loading: string;
  noMemory: string;
  noPaperSource: string;
  memorySearch: string;
  memoryPaperFilter: string;
  memoryTypeFilter: string;
  memorySort: string;
  sortRecentUpdated: string;
  sortRecentCreated: string;
  sortPaperTitle: string;
  createdAt: string;
  updatedAt: string;
  memoryCountLabel: string;
  expand: string;
  collapse: string;
  showMore: string;
  showLess: string;
  relationFrom: string;
  relationTo: string;
  memoryFilterAll: string;
  memoryFilterPaper: string;
  memoryFilterRelation: string;
  memoryFilterOpenQuestion: string;
  paperMemory: string;
  openQuestionMemory: string;
  relationMemory: string;
  noEvidenceText: string;
  deleteMemory: string;
  memoryKindLabel(kind: MemoryKind): string;
};

type MemoryBundlesViewProps = {
  ui: MemoryBundlesUi;
  loading: boolean;
  error: string | null;
  search: string;
  onSearchChange: (value: string) => void;
  paperFilter: string;
  onPaperFilterChange: (value: string) => void;
  memoryFilter: MemoryKind;
  onMemoryFilterChange: (value: MemoryKind) => void;
  sortMode: MemorySortMode;
  onSortModeChange: (value: MemorySortMode) => void;
  paperGroups: MemoryBundleGroup[];
  visiblePaperGroups: MemoryBundleGroup[];
  visibleUnscopedMemories: MemoryBundleItem[];
  expandedGroupIds: string[];
  onToggleGroup: (groupId: string) => void;
  expandedItemIds: string[];
  onToggleItem: (itemId: string) => void;
  onDeleteMemory: (item: MemoryBundleItem) => void;
};

function formatOptionalTime(value: string | null | undefined, language: UiLanguage) {
  if (!value) {
    return "—";
  }
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) {
    return "—";
  }
  return new Intl.DateTimeFormat(language === "zh" ? "zh-CN" : "en-US", {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(date);
}

function getCollapseLabel(ui: MemoryBundlesUi, expanded: boolean) {
  return expanded ? ui.collapse : ui.expand;
}

function renderMemoryMeta(item: MemoryBundleItem, ui: MemoryBundlesUi) {
  const metaParts = [
    `${ui.createdAt}: ${formatOptionalTime(item.created_at, ui.language)}`,
    `${ui.updatedAt}: ${formatOptionalTime(item.updated_at, ui.language)}`,
    item.evidence_count > 0 ? `evidence_count: ${item.evidence_count}` : null,
    item.source_chunk_ids.length > 0 ? `source_chunk_ids: ${item.source_chunk_ids.length}` : null,
  ].filter(Boolean);

  return metaParts.length > 0 ? (
    <div className="memory-card__meta">
      {metaParts.map((part, index) => (
        <span key={`${part}-${index}`}>{part}</span>
      ))}
    </div>
  ) : null;
}

function renderItemContent(item: MemoryBundleItem, ui: MemoryBundlesUi, expanded: boolean, onToggle: () => void) {
  const content = item.content?.trim() || ui.noEvidenceText;
  const shouldShowToggle = content.length > 160;

  return (
    <>
      <p className={expanded ? "memory-card__content memory-card__content--expanded" : "memory-card__content"}>{content}</p>
      {shouldShowToggle ? (
        <button className="memory-card__expand" type="button" onClick={onToggle}>
          {getCollapseLabel(ui, expanded)}
        </button>
      ) : null}
    </>
  );
}

function renderMemoryCard(
  item: MemoryBundleItem,
  ui: MemoryBundlesUi,
  expanded: boolean,
  onToggle: () => void,
  onDelete: () => void,
  extra: ReactNode | null = null,
) {
  return (
    <article key={item.id} className="memory-card">
      <header className="memory-card__header">
        <span className="memory-card__badge">{ui.memoryKindLabel(item.memory_type as MemoryKind)}</span>
        <div className="memory-card__actions">
          <span className="memory-card__id">{item.id.slice(0, 8)}</span>
          <button
            className="memory-card__delete"
            type="button"
            title={ui.deleteMemory}
            aria-label={ui.deleteMemory}
            onClick={(event) => {
              event.stopPropagation();
              onDelete();
            }}
          >
            <Trash2 size={14} />
          </button>
        </div>
      </header>
      {renderItemContent(item, ui, expanded, onToggle)}
      {renderMemoryMeta(item, ui)}
      {extra}
    </article>
  );
}

function renderRelationDetails(item: MemoryBundleItem, ui: MemoryBundlesUi) {
  return (
    <div className="memory-card__relations">
      <span>
        {ui.relationFrom}: {item.source_paper ?? "—"}
      </span>
      <span>
        {ui.relationTo}: {item.target_paper ?? "—"}
      </span>
      <span>relation_type: {item.relation_type ?? "—"}</span>
      <span>relation_direction: {item.relation_direction ?? "—"}</span>
      {item.related_papers.length > 0 ? <span>{item.related_papers.join(" · ")}</span> : null}
    </div>
  );
}

export function MemoryBundlesView({
  ui,
  loading,
  error,
  search,
  onSearchChange,
  paperFilter,
  onPaperFilterChange,
  memoryFilter,
  onMemoryFilterChange,
  sortMode,
  onSortModeChange,
  paperGroups,
  visiblePaperGroups,
  visibleUnscopedMemories,
  expandedGroupIds,
  onToggleGroup,
  expandedItemIds,
  onToggleItem,
  onDeleteMemory,
}: MemoryBundlesViewProps) {
  const hasContent = visiblePaperGroups.length > 0 || visibleUnscopedMemories.length > 0;

  return (
    <div className="memory-panel">
      <div className="memory-toolbar">
        <label className="memory-field">
          <span>{ui.memorySearch}</span>
          <div className="memory-field__control">
            <Search size={14} />
            <input type="search" value={search} placeholder={ui.memorySearch} onChange={(event) => onSearchChange(event.target.value)} />
          </div>
        </label>

        <label className="memory-field">
          <span>{ui.memoryPaperFilter}</span>
          <div className="memory-field__control memory-field__control--select">
            <ListFilter size={14} />
            <select value={paperFilter} onChange={(event) => onPaperFilterChange(event.target.value)}>
              <option value="all">{ui.memoryFilterAll}</option>
              {paperGroups.map((group) => (
                <option key={group.paper.paper_id} value={group.paper.paper_id}>
                  {group.paper.title}
                  {group.paper.file_name ? ` · ${group.paper.file_name}` : ""}
                </option>
              ))}
            </select>
          </div>
        </label>

        <label className="memory-field">
          <span>{ui.memoryTypeFilter}</span>
          <div className="memory-field__control memory-field__control--select">
            <ListFilter size={14} />
            <select value={memoryFilter} onChange={(event) => onMemoryFilterChange(event.target.value as MemoryKind)}>
              <option value="all">{ui.memoryFilterAll}</option>
              <option value="paper_memory">{ui.memoryFilterPaper}</option>
              <option value="relation_memory">{ui.memoryFilterRelation}</option>
              <option value="open_question_memory">{ui.memoryFilterOpenQuestion}</option>
            </select>
          </div>
        </label>

        <label className="memory-field">
          <span>{ui.memorySort}</span>
          <div className="memory-field__control memory-field__control--select">
            <ListFilter size={14} />
            <select value={sortMode} onChange={(event) => onSortModeChange(event.target.value as MemorySortMode)}>
              <option value="updated_at">{ui.sortRecentUpdated}</option>
              <option value="created_at">{ui.sortRecentCreated}</option>
              <option value="title">{ui.sortPaperTitle}</option>
            </select>
          </div>
        </label>
      </div>

      {loading ? <p className="muted-copy">{ui.loading}</p> : null}
      {error ? <div className="memory-error">{error}</div> : null}
      {!loading && !error && !hasContent ? <p className="muted-copy">{ui.noMemory}</p> : null}

      <div className="memory-groups">
        {visiblePaperGroups.map((group) => {
          const expanded = expandedGroupIds.includes(group.paper.paper_id);
          return (
            <section key={group.paper.paper_id} className="memory-group">
              <button
                className="memory-group__header"
                type="button"
                onClick={() => onToggleGroup(group.paper.paper_id)}
                aria-expanded={expanded}
              >
                <div className="memory-group__header-main">
                  <div className="memory-group__title-row">
                    <strong>{group.paper.title}</strong>
                    <span className="memory-group__toggle">{expanded ? <ChevronDown size={16} /> : <ChevronRight size={16} />}</span>
                  </div>
                  <div className="memory-group__source">
                    <span>{group.paper.paper_id}</span>
                    {group.paper.file_name ? <span>{group.paper.file_name}</span> : null}
                  </div>
                </div>
                <div className="memory-group__meta">
                  <span>
                    {ui.memoryCountLabel}: {group.paper.memory_count}
                  </span>
                  <span>
                    {ui.createdAt}: {formatOptionalTime(group.paper.created_at, ui.language)}
                  </span>
                  <span>
                    {ui.updatedAt}: {formatOptionalTime(group.paper.updated_at, ui.language)}
                  </span>
                </div>
              </button>

              {expanded ? (
                <div className="memory-group__body">
                  {group.paper_memories.length > 0 ? (
                    <section className="memory-section">
                      <h4>{ui.paperMemory}</h4>
                      <div className="memory-section__grid">
                        {group.paper_memories.map((item) =>
                          renderMemoryCard(
                            item,
                            ui,
                            expandedItemIds.includes(item.id),
                            () => onToggleItem(item.id),
                            () => onDeleteMemory(item),
                          ),
                        )}
                      </div>
                    </section>
                  ) : null}

                  {group.open_question_memories.length > 0 ? (
                    <section className="memory-section">
                      <h4>{ui.openQuestionMemory}</h4>
                      <div className="memory-section__grid">
                        {group.open_question_memories.map((item) =>
                          renderMemoryCard(
                            item,
                            ui,
                            expandedItemIds.includes(item.id),
                            () => onToggleItem(item.id),
                            () => onDeleteMemory(item),
                          ),
                        )}
                      </div>
                    </section>
                  ) : null}

                  {group.relation_memories.length > 0 ? (
                    <section className="memory-section">
                      <h4>{ui.relationMemory}</h4>
                      <div className="memory-section__grid">
                        {group.relation_memories.map((item) =>
                          renderMemoryCard(
                            item,
                            ui,
                            expandedItemIds.includes(item.id),
                            () => onToggleItem(item.id),
                            () => onDeleteMemory(item),
                            renderRelationDetails(item, ui),
                          ),
                        )}
                      </div>
                    </section>
                  ) : null}
                </div>
              ) : null}
            </section>
          );
        })}

        {visibleUnscopedMemories.length > 0 ? (
          <section className="memory-group memory-group--unscoped">
            <div className="memory-group__header">
              <div className="memory-group__header-main">
                <div className="memory-group__title-row">
                  <strong>{ui.noPaperSource}</strong>
                </div>
                <div className="memory-group__source">
                  <span>{visibleUnscopedMemories.length}</span>
                </div>
              </div>
              <div className="memory-group__meta">
                <span>
                  {ui.memoryCountLabel}: {visibleUnscopedMemories.length}
                </span>
              </div>
            </div>
            <div className="memory-group__body">
              <div className="memory-section__grid">
                {visibleUnscopedMemories.map((item) =>
                  renderMemoryCard(
                    item,
                    ui,
                    expandedItemIds.includes(item.id),
                    () => onToggleItem(item.id),
                    () => onDeleteMemory(item),
                  ),
                )}
              </div>
            </div>
          </section>
        ) : null}
      </div>
    </div>
  );
}
