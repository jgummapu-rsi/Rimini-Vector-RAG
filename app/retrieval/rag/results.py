"""Result models shared by retrieval and answer generation."""

from dataclasses import dataclass, field


@dataclass
class QueryResult:
    """A generated answer plus its evidence and attribution."""

    question: str
    answer: str
    contexts: list[str]
    chunk_ids: list[str]
    scores: list[float] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
    grounded: bool = True
    evidence_origin: str = "retrieved"
    answer_status: str = "answered"


@dataclass
class RetrievalResult:
    """Retrieval-only evidence returned by POST /query."""

    question: str
    contexts: list[str]
    chunk_ids: list[str]
    scores: list[float] = field(default_factory=list)
    sub_questions: list[str] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    trace: list[dict] = field(default_factory=list)
