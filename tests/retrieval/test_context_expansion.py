from types import SimpleNamespace

from psycopg2.extras import Json

from app.retrieval.rag.access import access_predicate
from app.retrieval.rag.context import asks_for_collection, expand_evidence, matching_roster
from app.retrieval.rag.grounding import pack_evidence
from app.retrieval.rag.query import RetrievalResult
from app.shared.adapters.postgres.db import transaction
from app.shared.domain.models import Document, Principal, Role
from app.shared.ids import new_object_id
from app.shared.ports.vector_store import SearchHit, VectorPoint


def test_collection_detection_and_schema_matching_are_domain_independent():
    for question, header in [
        ("List all suppliers", "Supplier name | ID"),
        ("How many products are listed?", "Product | Price"),
        ("Who are all the candidates?", "Candidate Name | Candidate ID"),
    ]:
        assert asks_for_collection(question)
        assert matching_roster(question, f"| {header} |\n| --- | --- |\n| A | 1 |\n| B | 2 |")
    assert not asks_for_collection("Where was product A manufactured?")
    assert not matching_roster(
        "List all data engineers",
        "| Senior data engineer who worked on many projects and built complex pipelines for clients |\n| --- |\n| Biography |\n| Experience |",
    )


def test_expansion_orders_heading_before_details_and_deduplicates_overlap():
    table1 = "| Product | ID |\n| --- | --- |\n| Alpha | 001 |\n| Beta | 002 |"
    table2 = "| Product | ID |\n| --- | --- |\n| Beta | 002 |\n| Gamma | 003 |"

    def hit(cid, text, ordinal):
        return SearchHit(
            cid,
            0.0,
            {
                "_id": "doc",
                "generation_id": "g",
                "content": text,
                "source_ordinal": ordinal,
                "location": "p.1",
            },
        )

    container = SimpleNamespace(
        vectors=SimpleNamespace(
            surrounding_chunks=lambda *a, **k: [hit("second", table2, 2), hit("first", table1, 1)]
        )
    )
    seed = RetrievalResult(
        "List all products",
        [table2],
        ["second"],
        [1.0],
        citations=[{"chunk_id": "second", "document_id": "doc", "generation_id": "g"}],
    )
    result = expand_evidence(container, "tenant", seed)
    assert result.chunk_ids == ["first", "second"]
    assert "\n".join(result.contexts).count("| Beta | 002 |") == 1
    assert "| Gamma | 003 |" in result.contexts[1]
    # Citation still exposes the original evidence, not a invented combined source.
    assert result.citations[0]["snippet"] == table1


def test_sql_neighbors_respect_acl_generation_section_and_numeric_order(container, tenant):

    tid = tenant["id"]
    doc = Document(
        new_object_id(),
        tid,
        tenant["admin_id"],
        "text",
        "unused",
        "hash",
        "text/plain",
        "manual.txt",
        "tenant",
        [],
    )
    container.metadata.create_document(doc)
    vector = [1.0] + [0.0] * (container.embedder.dim - 1)
    points = []
    for cid, ordinal, section in [
        ("z-header", 9, "Alpha"),
        ("a-details", 10, "Alpha"),
        ("b-next", 11, "Beta"),
    ]:
        points.append(
            VectorPoint(
                cid,
                tid,
                vector,
                {
                    "_id": doc.id,
                    "content": cid,
                    "meta": {"section_path": section},
                    "visibility": "tenant",
                },
            )
        )
        with transaction(container.settings.postgres_dsn) as cur:
            cur.execute(
                "INSERT INTO chunks(id,document_id,tenant_id,ordinal,modality,text,meta) VALUES(%s,%s,%s,%s,'text',%s,%s)",
                (cid, doc.id, tid, ordinal, cid, Json({"section_path": section})),
            )
    container.vectors.upsert(points)
    access = access_predicate(Principal(tid, tenant["member_id"], Role.MEMBER))
    hits = container.vectors.surrounding_chunks(tid, ["a-details"], access)
    assert {h.chunk_id for h in hits} == {"z-header", "a-details"}
    assert {h.payload["source_ordinal"] for h in hits} == {9, 10}
    assert container.vectors.surrounding_chunks("other-tenant", ["a-details"]) == []
    assert container.vectors.surrounding_chunks(tid, ["a-details"], lambda p: False) == []
    with transaction(container.settings.postgres_dsn) as cur:
        cur.execute("UPDATE documents SET visibility='private' WHERE id=%s", (doc.id,))
    assert container.vectors.surrounding_chunks(tid, ["a-details"], access) == []


def test_collection_channel_finds_roster_despite_irrelevant_dense_vectors(container):
    vector = [1.0] + [0.0] * (container.embedder.dim - 1)
    container.vectors.upsert(
        [
            VectorPoint(
                "roster",
                "t",
                vector,
                {
                    "content": "| Supplier | ID |\n| --- | --- |\n| Alpha | 001 |\n| Beta | 002 |",
                    "modality": "table",
                },
            ),
            VectorPoint(
                "biography",
                "t",
                vector,
                {"content": "Supplier services overview", "modality": "text"},
            ),
            VectorPoint(
                "hidden",
                "other",
                vector,
                {
                    "content": "| Supplier | ID |\n| --- | --- |\n| Secret | 1 |\n| Hidden | 2 |",
                    "modality": "table",
                },
            ),
        ]
    )
    hits = container.vectors.collection_candidates("t", "List all suppliers")
    assert [h.chunk_id for h in hits] == ["roster"]


def test_named_scope_does_not_discard_evidence_using_short_block_guesses():
    contexts = [
        "Engineer at Northwind\n2018-2020",
        "• Built warehouse pipelines.",
        "Engineer at Contoso\n2020-2023",
        "• Built search services.",
        "• Tested ingestion workflows.",
        "Engineer at Fabrikam\n2023-Present",
        "• Operates payment services.",
    ]
    citations = [
        {"document_id": "d", "generation_id": "g", "section_path": "Person A", "source_ordinal": i}
        for i in range(len(contexts))
    ]
    packed, selected = pack_evidence(
        "What did Person A do at Contoso?", contexts, "gpt-5-nano", citations
    )
    # Retrieval must not discard evidence using a short-block heading guess.
    assert selected == list(range(len(contexts)))
    assert [entry["source_id"] for entry in packed] == [str(i + 1) for i in range(len(contexts))]
    assert "search services" in packed[3]["text"]
    # Missing intervening evidence must not manufacture an employer relationship.
    citations[4]["source_ordinal"] = 20
    _, selected = pack_evidence(
        "What did Person A do at Contoso?", contexts, "gpt-5-nano", citations
    )
    assert selected == list(range(len(contexts)))
