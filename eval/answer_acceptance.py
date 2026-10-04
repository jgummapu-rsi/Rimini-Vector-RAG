from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path

from app.ingest.pipeline.runner import run_job
from app.retrieval.rag.access import access_predicate
from app.retrieval.rag.query import answer_query, retrieve_chunks
from app.shared.container import build_container
from app.shared.domain.models import Document, DocumentVersion, Job, Principal, Role
from app.shared.ids import new_object_id
from eval.storage import isolated_evaluation

log = logging.getLogger(__name__)


def score_case(case: dict, retrieval, answer) -> dict:
    expected = case["evidence"]
    retrieved = sum(any(span in context for context in retrieval.contexts) for span in expected)
    used = sum(any(span in context for context in answer.contexts) for span in expected)
    citations = answer.citations
    supported = sum(
        any(span in citation.get("snippet", "") for span in expected) for citation in citations
    )
    text = answer.answer
    invented = any(value in text for value in case.get("forbidden_values", []))
    required = all(value in text for value in case.get("required_values", []))
    abstained = answer.answer_status == "insufficient_evidence"
    return {
        "id": case["id"],
        "answerable": case["answerable"],
        "expected_spans": len(expected),
        "retrieved_spans_50": retrieved,
        "evidence_spans_10": used,
        "citations": len(citations),
        "supported_citations": supported,
        "correct_abstention": not case["answerable"] and abstained,
        "answered_correct_values": case["answerable"]
        and not abstained
        and required
        and not invented,
        "critical_invented_value": invented,
        "abstained": abstained,
    }


@isolated_evaluation
def main(settings) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if manifest.get("split") != "held_out" or not manifest.get("reviewed_by"):
        raise ValueError(
            "Acceptance requires a held-out manifest with explicit reviewer provenance"
        )
    if not settings.litellm_base_url or not settings.litellm_api_key:
        raise ValueError("EVAL_LITELLM_BASE_URL and EVAL_LITELLM_API_KEY are required")
    container = build_container(settings)
    try:
        tenant = container.metadata.create_tenant("held-out-acceptance")
        owner = container.metadata.create_user(
            tenant, "owner@acceptance.test", "admin", "acceptance-owner"
        )
        principal = Principal(tenant, owner, Role.ADMIN)
        for source in manifest["documents"]:
            path = (args.manifest.parent / source["path"]).resolve()
            data = path.read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest != source["sha256"]:
                raise ValueError("Held-out source hash differs from reviewed manifest")
            blob = container.blob.put(tenant, digest, path.suffix, data)
            document = Document(
                new_object_id(),
                tenant,
                owner,
                path.suffix.lstrip("."),
                blob,
                digest,
                "application/octet-stream",
                path.name,
                "private",
                [],
            )
            job = Job(new_object_id(), document.id, tenant, "parse", "queued", 0)
            version = DocumentVersion(
                new_object_id(),
                document.id,
                tenant,
                1,
                digest,
                blob,
                path.name,
                len(data),
                owner,
                job.id,
            )
            container.metadata.create_document_with_job(document, job, version)
            run_job(container, container.queue.claim_next())
        observations = []
        for case in manifest["cases"]:
            retrieval = retrieve_chunks(
                container, tenant, case["question"], top_k=50, access=access_predicate(principal)
            )
            answer = answer_query(
                container, tenant, case["question"], top_k=10, access=access_predicate(principal)
            )
            observations.append(score_case(case, retrieval, answer))
        totals = {
            key: sum(row[key] for row in observations)
            for key in (
                "expected_spans",
                "retrieved_spans_50",
                "evidence_spans_10",
                "citations",
                "supported_citations",
                "correct_abstention",
                "critical_invented_value",
            )
        }
        totals["unanswerable_cases"] = sum(not row["answerable"] for row in observations)
        sufficient = (
            totals["expected_spans"] > 0
            and totals["citations"] > 0
            and totals["unanswerable_cases"] > 0
        )
        passed = sufficient and (
            totals["retrieved_spans_50"] / totals["expected_spans"] >= 0.95
            and totals["evidence_spans_10"] / totals["expected_spans"] >= 0.90
            and totals["supported_citations"] / totals["citations"] >= 0.95
            and totals["correct_abstention"] / totals["unanswerable_cases"] >= 0.90
            and totals["critical_invented_value"] == 0
            and all(row["answered_correct_values"] for row in observations if row["answerable"])
        )
        print(
            json.dumps(
                {
                    "passed": passed,
                    "counts": totals,
                    "cases": observations,
                    "limitations": "Exact-span citation support is not a semantic entailment judge; human claim review is still required.",
                },
                indent=2,
            )
        )
        if not passed:
            raise RuntimeError("Held-out acceptance thresholds were not met")
    finally:
        container.close()


if __name__ == "__main__":
    main()
