from app.shared.domain.models import Modality
from app.ingest.pipeline.chunker import ChunkSpec, chunk_elements
from app.ingest.pipeline.elements import Element

SPEC = ChunkSpec(target_tokens=60, overlap_tokens=15, max_tokens=90, min_tokens=8)

# A deterministic stand-in for an embedder's tokenizer: one token per
# whitespace-separated word. Lets these tests assert exact sizing without
# loading a real model, and stands in for "a different embedder with a
# different tokenizer" than the MiniLM default the chunker falls back to.
def _word_count(text):
    return len((text or "").split())


def _text_el(text, **meta):
    return Element(text, Modality.TEXT.value, "pdf_text", "text_layer", 0, meta)


def test_no_chunk_exceeds_max_tokens():
    para = ("Alpha beta gamma delta epsilon zeta eta theta. " * 6).strip()
    doc = "\n\n".join([para, para, para])
    recs = chunk_elements([_text_el(doc, page=1)], SPEC)
    assert recs
    assert all(r.token_count <= SPEC.max_tokens for r in recs)


def test_sentence_aligned_overlap_present():
    para = ("Alpha beta gamma delta epsilon zeta eta theta. " * 6).strip()
    recs = chunk_elements([_text_el("\n\n".join([para, para]), page=1)], SPEC)
    assert len(recs) >= 2
    last_sentence = [s for s in recs[0].text.split(".") if s.strip()][-1].strip()
    assert last_sentence in recs[1].text


def test_table_split_repeats_header():
    header = "| id | name | value |"
    sep = "| --- | --- | --- |"
    rows = [f"| {i} | item{i} | {i*10} |" for i in range(40)]
    md = "\n".join([header, sep, *rows])
    recs = chunk_elements([Element(md, Modality.TABLE.value, "csv_table",
                                   "structured_csv", 0, {})], SPEC)
    assert len(recs) > 1
    assert all(r.text.splitlines()[0] == header for r in recs)
    assert all(r.token_count <= SPEC.max_tokens for r in recs)


def test_modalities_stay_isolated():
    md = "| a | b |\n| --- | --- |\n| 1 | 2 |"
    els = [
        _text_el("Intro paragraph."),
        Element(md, Modality.TABLE.value, "pdf_table", "structured_table", 1, {}),
        Element("A chart of revenue.", Modality.IMAGE.value, "vision", "figure", 2, {}),
    ]
    mods = {r.modality for r in chunk_elements(els, SPEC)}
    assert mods == {"text", "table", "image"}


def test_consecutive_text_elements_merge():
    els = [_text_el("First short para."), _text_el("Second short para.")]
    recs = chunk_elements(els, SPEC)
    assert len(recs) == 1


def test_empty_input_and_provenance():
    assert chunk_elements([], SPEC) == []
    recs = chunk_elements([_text_el("Hello world.", page=3)], SPEC)
    assert recs[0].extractor == "pdf_text"
    assert recs[0].meta.get("pages") == [3]


def test_no_lone_heading_chunk():
    """A heading with no body of its own (immediately followed by a deeper
    subheading) must never become its own content-free chunk -- its text
    already reaches the child section via the ancestors-prefix."""
    from app.ingest.pipeline.blocks import split_blocks

    md = "## Parent Section\n\n### Child Section\n\nActual body content goes here."
    els = split_blocks(md, text_extractor="markdown", table_extractor="markdown_table",
                        text_reason="prose", table_reason="structured_table")
    recs = chunk_elements(els, SPEC)
    assert not any(r.text.strip() == "## Parent Section" for r in recs)
    assert any("## Parent Section" in r.text and "### Child Section" in r.text for r in recs)


def test_tail_sentences_excludes_heading_line():
    """The overlap seed carried into the next chunk must never include a
    heading line -- it belongs to the chunk it introduces, not to a seed
    carried past it (a direct unit test since this is impractical to force
    deterministically through the public packing API)."""
    from app.ingest.pipeline.chunker import _tail_sentences

    text = "## A Heading\n\nShort sentence one. Short sentence two."
    seed = _tail_sentences(text, budget=100)
    assert "#" not in seed
    assert "Short sentence two." in seed


def test_numbered_list_markers_dont_trigger_sentence_split():
    """A numbered list item's own period ("7.") must not be treated as a
    sentence end -- otherwise list items get torn away from their own list
    (an observed real failure)."""
    from app.ingest.pipeline.chunker import _SENT_RE

    para = "1. Do the first thing. 2. Do the second thing. 3. Do the third thing."
    parts = _SENT_RE.split(para)
    assert not any(p.strip().rstrip(".").isdigit() for p in parts)


def test_numbered_list_items_survive_oversized_split():
    para = " ".join(f"{i}. Step number {i} in the process happens here." for i in range(1, 8))
    recs = chunk_elements([_text_el(para)], SPEC)
    combined = " ".join(r.text for r in recs)
    for i in range(1, 8):
        assert f"{i}. Step number {i}" in combined


def test_table_split_carries_overlap_row():
    header = "| id | name |"
    sep = "| --- | --- |"
    rows = [f"| {i} | item{i} |" for i in range(30)]
    md = "\n".join([header, sep, *rows])
    spec = ChunkSpec(target_tokens=40, overlap_tokens=10, max_tokens=60, min_tokens=8)
    recs = chunk_elements([Element(md, Modality.TABLE.value, "csv_table",
                                   "structured_csv", 0, {})], spec)
    assert len(recs) > 1
    for a, b in zip(recs, recs[1:]):
        a_rows = a.text.splitlines()[2:]   # skip header + separator
        b_rows = b.text.splitlines()[2:]
        assert a_rows[-1] == b_rows[0]


def test_heading_path_caps_depth_for_deeply_nested_docs():
    from app.ingest.pipeline.blocks import split_blocks

    text = "\n\n".join([
        "# L1", "body0.", "## L2", "body1.", "### L3", "body2.",
        "#### L4", "body3.", "##### L5", "body4.",
    ])
    els = split_blocks(text, text_extractor="markdown", table_extractor="markdown_table",
                        text_reason="prose", table_reason="structured_table")
    deepest = next(e for e in els if "body4." in e.text)
    path = deepest.meta["section_path"]
    assert path.count(" > ") <= 3       # capped depth, not all 5 levels
    assert path.startswith("# L1")      # document title always kept
    assert path.endswith("##### L5")    # innermost heading always kept


def test_disambiguates_repeated_template_sections():
    """Regression test for the core bug this whole fix addresses: a document
    that repeats near-identical boilerplate under many different topic
    headings (the real pattern in our own SAP corpus) must never produce two
    chunks that are indistinguishable from each other."""
    from app.ingest.pipeline.blocks import split_blocks

    topics = ["Pricing", "Availability", "Credit Management", "Output Determination"]
    sections = [
        f"## {t}\n\n### Common Failure Pattern\n\n"
        f"A typical failure related to {t} begins with a symptom."
        for t in topics
    ]
    text = "# SD Guide\n\n" + "\n\n".join(sections)
    els = split_blocks(text, text_extractor="markdown", table_extractor="markdown_table",
                        text_reason="prose", table_reason="structured_table")
    recs = chunk_elements(els, SPEC)
    failure_chunks = [r.text for r in recs if "Common Failure Pattern" in r.text]
    assert len(failure_chunks) == len(topics)
    assert len(set(failure_chunks)) == len(topics)  # every one pairwise distinct


def test_tab_separated_table_becomes_table_element():
    from app.ingest.pipeline.blocks import split_blocks

    text = "## Metrics\n\nRegion\tRevenue\tGrowth\nAPAC\t120\t18%\nEMEA\t90\t9%\n"
    els = split_blocks(text, text_extractor="markdown", table_extractor="markdown_table",
                        text_reason="prose", table_reason="structured_table")
    tbl = next((e for e in els if e.modality == Modality.TABLE.value), None)
    assert tbl is not None
    assert "| Region | Revenue | Growth |" in tbl.text
    assert "APAC" in tbl.text and "120" in tbl.text
    assert tbl.meta.get("section") == "## Metrics"


def test_auto_spec_reproduces_minilm_defaults_exactly():
    """The whole refactor's safety invariant: at the MiniLM limit (256), the
    derived spec must equal the hand-tuned defaults, so production chunking with
    MiniLM is byte-identical (no silent re-chunk of an existing corpus)."""
    assert ChunkSpec.auto(256) == ChunkSpec()
    assert ChunkSpec.auto(256) == ChunkSpec(
        target_tokens=180, overlap_tokens=20, max_tokens=220, min_tokens=16)


def test_auto_spec_scales_up_and_reserves_headroom():
    """A bigger embedder limit yields bigger chunks that still leave headroom
    under the model's hard limit (for the section-path prefix + special tokens)."""
    big = ChunkSpec.auto(512)
    small = ChunkSpec.auto(256)
    assert big.max_tokens < 512                    # headroom reserved under the limit
    assert big.max_tokens > small.max_tokens       # bigger model -> bigger chunks
    assert big.target_tokens > small.target_tokens
    assert 0 < big.overlap_tokens < big.target_tokens < big.max_tokens


def test_auto_spec_degenerate_tiny_limit_is_safe():
    """A pathologically small limit must not produce a spec with target/max at or
    below min (which would loop or emit empty chunks)."""
    s = ChunkSpec.auto(8, min_tokens=16)
    assert s.max_tokens > s.min_tokens
    assert s.target_tokens >= s.min_tokens
    assert s.overlap_tokens >= 1


def test_chunker_sizes_against_injected_counter():
    """The core of the fix: chunking must measure against the counter it's given
    (the active embedder's tokenizer), not a hardcoded one. With a word-counter
    and a small spec, every chunk must respect the spec AS MEASURED BY THAT
    COUNTER, and token_count must be that counter's value."""
    para = " ".join(f"word{i} more{i} filler{i}." for i in range(40))
    spec = ChunkSpec(target_tokens=10, overlap_tokens=2, max_tokens=15, min_tokens=2)
    recs = chunk_elements([_text_el(para)], spec, count=_word_count, embed_max=15)
    assert recs
    assert all(_word_count(r.text) <= spec.max_tokens for r in recs)
    assert all(r.token_count == _word_count(r.text) for r in recs)


def test_injected_counter_changes_chunk_boundaries():
    """Sanity that the counter actually drives packing: the same text under a
    coarser counter (fewer tokens per word) packs into fewer chunks than under
    the finer default WordPiece counter, for the same spec."""
    para = ("Alpha beta gamma delta epsilon zeta eta theta iota kappa. " * 6).strip()
    doc = "\n\n".join([para, para, para])
    spec = ChunkSpec(target_tokens=60, overlap_tokens=10, max_tokens=90, min_tokens=8)
    n_default = len(chunk_elements([_text_el(doc)], spec))                     # MiniLM WordPiece
    n_words = len(chunk_elements([_text_el(doc)], spec, count=_word_count, embed_max=90))
    assert n_words <= n_default   # coarser ruler -> more text fits per chunk -> fewer chunks


def test_over_limit_warning_uses_injected_embed_max(caplog):
    """The over-limit safety warning must fire against the embedder's OWN limit,
    not the MiniLM constant -- a single unsplittable token stream longer than
    embed_max should warn with that embed_max."""
    import logging
    long_line = "supercalifragilistic " * 30  # one long line, no sentence breaks
    spec = ChunkSpec(target_tokens=1000, overlap_tokens=0, max_tokens=1000, min_tokens=1)
    with caplog.at_level(logging.WARNING, logger="pipeline.chunker"):
        chunk_elements([_text_el(long_line)], spec, count=_word_count, embed_max=5)
    assert any(getattr(r, "embed_max_tokens", None) == 5 for r in caplog.records)


def test_minilm_embedder_token_contract():
    from app.shared.adapters.embedders.minilm import MiniLMEmbedder
    e = MiniLMEmbedder()
    assert e.max_tokens == 256
    assert e.count_tokens("") == 0
    assert e.count_tokens("hello world") > 0


def test_onnx_embedder_reports_its_own_limit_without_loading():
    """max_tokens must be known without downloading the model (container startup
    and chunk sizing can't afford a model load just to learn the limit)."""
    from app.shared.adapters.embedders.onnx_embedder import OnnxEmbedder
    e = OnnxEmbedder("Xenova/bge-base-en-v1.5", 768, max_length=512)
    assert e.max_tokens == 512


def test_count_tokens_for_default_repo_matches_default_counter():
    from app.ingest.pipeline.tokens import _DEFAULT_REPO, count_tokens, count_tokens_for
    assert count_tokens_for(_DEFAULT_REPO, "quarterly revenue report") == \
        count_tokens("quarterly revenue report")
