"""Prompt constants for ingest-time OCR/transcription and metadata extraction.

There is no dedicated OCR model on the gateway, so a vision LLM does the
transcription. `STRICT_TRANSCRIBE_PROMPT` is deliberately strict: transcribe
only what is literally visible, invent nothing. If the content is a table, it
asks for markdown so downstream chunking treats it as a table, and asks for
markdown heading syntax on genuinely visually-distinct section titles so a
scanned document still gets the heading-path/ancestor-context chunking benefit
markdown-sourced documents get (app.ingest.pipeline.blocks's heading stack has
nothing to find otherwise) -- consistent with the "don't guess" rule already in
place for illegible text.
"""

STRICT_TRANSCRIBE_PROMPT = (
    "You are a strict transcription engine, not an assistant. Transcribe ONLY the "
    "text that is actually visible in this image, exactly as written, preserving "
    "reading order. Do NOT summarize, translate, rephrase, explain, describe, or "
    "guess. Never add words, numbers, or punctuation that are not clearly present. "
    "If any part of the image is a table -- not only when the whole image is one -- "
    "output that part as a GitHub-flavored markdown table containing exactly the "
    "visible cell values, in the surrounding reading order. "
    "If a line is CLEARLY a section title or heading -- visually distinct from body "
    "text through larger size, bold weight, underline, or standing alone as a label "
    "for what follows -- prefix that exact line with markdown heading syntax ('#' "
    "for a top-level title, '##' for a subsection, '###' for a smaller heading "
    "below that), matching only its visual prominence. Never invent a hierarchy the "
    "image doesn't visually show, and if a line's role is unclear, leave it as plain "
    "text rather than guess. "
    "If a region is unreadable, write [illegible]. If there is no readable text, "
    "output nothing. Return only the transcription, with no commentary."
)

METADATA_EXTRACTION_PROMPT = (
    "You extract document metadata, not summaries. Read the document text below and "
    "extract ONLY what is explicitly stated or unambiguously evident in the text. "
    "Return STRICT JSON with EXACTLY these keys and nothing else:\n"
    '{"author": string or null, "date": string or null, "topics": string[], "entities": string[]}\n'
    "- author: the person or team who wrote/owns the document, if stated. Otherwise null.\n"
    "- date: the document's own date (not a date merely mentioned in passing), as "
    "YYYY-MM-DD if determinable, else the text as written, else null.\n"
    "- topics: up to 8 short topic phrases (2-4 words each) that the document is about.\n"
    "- entities: up to 15 named things mentioned (systems, products, standards, "
    "organizations, key terms) — proper nouns and specific identifiers, not generic words.\n"
    "If a field cannot be determined, use null (author/date) or [] (topics/entities) — "
    "never guess or invent a value. Return ONLY the JSON object, no commentary, no code fences."
)
