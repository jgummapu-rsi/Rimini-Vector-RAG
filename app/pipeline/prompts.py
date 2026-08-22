"""LLM transcription prompt.

There is no dedicated OCR model on the gateway, so a vision LLM does the
transcription. The prompt is deliberately STRICT: transcribe only what is
literally visible, invent nothing. If the content is a table, emit markdown so
downstream chunking treats it as a table (the "LLM only returns markdown for an
image table" rule). It also asks for markdown heading syntax on genuinely
visually-distinct section titles, for the same reason: without this, a scanned
document gets NONE of the heading-path/ancestor-context chunking benefit
markdown-sourced documents get (app.pipeline.blocks's heading stack has nothing
to find), even though scanned documents are exactly the harder, higher-value
ingestion path this pipeline is built around. The instruction is deliberately
conservative -- mark only what's visually unambiguous, never invent a hierarchy
the image doesn't show -- consistent with the "don't guess" rule already in
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

QUERY_DECOMPOSITION_PROMPT = (
    "You decide whether answering a user's question well genuinely requires "
    "searching for more than one distinct, independent piece of information, or "
    "whether it's really one focused question -- even if it's long or detailed. "
    "A question about ONE topic, ONE process, or ONE object is NOT multi-part, "
    "no matter how many details it asks for about that one thing. "
    "Return STRICT JSON with EXACTLY these keys and nothing else:\n"
    '{"decompose": boolean, "sub_questions": string[]}\n'
    "If it's one focused question, return {\"decompose\": false, \"sub_questions\": []}.\n"
    "If it genuinely needs 2-4 SEPARATE, independently-searchable pieces of "
    "information -- it compares two things, asks how two distinct topics relate "
    "to each other, or is really multiple questions joined into one -- set "
    "\"decompose\" to true and list each as its own complete, standalone question "
    "in \"sub_questions\" (never more than 4). Each sub-question must keep the "
    "exact codes, names, and numbers from the original question -- do not "
    "paraphrase them away. Return ONLY the JSON object, no commentary, no code fences."
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
