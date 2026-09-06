"""Prompt constants for retrieval-time query decomposition."""

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
