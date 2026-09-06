"""Document metadata extraction: best-effort, never raises."""
from app.shared.gateway.client import GatewayError
from app.ingest.pipeline.metadata_extract import TEXT_BUDGET_TOKENS, extract_metadata


class _StubGateway:
    def __init__(self, response=None, raises=None):
        self._response = response
        self._raises = raises
        self.calls = []

    def chat(self, messages, model, temperature=0.0):
        self.calls.append({"messages": messages, "model": model, "temperature": temperature})
        if self._raises is not None:
            raise self._raises
        return self._response


def test_valid_json_response_is_parsed():
    gw = _StubGateway(response=(
        '{"author": "Priya", "date": "2024-03-31", '
        '"topics": ["period close", "FI sub-ledger"], "entities": ["GR/IR", "SAP FI"]}'
    ))
    result = extract_metadata(gw, "gpt-5-nano", "GR/IR clearing entries must post by period close.")
    assert result == {
        "author": "Priya", "date": "2024-03-31",
        "topics": ["period close", "FI sub-ledger"], "entities": ["GR/IR", "SAP FI"],
    }
    assert gw.calls[0]["model"] == "gpt-5-nano"
    assert gw.calls[0]["temperature"] == 0


def test_response_wrapped_in_code_fence_is_parsed():
    gw = _StubGateway(response='```json\n{"author": null, "date": null, "topics": [], "entities": []}\n```')
    result = extract_metadata(gw, "gpt-5-nano", "some text")
    assert result == {"author": None, "date": None, "topics": [], "entities": []}


def test_malformed_json_degrades_to_blank():
    gw = _StubGateway(response="not json at all")
    result = extract_metadata(gw, "gpt-5-nano", "some text")
    assert result == {"author": None, "date": None, "topics": [], "entities": []}


def test_wrong_shape_json_degrades_to_blank():
    gw = _StubGateway(response='{"topics": "not-a-list", "entities": []}')
    result = extract_metadata(gw, "gpt-5-nano", "some text")
    assert result == {"author": None, "date": None, "topics": [], "entities": []}


def test_gateway_error_degrades_to_blank():
    gw = _StubGateway(raises=GatewayError("503: service unavailable"))
    result = extract_metadata(gw, "gpt-5-nano", "some text")
    assert result == {"author": None, "date": None, "topics": [], "entities": []}


def test_empty_text_short_circuits_without_calling_gateway():
    gw = _StubGateway(response="should never be reached")
    result = extract_metadata(gw, "gpt-5-nano", "   ")
    assert result == {"author": None, "date": None, "topics": [], "entities": []}
    assert gw.calls == []


def test_oversized_input_is_truncated_before_calling_chat():
    gw = _StubGateway(response='{"author": null, "date": null, "topics": [], "entities": []}')
    huge_text = "word " * 20000   # far more than TEXT_BUDGET_TOKENS
    extract_metadata(gw, "gpt-5-nano", huge_text)
    sent = gw.calls[0]["messages"][-1]["content"]
    assert len(sent) < len(huge_text)
