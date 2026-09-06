import pytest

from app.shared.ids import chunk_id, is_object_id, new_object_id


def test_object_id_is_24_hex():
    oid = new_object_id()
    assert len(oid) == 24
    int(oid, 16)
    assert is_object_id(oid)


def test_object_ids_are_unique_and_monotonic_counter():
    ids = {new_object_id() for _ in range(5000)}
    assert len(ids) == 5000


def test_is_object_id_rejects_bad_values():
    assert not is_object_id("nope")
    assert not is_object_id("zz" * 12)
    assert not is_object_id("abc")
    assert not is_object_id(12345)


@pytest.mark.parametrize("value", [
    "+" + "a" * 23,          # int(v, 16) accepts a leading sign
    "-" + "a" * 23,
    "ffff_fffffffffffffffff_f",  # int(v, 16) accepts underscore separators; 24 chars total
    "  " + "a" * 20 + "  ",  # int(v, 16) strips surrounding whitespace
    "0x" + "a" * 22,
])
def test_is_object_id_rejects_things_int_base16_would_have_accepted(value):
    """The old check was `len == 24 and int(value, 16)`, which let through
    signs, digit separators and padded whitespace -- none of which are valid
    ObjectIds."""
    assert len(value) == 24, "these are all exactly 24 chars, so only the hex test rejects them"
    assert not is_object_id(value)


# ------------------------------------------------------------- chunk ids --


def test_chunk_id_is_document_id_plus_padded_ordinal():
    assert chunk_id("DOC", 0, 2) == "DOC001"
    assert chunk_id("DOC", 1, 2) == "DOC002"


def test_chunk_ids_stay_sortable_past_999():
    """Zero-padding width scales with the count, so lexicographic order still
    matches ordinal order once a document exceeds 999 chunks."""
    ids = [chunk_id("DOC", i, 1000) for i in range(1000)]
    assert ids == sorted(ids)
    assert len(set(ids)) == 1000
    assert ids[0] == "DOC0001" and ids[-1] == "DOC1000"


def test_chunk_id_is_deterministic():
    assert chunk_id("DOC", 5, 20) == chunk_id("DOC", 5, 20)
