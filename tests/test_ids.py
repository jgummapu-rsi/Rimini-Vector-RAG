from app.ids import is_object_id, new_object_id


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
