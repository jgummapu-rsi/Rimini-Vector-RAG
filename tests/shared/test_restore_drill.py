import os

from eval.restore_drill import exercise_restore
from scripts.disposable_storage import disposable_settings


def test_database_and_blob_restore_preserves_evidence_acl_and_profile(storage_settings, tmp_path):
    with disposable_settings(
        os.environ["TEST_DATABASE_URL"], os.environ["TEST_REDIS_URL"]
    ) as target:
        result = exercise_restore(storage_settings, target, tmp_path)
    assert result["versions"] == result["generations"] == result["blobs_verified"] == 2
    assert result["documents"] == 1
    assert result["unauthorized_hits"] == result["queued_jobs"] == 0
    assert result["encrypted_dump_verified"] is True
