from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import make_dsn, parse_dsn
from psycopg2.extras import Json, execute_values

from app.ingest.pipeline.runner import run_job
from app.retrieval.rag.access import access_predicate
from app.shared.adapters.postgres.db import close_pool, transaction
from app.shared.container import build_container
from app.shared.domain.models import Document, DocumentVersion, Job, Principal, Role
from app.shared.ids import new_object_id
from scripts.disposable_storage import disposable_settings, evaluation_settings
from scripts.encrypted_backup import decrypt, encrypt

log = logging.getLogger(__name__)


def _database_command(program: str, dsn: str, *arguments: str) -> None:
    parameters = parse_dsn(dsn)
    variables = {
        "host": "PGHOST",
        "port": "PGPORT",
        "user": "PGUSER",
        "password": "PGPASSWORD",
        "dbname": "PGDATABASE",
        "sslmode": "PGSSLMODE",
        "options": "PGOPTIONS",
    }
    environment = {key: value for key, value in os.environ.items() if not key.startswith("PG")}
    for name, value in parameters.items():
        if name in variables:
            environment[variables[name]] = value
    environment["PGCONNECT_TIMEOUT"] = "5"
    container = os.environ.get("RESTORE_DRILL_TOOLS_CONTAINER")
    if container:
        command = ["docker", "exec", "-i"]
        environment["PGHOST"] = "127.0.0.1"
        environment["PGPORT"] = "5432"
        for key in variables.values():
            if key in environment:
                command.extend(["--env", key])
        command.extend([container, program])
        if program == "pg_dump":
            destination = next(
                argument[7:] for argument in arguments if argument.startswith("--file=")
            )
            command.extend(argument for argument in arguments if not argument.startswith("--file="))
            with open(destination, "wb") as output:
                result = subprocess.run(
                    command, env=environment, stdout=output, stderr=subprocess.PIPE, timeout=120
                )
        else:
            command.extend(arguments[:-1])
            with open(arguments[-1], "rb") as archive:
                result = subprocess.run(
                    command, env=environment, stdin=archive, capture_output=True, timeout=120
                )
    else:
        result = subprocess.run(
            [program, *arguments], env=environment, capture_output=True, timeout=120
        )
    if result.returncode:
        error = result.stderr.decode(errors="replace")
        for name in ("password", "user", "host"):
            if parameters.get(name):
                error = error.replace(parameters[name], "[redacted]")
        raise RuntimeError(f"{program} failed with exit code {result.returncode}: {error[:2000]}")


def exercise_restore(
    source_settings, target_settings, directory: Path, scale_documents: int = 0
) -> dict:
    started = time.monotonic()
    source = build_container(source_settings.model_copy(update={"reranker_provider": "none"}))
    source.gateway.chat = lambda *args, **kwargs: (
        '{"author":null,"date":null,"topics":[],"entities":[]}'
    )
    tenant_id = source.metadata.create_tenant("restore-drill")
    owner = source.metadata.create_user(tenant_id, "owner@drill.test", "admin", "drill-owner-token")
    reader = source.metadata.create_user(
        tenant_id, "reader@drill.test", "member", "drill-reader-token"
    )
    document_id = new_object_id()
    for version, amount in ((1, "007.00"), (2, "009.00")):
        content = f"Invoice 00123 amount {amount} USD. Payment requires approval.".encode()
        digest = hashlib.sha256(content).hexdigest()
        blob = source.blob.put(tenant_id, digest, ".txt", content)
        job = Job(new_object_id(), document_id, tenant_id, "parse", "queued", 0)
        if version == 1:
            document = Document(
                document_id,
                tenant_id,
                owner,
                "text",
                blob,
                digest,
                "text/plain",
                "invoice.txt",
                "private",
                [],
            )
            history = DocumentVersion(
                new_object_id(),
                document_id,
                tenant_id,
                1,
                digest,
                blob,
                "invoice.txt",
                len(content),
                owner,
                job.id,
            )
            source.metadata.create_document_with_job(document, job, history)
        else:
            source.metadata.update_document_content_and_queue(
                tenant_id,
                document_id,
                blob_path=blob,
                content_sha256=digest,
                mime="text/plain",
                source_type="text",
                filename="invoice.txt",
                visibility="private",
                scope="tenant",
                job=job,
                uploaded_by=owner,
                byte_size=len(content),
                expected_version=1,
            )
        run_job(source, source.queue.claim_next())
    active = source.metadata.get_document(tenant_id, document_id)
    profile_id = source.embedder.profile.id
    scale_bytes = 0
    if scale_documents:
        records = []
        for index in range(scale_documents):
            content = (
                f"Scale fixture {index:08d}\n"
                + "Evidence preservation and restore checksum.\n" * 1600
            ).encode()
            digest = hashlib.sha256(content).hexdigest()
            blob = source.blob.put(tenant_id, digest, ".txt", content)
            scale_bytes += len(content)
            records.append(
                (
                    new_object_id(),
                    tenant_id,
                    owner,
                    "text",
                    blob,
                    digest,
                    "text/plain",
                    f"scale-{index}.txt",
                )
            )
        with transaction(source_settings.postgres_dsn) as cursor:
            execute_values(
                cursor,
                "INSERT INTO documents(id,tenant_id,owner_user_id,source_type,blob_path,content_sha256,mime,filename) VALUES %s",
                records,
            )
            history = [
                (new_object_id(), row[0], tenant_id, 1, row[5], row[4], row[7], len(content), owner)
                for row in records
            ]
            execute_values(
                cursor,
                "INSERT INTO document_versions(id,document_id,tenant_id,version,content_sha256,blob_path,filename,byte_size,uploaded_by) VALUES %s",
                history,
            )
            cursor.execute("SELECT embedding::text FROM vector_chunks LIMIT 1")
            seed_vector = cursor.fetchone()["embedding"]
            vectors = [
                (
                    f"{row[0]}-{ordinal}",
                    tenant_id,
                    row[0],
                    seed_vector,
                    Json(
                        {
                            "_id": row[0],
                            "user_id": owner,
                            "visibility": "private",
                            "content": f"Scale fixture {index} passage {ordinal}",
                        }
                    ),
                )
                for index, row in enumerate(records)
                for ordinal in range(20)
            ]
            execute_values(
                cursor,
                "INSERT INTO vector_chunks(chunk_id,tenant_id,document_id,embedding,payload) VALUES %s",
                vectors,
                page_size=500,
            )
    with transaction(source_settings.postgres_dsn) as cursor:
        cursor.execute("SELECT count(*) AS n FROM vector_chunks")
        expected_vectors = cursor.fetchone()["n"]
        cursor.execute("SELECT pg_database_size(current_database()) AS bytes")
        database_bytes = cursor.fetchone()["bytes"]
    archive = directory / "metadata.dump"
    backup_blobs = directory / "blobs"
    backup_blobs.mkdir()
    manifest = []
    with psycopg2.connect(source_settings.postgres_dsn) as connection:
        connection.set_session(isolation_level="REPEATABLE READ")
        with connection.cursor() as cursor:
            cursor.execute("SELECT current_schema()")
            schema = cursor.fetchone()[0]
            cursor.execute(
                "LOCK TABLE documents,document_versions,ingestion_jobs,ingestion_generations,vector_chunks IN SHARE MODE"
            )
            cursor.execute("SELECT pg_export_snapshot()")
            snapshot = cursor.fetchone()[0]
            _database_command(
                "pg_dump",
                source_settings.postgres_dsn,
                "--format=custom",
                "--no-owner",
                "--no-acl",
                f"--schema={schema}",
                f"--snapshot={snapshot}",
                f"--file={archive}",
            )
            cursor.execute("SELECT DISTINCT blob_path,content_sha256 FROM document_versions")
            for index, (path, digest) in enumerate(cursor.fetchall()):
                data = source.blob.get(path)
                if hashlib.sha256(data).hexdigest() != digest:
                    raise ValueError("Backup blob does not match metadata content identity")
                backup = backup_blobs / str(index)
                backup.write_bytes(data)
                manifest.append(
                    {"original": path, "backup": backup.name, "sha256": digest, "bytes": len(data)}
                )
    (directory / "manifest.json").write_text(
        json.dumps({"profile_id": profile_id, "blobs": manifest}, indent=2)
    )
    encryption_key = directory / "restore-test.key"
    encryption_key.write_bytes(os.urandom(32))
    encryption_key.chmod(0o600)
    encrypted_archive = directory / "metadata.dump.enc"
    encrypted = encrypt(archive, encrypted_archive, encryption_key)
    archive.unlink()
    decrypted = decrypt(encrypted_archive, archive, encryption_key)
    if decrypted != encrypted:
        raise ValueError("Encrypted dump restore integrity mismatch")
    backup_ms = (time.monotonic() - started) * 1000
    restored_at = time.monotonic()
    _database_command(
        "pg_restore",
        target_settings.postgres_dsn,
        "--no-owner",
        "--no-acl",
        "--exit-on-error",
        "--single-transaction",
        "--dbname",
        parse_dsn(target_settings.postgres_dsn)["dbname"],
        str(archive),
    )
    restored_dsn = make_dsn(target_settings.postgres_dsn, options=f"-csearch_path={schema},public")
    restored_settings = target_settings.model_copy(
        update={"database_url": restored_dsn, "reranker_provider": "none"}
    )
    restored = build_container(restored_settings)
    try:
        with transaction(restored_dsn) as cursor:
            cursor.execute("ALTER TABLE ingestion_generations DISABLE TRIGGER generation_immutable")
            for entry in manifest:
                data = (backup_blobs / entry["backup"]).read_bytes()
                path = restored.blob.put(tenant_id, entry["sha256"], ".txt", data)
                for table in (
                    "documents",
                    "document_versions",
                    "ingestion_jobs",
                    "ingestion_generations",
                ):
                    cursor.execute(
                        sql.SQL("UPDATE {} SET blob_path=%s WHERE blob_path=%s").format(
                            sql.Identifier(table)
                        ),
                        (path, entry["original"]),
                    )
            cursor.execute("ALTER TABLE ingestion_generations ENABLE TRIGGER generation_immutable")
        source.close()
        shutil.rmtree(source_settings.blob_dir)
        recovered = restored.metadata.get_document(tenant_id, document_id)
        assert recovered.active_generation_id == active.active_generation_id
        assert recovered.indexed_version == 2
        assert restored.embedder.profile.id == profile_id
        assert restored.metadata.get_principal_by_token("drill-owner-token").user_id == owner
        histories = restored.metadata.list_document_versions(tenant_id, document_id)
        assert len(histories) == 2
        for history in histories:
            assert (
                hashlib.sha256(restored.blob.get(history.blob_path)).hexdigest()
                == history.content_sha256
            )
        vector = restored.embedder.embed_query("Invoice amount")
        owner_hits = restored.vectors.search(
            tenant_id,
            vector,
            top_k=10,
            access=access_predicate(Principal(tenant_id, owner, Role.ADMIN)),
            query_text="Invoice 00123",
        )
        assert any("009.00" in hit.payload["content"] for hit in owner_hits)
        assert all("007.00" not in hit.payload["content"] for hit in owner_hits)
        assert (
            restored.vectors.search(
                tenant_id,
                vector,
                access=access_predicate(Principal(tenant_id, reader, Role.MEMBER)),
            )
            == []
        )
        assert restored.queue.claim_next() is None
        with transaction(restored_dsn) as cursor:
            cursor.execute(
                "SELECT count(*) AS n FROM ingestion_generations WHERE document_id=%s",
                (document_id,),
            )
            assert cursor.fetchone()["n"] == 2
            cursor.execute("SELECT count(*) AS n FROM vector_chunks")
            assert cursor.fetchone()["n"] == expected_vectors
            cursor.execute("SELECT count(*) AS n FROM documents")
            assert cursor.fetchone()["n"] == scale_documents + 1
        for entry in manifest:
            restored_path = restored.settings.blob_dir / tenant_id / (entry["sha256"] + ".txt")
            assert (
                hashlib.sha256(restored.blob.get(str(restored_path))).hexdigest() == entry["sha256"]
            )
        return {
            "documents": scale_documents + 1,
            "versions": len(histories) + scale_documents,
            "generations": 2,
            "blobs_verified": len(manifest),
            "vector_rows_verified": expected_vectors,
            "database_bytes": database_bytes,
            "scale_blob_bytes": scale_bytes,
            "dump_bytes": archive.stat().st_size,
            "encrypted_dump_verified": True,
            "active_hits": len(owner_hits),
            "unauthorized_hits": 0,
            "queued_jobs": 0,
            "profile_id": profile_id,
            "backup_ms": round(backup_ms, 2),
            "restore_verify_ms": round((time.monotonic() - restored_at) * 1000, 2),
        }
    finally:
        restored.close()
        source.close()
        close_pool(restored_dsn)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale-documents", type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.scale_documents <= 100000:
        parser.error("scale-documents must be between 0 and 100000")
    with evaluation_settings() as source:
        with disposable_settings(
            os.environ["EVAL_DATABASE_URL"], os.environ["EVAL_REDIS_URL"]
        ) as target:
            with TemporaryDirectory(prefix="restore-drill-") as directory:
                result = exercise_restore(source, target, Path(directory), args.scale_documents)
                print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
