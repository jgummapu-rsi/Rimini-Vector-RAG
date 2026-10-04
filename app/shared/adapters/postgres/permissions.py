from __future__ import annotations

from psycopg2 import sql

RUNTIME_TABLES = (
    "tenants",
    "users",
    "documents",
    "document_versions",
    "ingestion_jobs",
    "extraction_artifacts",
    "job_events",
    "chunks",
    "connectors",
    "audit_log",
    "metrics",
    "system_config",
    "corpus_epochs",
    "deleted_documents",
    "vector_chunks",
)


def apply_runtime_grants(cursor, schema: str, role: str) -> None:
    cursor.execute(
        "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname=%s AND c.relkind IN ('r','p','v','m','f')",
        (schema,),
    )
    for row in cursor.fetchall():
        name = row["relname"] if isinstance(row, dict) else row[0]
        cursor.execute(
            sql.SQL("REVOKE ALL ON {}.{} FROM {}").format(
                sql.Identifier(schema), sql.Identifier(name), sql.Identifier(role)
            )
        )
    cursor.execute(
        sql.SQL("REVOKE ALL ON SCHEMA {} FROM {}").format(
            sql.Identifier(schema), sql.Identifier(role)
        )
    )
    cursor.execute(
        sql.SQL("GRANT USAGE ON SCHEMA {} TO {}").format(
            sql.Identifier(schema), sql.Identifier(role)
        )
    )
    for table in RUNTIME_TABLES:
        cursor.execute(
            sql.SQL("GRANT SELECT,INSERT,UPDATE,DELETE ON {}.{} TO {}").format(
                sql.Identifier(schema), sql.Identifier(table), sql.Identifier(role)
            )
        )
    for table in ("embedding_profile", "ingestion_generations", "retrieval_vectors"):
        cursor.execute(
            sql.SQL("GRANT SELECT ON {}.{} TO {}").format(
                sql.Identifier(schema), sql.Identifier(table), sql.Identifier(role)
            )
        )
    cursor.execute(
        sql.SQL("GRANT INSERT ON {}.ingestion_generations TO {}").format(
            sql.Identifier(schema), sql.Identifier(role)
        )
    )
    cursor.execute(
        sql.SQL("GRANT UPDATE(singleton) ON {}.embedding_profile TO {}").format(
            sql.Identifier(schema), sql.Identifier(role)
        )
    )
    cursor.execute(
        sql.SQL("GRANT USAGE,SELECT ON ALL SEQUENCES IN SCHEMA {} TO {}").format(
            sql.Identifier(schema), sql.Identifier(role)
        )
    )


def validate_runtime_role(cursor, schema: str, role: str) -> None:
    cursor.execute(
        "SELECT rolsuper,rolcreatedb,rolcreaterole,rolbypassrls FROM pg_roles WHERE rolname=%s",
        (role,),
    )
    row = cursor.fetchone()
    if row is None or any(row.values() if isinstance(row, dict) else row):
        raise ValueError("Runtime role must exist without administrative privileges")
    cursor.execute(
        "SELECT 1 FROM pg_auth_members WHERE member=(SELECT oid FROM pg_roles WHERE rolname=%s) LIMIT 1",
        (role,),
    )
    if cursor.fetchone() is not None:
        raise ValueError("Runtime role must be dedicated without inherited role membership")
    cursor.execute(
        "SELECT 1 FROM pg_namespace WHERE nspname=%s AND nspowner=(SELECT oid FROM pg_roles WHERE rolname=%s)",
        (schema, role),
    )
    if cursor.fetchone() is not None:
        raise ValueError("Runtime role must not own the application schema")
    cursor.execute(
        "SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname=%s AND c.relowner=(SELECT oid FROM pg_roles WHERE rolname=%s) "
        "UNION ALL SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "WHERE n.nspname=%s AND p.proowner=(SELECT oid FROM pg_roles WHERE rolname=%s) LIMIT 1",
        (schema, role, schema, role),
    )
    if cursor.fetchone() is not None:
        raise ValueError("Runtime role must not own application objects")
