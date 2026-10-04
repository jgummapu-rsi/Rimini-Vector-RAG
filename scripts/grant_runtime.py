from __future__ import annotations

import argparse
import logging

import psycopg2
from psycopg2 import sql

from app.shared.adapters.postgres.permissions import apply_runtime_grants, validate_runtime_role
from app.shared.config import settings

log = logging.getLogger(__name__)


def grant_runtime(dsn: str, schema: str, role: str) -> None:
    with psycopg2.connect(dsn) as connection, connection.cursor() as cursor:
        validate_runtime_role(cursor, schema, role)
        apply_runtime_grants(cursor, schema, role)
        cursor.execute("SELECT has_schema_privilege(%s,%s,'CREATE')", (role, schema))
        if cursor.fetchone()[0]:
            raise ValueError("Runtime role still has schema CREATE through a public grant")
        cursor.execute(
            sql.SQL(
                "CREATE TABLE IF NOT EXISTS {}.runtime_database_roles (role_name TEXT PRIMARY KEY)"
            ).format(sql.Identifier(schema))
        )
        cursor.execute(
            sql.SQL(
                "INSERT INTO {}.runtime_database_roles VALUES (%s) ON CONFLICT DO NOTHING"
            ).format(sql.Identifier(schema)),
            (role,),
        )
    log.info(
        "Runtime database permissions granted",
        extra={"event": "runtime_permissions_granted", "schema": schema, "database_role": role},
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default="public")
    parser.add_argument("--role", required=True)
    args = parser.parse_args()
    grant_runtime(settings.postgres_dsn, args.schema, args.role)


if __name__ == "__main__":
    main()
