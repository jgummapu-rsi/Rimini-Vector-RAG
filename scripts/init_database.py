from __future__ import annotations

import argparse
import logging

import psycopg2

from app.shared.config import settings
from app.shared.container import build_container
from app.shared.observability import configure_logging

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Provision pgvector and optionally initialize application schema."
    )
    parser.add_argument(
        "--schema",
        action="store_true",
        help="Also initialize application tables and embedding profile, and prepare models",
    )
    args = parser.parse_args()
    configure_logging()
    with psycopg2.connect(settings.postgres_dsn) as connection:
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA public")
            cursor.execute("SELECT extversion FROM pg_extension WHERE extname='vector'")
            version = cursor.fetchone()[0]
            if tuple(map(int, version.split("."))) < (0, 8, 0):
                raise RuntimeError("pgvector >= 0.8.0 is required")
    log.info(
        "Database vector extension provisioned",
        extra={"event": "database_initialized", "extension_version": version},
    )
    if args.schema:
        container = build_container(settings.model_copy(update={"initialize_schema": True}))
        container.close()


if __name__ == "__main__":
    main()
