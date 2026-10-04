from __future__ import annotations

import argparse
import logging

from app.shared.adapters.embedders.gateway import GatewayEmbedder
from app.shared.adapters.embedders.minilm import MiniLMEmbedder
from app.shared.adapters.pgvector.migration import activate_shadow, build_shadow
from app.shared.adapters.postgres.metadata_store import PostgresMetadataStore
from app.shared.config import settings
from app.shared.gateway.client import LiteLLMClient
from app.shared.observability import configure_logging

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("build", "activate"))
    parser.add_argument("--migration")
    parser.add_argument("--model")
    parser.add_argument("--dimensions", type=int, default=1536)
    parser.add_argument("--revision")
    parser.add_argument("--prepare-legacy", action="store_true")
    args = parser.parse_args()

    configure_logging()
    if args.action == "activate":
        if not args.migration:
            parser.error("--migration is required for activation")
        previous = activate_shadow(settings.postgres_dsn, args.migration)
        print(f"Activated {args.migration}; previous index retained as {previous}")
        return
    if args.model:
        if not args.revision:
            parser.error("--revision is required for gateway embeddings")
        client = LiteLLMClient(
            settings.litellm_base_url, settings.litellm_api_key, settings.vision_model, args.model
        )
        embedder = GatewayEmbedder(client, args.dimensions, revision=args.revision)
    else:
        embedder = MiniLMEmbedder()
    if args.prepare_legacy:
        PostgresMetadataStore(settings.postgres_dsn).init_schema()
    migration = build_shadow(settings.postgres_dsn, embedder)
    print(f"Migration ready: {migration}")


if __name__ == "__main__":
    main()
