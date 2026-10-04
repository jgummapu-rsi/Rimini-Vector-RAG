from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import math
import re
import ssl
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from cryptography import x509
from cryptography.x509.oid import ExtensionOID

log = logging.getLogger(__name__)


def validate_plan(plan: dict) -> list[str]:
    if not isinstance(plan, dict):
        return ["Rollout plan must be a JSON object"]
    errors = []
    required = (
        "origin",
        "tls_certificate",
        "tls_key",
        "database_secret",
        "redis_secret",
        "backup_destination",
        "backup_key_file",
        "backup_key_custodian",
        "rollback_image",
        "corpus_documents",
        "corpus_vectors",
        "concurrency",
        "p95_latency_ms",
        "rto_seconds",
        "rpo_seconds",
        "backup_retention_days",
    )
    for field in required:
        if field not in plan or plan[field] in (None, ""):
            errors.append(f"Missing {field}")
    origin = urlsplit(plan.get("origin", ""))
    if (
        origin.scheme != "https"
        or not origin.hostname
        or origin.username
        or origin.password
        or origin.query
        or origin.fragment
        or origin.path not in ("", "/")
    ):
        errors.append("origin must be an HTTPS application origin")
    destination = urlsplit(plan.get("backup_destination", ""))
    if destination.scheme not in {"sftp", "s3", "https", "az"} or not destination.netloc:
        errors.append("backup_destination must identify off-host storage")
    for field in (
        "corpus_documents",
        "corpus_vectors",
        "concurrency",
        "p95_latency_ms",
        "rto_seconds",
        "rpo_seconds",
        "backup_retention_days",
    ):
        if (
            type(plan.get(field)) not in (int, float)
            or not math.isfinite(plan[field])
            or plan[field] <= 0
        ):
            errors.append(f"{field} must be a positive approved target")
    for field in (
        "tls_certificate",
        "tls_key",
        "database_secret",
        "redis_secret",
        "backup_key_file",
    ):
        value = plan.get(field)
        if value and not Path(value).is_file():
            errors.append(f"{field} file is unavailable")
    if not errors:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        try:
            context.load_cert_chain(plan["tls_certificate"], plan["tls_key"])
        except (OSError, ssl.SSLError):
            errors.append("TLS certificate and key cannot be loaded as a pair")
        try:
            certificate = x509.load_pem_x509_certificate(Path(plan["tls_certificate"]).read_bytes())
            now = datetime.now(UTC)
            if not certificate.not_valid_before_utc <= now < certificate.not_valid_after_utc:
                errors.append("TLS certificate is not currently valid")
            names = certificate.extensions.get_extension_for_oid(
                ExtensionOID.SUBJECT_ALTERNATIVE_NAME
            ).value
            hostname = origin.hostname.lower()
            try:
                address = ipaddress.ip_address(hostname)
                matches = address in names.get_values_for_type(x509.IPAddress)
            except ValueError:
                matches = any(
                    hostname == name.lower()
                    or (
                        name.startswith("*.")
                        and hostname.count(".") == name.count(".")
                        and hostname.endswith(name[1:].lower())
                    )
                    for name in names.get_values_for_type(x509.DNSName)
                )
            if not matches:
                errors.append("TLS certificate does not cover the application hostname")
        except (ValueError, x509.ExtensionNotFound):
            errors.append("TLS certificate requires a valid subject alternative name")
        key = Path(plan["backup_key_file"])
        if key.stat().st_mode & 0o077 or key.stat().st_size != 32:
            errors.append("Backup key must be 32 bytes with owner-only permissions")
    if plan.get("rollback_image") and not re.fullmatch(
        r".+@sha256:[a-f0-9]{64}", plan["rollback_image"]
    ):
        errors.append("rollback_image must be an immutable digest reference")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    args = parser.parse_args()
    errors = validate_plan(json.loads(args.plan.read_text()))
    print(json.dumps({"ready": not errors, "blockers": errors}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
