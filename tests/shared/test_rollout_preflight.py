import os
from datetime import UTC, datetime, timedelta

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID

from scripts.production_preflight import validate_plan


def test_rollout_preflight_never_invents_missing_production_targets():
    errors = validate_plan({})
    assert "Missing backup_key_custodian" in errors
    assert "Missing origin" in errors
    assert any("rto_seconds" in error for error in errors)


def test_local_backup_path_is_not_offhost():
    errors = validate_plan({"backup_destination": "/tmp/backup", "origin": "http://localhost"})
    assert "backup_destination must identify off-host storage" in errors
    assert "origin must be an HTTPS application origin" in errors


def test_certificate_hostname_validity_and_immutable_rollback(tmp_path):

    private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    key = tmp_path / "tls.key"
    key.write_bytes(
        private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
    )
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "rag.example.test")])
    certificate = (
        x509.CertificateBuilder()
        .subject_name(name)
        .issuer_name(name)
        .public_key(private.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.now(UTC) - timedelta(minutes=1))
        .not_valid_after(datetime.now(UTC) + timedelta(days=1))
        .add_extension(
            x509.SubjectAlternativeName([x509.DNSName("rag.example.test")]), critical=False
        )
        .sign(private, hashes.SHA256())
    )
    cert = tmp_path / "tls.pem"
    cert.write_bytes(certificate.public_bytes(serialization.Encoding.PEM))
    backup_key = tmp_path / "backup.key"
    backup_key.write_bytes(os.urandom(32))
    backup_key.chmod(0o600)
    secret = tmp_path / "secret"
    secret.write_text("secret reference")
    plan = {
        "origin": "https://rag.example.test",
        "tls_certificate": str(cert),
        "tls_key": str(key),
        "database_secret": str(secret),
        "redis_secret": str(secret),
        "backup_destination": "s3://approved-bucket/prefix",
        "backup_key_file": str(backup_key),
        "backup_key_custodian": "platform-team",
        "rollback_image": "registry.example.test/rag@sha256:" + "a" * 64,
        "corpus_documents": 100,
        "corpus_vectors": 1000,
        "concurrency": 8,
        "p95_latency_ms": 500,
        "rto_seconds": 600,
        "rpo_seconds": 60,
        "backup_retention_days": 30,
    }
    assert validate_plan(plan) == []
    assert "TLS certificate does not cover the application hostname" in validate_plan(
        dict(plan, origin="https://wrong.example.test")
    )
    assert any(
        "immutable" in error for error in validate_plan(dict(plan, rollback_image="rag:latest"))
    )
    assert any("positive" in error for error in validate_plan(dict(plan, rto_seconds=float("nan"))))
