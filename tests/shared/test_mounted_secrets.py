import pytest

from app.shared.config import Settings


def test_mounted_secret_wins_over_environment_but_not_explicit_configuration(tmp_path, monkeypatch):
    secret = tmp_path / "database"
    secret.write_text("postgresql://mounted/db\n")
    monkeypatch.setenv("DATABASE_URL_FILE", str(secret))
    monkeypatch.setenv("DATABASE_URL", "postgresql://environment/db")
    assert Settings(_env_file=None).database_url == "postgresql://mounted/db"
    assert (
        Settings(_env_file=None, database_url="postgresql://explicit/db").database_url
        == "postgresql://explicit/db"
    )


def test_missing_or_empty_secret_fails_startup(tmp_path, monkeypatch):
    secret = tmp_path / "missing"
    monkeypatch.setenv("REDIS_URL_FILE", str(secret))
    with pytest.raises(FileNotFoundError):
        Settings(_env_file=None)
    secret.write_text("\n")
    with pytest.raises(ValueError, match="empty"):
        Settings(_env_file=None)
