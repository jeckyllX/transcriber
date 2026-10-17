"""Pytest test configuration and isolation fixtures."""

import pytest
from pathlib import Path
from storage import JobStore
import storage
import jobs
import config


@pytest.fixture(autouse=True)
def isolate_test_job_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Ensure every test runs with an isolated temporary SQLite store and config file.

    Patches the global singleton job_store, job_manager, and CONFIG_JSON_PATH to prevent test fixtures
    and mocks from polluting the workspace database and config.json.
    """
    test_db = tmp_path / "test_transcriptions.db"
    test_store = JobStore(db_path=test_db, enabled=True)

    monkeypatch.setattr(storage, "job_store", test_store)
    monkeypatch.setattr(jobs.job_manager, "_store", test_store)

    test_config = tmp_path / "test_config.json"
    if config.CONFIG_JSON_PATH.is_file():
        test_config.write_text(config.CONFIG_JSON_PATH.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_JSON_PATH", test_config)

    jobs.job_manager._jobs.clear()
    jobs.job_manager._batches.clear()

    yield test_store
