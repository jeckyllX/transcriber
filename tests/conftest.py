"""Pytest test configuration and isolation fixtures."""

import pytest
from pathlib import Path
from storage import JobStore
import storage
import jobs


@pytest.fixture(autouse=True)
def isolate_test_job_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Ensure every test runs with an isolated, temporary SQLite store.

    Patches the global singleton job_store and job_manager to prevent test fixtures
    and mocks from polluting the production database.
    """
    test_db = tmp_path / "test_transcriptions.db"
    test_store = JobStore(db_path=test_db, enabled=True)

    monkeypatch.setattr(storage, "job_store", test_store)
    monkeypatch.setattr(jobs.job_manager, "_store", test_store)

    jobs.job_manager._jobs.clear()
    jobs.job_manager._batches.clear()

    yield test_store
