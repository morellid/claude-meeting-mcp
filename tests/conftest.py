"""Shared test fixtures.

The job registry resolves its directory at call time from CLAUDE_MEETING_DATA_DIR,
so without this the suite would create claim and lock files in the real
~/Library/Application Support directory.
"""

import pytest

from claude_meeting_mcp import jobs


@pytest.fixture(autouse=True)
def isolate_data_dir(tmp_path, monkeypatch):
    """Redirect the data dir and drop any job locks left behind by a test."""
    monkeypatch.setenv("CLAUDE_MEETING_DATA_DIR", str(tmp_path))
    yield
    jobs._release_all_for_tests()
