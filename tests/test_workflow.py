"""Tests for the JEDI workflow CLI."""

import subprocess
import sys
from importlib.metadata import version


def test_version_flag():
    """Test that --version flag works correctly."""
    _version = version("jedireporter")
    result = subprocess.run(
        [sys.executable, "-m", "jedireporter.workflow", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert _version in result.stdout
