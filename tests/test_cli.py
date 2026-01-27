"""Tests for the JEDI CLI."""

import subprocess
import sys
from importlib.metadata import version


def test_version_matches_package():
    """Test that CLI version matches package version."""
    _version = version("jedireporter")
    result = subprocess.run(
        [sys.executable, "-m", "jedireporter.cli"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert _version in result.stdout


def test_version_flag():
    """Test that --version flag works correctly."""
    _version = version("jedireporter")
    result = subprocess.run(
        [sys.executable, "-m", "jedireporter.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert _version in result.stdout
