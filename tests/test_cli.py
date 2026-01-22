"""Tests for the JEDI CLI."""

import subprocess
import sys

from jedi import __version__


def test_version_matches_package():
    """Test that CLI version matches package version."""
    result = subprocess.run(
        [sys.executable, "-m", "jedi.cli"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert __version__ in result.stdout


def test_version_flag():
    """Test that --version flag works correctly."""
    result = subprocess.run(
        [sys.executable, "-m", "jedi.cli", "--version"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert __version__ in result.stdout
