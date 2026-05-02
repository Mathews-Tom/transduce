"""Sanity checks for the test harness itself."""

from __future__ import annotations

import sys

import pytest

import transduce


@pytest.mark.unit
def test_pytest_works_returns_truthy() -> None:
    """Trivial assertion to prove pytest, importlib mode, and markers are wired."""
    assert True


@pytest.mark.unit
def test_transduce_package_importable_exposes_version() -> None:
    """The top-level package exposes a SemVer-shaped __version__ string."""
    assert isinstance(transduce.__version__, str)
    assert len(transduce.__version__.split(".")) == 3


@pytest.mark.unit
def test_python_runtime_meets_minimum_version_requirement() -> None:
    """Tests must run on Python 3.12 or newer per pyproject requires-python."""
    assert sys.version_info >= (3, 12)
