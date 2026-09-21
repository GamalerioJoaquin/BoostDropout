"""Smoke tests for the distributable package boundary."""

import boostdropout


def test_package_exposes_a_semantic_version() -> None:
    """The package can be imported before experimental modules are introduced."""
    assert boostdropout.__version__ == "0.1.0"
