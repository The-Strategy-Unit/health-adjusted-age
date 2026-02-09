"""Tests for my-python-project."""

from health_adjusted_age import hello


def test_hello_default():
    """Test hello with default argument."""
    assert hello() == "Hello, World!"


def test_hello_custom():
    """Test hello with custom name."""
    assert hello("Alice") == "Hello, Alice!"
