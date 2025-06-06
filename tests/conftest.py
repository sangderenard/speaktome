import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "stub: placeholder test requiring implementation")
