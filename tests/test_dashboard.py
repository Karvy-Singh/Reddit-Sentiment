from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_dashboard_starts_without_exception() -> None:
    path = Path(__file__).resolve().parents[1] / "dashboard.py"
    dashboard = AppTest.from_file(path).run(timeout=30)
    assert not dashboard.exception
