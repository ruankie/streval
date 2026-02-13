import json
from pathlib import Path
import pytest

THIS_PATH = Path(__file__).resolve().parent
INVOICE_EXAMPLE_PATH = THIS_PATH / "files" / "invoice"


def load_json(path: Path) -> dict:
    """Load a JSON file as dict."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture
def ground_truth():
    return load_json(INVOICE_EXAMPLE_PATH / "ground_truth.json")


@pytest.fixture
def prediction_all_correct():
    return load_json(INVOICE_EXAMPLE_PATH / "predictions" / "all_correct.json")


@pytest.fixture
def prediction_all_incorrect():
    return load_json(INVOICE_EXAMPLE_PATH / "predictions" / "all_incorrect.json")


@pytest.fixture
def prediction_half_correct():
    return load_json(INVOICE_EXAMPLE_PATH / "predictions" / "half_correct.json")


@pytest.fixture
def prediction_missing_items():
    return load_json(INVOICE_EXAMPLE_PATH / "predictions" / "missing_items.json")


@pytest.fixture
def prediction_extra_items():
    return load_json(INVOICE_EXAMPLE_PATH / "predictions" / "extra_items.json")
