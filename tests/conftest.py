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
def ground_truth_1():
    return load_json(INVOICE_EXAMPLE_PATH / "ground_truth_1.json")


@pytest.fixture
def ground_truth_2():
    return load_json(INVOICE_EXAMPLE_PATH / "ground_truth_2.json")


@pytest.fixture
def prediction_all_correct_gt1():
    return load_json(INVOICE_EXAMPLE_PATH / "predictions" / "all_correct_gt1.json")


@pytest.fixture
def prediction_all_correct_gt2():
    return load_json(INVOICE_EXAMPLE_PATH / "predictions" / "all_correct_gt2.json")


@pytest.fixture
def prediction_all_incorrect_gt1():
    return load_json(INVOICE_EXAMPLE_PATH / "predictions" / "all_incorrect_gt1.json")


@pytest.fixture
def prediction_all_incorrect_gt2():
    return load_json(INVOICE_EXAMPLE_PATH / "predictions" / "all_incorrect_gt2.json")


@pytest.fixture
def prediction_half_correct_gt1():
    return load_json(INVOICE_EXAMPLE_PATH / "predictions" / "half_correct_gt1.json")


@pytest.fixture
def prediction_missing_items_gt1():
    return load_json(INVOICE_EXAMPLE_PATH / "predictions" / "missing_items_gt1.json")


@pytest.fixture
def prediction_extra_items_gt1():
    return load_json(INVOICE_EXAMPLE_PATH / "predictions" / "extra_items_gt1.json")
