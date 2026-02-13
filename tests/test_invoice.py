import json
from pathlib import Path

from streval.evaluators import StructuredExtractionEvaluator


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_invoice_evaluation():
    """
    Run evaluation against the invoice example and print results.
    """

    base_path = Path(__file__).resolve().parent.parent
    example_path = base_path / "examples" / "invoice"

    ground_truth_path = example_path / "ground_truth.json"
    prediction_all_correct_path = example_path / "predictions" / "all_correct.json"
    prediction_all_incorrect_path = example_path / "predictions" / "all_incorrect.json"
    prediction_half_correct_path = example_path / "predictions" / "half_correct.json"

    ground_truth = load_json(ground_truth_path)
    prediction_all_correct = load_json(prediction_all_correct_path)
    prediction_all_incorrect = load_json(prediction_all_incorrect_path)
    prediction_half_correct = load_json(prediction_half_correct_path)
    predictions = [
        prediction_all_correct,
        prediction_all_incorrect,
        prediction_half_correct,
    ]

    evaluator = StructuredExtractionEvaluator()
    results = evaluator.evaluate(
        ground_truth=ground_truth,
        predictions=predictions,
    )

    print("\n===== RAW RESULTS =====")
    print(json.dumps(results, indent=4))

    print("\n===== STREVAL INVOICE RESULTS =====")
    print(f"Avg field-level Accuracy:  {results['avg_field_accuracy']:.4f}")
    print(f"Avg object-level Accuracy: {results['avg_object_accuracy']:.4f}")
    print(f"Total Fields Compared: {results['total_fields_compared']}")
    print(f"Number of samples: {results['nb_samples']}\n")

    print("Field-level Accuracy:")
    for field, acc in sorted(results["per_field_accuracy"].items()):
        print(f"  {field}: {acc:.4f}")

    print("\n====================================\n")

    # Minimal assertion so pytest doesn't complain
    assert results["nb_samples"] == len(predictions)

if __name__ == "__main__":
    test_invoice_evaluation()