from streval.evaluators import StructuredExtractionEvaluator
from files.invoice.schema import Invoice


def test_all_correct_gives_highest_score(ground_truth, prediction_all_correct):
    evaluator = StructuredExtractionEvaluator()
    results = evaluator.evaluate(
        ground_truth=ground_truth,
        predictions=[prediction_all_correct],
    )

    assert results["avg_field_accuracy"] == 1.0
    assert results["avg_object_accuracy"] == 1.0
    for _, acc in results["per_field_accuracy"].items():
        assert acc == 1.0


def test_all_incorrect_gives_zero(ground_truth, prediction_all_incorrect):
    evaluator = StructuredExtractionEvaluator()
    results = evaluator.evaluate(
        ground_truth=ground_truth,
        predictions=[prediction_all_incorrect],
    )

    assert results["avg_field_accuracy"] == 0.0
    assert results["avg_object_accuracy"] == 0.0
    for _, acc in results["per_field_accuracy"].items():
        assert acc == 0.0


def test_half_correct_half_incorrect_gives_fifty_percent(
    ground_truth, prediction_all_correct, prediction_all_incorrect
):
    evaluator = StructuredExtractionEvaluator()
    results = evaluator.evaluate(
        ground_truth=ground_truth,
        predictions=[prediction_all_correct, prediction_all_incorrect],
    )

    assert results["avg_field_accuracy"] == 0.5
    assert results["avg_object_accuracy"] == 0.5


def test_penalises_missing_extracted_items(ground_truth, prediction_missing_items):
    evaluator = StructuredExtractionEvaluator()
    results = evaluator.evaluate(
        ground_truth=ground_truth,
        predictions=[prediction_missing_items],
    )

    assert results["avg_field_accuracy"] < 1.0
    assert results["avg_object_accuracy"] == 0.0


def test_penalises_extra_extracted_items(ground_truth, prediction_extra_items):
    evaluator = StructuredExtractionEvaluator()
    results = evaluator.evaluate(
        ground_truth=ground_truth,
        predictions=[prediction_extra_items],
    )

    assert results["avg_field_accuracy"] < 1.0
    assert results["avg_object_accuracy"] == 0.0


def test_correct_average_calculation(
    ground_truth, prediction_all_correct, prediction_half_correct
):
    evaluator = StructuredExtractionEvaluator()
    results = evaluator.evaluate(
        ground_truth=ground_truth,
        predictions=[prediction_all_correct, prediction_half_correct],
    )

    assert round(results["avg_field_accuracy"], 3) == 0.727
    assert round(results["avg_object_accuracy"], 3) == 0.5


def test_count_total_fields_assessed_is_correct(ground_truth, prediction_all_correct):
    evaluator = StructuredExtractionEvaluator()
    results = evaluator.evaluate(
        ground_truth=ground_truth,
        predictions=[prediction_all_correct],
    )

    assert results["total_fields_compared"] == 11


def test_count_total_samples(
    ground_truth,
    prediction_all_correct,
    prediction_all_incorrect,
    prediction_half_correct,
):
    evaluator = StructuredExtractionEvaluator()

    # 1 sample
    results = evaluator.evaluate(
        ground_truth=ground_truth,
        predictions=[prediction_all_correct],
    )
    assert results["nb_samples"] == 1

    # 2 samples
    results = evaluator.evaluate(
        ground_truth=ground_truth,
        predictions=[prediction_all_correct, prediction_all_incorrect],
    )
    assert results["nb_samples"] == 2

    # 3 samples
    results = evaluator.evaluate(
        ground_truth=ground_truth,
        predictions=[
            prediction_all_correct,
            prediction_all_incorrect,
            prediction_half_correct,
        ],
    )
    assert results["nb_samples"] == 3


def test_accepts_pydantic_models(ground_truth, prediction_all_correct):
    evaluator = StructuredExtractionEvaluator()
    ground_truth_pydantic = Invoice.model_validate(ground_truth)
    prediction_all_correct_pydantic = Invoice.model_validate(prediction_all_correct)

    results = evaluator.evaluate(
        ground_truth=ground_truth_pydantic,
        predictions=[prediction_all_correct_pydantic],
    )
    assert results["avg_field_accuracy"] == 1.0
    assert results["avg_object_accuracy"] == 1.0
