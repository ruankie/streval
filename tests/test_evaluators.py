import pytest
from streval.evaluators import StructuredExtractionEvaluator
from files.invoice.schema import Invoice


def test_all_correct_gives_highest_score_single_sample(
    ground_truth_1, prediction_all_correct_gt1
):
    evaluator = StructuredExtractionEvaluator()
    results = evaluator.evaluate(
        ground_truths=[ground_truth_1],
        predictions=[prediction_all_correct_gt1],
    )

    assert results["avg_field_accuracy"] == 1.0
    assert results["avg_object_accuracy"] == 1.0
    for _, acc in results["per_field_accuracy"].items():
        assert acc == 1.0


def test_all_correct_gives_highest_score_multiple_samples(
    ground_truth_1,
    ground_truth_2,
    prediction_all_correct_gt1,
    prediction_all_correct_gt2,
):
    evaluator = StructuredExtractionEvaluator()
    results = evaluator.evaluate(
        ground_truths=[ground_truth_1, ground_truth_2],
        predictions=[prediction_all_correct_gt1, prediction_all_correct_gt2],
    )

    assert results["avg_field_accuracy"] == 1.0
    assert results["avg_object_accuracy"] == 1.0
    for _, acc in results["per_field_accuracy"].items():
        assert acc == 1.0


def test_raises_value_error_for_sample_mismatch(
    ground_truth_1,
    ground_truth_2,
    prediction_all_correct_gt1,
    prediction_all_correct_gt2,
):
    evaluator = StructuredExtractionEvaluator()
    with pytest.raises(ValueError) as excinfo:
        evaluator.evaluate(
            ground_truths=[ground_truth_1, ground_truth_2],
            predictions=[prediction_all_correct_gt1],
        )


def test_all_incorrect_gives_zero_single_sample(
    ground_truth_1, prediction_all_incorrect_gt1
):
    evaluator = StructuredExtractionEvaluator()
    results = evaluator.evaluate(
        ground_truths=[ground_truth_1],
        predictions=[prediction_all_incorrect_gt1],
    )

    assert results["avg_field_accuracy"] == 0.0
    assert results["avg_object_accuracy"] == 0.0
    for _, acc in results["per_field_accuracy"].items():
        assert acc == 0.0


def test_all_incorrect_gives_zero_multiples_samples(
    ground_truth_1,
    ground_truth_2,
    prediction_all_incorrect_gt1,
    prediction_all_incorrect_gt2,
):
    evaluator = StructuredExtractionEvaluator()
    results = evaluator.evaluate(
        ground_truths=[ground_truth_1, ground_truth_2],
        predictions=[prediction_all_incorrect_gt1, prediction_all_incorrect_gt2],
    )

    assert results["avg_field_accuracy"] == 0.0
    assert results["avg_object_accuracy"] == 0.0
    for _, acc in results["per_field_accuracy"].items():
        assert acc == 0.0


def test_half_fields_correct_gives_fifty_percent_single_sample(
    ground_truth_1, prediction_half_correct_gt1
):
    evaluator = StructuredExtractionEvaluator()
    results = evaluator.evaluate(
        ground_truths=[ground_truth_1],
        predictions=[prediction_half_correct_gt1],
    )

    assert round(results["avg_field_accuracy"], 3) == 0.455  # 5/11 correct
    assert results["avg_object_accuracy"] == 0.0  # No full obj matches


def test_one_fully_correct_one_fully_incorrect_gives_fifty_percent(
    ground_truth_1, prediction_all_correct_gt1, prediction_all_incorrect_gt1
):
    evaluator = StructuredExtractionEvaluator()
    results = evaluator.evaluate(
        ground_truths=[ground_truth_1, ground_truth_1],
        predictions=[prediction_all_correct_gt1, prediction_all_incorrect_gt1],
    )

    assert results["avg_field_accuracy"] == 0.5
    assert results["avg_object_accuracy"] == 0.5


def test_penalises_missing_extracted_items(
    ground_truth_1, prediction_missing_items_gt1
):
    evaluator = StructuredExtractionEvaluator()
    results = evaluator.evaluate(
        ground_truths=[ground_truth_1],
        predictions=[prediction_missing_items_gt1],
    )

    assert results["avg_field_accuracy"] < 1.0
    assert results["avg_object_accuracy"] == 0.0


def test_penalises_extra_extracted_items(ground_truth_1, prediction_extra_items_gt1):
    evaluator = StructuredExtractionEvaluator()
    results = evaluator.evaluate(
        ground_truths=[ground_truth_1],
        predictions=[prediction_extra_items_gt1],
    )

    assert results["avg_field_accuracy"] < 1.0
    assert results["avg_object_accuracy"] == 0.0


def test_correct_average_calculation(
    ground_truth_1,
    ground_truth_2,
    prediction_all_correct_gt1,
    prediction_half_correct_gt1,
    prediction_all_correct_gt2,
):
    evaluator = StructuredExtractionEvaluator()
    results = evaluator.evaluate(
        ground_truths=[ground_truth_1, ground_truth_1, ground_truth_2],
        predictions=[
            prediction_all_correct_gt1,
            prediction_half_correct_gt1,
            prediction_all_correct_gt2,
        ],
    )

    assert round(results["avg_field_accuracy"], 3) == 0.833
    assert round(results["avg_object_accuracy"], 3) == 0.667


def test_count_total_fields_assessed_is_correct_single_sample(
    ground_truth_1, prediction_all_correct_gt1
):
    evaluator = StructuredExtractionEvaluator()
    results = evaluator.evaluate(
        ground_truths=[ground_truth_1],
        predictions=[prediction_all_correct_gt1],
    )

    assert results["total_fields_compared"] == 11


def test_count_total_fields_assessed_is_correct_multiple_samples(
    ground_truth_1, prediction_all_correct_gt1
):
    evaluator = StructuredExtractionEvaluator()
    results = evaluator.evaluate(
        ground_truths=[ground_truth_1, ground_truth_1],
        predictions=[prediction_all_correct_gt1, prediction_all_correct_gt1],
    )

    assert results["total_fields_compared"] == 22


def test_count_total_samples(
    ground_truth_1,
    prediction_all_correct_gt1,
    prediction_all_incorrect_gt1,
    prediction_half_correct_gt1,
):
    evaluator = StructuredExtractionEvaluator()

    # 1 sample
    results = evaluator.evaluate(
        ground_truths=[ground_truth_1],
        predictions=[prediction_all_correct_gt1],
    )
    assert results["nb_samples"] == 1

    # 2 samples
    results = evaluator.evaluate(
        ground_truths=[ground_truth_1, ground_truth_1],
        predictions=[prediction_all_correct_gt1, prediction_all_incorrect_gt1],
    )
    assert results["nb_samples"] == 2

    # 3 samples
    results = evaluator.evaluate(
        ground_truths=[ground_truth_1, ground_truth_1, ground_truth_1],
        predictions=[
            prediction_all_correct_gt1,
            prediction_all_incorrect_gt1,
            prediction_half_correct_gt1,
        ],
    )
    assert results["nb_samples"] == 3


def test_accepts_pydantic_models(ground_truth_1, prediction_all_correct_gt1):
    evaluator = StructuredExtractionEvaluator()
    ground_truth_pydantic = Invoice.model_validate(ground_truth_1)
    prediction_all_correct_pydantic = Invoice.model_validate(prediction_all_correct_gt1)

    results = evaluator.evaluate(
        ground_truths=[ground_truth_pydantic],
        predictions=[prediction_all_correct_pydantic],
    )
    assert results["avg_field_accuracy"] == 1.0
    assert results["avg_object_accuracy"] == 1.0


def test_accepts_mixture_of_dict_and_pydantic_models(
    ground_truth_1, prediction_all_correct_gt1
):
    evaluator = StructuredExtractionEvaluator()
    prediction_all_correct_pydantic = Invoice.model_validate(prediction_all_correct_gt1)

    results = evaluator.evaluate(
        ground_truths=[ground_truth_1],
        predictions=[prediction_all_correct_pydantic],
    )
    assert results["avg_field_accuracy"] == 1.0
    assert results["avg_object_accuracy"] == 1.0
