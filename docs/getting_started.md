# Getting Started

This guide shows how to install Streval and run a minimal evaluation.

## Installation

Install Streval from PyPI:

```bash
pip install streval
```

## Minimal Example

### Run Evaluation

=== "Dict"
    ```py
    from streval.evaluators import StructuredExtractionEvaluator

    # Ground truth
    ground_truth = {
        "invoice_number": "INV-001",
        "summary": {
            "total_amount": 100.0,
        },
    }

    # Predictions
    predictions = [
        {"invoice_number": "INV-001", "summary": {"total_amount": 100.0}},  # Correct
        {"invoice_number": "001", "summary": {"total_amount": 50.0}},  # Incorrect
    ]

    # Initialize evaluator
    evaluator = StructuredExtractionEvaluator()

    # Run evaluation
    results = evaluator.evaluate(
        ground_truth=ground_truth,
        predictions=predictions,
    )
    ```

=== "Pydantic"
    ```py
    from pydantic import BaseModel
    from streval.evaluators import StructuredExtractionEvaluator


    # Define Models
    class SpendSummary(BaseModel):
        total_amount: float


    class Invoice(BaseModel):
        invoice_number: str
        summary: SpendSummary


    # Ground truth
    ground_truth = Invoice(
        invoice_number="INV-001",
        summary=SpendSummary(total_amount=100.0),
    )

    # Predictions
    predictions = [
        Invoice(
            invoice_number="INV-001", summary=SpendSummary(total_amount=100.0)
        ),  # Correct
        Invoice(invoice_number="001", summary=SpendSummary(total_amount=50.0)),  # Incorrect
    ]

    # Initialize evaluator
    evaluator = StructuredExtractionEvaluator()

    # Run evaluation
    results = evaluator.evaluate(
        ground_truth=ground_truth,
        predictions=predictions,
    )
    ```

### Results

```json
{
    "avg_field_accuracy": 0.5,
    "avg_object_accuracy": 0.5,
    "total_fields_compared": 4,
    "nb_samples": 2,
    "per_field_accuracy": {
        "summary.total_amount": 0.5,
        "invoice_number": 0.5
    }
}
```
