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

    # Ground truths
    ## Invoice 1
    gt1 = {
        "invoice_number": "INV-001",
        "summary": {
            "total_amount": 100.0,
        },
    }

    ## Invoice 2
    gt2 = {
        "invoice_number": "INV-002",
        "summary": {
            "total_amount": 200.0,
        },
    }

    # Predictions
    ## Invoice 1 (correct)
    p1 = {"invoice_number": "INV-001", "summary": {"total_amount": 100.0}}

    ## Invoice 2 (incorrect)
    p2 = {"invoice_number": "INV-001", "summary": {"total_amount": 100.0}}

    # Initialize evaluator
    evaluator = StructuredExtractionEvaluator()

    # Run evaluation
    results = evaluator.evaluate(
        ground_truths=[gt1, gt2],
        predictions=[p1, p2],
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


    # Ground truths
    ## Invoice 1
    gt1 = Invoice(
        invoice_number="INV-001",
        summary=SpendSummary(total_amount=100.0),
    )

    ## Invoice 2
    gt2 = Invoice(
        invoice_number="INV-002",
        summary=SpendSummary(total_amount=200.0),
    )

    # Predictions
    ## Invoice 1 (correct)
    p1 = Invoice(invoice_number="INV-001", summary=SpendSummary(total_amount=100.0))

    ## Invoice 2 (incorrect)
    p2 = Invoice(invoice_number="001", summary=SpendSummary(total_amount=50.0))

    # Initialize evaluator
    evaluator = StructuredExtractionEvaluator()

    # Run evaluation
    results = evaluator.evaluate(
        ground_truths=[gt1, gt2],
        predictions=[p1, p2],
    )
    ```

### Results

```json
{
    "avg_field_accuracy": 0.5,
    "avg_object_accuracy": 0.5,
    "nb_samples": 2,
    "total_fields_compared": 4,
    "per_field_accuracy": {
        "summary.total_amount": 0.5,
        "invoice_number": 0.5
    }
}
```
