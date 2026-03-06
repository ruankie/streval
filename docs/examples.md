# Getting Started

This guide shows several minimal examples to get you started with Streval.

## Excluding Fields

Sometimes you may want to exclude certain fields from evaluation. You can do this by passing dot-notation paths to the `exclude_fields` parameter.

In this example, we will show how to exclude leaf fields as well as entire sections from the evaluation. As a simple demonstration, we can exclude all incorrect fields to get a perfect score.

> **Note**: Excluded fields will be excluded from all ground truths and predictions of an evaluation run.

### Prepare Evaluation
```py
from streval.evaluators import StructuredExtractionEvaluator

# Ground truth
gt = {
    "order_id": "ORD-001",
    "customer": {"name": "Jane Doe", "address": {"city": "Austin", "country": "US"}},
    "total": 125.86,
}

# Prediction
pred = {
    "order_id": "ORD-002",  # Incorrect
    "customer": {
        "name": "Jane Doe",
        "address": {"city": "London", "country": "UK"},  # Incorrect
    },
    "total": 125.86,
}

evaluator = StructuredExtractionEvaluator()
results = evaluator.evaluate(
    ground_truths=[gt],
    predictions=[pred],
    # Exclude all incorrect fields to get perfect score
    exclude_fields=[
        "order_id",  # Leaf field
        "customer.address",  # Section
    ],
)
```

### Results

```json
{
    "nb_samples": 1,
    "total_fields_compared": 2,
    "avg_field_accuracy": 1.0,
    "avg_object_accuracy": 1.0,
    "per_field_accuracy": {
        "customer.name": 1.0,
        "total": 1.0
    }
}
```

## Using Pydantic Models

### Prepare Evaluation
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