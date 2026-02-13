from typing import Any, Dict, List, Tuple, Union
from pydantic import BaseModel


class StructuredExtractionEvaluator:
    def __init__(self):
        # Future: allow comparator registry injection
        pass

    # -------------------------
    # Public API
    # -------------------------
    def evaluate(
        self,
        ground_truth: Union[Dict, BaseModel],
        predictions: List[Union[Dict, BaseModel]],
    ) -> Dict[str, Any]:
        """
        Evaluate multiple predictions against a single ground truth object.
        """

        gt_dict = self._to_dict(ground_truth)
        gt_flat = self._flatten(gt_dict)

        all_results = []
        total_correct = 0
        total_fields = 0
        object_exact_matches = 0

        per_field_counts = {}  # path -> {"correct": int, "total": int}

        for pred in predictions:
            pred_dict = self._to_dict(pred)
            pred_flat = self._flatten(pred_dict)

            result = self._compare_flat_dicts(gt_flat, pred_flat)
            all_results.append(result)

            total_correct += result["correct_fields"]
            total_fields += result["total_fields"]

            if result["correct_fields"] == result["total_fields"]:
                object_exact_matches += 1

            # Aggregate per-field stats
            for path, is_correct in result["field_results"].items():
                if path not in per_field_counts:
                    per_field_counts[path] = {"correct": 0, "total": 0}

                per_field_counts[path]["correct"] += int(is_correct)
                per_field_counts[path]["total"] += 1

        field_accuracy = (
            total_correct / total_fields if total_fields > 0 else 0.0
        )
        object_accuracy = (
            object_exact_matches / len(predictions)
            if predictions
            else 0.0
        )

        per_field_accuracy = {
            path: counts["correct"] / counts["total"]
            for path, counts in per_field_counts.items()
        }

        return {
            "field_accuracy": field_accuracy,
            "object_accuracy": object_accuracy,
            "total_fields_compared": total_fields,
            "total_predictions": len(predictions),
            "per_field_accuracy": per_field_accuracy,
            "detailed_results": all_results,  # useful for debugging
        }

    # -------------------------
    # Core Comparison Logic
    # -------------------------
    def _compare_flat_dicts(
        self,
        gt_flat: Dict[str, Any],
        pred_flat: Dict[str, Any],
    ) -> Dict[str, Any]:

        all_paths = set(gt_flat.keys()).union(set(pred_flat.keys()))

        field_results = {}
        correct_fields = 0
        total_fields = len(all_paths)

        for path in all_paths:
            gt_value = gt_flat.get(path, None)
            pred_value = pred_flat.get(path, None)

            is_correct = self._compare_values(gt_value, pred_value)
            field_results[path] = is_correct

            if is_correct:
                correct_fields += 1

        return {
            "correct_fields": correct_fields,
            "total_fields": total_fields,
            "field_results": field_results,
        }

    # -------------------------
    # Value Comparison
    # -------------------------
    def _compare_values(self, gt_value: Any, pred_value: Any) -> bool:
        """
        Strict equality comparison for V1.
        Designed to be replaceable later.
        """
        return gt_value == pred_value

    # -------------------------
    # Utilities
    # -------------------------
    def _to_dict(self, obj: Union[Dict, BaseModel]) -> Dict:
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        elif isinstance(obj, dict):
            return obj
        else:
            raise TypeError("Input must be dict or Pydantic BaseModel")

    def _flatten(
        self,
        obj: Any,
        parent_key: str = "",
    ) -> Dict[str, Any]:

        items = {}

        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}.{key}" if parent_key else key
                items.update(self._flatten(value, new_key))

        elif isinstance(obj, list):
            for idx, value in enumerate(obj):
                new_key = f"{parent_key}.{idx}" if parent_key else str(idx)
                items.update(self._flatten(value, new_key))

        else:
            # Leaf node
            items[parent_key] = obj

        return items
