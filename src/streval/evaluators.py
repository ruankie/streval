from typing import Any, Dict, List, Union
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
        ground_truths: List[Union[Dict, BaseModel]],
        predictions: List[Union[Dict, BaseModel]],
        exclude_fields: List[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate paired ground truths and predictions across a dataset.

        Args:
            ground_truths (List[Union[Dict, BaseModel]]): The ground truth objects (y_true).
            predictions (List[Union[Dict, BaseModel]]): The prediction objects (y_hat).
                Must be the same length as ground_truths.
            exclude_fields (List[str], optional): Dot-notation paths to exclude from
                evaluation. Entire subsections can be excluded (e.g. "address" excludes
                "address.street", "address.city", etc.). Defaults to None.

        Returns:
            Dict[str, Any]: The evaluation results, including:
                - nb_samples (int): The number of samples evaluated.
                - total_fields_compared (int): The total number of fields compared.
                - avg_field_accuracy (float): The average field accuracy.
                - avg_object_accuracy (float): The average object accuracy.
                - per_field_accuracy (Dict[str, float]): The per-field accuracy.

        Raises:
            ValueError: If ground_truths and predictions have different lengths.
        """

        if len(ground_truths) != len(predictions):
            raise ValueError(
                f"ground_truths and predictions must have the same length, "
                f"got {len(ground_truths)} and {len(predictions)}"
            )

        exclude_fields = exclude_fields or []
        total_correct = 0
        total_fields = 0
        object_exact_matches = 0
        per_field_counts: Dict[str, Dict[str, int]] = {}

        for gt, pred in zip(ground_truths, predictions):
            gt_flat = self._exclude_fields(
                self._flatten(self._to_dict(gt)), exclude_fields
            )
            pred_flat = self._exclude_fields(
                self._flatten(self._to_dict(pred)), exclude_fields
            )

            result = self._compare_flat_dicts(gt_flat, pred_flat)

            total_correct += result["correct_fields"]
            total_fields += result["total_fields"]

            if result["correct_fields"] == result["total_fields"]:
                object_exact_matches += 1

            for path, is_correct in result["field_results"].items():
                if path not in per_field_counts:
                    per_field_counts[path] = {"correct": 0, "total": 0}
                per_field_counts[path]["correct"] += int(is_correct)
                per_field_counts[path]["total"] += 1

        nb_samples = len(predictions)

        return {
            "nb_samples": nb_samples,
            "total_fields_compared": total_fields,
            "avg_field_accuracy": (
                total_correct / total_fields if total_fields > 0 else 0.0
            ),
            "avg_object_accuracy": (
                object_exact_matches / nb_samples if nb_samples > 0 else 0.0
            ),
            "per_field_accuracy": {
                path: counts["correct"] / counts["total"]
                for path, counts in per_field_counts.items()
            },
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

    def _exclude_fields(
        self,
        flat_dict: Dict[str, Any],
        excluded_fields: Union[List[str], None] = None,
    ) -> Dict[str, Any]:
        if not excluded_fields:
            return flat_dict
        return {
            key: value
            for key, value in flat_dict.items()
            if not any(
                key == excl or key.startswith(excl + ".") for excl in excluded_fields
            )
        }
