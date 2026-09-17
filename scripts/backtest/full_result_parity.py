"""Bounded full-result evidence at the persisted API top boundary (no financial oracle)."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

FLOAT_TOLERANCE = 1e-5  # Existing API-runner absolute tolerance; no relative tolerance.
MAX_ROWS = 50
CONTEXT_KEYS = (
    "request_hash",
    "engine_params_hash",
    "backtest_runtime_config_hash",
    "artifact_manifest_hash",
)
ROW_FIELDS = (
    "rank",
    "variant_hash",
    "canonical_variant_params",
    "summary_metrics",
    "best_tp_pct",
    "best_sl_pct",
)


def normalize(value: Any) -> Any:
    """Match the existing persisted JSON policy: nonfinite numbers become null."""
    if isinstance(value, Mapping):
        return {str(k): normalize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def metric_equal(left: Any, right: Any) -> bool:
    left, right = normalize(left), normalize(right)
    if left is None or right is None:
        return left is right
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return abs(left - right) <= FLOAT_TOLERANCE
    return type(left) is type(right) and left == right


def compare_rows(
    actual: Sequence[Mapping[str, Any]], expected: Sequence[Mapping[str, Any]]
) -> list:
    """Compare membership, sequence, semantic identity, full metric keys and selected cells."""
    mismatches = []
    if len(actual) != len(expected):
        mismatches.append(
            {"field": "cardinality", "expected": len(expected), "actual": len(actual)}
        )
    seen = set()
    for index, row in enumerate(actual):
        identity = json.dumps(row.get("canonical_variant_params"), sort_keys=True)
        if identity in seen:
            mismatches.append({"rank": index + 1, "field": "duplicate_identity"})
        seen.add(identity)
    for index, (a, e) in enumerate(zip(actual, expected, strict=False), start=1):
        for field in ROW_FIELDS:
            if field not in a or field not in e:
                mismatches.append({"rank": index, "field": field, "reason": "missing_field"})
            elif field == "summary_metrics":
                am, em = a[field], e[field]
                if not isinstance(am, Mapping) or not isinstance(em, Mapping):
                    mismatches.append({"rank": index, "field": field, "reason": "invalid_metrics"})
                    continue
                for key in sorted(set(am) | set(em)):
                    if key not in am or key not in em or not metric_equal(am[key], em[key]):
                        mismatches.append(
                            {"rank": index, "field": key, "reason": "metric_mismatch"}
                        )
            elif field in ("best_tp_pct", "best_sl_pct"):
                if not metric_equal(a[field], e[field]):
                    mismatches.append({"rank": index, "field": field})
            elif normalize(a[field]) != normalize(e[field]):
                mismatches.append({"rank": index, "field": field})
        if a.get("rank") != index or e.get("rank") != index:
            mismatches.append({"rank": index, "field": "order"})
    return mismatches


def assess_full_parity(
    *,
    api_top: Mapping[str, Any],
    reference: Mapping[str, Any],
    context: Mapping[str, Any],
    requested_top_n: int,
    available_count: int | None,
) -> dict[str, Any]:
    """A missing, partial or semantically incompatible oracle is never assessed success."""
    rows = api_top.get("items")
    expected = reference.get("items")
    result: dict[str, Any] = {
        "status": "not_assessed",
        "pass": False,
        "requested_top_n": requested_top_n,
        "available_count": available_count,
        "expected_count": reference.get("expected_count"),
        "retrieved_count": len(rows) if isinstance(rows, list) else None,
        "compared_count": 0,
        "complete": False,
        "mismatches": [],
        "reasons": [],
        "boundary": "persisted_api_top",
        "absolute_tolerance": FLOAT_TOLERANCE,
    }
    reasons = result["reasons"]
    if not 1 <= requested_top_n <= MAX_ROWS:
        reasons.append("requested_top_n_outside_bounded_benchmark")
    if api_top.get("_status", 200) != 200 or not isinstance(rows, list):
        reasons.append("api_rows_unavailable")
    # The current /top contract is unpaginated. Fail closed on an unfamiliar partial response.
    if api_top.get("next_cursor") or api_top.get("has_more") or api_top.get("pagination"):
        reasons.append("incomplete_or_unsupported_pagination")
    if reference.get("schema") != "backtest_full_top_reference_v1":
        reasons.append("complete_reference_schema_missing")
    provenance = reference.get("provenance", {})
    if not isinstance(provenance, Mapping) or (
        provenance.get("method") not in ("independent_expectations", "trusted_baseline")
        or not provenance.get("source")
        or not provenance.get("revision")
    ):
        reasons.append("trusted_reference_provenance_missing")
    ref_context = reference.get("context", {})
    if not isinstance(ref_context, Mapping) or any(
        not context.get(k) or context.get(k) != ref_context.get(k) for k in CONTEXT_KEYS
    ):
        reasons.append("reference_semantics_or_artifact_identity_unavailable_or_incompatible")
    count = reference.get("expected_count")
    if (
        reference.get("complete") is not True
        or reference.get("requested_top_n") != requested_top_n
        or type(count) is not int
        or not 0 <= count <= requested_top_n
        or not isinstance(expected, list)
        or len(expected) != count
    ):
        reasons.append("reference_incomplete")
    if isinstance(expected, list):
        for row in expected:
            if (
                not isinstance(row, Mapping)
                or any(k not in row for k in ROW_FIELDS)
                or not row.get("variant_hash")
                or not row.get("canonical_variant_params")
                or not isinstance(row.get("summary_metrics"), Mapping)
                or "total_return_pct" not in row["summary_metrics"]
            ):
                reasons.append("reference_required_fields_missing")
                break
        if not reasons and compare_rows(expected, expected):
            reasons.append("reference_identity_or_rank_invalid")
    if reasons:
        return result
    assert isinstance(rows, list) and isinstance(expected, list)
    if not all(isinstance(row, Mapping) for row in rows):
        result["status"] = "failed"
        result["mismatches"] = [{"field": "rows", "reason": "invalid_row"}]
        return result
    mismatches = compare_rows(rows, expected)
    if available_count is not None and len(rows) != available_count:
        mismatches.append(
            {"field": "available_count", "expected": available_count, "actual": len(rows)}
        )
    result.update(
        status="failed" if mismatches else "passed",
        pass_=not mismatches,
        complete=len(rows) == count,
        compared_count=min(len(rows), len(expected)),
        mismatches=mismatches,
    )
    result["pass"] = result.pop("pass_")
    return result
