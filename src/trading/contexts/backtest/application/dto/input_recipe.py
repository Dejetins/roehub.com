"""Durable semantic preparation recipe; legacy jobs have no recipe."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Mapping

from trading.contexts.backtest_artifacts.application.services.v2.contracts import (
    ArtifactCandleSnapshot,
    ArtifactRequestedRow,
)


@dataclass(frozen=True, slots=True)
class BacktestInputRecipe:
    """Financial identity excludes file location, generation and preparation choices.

    The original source file/domain identities remain immutable after attestation;
    verified prefix proof is added without changing semantic_sha256. Replay may
    resolve different physical files only after checking the stored payload proof.
    """

    snapshot: ArtifactCandleSnapshot
    requested_start_utc: str
    requested_end_utc: str
    rows: tuple[ArtifactRequestedRow, ...]
    rule_version: str
    defaults_sha256: str
    compute_version: str
    precision_version: str
    funding_policy: str
    funding_fingerprint: str | None
    tp_levels_pct: tuple[float, ...] = ()
    sl_levels_pct: tuple[float, ...] = ()
    index_domain: str = "source-global-u32-sentinel-length/v1"
    schema: str = "backtest-input-recipe/v1"

    def __post_init__(self) -> None:
        import math

        if self.schema != "backtest-input-recipe/v1":
            raise ValueError("unsupported input recipe schema")
        if not isinstance(self.snapshot, ArtifactCandleSnapshot):
            raise ValueError("recipe requires a typed candle snapshot")
        if self.index_domain != "source-global-u32-sentinel-length/v1":
            raise ValueError("unsupported recipe index domain")
        dates = []
        for value in (self.requested_start_utc, self.requested_end_utc):
            if not isinstance(value, str) or not value.endswith("Z"):
                raise ValueError("recipe interval requires UTC Z timestamps")
            dates.append(datetime.fromisoformat(value.replace("Z", "+00:00")))
        if dates[0] >= dates[1]:
            raise ValueError("recipe interval must be nonempty and half-open")
        for domain in self.snapshot.consumed_domains:
            if "time" in domain.axis_order:
                start = datetime.fromisoformat(domain.origin_utc.replace("Z", "+00:00"))
                end = datetime.fromisoformat(domain.end_utc.replace("Z", "+00:00"))
                if dates[0] < start or dates[1] > end:
                    raise ValueError("requested interval exceeds recorded source domain")
        keys = [(row.indicator_id, row.row_id) for row in self.rows]
        if not keys or keys != sorted(set(keys)):
            raise ValueError("recipe rows must have unique canonical indicator/row order")
        for value in (self.rule_version, self.compute_version, self.precision_version):
            if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.:/-]+", value):
                raise ValueError("invalid recipe version identity")
        if not re.fullmatch(r"[a-f0-9]{64}", self.defaults_sha256):
            raise ValueError("invalid defaults SHA-256")
        if self.funding_policy not in (
            "not_applicable",
            "disabled",
            "strict",
            "degraded_with_warning",
        ):
            raise ValueError("unsupported effective funding policy")
        if self.funding_policy in ("not_applicable", "disabled"):
            if self.funding_fingerprint is not None:
                raise ValueError("inapplicable funding cannot claim a series proof")
        elif not isinstance(self.funding_fingerprint, str) or not re.fullmatch(
            r"[a-f0-9]{64}", self.funding_fingerprint
        ):
            raise ValueError("effective funding requires an explicit fingerprint, including empty")
        for levels in (self.tp_levels_pct, self.sl_levels_pct):
            if any(type(v) not in (int, float) or not math.isfinite(v) or v <= 0 for v in levels):
                raise ValueError("invalid recipe risk levels")
            if tuple(levels) != tuple(sorted(set(levels))):
                raise ValueError("recipe risk levels must be sorted and unique")
        object.__setattr__(self, "rows", tuple(self.rows))
        object.__setattr__(self, "tp_levels_pct", tuple(self.tp_levels_pct))
        object.__setattr__(self, "sl_levels_pct", tuple(self.sl_levels_pct))

    def as_mapping(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "snapshot": self.snapshot.as_mapping(),
            "requested_start_utc": self.requested_start_utc,
            "requested_end_utc": self.requested_end_utc,
            "rows": [row.as_mapping() for row in self.rows],
            "rule_version": self.rule_version,
            "defaults_sha256": self.defaults_sha256,
            "compute_version": self.compute_version,
            "precision_version": self.precision_version,
            "funding_policy": self.funding_policy,
            "funding_fingerprint": self.funding_fingerprint,
            "tp_levels_pct": list(self.tp_levels_pct),
            "sl_levels_pct": list(self.sl_levels_pct),
            "index_domain": self.index_domain,
        }

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> BacktestInputRecipe:
        keys = {
            "schema",
            "snapshot",
            "requested_start_utc",
            "requested_end_utc",
            "rows",
            "rule_version",
            "defaults_sha256",
            "compute_version",
            "precision_version",
            "funding_policy",
            "funding_fingerprint",
            "tp_levels_pct",
            "sl_levels_pct",
            "index_domain",
        }
        if not isinstance(payload, Mapping) or set(payload) != keys:
            raise ValueError("recipe requires exact versioned keys")
        for key in ("rows", "tp_levels_pct", "sl_levels_pct"):
            if not isinstance(payload[key], (list, tuple)):
                raise ValueError(f"recipe {key} must be an array")
        return cls(
            **{
                **payload,
                "snapshot": ArtifactCandleSnapshot.from_mapping(payload["snapshot"]),
                "rows": tuple(ArtifactRequestedRow.from_mapping(row) for row in payload["rows"]),
                "tp_levels_pct": tuple(payload["tp_levels_pct"]),
                "sl_levels_pct": tuple(payload["sl_levels_pct"]),
            }
        )

    def semantic_mapping(self) -> dict[str, Any]:
        payload = self.as_mapping()
        snapshot = payload.pop("snapshot")
        payload["source"] = {
            "coordinates": snapshot["coordinates"],
            "signal_timeframe": self.snapshot.signal_timeframe,
            "execution_timeframe": self.snapshot.execution_timeframe,
            "consumed_domains": [
                domain.as_mapping()
                for domain in sorted(self.snapshot.consumed_domains, key=lambda domain: domain.role)
            ],
            "file_identities": [
                {"sha256": ref.sha256, "domain": ref.domain.as_mapping()}
                for ref in sorted(
                    self.snapshot.source_file_identities, key=lambda ref: ref.domain.role
                )
            ],
        }
        return {"schema": "backtest-semantic-input/v1", "recipe": payload}

    @property
    def semantic_sha256(self) -> str:
        encoded = json.dumps(
            self.semantic_mapping(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()
