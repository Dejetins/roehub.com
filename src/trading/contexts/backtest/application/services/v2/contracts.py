"""Contracts for deterministic backtest artifact store v2 layout (R2-01)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Literal, Mapping, Protocol

ARTIFACT_STORE_V2_ROOT_LITERAL = "artifacts/backtest/v2"
CURRENT_ARTIFACT_POINTER_FILENAME_V2 = "current.yaml"
ARTIFACT_MANIFEST_FILENAME_V2 = "manifest.yaml"
PRICES_DIRECTORY_LITERAL_V2 = "prices"
SIGNALS_DIRECTORY_LITERAL_V2 = "signals"
MAPPINGS_DIRECTORY_LITERAL_V2 = "mappings"
HIT_TIMES_DIRECTORY_LITERAL_V2 = "hit_times"
ARTIFACT_SLOT_A_LITERAL_V2 = "slot_a"
ARTIFACT_SLOT_B_LITERAL_V2 = "slot_b"
HIT_TIMES_TIMEFRAME_LITERAL_V2 = "1m"
BAR_OPEN_MAPPING_FILENAME_V2 = "bar_open_1m_idx.u32.npy"
BAR_CLOSE_MAPPING_FILENAME_V2 = "bar_close_1m_idx.u32.npy"
OPEN_TIME_FILENAME_V2 = "open_time.i64.npy"
CLOSE_TIME_FILENAME_V2 = "close_time.i64.npy"
OHLCV_FILENAME_V2 = "ohlcv.f32.npy"
SIGNALS_FILENAME_V2 = "signals.i8.npy"
CURRENT_ARTIFACT_POINTER_SCHEMA_VERSION_V2 = 1
SUPPORTED_CURRENT_ARTIFACT_POINTER_SCHEMA_VERSIONS_V2: tuple[int, ...] = (
    CURRENT_ARTIFACT_POINTER_SCHEMA_VERSION_V2,
)
CURRENT_ARTIFACT_POINTER_REQUIRED_KEYS_V2: tuple[str, ...] = (
    "schema_version",
    "active_slot",
    "slot_generation",
    "asof_date",
    "manifest_sha256",
    "published_at_utc",
)
SUPPORTED_ARTIFACT_MARKETS_BY_ID_V2: Mapping[int, tuple[str, str]] = MappingProxyType(
    {
        1: ("binance", "spot"),
        2: ("binance", "futures"),
        3: ("bybit", "spot"),
        4: ("bybit", "futures"),
    }
)

_STRICT_DATE_LITERAL_PATTERN_V2 = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_STRICT_UTC_TIMESTAMP_LITERAL_PATTERN_V2 = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
)
_SHA256_HEX_PATTERN_V2 = re.compile(r"^[0-9a-f]{64}$")

type ArtifactSlotLiteralV2 = Literal["slot_a", "slot_b"]

ALLOWED_ARTIFACT_SLOTS_V2: tuple[ArtifactSlotLiteralV2, ...] = (
    ARTIFACT_SLOT_A_LITERAL_V2,
    ARTIFACT_SLOT_B_LITERAL_V2,
)
ARTIFACT_PRICE_TIMEFRAMES_V2: tuple[str, ...] = (
    "1m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "1d",
    "2d",
    "3d",
)
ARTIFACT_SIGNAL_TIMEFRAMES_V2: tuple[str, ...] = (
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "1d",
    "2d",
    "3d",
)
ARTIFACT_MAPPING_TIMEFRAMES_V2: tuple[str, ...] = ARTIFACT_SIGNAL_TIMEFRAMES_V2
ARTIFACT_HIT_TIMES_TIMEFRAMES_V2: tuple[str, ...] = (HIT_TIMES_TIMEFRAME_LITERAL_V2,)


def ordered_artifact_slots_v2() -> tuple[ArtifactSlotLiteralV2, ...]:
    """
    Return the canonical artifact slot order for R2-01.

    Args:
        None.
    Returns:
        tuple[ArtifactSlotLiteralV2, ...]: Stable ordered slot literals.
    Assumptions:
        The active dataset always lives in one of two fixed slots.
    Raises:
        None.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    return ALLOWED_ARTIFACT_SLOTS_V2


def inactive_artifact_slot_v2(active_slot: str) -> ArtifactSlotLiteralV2:
    """
    Resolve the deterministic inactive slot opposite to the current active slot.

    Args:
        active_slot: Current active slot literal.
    Returns:
        ArtifactSlotLiteralV2: The opposite fixed slot literal.
    Assumptions:
        Milestone R2 uses exactly two slots and publish always targets the inactive one.
    Raises:
        ValueError: If the active slot literal is outside the fixed slot contract.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    validated_active_slot = validate_artifact_slot_v2(active_slot)
    if validated_active_slot == ARTIFACT_SLOT_A_LITERAL_V2:
        return ARTIFACT_SLOT_B_LITERAL_V2
    return ARTIFACT_SLOT_A_LITERAL_V2


def validate_artifact_coordinate_token_v2(token: str, *, field_name: str) -> str:
    """
    Validate one artifact coordinate token with fail-fast filesystem-safe rules.

    Args:
        token: Candidate coordinate literal for exchange, market_type, or symbol.
        field_name: Human-readable coordinate field name used in error messages.
    Returns:
        str: The original token when it satisfies the R2-01 contract.
    Assumptions:
        Coordinates are single path components and must never require normalization.
    Raises:
        ValueError: If the token is empty, contains whitespace, contains path separators,
            or includes traversal patterns.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """
    return _validate_safe_path_token_v2(token=token, field_name=f"coordinate {field_name}")


def validate_indicator_id_v2(indicator_id: str) -> str:
    """
    Validate one indicator identifier token used inside `signals/<tf>/<indicator_id>/`.

    Args:
        indicator_id: Candidate indicator identifier literal.
    Returns:
        str: The original indicator identifier when valid.
    Assumptions:
        Indicator ids may contain dots such as `ma.ema`, but remain one safe path token.
    Raises:
        ValueError: If the identifier is empty, contains whitespace, separators, or traversal.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """
    return _validate_safe_path_token_v2(token=indicator_id, field_name="indicator_id")


def validate_artifact_slot_v2(slot: str) -> ArtifactSlotLiteralV2:
    """
    Validate one artifact slot literal against the fixed R2-01 slot set.

    Args:
        slot: Candidate slot literal.
    Returns:
        ArtifactSlotLiteralV2: Canonical slot literal.
    Assumptions:
        Only `slot_a` and `slot_b` are valid during Milestone R2.
    Raises:
        ValueError: If the slot is not one of the fixed allowed literals.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    _validate_allowed_literal_v2(
        value=slot,
        field_name="slot",
        allowed_literals=ALLOWED_ARTIFACT_SLOTS_V2,
    )
    if slot == ARTIFACT_SLOT_A_LITERAL_V2:
        return ARTIFACT_SLOT_A_LITERAL_V2
    return ARTIFACT_SLOT_B_LITERAL_V2


def validate_current_pointer_schema_version_v2(schema_version: int) -> int:
    """
    Validate `current.yaml.schema_version` against the supported R2-02 set.

    Args:
        schema_version: Candidate pointer schema version value.
    Returns:
        int: Supported schema version literal.
    Assumptions:
        R2-02 supports only one strict pointer schema version.
    Raises:
        ValueError: If the value is not an integer schema version supported by runtime.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    if isinstance(schema_version, bool) or not isinstance(schema_version, int):
        raise ValueError("current.yaml field 'schema_version' must be int")
    if schema_version not in SUPPORTED_CURRENT_ARTIFACT_POINTER_SCHEMA_VERSIONS_V2:
        raise ValueError(
            "current.yaml field 'schema_version' must be one of "
            f"{SUPPORTED_CURRENT_ARTIFACT_POINTER_SCHEMA_VERSIONS_V2}; "
            f"got {schema_version!r}"
        )
    return schema_version


def validate_current_pointer_slot_generation_v2(slot_generation: int) -> int:
    """
    Validate `current.yaml.slot_generation` as a positive integer.

    Args:
        slot_generation: Candidate slot generation scalar.
    Returns:
        int: Validated positive slot generation.
    Assumptions:
        Slot generation increments monotonically on each successful publish switch.
    Raises:
        ValueError: If the value is not a positive integer.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    if isinstance(slot_generation, bool) or not isinstance(slot_generation, int):
        raise ValueError("current.yaml field 'slot_generation' must be int")
    if slot_generation <= 0:
        raise ValueError("current.yaml field 'slot_generation' must be > 0")
    return slot_generation


def validate_current_pointer_asof_date_v2(asof_date: str) -> str:
    """
    Validate `current.yaml.asof_date` as a strict `YYYY-MM-DD` literal.

    Args:
        asof_date: Candidate as-of date literal.
    Returns:
        str: Canonical date literal.
    Assumptions:
        R2-02 serializes pointer identity with exact date-only precision.
    Raises:
        ValueError: If the literal is not a valid strict calendar date.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    if not isinstance(asof_date, str):
        raise ValueError("current.yaml field 'asof_date' must be str")
    if _STRICT_DATE_LITERAL_PATTERN_V2.fullmatch(asof_date) is None:
        raise ValueError("current.yaml field 'asof_date' must be YYYY-MM-DD")
    try:
        date.fromisoformat(asof_date)
    except ValueError as error:
        raise ValueError("current.yaml field 'asof_date' must be valid YYYY-MM-DD") from error
    return asof_date


def validate_current_pointer_manifest_sha256_v2(manifest_sha256: str) -> str:
    """
    Validate `current.yaml.manifest_sha256` as a strict lowercase SHA-256 literal.

    Args:
        manifest_sha256: Candidate manifest hash literal.
    Returns:
        str: Canonical lowercase SHA-256 literal.
    Assumptions:
        Pointer identity stores manifest hashes as 64 lowercase hexadecimal characters.
    Raises:
        ValueError: If the hash is not 64 lowercase hexadecimal characters.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    if not isinstance(manifest_sha256, str):
        raise ValueError("current.yaml field 'manifest_sha256' must be str")
    if _SHA256_HEX_PATTERN_V2.fullmatch(manifest_sha256) is None:
        raise ValueError(
            "current.yaml field 'manifest_sha256' must be 64 lowercase hex chars"
        )
    return manifest_sha256


def validate_current_pointer_published_at_utc_v2(published_at_utc: str) -> str:
    """
    Validate `current.yaml.published_at_utc` as a strict UTC timestamp literal.

    Args:
        published_at_utc: Candidate UTC timestamp literal.
    Returns:
        str: Canonical UTC timestamp literal with `Z` suffix.
    Assumptions:
        Pointer timestamps are serialized with second precision and explicit UTC marker.
    Raises:
        ValueError: If the literal is not `YYYY-MM-DDTHH:MM:SSZ` in UTC.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    if not isinstance(published_at_utc, str):
        raise ValueError("current.yaml field 'published_at_utc' must be str")
    if _STRICT_UTC_TIMESTAMP_LITERAL_PATTERN_V2.fullmatch(published_at_utc) is None:
        raise ValueError(
            "current.yaml field 'published_at_utc' must be YYYY-MM-DDTHH:MM:SSZ"
        )
    parsed = datetime.fromisoformat(published_at_utc.replace("Z", "+00:00"))
    if parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise ValueError("current.yaml field 'published_at_utc' must be UTC")
    return published_at_utc


def validate_price_timeframe_v2(timeframe: str) -> str:
    """
    Validate one price artifact timeframe literal.

    Args:
        timeframe: Candidate timeframe literal for `prices/<tf>/`.
    Returns:
        str: Canonical timeframe literal.
    Assumptions:
        Price artifacts exist for base `1m` and every supported request timeframe.
    Raises:
        ValueError: If the timeframe is outside the documented price artifact contract.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """
    _validate_allowed_literal_v2(
        value=timeframe,
        field_name="price timeframe",
        allowed_literals=ARTIFACT_PRICE_TIMEFRAMES_V2,
    )
    return timeframe


def validate_signal_timeframe_v2(timeframe: str) -> str:
    """
    Validate one signal artifact timeframe literal.

    Args:
        timeframe: Candidate timeframe literal for `signals/<tf>/`.
    Returns:
        str: Canonical timeframe literal.
    Assumptions:
        Signal artifacts are generated only for supported request timeframes, not for `1m`.
    Raises:
        ValueError: If the timeframe is outside the documented signal artifact contract.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """
    _validate_allowed_literal_v2(
        value=timeframe,
        field_name="signal timeframe",
        allowed_literals=ARTIFACT_SIGNAL_TIMEFRAMES_V2,
    )
    return timeframe


def validate_mapping_timeframe_v2(timeframe: str) -> str:
    """
    Validate one mapping artifact timeframe literal.

    Args:
        timeframe: Candidate timeframe literal for `mappings/<tf>/`.
    Returns:
        str: Canonical timeframe literal.
    Assumptions:
        Mapping artifacts are generated for every supported request timeframe.
    Raises:
        ValueError: If the timeframe is outside the documented mapping artifact contract.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """
    _validate_allowed_literal_v2(
        value=timeframe,
        field_name="mapping timeframe",
        allowed_literals=ARTIFACT_MAPPING_TIMEFRAMES_V2,
    )
    return timeframe


def validate_hit_times_timeframe_v2(timeframe: str) -> str:
    """
    Validate one hit-times artifact timeframe literal.

    Args:
        timeframe: Candidate timeframe literal for `hit_times/<tf>/`.
    Returns:
        str: Canonical timeframe literal.
    Assumptions:
        R2-01 fixes hit-times manifests under `hit_times/1m/`.
    Raises:
        ValueError: If the timeframe differs from the fixed `1m` contract.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """
    _validate_allowed_literal_v2(
        value=timeframe,
        field_name="hit-times timeframe",
        allowed_literals=ARTIFACT_HIT_TIMES_TIMEFRAMES_V2,
    )
    return timeframe


def freeze_artifact_payload_mapping_v2(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    Freeze one YAML payload into a stable key-sorted read-only mapping.

    Args:
        payload: Parsed YAML mapping with string keys.
    Returns:
        Mapping[str, Any]: Shallow immutable mapping with deterministic key order.
    Assumptions:
        Nested YAML values are preserved as loaded because R2-01 does not yet impose schema
        coercion for manifests.
    Raises:
        ValueError: If a payload key is not a string.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """
    normalized_keys: list[str] = []
    for key in payload.keys():
        if not isinstance(key, str):
            raise ValueError("artifact YAML payload keys must be strings")
        normalized_keys.append(key)
    normalized_payload: dict[str, Any] = {}
    for key in sorted(normalized_keys):
        normalized_payload[key] = payload[key]
    return MappingProxyType(normalized_payload)


def artifact_coordinates_from_market_id_v2(*, market_id: int, symbol: str) -> ArtifactCoordinatesV2:
    """
    Resolve artifact coordinates from canonical `ref_market.market_id` and symbol.

    Args:
        market_id: Stable market identifier from request/spec payload.
        symbol: Instrument symbol literal.
    Returns:
        ArtifactCoordinatesV2: Deterministic artifact coordinates for symbol-root resolution.
    Assumptions:
        R2-02 bridges `market_id` to `(exchange, market_type)` via the canonical seeded
        `ref_market` ids until R2-04 introduces dedicated artifact config loading.
    Raises:
        ValueError: If the market id is unsupported by the current fixed bridge mapping.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/market_data/application/use_cases/seed_ref_market.py
      - src/trading/contexts/backtest/application/use_cases/backtest_jobs_api_v1.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    resolved_scope = SUPPORTED_ARTIFACT_MARKETS_BY_ID_V2.get(market_id)
    if resolved_scope is None:
        raise ValueError(
            "artifact market bridge does not support market_id "
            f"{market_id!r}; expected one of {tuple(sorted(SUPPORTED_ARTIFACT_MARKETS_BY_ID_V2))}"
        )
    exchange, market_type = resolved_scope
    return ArtifactCoordinatesV2(exchange=exchange, market_type=market_type, symbol=symbol)


def artifact_market_id_from_coordinates_v2(coordinates: ArtifactCoordinatesV2) -> int:
    """
    Resolve canonical `market_id` from artifact coordinates using the fixed R2-02 bridge.

    Args:
        coordinates: Deterministic artifact coordinates.
    Returns:
        int: Canonical market id matching the artifact symbol-root market scope.
    Assumptions:
        Coordinate-to-market resolution stays aligned with `seed_ref_market` during R2-02.
    Raises:
        ValueError: If the coordinate scope has no canonical market id mapping.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/market_data/application/use_cases/seed_ref_market.py
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    for market_id, scope in SUPPORTED_ARTIFACT_MARKETS_BY_ID_V2.items():
        if scope == (coordinates.exchange, coordinates.market_type):
            return market_id
    raise ValueError(
        "artifact market bridge does not support coordinates "
        f"{coordinates.exchange!r}/{coordinates.market_type!r}"
    )


@dataclass(frozen=True, slots=True)
class ArtifactCoordinatesV2:
    """
    Deterministic artifact coordinates that select one backtest dataset namespace.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """

    exchange: str
    market_type: str
    symbol: str

    def __post_init__(self) -> None:
        """
        Validate coordinate tokens for deterministic path composition.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Coordinates are passed directly into filesystem path builders without normalization.
        Raises:
            ValueError: If one coordinate violates the filesystem-safe token contract.
        Side Effects:
            Normalizes the stored coordinates to validated canonical literals.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
        """
        object.__setattr__(
            self,
            "exchange",
            validate_artifact_coordinate_token_v2(self.exchange, field_name="exchange"),
        )
        object.__setattr__(
            self,
            "market_type",
            validate_artifact_coordinate_token_v2(self.market_type, field_name="market_type"),
        )
        object.__setattr__(
            self,
            "symbol",
            validate_artifact_coordinate_token_v2(self.symbol, field_name="symbol"),
        )


@dataclass(frozen=True, slots=True)
class ArtifactPricePathsV2:
    """
    Explicit paths for one `prices/<tf>/` artifact directory.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """

    open_time: Path
    close_time: Path
    ohlcv: Path


@dataclass(frozen=True, slots=True)
class ArtifactSignalPathsV2:
    """
    Explicit paths for one `signals/<tf>/<indicator_id>/` artifact directory.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """

    manifest: Path
    signals: Path


@dataclass(frozen=True, slots=True)
class ArtifactMappingPathsV2:
    """
    Explicit paths for one `mappings/<tf>/` artifact directory.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """

    bar_open_1m_idx: Path
    bar_close_1m_idx: Path


@dataclass(frozen=True, slots=True)
class ArtifactCurrentPointerV2:
    """
    Parsed strict `current.yaml` payload with the typed identity fields required by R2-02.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """

    path: Path
    active_slot: ArtifactSlotLiteralV2
    raw_payload: Mapping[str, Any]
    schema_version: int
    slot_generation: int
    asof_date: str
    manifest_sha256: str
    published_at_utc: str

    def __post_init__(self) -> None:
        """
        Re-validate the strict pointer identity contract and freeze the raw payload.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            `current.yaml` contains exactly the required R2-02 fields with no extras.
        Raises:
            ValueError: If the slot literal or payload shape violates the contract.
        Side Effects:
            Replaces `raw_payload` with a stable read-only mapping.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/backtest-refactor-final-plan-v2.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
        """
        raw_keys = tuple(sorted(self.raw_payload.keys()))
        required_keys = tuple(sorted(CURRENT_ARTIFACT_POINTER_REQUIRED_KEYS_V2))
        if raw_keys != required_keys:
            missing_keys = tuple(
                key
                for key in CURRENT_ARTIFACT_POINTER_REQUIRED_KEYS_V2
                if key not in self.raw_payload
            )
            extra_keys = tuple(
                key
                for key in raw_keys
                if key not in CURRENT_ARTIFACT_POINTER_REQUIRED_KEYS_V2
            )
            details: list[str] = []
            if len(missing_keys) > 0:
                details.append(f"missing keys {missing_keys}")
            if len(extra_keys) > 0:
                details.append(f"unexpected keys {extra_keys}")
            raise ValueError(
                f"{self.path} must contain exactly keys "
                f"{CURRENT_ARTIFACT_POINTER_REQUIRED_KEYS_V2}"
                + (f"; {'; '.join(details)}" if len(details) > 0 else "")
            )
        object.__setattr__(self, "active_slot", validate_artifact_slot_v2(self.active_slot))
        object.__setattr__(
            self,
            "schema_version",
            validate_current_pointer_schema_version_v2(self.schema_version),
        )
        object.__setattr__(
            self,
            "slot_generation",
            validate_current_pointer_slot_generation_v2(self.slot_generation),
        )
        object.__setattr__(
            self,
            "asof_date",
            validate_current_pointer_asof_date_v2(self.asof_date),
        )
        object.__setattr__(
            self,
            "manifest_sha256",
            validate_current_pointer_manifest_sha256_v2(self.manifest_sha256),
        )
        object.__setattr__(
            self,
            "published_at_utc",
            validate_current_pointer_published_at_utc_v2(self.published_at_utc),
        )
        object.__setattr__(
            self, "raw_payload", freeze_artifact_payload_mapping_v2(self.raw_payload)
        )


@dataclass(frozen=True, slots=True)
class ArtifactManifestDocumentV2:
    """
    Parsed slot-level `manifest.yaml` document returned by explicit-path loaders.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """

    path: Path
    raw_payload: Mapping[str, Any]
    slot: ArtifactSlotLiteralV2 | None = None

    def __post_init__(self) -> None:
        """
        Re-validate the optional slot and freeze the raw manifest payload.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Manifest schema validation beyond mapping shape is deferred to later milestones.
        Raises:
            ValueError: If the optional slot literal or payload shape violates the contract.
        Side Effects:
            Replaces `raw_payload` with a stable read-only mapping.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
        """
        if self.slot is not None:
            object.__setattr__(self, "slot", validate_artifact_slot_v2(self.slot))
        object.__setattr__(
            self, "raw_payload", freeze_artifact_payload_mapping_v2(self.raw_payload)
        )


@dataclass(frozen=True, slots=True)
class ArtifactSignalValidationSpecV2:
    """
    Explicit one-indicator validation target used by R2-02 slot publishing checks.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """

    timeframe: str
    indicator_id: str

    def __post_init__(self) -> None:
        """
        Validate one explicit signal artifact validation target.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Signal validation targets remain explicit because R2-04 config loading is not part of
            this milestone.
        Raises:
            ValueError: If timeframe or indicator id violates the deterministic path contract.
        Side Effects:
            Normalizes stored literals to validated canonical values.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
        """
        object.__setattr__(self, "timeframe", validate_signal_timeframe_v2(self.timeframe))
        object.__setattr__(self, "indicator_id", validate_indicator_id_v2(self.indicator_id))


@dataclass(frozen=True, slots=True)
class ArtifactSlotValidationSpecV2:
    """
    Explicit validation plan for an already-built inactive slot in R2-02 publish flow.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """

    price_timeframes: tuple[str, ...] = ()
    mapping_timeframes: tuple[str, ...] = ()
    signal_artifacts: tuple[ArtifactSignalValidationSpecV2, ...] = ()
    require_hit_times_manifest: bool = True

    def __post_init__(self) -> None:
        """
        Validate and deterministically order the explicit slot validation plan.

        Args:
            None.
        Returns:
            None.
        Assumptions:
            Validation order must stay stable regardless of caller tuple ordering.
        Raises:
            ValueError: If one timeframe or signal artifact violates the path contract.
        Side Effects:
            Replaces stored tuples with deterministic canonical ordering.
        Docs:
          - docs/architecture/backtest/backtest-artifact-store-v2.md
          - docs/architecture/roadmap/base_refactor_plan.md
        Related:
          - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
        """
        object.__setattr__(
            self,
            "price_timeframes",
            _sorted_unique_timeframes_v2(
                values=self.price_timeframes,
                allowed_literals=ARTIFACT_PRICE_TIMEFRAMES_V2,
                field_name="price_timeframes",
                validator=validate_price_timeframe_v2,
            ),
        )
        object.__setattr__(
            self,
            "mapping_timeframes",
            _sorted_unique_timeframes_v2(
                values=self.mapping_timeframes,
                allowed_literals=ARTIFACT_MAPPING_TIMEFRAMES_V2,
                field_name="mapping_timeframes",
                validator=validate_mapping_timeframe_v2,
            ),
        )
        object.__setattr__(
            self,
            "signal_artifacts",
            _sorted_signal_validation_specs_v2(self.signal_artifacts),
        )


@dataclass(frozen=True, slots=True)
class ArtifactPublishPrecheckV2:
    """
    Deterministic precheck diagnostics for `build inactive slot` before publish switch.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
      - src/trading/contexts/backtest/application/ports/backtest_job_repositories.py
    """

    coordinates: ArtifactCoordinatesV2
    current_pointer: ArtifactCurrentPointerV2
    inactive_slot: ArtifactSlotLiteralV2
    inactive_manifest_path: Path
    inactive_manifest_hash: str | None
    blocking_active_run_count: int
    ready: bool
    failure_code: str | None = None
    failure_message: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactSlotValidationResultV2:
    """
    Validation output for a prepared inactive slot just before pointer switch.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """

    slot: ArtifactSlotLiteralV2
    slot_manifest: ArtifactManifestDocumentV2
    manifest_sha256: str
    validation_spec: ArtifactSlotValidationSpecV2


@dataclass(frozen=True, slots=True)
class ArtifactPublishResultV2:
    """
    Structured result payload for successful R2-02 current-pointer publish switch.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """

    coordinates: ArtifactCoordinatesV2
    previous_pointer: ArtifactCurrentPointerV2
    published_pointer: ArtifactCurrentPointerV2
    precheck: ArtifactPublishPrecheckV2
    validation: ArtifactSlotValidationResultV2


class BacktestArtifactPathResolverV2(Protocol):
    """
    Port for deterministic filesystem path resolution in artifact store v2.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """

    def ordered_slots(self) -> tuple[ArtifactSlotLiteralV2, ...]:
        """Return the fixed slot order used by runtime-facing callers."""
        ...

    def symbol_root(self, coordinates: ArtifactCoordinatesV2) -> Path:
        """Resolve the `<exchange>/<market_type>/<symbol>/` root."""
        ...

    def current_pointer_path(self, coordinates: ArtifactCoordinatesV2) -> Path:
        """Resolve the `current.yaml` path for one symbol root."""
        ...

    def slot_root(self, coordinates: ArtifactCoordinatesV2, slot: str) -> Path:
        """Resolve the path of one fixed artifact slot root."""
        ...

    def slot_manifest_path(self, coordinates: ArtifactCoordinatesV2, slot: str) -> Path:
        """Resolve the `manifest.yaml` path for one fixed artifact slot."""
        ...

    def price_paths(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
        timeframe: str,
    ) -> ArtifactPricePathsV2:
        """Resolve explicit price artifact paths for one timeframe."""
        ...

    def signal_paths(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
        timeframe: str,
        indicator_id: str,
    ) -> ArtifactSignalPathsV2:
        """Resolve explicit signal artifact paths for one indicator and timeframe."""
        ...

    def mapping_paths(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
        timeframe: str,
    ) -> ArtifactMappingPathsV2:
        """Resolve explicit bar mapping artifact paths for one timeframe."""
        ...

    def hit_times_manifest_path(self, coordinates: ArtifactCoordinatesV2, slot: str) -> Path:
        """Resolve the fixed `hit_times/1m/manifest.yaml` path."""
        ...


class BacktestArtifactLoaderV2(Protocol):
    """
    Port for explicit-path metadata reads in artifact store v2.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_manifest_loader.py
    """

    def load_current_pointer(self, coordinates: ArtifactCoordinatesV2) -> ArtifactCurrentPointerV2:
        """Read one `current.yaml` document by deterministic coordinates."""
        ...

    def load_current_pointer_from_path(self, path: Path) -> ArtifactCurrentPointerV2:
        """Read one `current.yaml` document from an explicit path."""
        ...

    def load_slot_manifest(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
    ) -> ArtifactManifestDocumentV2:
        """Read one slot `manifest.yaml` document by deterministic coordinates."""
        ...

    def load_manifest_from_path(
        self,
        path: Path,
        *,
        slot: ArtifactSlotLiteralV2 | None = None,
    ) -> ArtifactManifestDocumentV2:
        """Read one slot `manifest.yaml` document from an explicit path."""
        ...

    def load_active_slot_manifest(
        self,
        coordinates: ArtifactCoordinatesV2,
    ) -> ArtifactManifestDocumentV2:
        """Read the active slot manifest by first resolving `current.yaml`."""
        ...

    def resolve_current_pointer_path(self, coordinates: ArtifactCoordinatesV2) -> Path:
        """Resolve the `current.yaml` path without touching disk."""
        ...

    def resolve_slot_manifest_path(self, coordinates: ArtifactCoordinatesV2, slot: str) -> Path:
        """Resolve one slot `manifest.yaml` path without touching disk."""
        ...

    def resolve_price_paths(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
        timeframe: str,
    ) -> ArtifactPricePathsV2:
        """Resolve price artifact paths without touching disk."""
        ...

    def resolve_signal_paths(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
        timeframe: str,
        indicator_id: str,
    ) -> ArtifactSignalPathsV2:
        """Resolve signal artifact paths without touching disk."""
        ...

    def resolve_mapping_paths(
        self,
        coordinates: ArtifactCoordinatesV2,
        slot: str,
        timeframe: str,
    ) -> ArtifactMappingPathsV2:
        """Resolve mapping artifact paths without touching disk."""
        ...

    def resolve_hit_times_manifest_path(
        self, coordinates: ArtifactCoordinatesV2, slot: str
    ) -> Path:
        """Resolve the `hit_times/1m/manifest.yaml` path without touching disk."""
        ...


class BacktestArtifactCurrentPointerWriterV2(Protocol):
    """
    Port for deterministic atomic `current.yaml` replacement in artifact store v2.

    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/current_pointer_writer.py
    """

    def write_current_pointer_atomically(
        self,
        coordinates: ArtifactCoordinatesV2,
        pointer: ArtifactCurrentPointerV2,
    ) -> Path:
        """Atomically replace one symbol-root `current.yaml` with deterministic payload bytes."""
        ...


def _validate_safe_path_token_v2(token: str, *, field_name: str) -> str:
    """
    Enforce the shared filesystem-safe token rules used by R2-01 path builders.

    Args:
        token: Candidate path token.
        field_name: Human-readable field name used in stable error messages.
    Returns:
        str: The original token when valid.
    Assumptions:
        Tokens are stored and reused verbatim, so implicit normalization is forbidden.
    Raises:
        ValueError: If the token is empty, contains whitespace, separators, or traversal.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """
    if token == "" or token.strip() == "":
        raise ValueError(
            f"artifact {field_name} must be a non-empty safe token without whitespace, "
            f"separators, or '..'; got {token!r}"
        )
    if any(character.isspace() for character in token):
        raise ValueError(
            f"artifact {field_name} must be a non-empty safe token without whitespace, "
            f"separators, or '..'; got {token!r}"
        )
    if token in {".", ".."} or ".." in token or "/" in token or "\\" in token or "\x00" in token:
        raise ValueError(
            f"artifact {field_name} must be a non-empty safe token without whitespace, "
            f"separators, or '..'; got {token!r}"
        )
    return token


def _validate_allowed_literal_v2(
    *,
    value: str,
    field_name: str,
    allowed_literals: tuple[str, ...],
) -> None:
    """
    Enforce one fixed literal set with deterministic error messages.

    Args:
        value: Candidate literal to validate.
        field_name: Human-readable field name used in error messages.
        allowed_literals: Canonical ordered literal set.
    Returns:
        None.
    Assumptions:
        Allowed literals are already ordered for deterministic diagnostics.
    Raises:
        ValueError: If the candidate literal is not present in the allowed set.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/adapters/outbound/artifacts_fs/path_builder.py
    """
    if value not in allowed_literals:
        raise ValueError(f"artifact {field_name} must be one of {allowed_literals}; got {value!r}")


def _sorted_unique_timeframes_v2(
    *,
    values: tuple[str, ...],
    allowed_literals: tuple[str, ...],
    field_name: str,
    validator: Callable[[str], str],
) -> tuple[str, ...]:
    """
    Validate and deterministically order one timeframe tuple against the canonical contract.

    Args:
        values: Candidate timeframe tuple.
        allowed_literals: Canonical ordered timeframe literals for this scope.
        field_name: Human-readable field name used in error messages.
        validator: Scope-specific timeframe validator callable.
    Returns:
        tuple[str, ...]: Deterministically ordered unique timeframe tuple.
    Assumptions:
        Validation plans must use canonical ordering even if callers provide arbitrary tuples.
    Raises:
        ValueError: If one timeframe is invalid or appears more than once.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    seen: set[str] = set()
    validated_values: list[str] = []
    for raw_value in values:
        validated_value = validator(raw_value)
        if validated_value in seen:
            raise ValueError(
                f"artifact validation field '{field_name}' contains duplicate "
                f"{validated_value!r}"
            )
        seen.add(validated_value)
        validated_values.append(validated_value)
    ordered_values: list[str] = []
    for allowed_literal in allowed_literals:
        if allowed_literal in seen:
            ordered_values.append(allowed_literal)
    return tuple(ordered_values)


def _sorted_signal_validation_specs_v2(
    values: tuple[ArtifactSignalValidationSpecV2, ...],
) -> tuple[ArtifactSignalValidationSpecV2, ...]:
    """
    Validate and deterministically order explicit signal validation targets.

    Args:
        values: Candidate signal validation tuple.
    Returns:
        tuple[ArtifactSignalValidationSpecV2, ...]: Canonically ordered unique signal targets.
    Assumptions:
        Signal validation order is deterministic by timeframe contract and indicator id.
    Raises:
        ValueError: If one `(timeframe, indicator_id)` pair is duplicated.
    Side Effects:
        None.
    Docs:
      - docs/architecture/backtest/backtest-artifact-store-v2.md
      - docs/architecture/roadmap/base_refactor_plan.md
    Related:
      - src/trading/contexts/backtest/application/services/v2/artifact_slot_publisher.py
    """
    seen: set[tuple[str, str]] = set()
    validated_values: list[ArtifactSignalValidationSpecV2] = []
    for item in values:
        validated_item = ArtifactSignalValidationSpecV2(
            timeframe=item.timeframe,
            indicator_id=item.indicator_id,
        )
        identity = (validated_item.timeframe, validated_item.indicator_id)
        if identity in seen:
            raise ValueError(
                "artifact validation field 'signal_artifacts' contains duplicate "
                f"{identity!r}"
            )
        seen.add(identity)
        validated_values.append(validated_item)

    timeframe_order = {
        literal: index for index, literal in enumerate(ARTIFACT_SIGNAL_TIMEFRAMES_V2)
    }
    ordered_values = sorted(
        validated_values,
        key=lambda item: (timeframe_order[item.timeframe], item.indicator_id),
    )
    return tuple(ordered_values)
