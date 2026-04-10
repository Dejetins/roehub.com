from dataclasses import dataclass
from typing import Any
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from apps.api.common import register_api_error_handlers
from apps.api.dto import (
    BacktestRuntimeDefaultsResponse,
    build_backtest_runtime_defaults_response,
    build_sha256_from_payload,
)
from apps.api.routes import build_backtests_router
from trading.contexts.backtest.adapters.outbound import (
    BacktestExecutionRuntimeConfig,
    BacktestJobsRuntimeConfig,
    BacktestRankingRuntimeConfig,
    BacktestReportingRuntimeConfig,
    BacktestRuntimeConfig,
    BacktestSyncRuntimeConfig,
)
from trading.contexts.backtest.application.dto import (
    BacktestMetricRowV1,
    BacktestReportV1,
    BacktestVariantPayloadV1,
    BacktestVariantPreview,
    RunBacktestResponse,
)
from trading.contexts.backtest.application.ports import BacktestStrategySnapshot
from trading.contexts.backtest.domain.entities import TradeV1
from trading.contexts.backtest.domain.errors import BacktestValidationError
from trading.contexts.indicators.application.dto import IndicatorVariantSelection
from trading.contexts.indicators.domain.entities import IndicatorId
from trading.contexts.indicators.domain.specifications import ExplicitValuesSpec, GridSpec
from trading.platform.errors import RoehubError
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    PaidLevel,
    Symbol,
    Timeframe,
    UserId,
)


class _HeaderCurrentUserDependency:
    """
    Request dependency resolving authenticated principal from `X-User-Id` header.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - tests/unit/apps/api/test_backtests_routes.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/identity/adapters/inbound/api/deps/current_user.py
    """

    def __call__(self, request: Request):
        """
        Resolve principal or raise deterministic HTTP 401 payload.

        Args:
            request: HTTP request object.
        Returns:
            object: CurrentUserPrincipal-compatible object.
        Assumptions:
            Header contains UUID string when provided.
        Raises:
            HTTPException: If authentication header is missing.
        Side Effects:
            None.
        """
        raw_user_id = request.headers.get("x-user-id")
        if raw_user_id is None:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "unauthorized",
                    "message": "Authentication required",
                },
            )

        from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal

        return CurrentUserPrincipal(
            user_id=UserId.from_string(raw_user_id),
            paid_level=PaidLevel.free(),
        )


@dataclass(frozen=True, slots=True)
class _StaticStrategyReader:
    """
    Minimal strategy reader fake returning one preconfigured snapshot.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - tests/unit/apps/api/test_backtests_routes.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/application/ports/strategy_reader.py
    """

    snapshot: BacktestStrategySnapshot | None = None

    def load_any(self, *, strategy_id: UUID) -> BacktestStrategySnapshot | None:
        """
        Return preconfigured snapshot independent from requested strategy id.

        Args:
            strategy_id: Requested strategy identifier.
        Returns:
            BacktestStrategySnapshot | None: Configured snapshot value.
        Assumptions:
            Tests verify route behavior, not repository lookup semantics.
        Raises:
            None.
        Side Effects:
            None.
        """
        _ = strategy_id
        return self.snapshot


class _RuntimeDefaultsProvider:
    """
    Minimal defaults-provider fake exposing deterministic launch catalog values.
    """

    def compute_defaults(self, *, indicator_id: str) -> GridSpec | None:
        """
        Return compute defaults for supported indicators used in route tests.

        Args:
            indicator_id: Requested indicator identifier.
        Returns:
            GridSpec | None: Minimal compute defaults payload for supported indicators.
        Assumptions:
            Runtime-defaults route currently needs only supported ids and source catalogs.
        Raises:
            None.
        Side Effects:
            None.
        """
        normalized_id = indicator_id.strip().lower()
        if normalized_id == "ma.sma":
            return GridSpec(
                indicator_id=IndicatorId("ma.sma"),
                source=ExplicitValuesSpec(name="source", values=("close", "hlc3")),
                params={"window": ExplicitValuesSpec(name="window", values=(20,))},
            )
        if normalized_id == "momentum.trix":
            return GridSpec(
                indicator_id=IndicatorId("momentum.trix"),
                source=ExplicitValuesSpec(name="source", values=("close", "hlc3", "ohlc4")),
                params={
                    "signal_window": ExplicitValuesSpec(name="signal_window", values=(9,)),
                    "window": ExplicitValuesSpec(name="window", values=(15,)),
                },
            )
        if normalized_id == "volume.obv":
            return GridSpec(
                indicator_id=IndicatorId("volume.obv"),
                params={},
            )
        return None

    def signal_param_defaults(self, *, indicator_id: str) -> dict[str, ExplicitValuesSpec]:
        """
        Return deterministic signal defaults mapping for supported indicators.

        Args:
            indicator_id: Requested indicator identifier.
        Returns:
            dict[str, ExplicitValuesSpec]: Signal defaults mapping or empty mapping.
        Assumptions:
            Route tests do not inspect signal defaults directly.
        Raises:
            None.
        Side Effects:
            None.
        """
        if indicator_id.strip().lower() != "ma.sma":
            return {}
        return {"cross_up": ExplicitValuesSpec(name="cross_up", values=(0.5,))}

    def supported_indicator_ids(self) -> tuple[str, ...]:
        """
        Return deterministic supported indicator ids for runtime-defaults contract tests.

        Args:
            None.
        Returns:
            tuple[str, ...]: Sorted supported indicator ids.
        Assumptions:
            Launch catalog must be stable across repeated calls.
        Raises:
            None.
        Side Effects:
            None.
        """
        return ("ma.sma", "momentum.trix", "volume.obv")

    def allowed_source_values(self, *, indicator_id: str) -> tuple[str, ...]:
        """
        Return deterministic allowed `inputs.source` catalog for one indicator id.

        Args:
            indicator_id: Requested indicator identifier.
        Returns:
            tuple[str, ...]: Stable source literal tuple, empty when not applicable.
        Assumptions:
            Values are already sorted in contract order.
        Raises:
            None.
        Side Effects:
            None.
        """
        normalized_id = indicator_id.strip().lower()
        if normalized_id == "ma.sma":
            return ("close", "hlc3")
        if normalized_id == "momentum.trix":
            return ("close", "hlc3", "ohlc4")
        return ()


class _FakeRunBacktestUseCase:
    """
    Minimal use-case fake returning preconfigured result or raising configured error.

    Docs:
      - docs/architecture/backtest/backtest-api-post-backtests-v1.md
    Related:
      - tests/unit/apps/api/test_backtests_routes.py
      - apps/api/routes/backtests.py
      - src/trading/contexts/backtest/application/use_cases/run_backtest.py
    """

    def __init__(
        self,
        *,
        result: Any = None,
        error: Exception | None = None,
        variant_report_result: BacktestReportV1 | None = None,
        variant_report_error: Exception | None = None,
    ) -> None:
        """
        Store deterministic fake behavior for endpoint tests.

        Args:
            result: Value returned by execute when no error is configured.
            error: Optional exception raised by execute.
            variant_report_result: Optional payload returned by variant-report call.
            variant_report_error: Optional exception raised by variant-report call.
        Returns:
            None.
        Assumptions:
            Endpoint tests inspect only routing/mapping/error behavior.
        Raises:
            None.
        Side Effects:
            None.
        """
        self._result = result
        self._error = error
        self._variant_report_result = variant_report_result
        self._variant_report_error = variant_report_error

    def execute(self, *, request, current_user, request_payload=None, run_control=None):
        """
        Return configured result or raise configured error.

        Args:
            request: Application request DTO.
            current_user: Current user port object.
            request_payload: Optional strict API payload snapshot for persisted sync tests.
            run_control: Optional cooperative cancellation/deadline control object.
        Returns:
            Any: Configured result payload.
        Assumptions:
            Request/current_user are ignored in route-contract unit tests.
        Raises:
            Exception: Configured exception.
        Side Effects:
            None.
        """
        _ = request, current_user, request_payload, run_control
        if self._error is not None:
            raise self._error
        return self._result

    def build_variant_report(
        self,
        *,
        request,
        current_user,
        variant_payload,
        include_trades,
        run_control=None,
    ) -> BacktestReportV1:
        """
        Return configured variant report payload or raise configured variant-report error.

        Args:
            request: Application request DTO.
            current_user: Current user port object.
            variant_payload: Explicit variant payload DTO.
            include_trades: Include-trades flag from endpoint request.
            run_control: Optional cooperative cancellation/deadline control object.
        Returns:
            BacktestReportV1: Configured report payload.
        Assumptions:
            Endpoint tests verify route wiring and DTO mapping only.
        Raises:
            Exception: Configured variant-report exception.
        Side Effects:
            None.
        """
        _ = request, current_user, variant_payload, include_trades, run_control
        if self._variant_report_error is not None:
            raise self._variant_report_error
        if self._variant_report_result is None:
            raise AssertionError("variant_report_result is not configured")
        return self._variant_report_result


def _build_client(
    *,
    use_case: _FakeRunBacktestUseCase,
    strategy_reader: _StaticStrategyReader | None = None,
    runtime_defaults_response: BacktestRuntimeDefaultsResponse | None = None,
    eager_top_reports_enabled: bool = False,
) -> TestClient:
    """
    Build minimal FastAPI TestClient with backtests router and shared error handlers.

    Args:
        use_case: Fake use-case used by endpoint handler.
        strategy_reader: Optional strategy reader fake.
        runtime_defaults_response: Optional runtime defaults payload for GET endpoint tests.
        eager_top_reports_enabled: Feature flag toggling eager report payload in sync response.
    Returns:
        TestClient: Configured client instance.
    Assumptions:
        Shared API error handlers provide deterministic Roehub/422 payloads.
    Raises:
        ValueError: If router dependencies are invalid.
    Side Effects:
        None.
    """
    app = FastAPI()
    register_api_error_handlers(app=app)
    app.include_router(
        build_backtests_router(
            run_use_case=use_case,  # type: ignore[arg-type]
            strategy_reader=strategy_reader or _StaticStrategyReader(),
            runtime_defaults_response=runtime_defaults_response
            or _runtime_defaults_response(),
            current_user_dependency=_HeaderCurrentUserDependency(),
            sync_deadline_seconds=55.0,
            eager_top_reports_enabled=eager_top_reports_enabled,
        )
    )
    return TestClient(app)


def _runtime_defaults_response() -> BacktestRuntimeDefaultsResponse:
    """
    Build deterministic runtime defaults fixture matching `/backtests/runtime-defaults` contract.

    Args:
        None.
    Returns:
        BacktestRuntimeDefaultsResponse: Runtime defaults response fixture.
    Assumptions:
        Fee defaults input order is intentionally unsorted to verify deterministic ordering.
    Raises:
        ValueError: If fixture violates runtime config invariants.
    Side Effects:
        None.
    """
    return build_backtest_runtime_defaults_response(
        config=BacktestRuntimeConfig(
            version=1,
            warmup_bars_default=200,
            top_k_default=300,
            preselect_default=20000,
            ranking=BacktestRankingRuntimeConfig(
                primary_metric_default="total_return_pct",
                secondary_metric_default=None,
            ),
            sync=BacktestSyncRuntimeConfig(sync_deadline_seconds=55.0),
            reporting=BacktestReportingRuntimeConfig(top_trades_n_default=3),
            execution=BacktestExecutionRuntimeConfig(
                init_cash_quote_default=10000.0,
                fixed_quote_default=100.0,
                safe_profit_percent_default=30.0,
                slippage_pct_default=0.01,
                fee_pct_default_by_market_id={
                    4: 0.1,
                    2: 0.1,
                    1: 0.075,
                    3: 0.075,
                },
            ),
            jobs=BacktestJobsRuntimeConfig(
                enabled=True,
                top_k_persisted_default=300,
                max_active_jobs_per_user=3,
                claim_poll_seconds=1.0,
                lease_seconds=60,
                heartbeat_seconds=15,
                worker_processes=1,
                snapshot_seconds=30,
                snapshot_variants_step=1000,
            ),
        ),
        defaults_provider=_RuntimeDefaultsProvider(),
    )


def test_get_backtests_runtime_defaults_returns_deterministic_payload() -> None:
    """
    Verify runtime defaults endpoint returns stable shape and deterministic fee-map ordering.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Response keys should be stable across repeated calls for browser prefill logic.
    Raises:
        AssertionError: If shape, values, or ordering deviate from endpoint contract.
    Side Effects:
        None.
    """
    client = _build_client(use_case=_FakeRunBacktestUseCase(result=_template_mode_response()))

    response_one = client.get(
        "/backtests/runtime-defaults",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000777"},
    )
    response_two = client.get(
        "/backtests/runtime-defaults",
        headers={"x-user-id": "00000000-0000-0000-0000-000000000777"},
    )

    assert response_one.status_code == 200
    assert response_one.json() == response_two.json()
    assert response_one.json() == {
        "top_k_default": 300,
        "preselect_default": 20000,
        "ranking": {
            "primary_metric_default": "total_return_pct",
        },
        "execution": {
            "init_cash_quote_default": 10000.0,
            "fixed_quote_default": 100.0,
            "safe_profit_percent_default": 30.0,
            "slippage_pct_default": 0.01,
            "fee_pct_default_by_market_id": {
                "1": 0.075,
                "2": 0.1,
                "3": 0.075,
                "4": 0.1,
            },
        },
        "jobs": {
            "top_k_persisted_default": 300,
        },
        "contracts": {
            "request_timeframes": {
                "allowed": [
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
                ],
                "forbidden": ["1m", "5m"],
            },
            "summary": {
                "top_n_default": 100,
                "top_n_max": 300,
                "ranking_metrics": [
                    "total_return_pct",
                    "max_drawdown_pct",
                    "return_over_max_drawdown",
                    "profit_factor",
                    "sharpe_trades",
                    "win_rate_pct",
                ],
                "sortable_columns": [
                    "total_return_pct",
                    "max_drawdown_pct",
                    "return_over_max_drawdown",
                    "profit_factor",
                    "sharpe_trades",
                    "win_rate_pct",
                    "trade_count",
                    "avg_trade_ret_pct",
                    "avg_trade_exec_bars",
                    "exposure_pct",
                    "best_tp_pct",
                    "best_sl_pct",
                ],
            },
            "signals": {
                "params_path": "signals.v1.params",
                "params_policy": "default-only",
            },
            "execution": {
                "risk_model": "signal_tf + 1m_risk",
                "default_execution_profile": "exact_small",
                "available_execution_profiles": [
                    {
                        "mode": "exact_small",
                        "shortlist_config": {
                            "enabled": False,
                            "max_candidates": None,
                            "scoring": {
                                "activity_ratio_weight": 0.4,
                                "direction_balance_weight": 0.25,
                                "transition_ratio_weight": 0.25,
                                "active_span_ratio_weight": 0.1,
                            },
                            "retention": {
                                "diversity_buckets": [
                                    "activity_band",
                                    "direction_band",
                                ],
                                "max_per_bucket": None,
                            },
                        },
                        "parallelism": {
                            "stage_a_workers": 1,
                            "stage_b_workers": 1,
                        },
                        "feature_flags": {
                            "runtime_enabled": True,
                            "heuristic_shortlist_enabled": False,
                            "parallel_stage_b_enabled": False,
                            "family_plugin_enabled": False,
                        },
                        "launch_budget": {
                            "max_stage_a_variants_total": 1500,
                            "max_stage_b_variants_total": 12000,
                            "max_estimated_memory_bytes": 268435456,
                        },
                        "progress_weights": {
                            "stage_a": 25,
                            "stage_b": 70,
                            "finalizing": 5,
                        },
                        "family_plugin_budget_ms": 10,
                        "planning_budget_ms": 25,
                    },
                    {
                        "mode": "exact_parallel",
                        "shortlist_config": {
                            "enabled": False,
                            "max_candidates": None,
                            "scoring": {
                                "activity_ratio_weight": 0.4,
                                "direction_balance_weight": 0.25,
                                "transition_ratio_weight": 0.25,
                                "active_span_ratio_weight": 0.1,
                            },
                            "retention": {
                                "diversity_buckets": [
                                    "activity_band",
                                    "direction_band",
                                ],
                                "max_per_bucket": None,
                            },
                        },
                        "parallelism": {
                            "stage_a_workers": 1,
                            "stage_b_workers": 4,
                        },
                        "feature_flags": {
                            "runtime_enabled": True,
                            "heuristic_shortlist_enabled": False,
                            "parallel_stage_b_enabled": True,
                            "family_plugin_enabled": False,
                        },
                        "launch_budget": {
                            "max_stage_a_variants_total": 25000,
                            "max_stage_b_variants_total": 180000,
                            "max_estimated_memory_bytes": 1610612736,
                        },
                        "progress_weights": {
                            "stage_a": 35,
                            "stage_b": 60,
                            "finalizing": 5,
                        },
                        "family_plugin_budget_ms": 20,
                        "planning_budget_ms": 50,
                    },
                    {
                        "mode": "hybrid_conservative",
                        "shortlist_config": {
                            "enabled": True,
                            "max_candidates": 5000,
                            "scoring": {
                                "activity_ratio_weight": 0.4,
                                "direction_balance_weight": 0.25,
                                "transition_ratio_weight": 0.25,
                                "active_span_ratio_weight": 0.1,
                            },
                            "retention": {
                                "diversity_buckets": [
                                    "activity_band",
                                    "direction_band",
                                ],
                                "max_per_bucket": 750,
                            },
                        },
                        "parallelism": {
                            "stage_a_workers": 1,
                            "stage_b_workers": 4,
                        },
                        "feature_flags": {
                            "runtime_enabled": False,
                            "heuristic_shortlist_enabled": False,
                            "parallel_stage_b_enabled": False,
                            "family_plugin_enabled": False,
                        },
                        "launch_budget": {
                            "max_stage_a_variants_total": 50000,
                            "max_stage_b_variants_total": 250000,
                            "max_estimated_memory_bytes": 2147483648,
                        },
                        "progress_weights": {
                            "stage_a": 50,
                            "stage_b": 45,
                            "finalizing": 5,
                        },
                        "family_plugin_budget_ms": 30,
                        "planning_budget_ms": 75,
                    },
                    {
                        "mode": "hybrid_family",
                        "shortlist_config": {
                            "enabled": True,
                            "max_candidates": 2000,
                            "scoring": {
                                "activity_ratio_weight": 0.35,
                                "direction_balance_weight": 0.2,
                                "transition_ratio_weight": 0.3,
                                "active_span_ratio_weight": 0.15,
                            },
                            "retention": {
                                "diversity_buckets": [
                                    "activity_band",
                                    "transition_band",
                                ],
                                "max_per_bucket": 300,
                            },
                        },
                        "parallelism": {
                            "stage_a_workers": 1,
                            "stage_b_workers": 4,
                        },
                        "feature_flags": {
                            "runtime_enabled": False,
                            "heuristic_shortlist_enabled": False,
                            "parallel_stage_b_enabled": False,
                            "family_plugin_enabled": False,
                        },
                        "launch_budget": {
                            "max_stage_a_variants_total": 75000,
                            "max_stage_b_variants_total": 300000,
                            "max_estimated_memory_bytes": 2684354560,
                        },
                        "progress_weights": {
                            "stage_a": 60,
                            "stage_b": 35,
                            "finalizing": 5,
                        },
                        "family_plugin_budget_ms": 40,
                        "planning_budget_ms": 100,
                    },
                ],
                "adaptive_selector": {
                    "mode": "disabled",
                    "hybrid_conservative": {
                        "rollout_mode": "active",
                        "min_grid_cardinality": 6000,
                        "min_stage_a_variants_total": 6000,
                        "min_stage_b_variants_total": 40000,
                        "min_estimated_memory_bytes": 805306368,
                        "minimum_exceeded_signals": 3,
                    },
                    "hybrid_family": {
                        "rollout_mode": "active",
                        "min_grid_cardinality": 12000,
                        "min_stage_a_variants_total": 12000,
                        "min_stage_b_variants_total": 80000,
                        "min_estimated_memory_bytes": 1073741824,
                        "minimum_exceeded_signals": 3,
                    },
                },
            },
            "launch": {
                "execution_mode": "auto",
                "auto_preflight_enabled": True,
                "auto_fallback_to_background_enabled": True,
                "supported_indicator_ids": ["ma.sma", "momentum.trix", "volume.obv"],
                "source_values_by_indicator_id": {
                    "ma.sma": ["close", "hlc3"],
                    "momentum.trix": ["close", "hlc3", "ohlc4"],
                    "volume.obv": [],
                },
            },
        },
    }
    assert "secondary_metric_default" not in response_one.json()["ranking"]
    assert list(
        response_one.json()["execution"]["fee_pct_default_by_market_id"].keys()
    ) == ["1", "2", "3", "4"]
    assert [
        profile["mode"]
        for profile in response_one.json()["contracts"]["execution"][
            "available_execution_profiles"
        ]
    ] == [
        "exact_small",
        "exact_parallel",
        "hybrid_conservative",
        "hybrid_family",
    ]


def test_get_backtests_runtime_defaults_returns_401_when_unauthenticated() -> None:
    """
    Verify runtime defaults endpoint remains protected by shared current-user dependency.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Backtests API module keeps auth behavior consistent across routes.
    Raises:
        AssertionError: If endpoint does not return deterministic unauthorized payload.
    Side Effects:
        None.
    """
    client = _build_client(use_case=_FakeRunBacktestUseCase(result=_template_mode_response()))

    response = client.get("/backtests/runtime-defaults")

    assert response.status_code == 401
    assert response.json() == {
        "detail": {
            "error": "unauthorized",
            "message": "Authentication required",
        }
    }


def test_post_backtests_forbids_extra_fields_with_deterministic_422_payload() -> None:
    """
    Verify strict request DTO rejects extra fields with deterministic validation payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Pydantic `extra=forbid` is enabled for `BacktestsPostRequest` models.
    Raises:
        AssertionError: If payload or status code deviates from deterministic contract.
    Side Effects:
        None.
    """
    client = _build_client(use_case=_FakeRunBacktestUseCase(result=_template_mode_response()))

    payload = {
        "time_range": {
            "start": "2026-02-16T00:00:00Z",
            "end": "2026-02-16T01:00:00Z",
        },
        "template": _template_payload(),
        "unexpected_field": "boom",
    }
    response = client.post(
        "/backtests",
        json=payload,
        headers={"x-user-id": "00000000-0000-0000-0000-000000000777"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "validation_error",
            "message": "Validation failed",
            "details": {
                "errors": [
                    {
                        "path": "body.unexpected_field",
                        "code": "extra_forbidden",
                        "message": "Extra inputs are not permitted",
                    }
                ]
            },
        }
    }


def test_post_backtests_rejects_mode_conflict_with_deterministic_validation_error() -> None:
    """
    Verify route returns deterministic `validation_error` for `strategy_id xor template` conflict.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Mode exclusivity is validated by API->application request mapper.
    Raises:
        AssertionError: If payload or status code deviates from deterministic contract.
    Side Effects:
        None.
    """
    client = _build_client(use_case=_FakeRunBacktestUseCase(result=_template_mode_response()))

    payload = {
        "time_range": {
            "start": "2026-02-16T00:00:00Z",
            "end": "2026-02-16T01:00:00Z",
        },
        "strategy_id": "00000000-0000-0000-0000-000000000123",
        "template": _template_payload(),
    }
    response = client.post(
        "/backtests",
        json=payload,
        headers={"x-user-id": "00000000-0000-0000-0000-000000000777"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "validation_error",
            "message": "POST /backtests requires exactly one mode: strategy_id xor template",
            "details": {
                "errors": [
                    {
                        "path": "body.strategy_id",
                        "code": "mode_conflict",
                        "message": "Provide exactly one of strategy_id or template",
                    },
                    {
                        "path": "body.template",
                        "code": "mode_conflict",
                        "message": "Provide exactly one of strategy_id or template",
                    },
                ]
            },
        }
    }


def test_post_backtests_returns_401_when_unauthenticated() -> None:
    """
    Verify endpoint is protected by identity dependency and returns deterministic 401 payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Authentication dependency requires `X-User-Id` header in route tests.
    Raises:
        AssertionError: If status code or payload deviates from expected unauthorized contract.
    Side Effects:
        None.
    """
    client = _build_client(use_case=_FakeRunBacktestUseCase(result=_template_mode_response()))

    response = client.post(
        "/backtests",
        json={
            "time_range": {
                "start": "2026-02-16T00:00:00Z",
                "end": "2026-02-16T01:00:00Z",
            },
            "template": _template_payload(),
        },
    )

    assert response.status_code == 401
    assert response.json() == {
        "detail": {
            "error": "unauthorized",
            "message": "Authentication required",
        }
    }


def test_post_backtests_maps_saved_mode_forbidden_error() -> None:
    """
    Verify saved-mode ownership failure is mapped to deterministic `forbidden` payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Backtest use-case is source of truth for ownership policy.
    Raises:
        AssertionError: If status code or payload differs from Roehub contract.
    Side Effects:
        None.
    """
    client = _build_client(
        use_case=_FakeRunBacktestUseCase(
            error=RoehubError(
                code="forbidden",
                message="Backtest strategy does not belong to current user",
                details={"strategy_id": "00000000-0000-0000-0000-000000000123"},
            )
        )
    )

    response = client.post(
        "/backtests",
        json=_saved_mode_payload(),
        headers={"x-user-id": "00000000-0000-0000-0000-000000000777"},
    )

    assert response.status_code == 403
    assert response.json() == {
        "error": {
            "code": "forbidden",
            "message": "Backtest strategy does not belong to current user",
            "details": {"strategy_id": "00000000-0000-0000-0000-000000000123"},
        }
    }


def test_post_backtests_maps_saved_mode_not_found_error() -> None:
    """
    Verify saved-mode missing strategy is mapped to deterministic `not_found` payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Backtest use-case emits Roehub `not_found` for missing/deleted strategy.
    Raises:
        AssertionError: If status code or payload differs from Roehub contract.
    Side Effects:
        None.
    """
    client = _build_client(
        use_case=_FakeRunBacktestUseCase(
            error=RoehubError(
                code="not_found",
                message="Backtest strategy was not found",
                details={"strategy_id": "00000000-0000-0000-0000-000000000123"},
            )
        )
    )

    response = client.post(
        "/backtests",
        json=_saved_mode_payload(),
        headers={"x-user-id": "00000000-0000-0000-0000-000000000777"},
    )

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "code": "not_found",
            "message": "Backtest strategy was not found",
            "details": {"strategy_id": "00000000-0000-0000-0000-000000000123"},
        }
    }


def test_post_backtests_maps_runtime_contract_validation_error() -> None:
    """
    Verify sync route maps R1 runtime-contract violations to canonical 422 payload.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Backtest use-case raises `BacktestValidationError` with deterministic items.
    Raises:
        AssertionError: If status code or payload deviates from canonical validation contract.
    Side Effects:
        None.
    """
    client = _build_client(
        use_case=_FakeRunBacktestUseCase(
            error=BacktestValidationError(
                "Backtest request violates runtime defaults contract",
                errors=(
                    {
                        "path": "body.template.timeframe",
                        "code": "unsupported_value",
                        "message": "timeframe must be one of: 15m, 30m, 1h",
                    },
                ),
            )
        )
    )

    response = client.post(
        "/backtests",
        json={
            "time_range": {
                "start": "2026-02-16T00:00:00Z",
                "end": "2026-02-16T01:00:00Z",
            },
            "template": _template_payload(),
        },
        headers={"x-user-id": "00000000-0000-0000-0000-000000000777"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "validation_error",
            "message": "Backtest request violates runtime defaults contract",
            "details": {
                "errors": [
                    {
                        "path": "body.template.timeframe",
                        "code": "unsupported_value",
                        "message": "timeframe must be one of: 15m, 30m, 1h",
                    }
                ]
            },
        }
    }


def test_post_backtests_saved_response_includes_hashes_and_explicit_payload() -> None:
    """
    Verify saved-mode response includes `spec_hash`, `engine_params_hash`, and payload blocks.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Route computes hashes from canonical JSON payloads.
    Raises:
        AssertionError: If payload misses required fields or hashes.
    Side Effects:
        None.
    """
    snapshot_payload = {
        "schema_version": 1,
        "instrument_id": {"market_id": 1, "symbol": "BTCUSDT"},
        "timeframe": "1m",
        "indicators": [
            {"id": "ma.sma", "inputs": {"source": "close"}, "params": {"window": 20}}
        ],
    }
    strategy_snapshot = BacktestStrategySnapshot(
        strategy_id=UUID("00000000-0000-0000-0000-000000000123"),
        user_id=UserId.from_string("00000000-0000-0000-0000-000000000777"),
        is_deleted=False,
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        indicator_grids=(
            GridSpec(
                indicator_id=IndicatorId("ma.sma"),
                params={"window": ExplicitValuesSpec(name="window", values=(20,))},
            ),
        ),
        indicator_selections=(
            IndicatorVariantSelection(
                indicator_id="ma.sma",
                inputs={"source": "close"},
                params={"window": 20},
            ),
        ),
        spec_payload=snapshot_payload,
    )

    client = _build_client(
        use_case=_FakeRunBacktestUseCase(result=_saved_mode_response()),
        strategy_reader=_StaticStrategyReader(snapshot=strategy_snapshot),
    )

    response = client.post(
        "/backtests",
        json=_saved_mode_payload(),
        headers={"x-user-id": "00000000-0000-0000-0000-000000000777"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["run_id"] == "00000000-0000-0000-0000-000000000910"
    assert body["state"] == "succeeded"
    assert body["execution_mode"] == "sync_inline"
    assert body["engine_version"] == "signal_tf + 1m_risk"
    assert body["artifact_slot"] == "slot_b"
    assert body["artifact_slot_generation"] == 11
    assert body["artifact_asof_date"] == "2026-03-28"
    assert body["artifact_manifest_hash"] == "c" * 64
    assert body["spec_hash"] == build_sha256_from_payload(payload=snapshot_payload)
    assert body["grid_request_hash"] is None
    assert body["engine_params_hash"] == "e" * 64
    assert "warmup_bars" not in body
    assert body["variants"][0]["payload"] == {
        "indicator_selections": [
            {
                "indicator_id": "ma.sma",
                "inputs": {"source": "close"},
                "params": {"window": 20},
            }
        ],
        "signal_params": {"ma.sma": {"cross_up": 0.5}},
        "risk_params": {
            "sl_enabled": True,
            "sl_pct": 2.0,
            "tp_enabled": True,
            "tp_pct": 4.0,
        },
        "execution_params": {
            "fee_pct": 0.075,
            "fixed_quote": 100.0,
            "init_cash_quote": 10000.0,
            "safe_profit_percent": 30.0,
            "slippage_pct": 0.01,
        },
        "direction_mode": "long-short",
        "sizing_mode": "all_in",
    }


def test_post_backtests_returns_202_for_explicit_background_auto_launch() -> None:
    """
    Verify `/backtests` exposes queued auto-fallback launch via explicit `202 Accepted`.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Background auto launch does not include inline-ranked variants in the response body.
    Raises:
        AssertionError: If route returns `200` or hides explicit `background_auto` metadata.
    Side Effects:
        None.
    """
    client = _build_client(
        use_case=_FakeRunBacktestUseCase(result=_template_background_auto_response())
    )

    response = client.post(
        "/backtests",
        json={
            "time_range": {
                "start": "2026-02-16T00:00:00Z",
                "end": "2026-02-16T01:00:00Z",
            },
            "template": _template_payload(),
        },
        headers={"x-user-id": "00000000-0000-0000-0000-000000000777"},
    )

    assert response.status_code == 202
    body = response.json()
    assert body["run_id"] == "00000000-0000-0000-0000-000000000911"
    assert body["state"] == "queued"
    assert body["execution_mode"] == "background_auto"
    assert body["engine_version"] == "signal_tf + 1m_risk"
    assert body["artifact_slot"] == "slot_b"
    assert body["artifact_slot_generation"] == 11
    assert body["artifact_asof_date"] == "2026-03-28"
    assert body["artifact_manifest_hash"] == "c" * 64
    assert body["grid_request_hash"] is not None
    assert body["engine_params_hash"] == "e" * 64
    assert "warmup_bars" not in body
    assert body["variants"] == []


def test_post_backtests_preserves_application_variant_order() -> None:
    """
    Verify route preserves deterministic variant ordering provided by application DTO.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Application response DTO already enforces deterministic ranking/tie-break ordering.
    Raises:
        AssertionError: If response order breaks deterministic tie-break contract.
    Side Effects:
        None.
    """
    client = _build_client(
        use_case=_FakeRunBacktestUseCase(result=_template_mode_two_variant_response())
    )

    response = client.post(
        "/backtests",
        json={
            "time_range": {
                "start": "2026-02-16T00:00:00Z",
                "end": "2026-02-16T01:00:00Z",
            },
            "template": _template_payload(),
        },
        headers={"x-user-id": "00000000-0000-0000-0000-000000000777"},
    )

    assert response.status_code == 200
    variant_keys = [item["variant_key"] for item in response.json()["variants"]]
    assert variant_keys == ["a" * 64, "b" * 64]


def test_post_backtests_lazy_mode_hides_eager_report_payloads() -> None:
    """
    Verify lazy sync policy strips variant report bodies when eager flag is disabled.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Route enforces lazy payload policy even if use-case returns report blocks.
    Raises:
        AssertionError: If variant `report` is present in lazy mode response.
    Side Effects:
        None.
    """
    client = _build_client(use_case=_FakeRunBacktestUseCase(result=_template_mode_response()))

    response = client.post(
        "/backtests",
        json={
            "time_range": {
                "start": "2026-02-16T00:00:00Z",
                "end": "2026-02-16T01:00:00Z",
            },
            "template": _template_payload(),
        },
        headers={"x-user-id": "00000000-0000-0000-0000-000000000777"},
    )

    assert response.status_code == 200
    assert response.json()["variants"][0]["report"] is None
    assert response.json()["variants"][0]["total_return_pct"] == 12.0


def test_post_backtests_eager_flag_does_not_break_summary_only_launch() -> None:
    """
    Verify legacy eager flag no longer overrides the summary-only launch contract.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Summary rows stay report-free even when old eager wiring remains enabled.
    Raises:
        AssertionError: If sync launch still materializes report bodies.
    Side Effects:
        None.
    """
    client = _build_client(
        use_case=_FakeRunBacktestUseCase(result=_template_mode_response()),
        eager_top_reports_enabled=True,
    )

    response = client.post(
        "/backtests",
        json={
            "time_range": {
                "start": "2026-02-16T00:00:00Z",
                "end": "2026-02-16T01:00:00Z",
            },
            "template": _template_payload(),
        },
        headers={"x-user-id": "00000000-0000-0000-0000-000000000777"},
    )

    assert response.status_code == 200
    assert response.json()["variants"][0]["report"] is None


def test_post_backtests_variant_report_returns_rows_table_and_trades() -> None:
    """
    Verify variant-report endpoint returns strict report payload shape for one variant.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Endpoint payload contains explicit run-context and selected variant payload.
    Raises:
        AssertionError: If status code or response shape deviates from contract.
    Side Effects:
        None.
    """
    client = _build_client(
        use_case=_FakeRunBacktestUseCase(variant_report_result=_variant_report_response())
    )
    response = client.post(
        "/backtests/variant-report",
        json=_variant_report_payload(),
        headers={"x-user-id": "00000000-0000-0000-0000-000000000777"},
    )

    assert response.status_code == 200
    assert response.json()["rows"] == [{"metric": "Total Return [%]", "value": "12.00"}]
    assert response.json()["table_md"].startswith("|Metric|Value|")
    assert response.json()["trades"] is not None
    assert response.json()["trades"][0]["trade_id"] == 1


def test_post_backtests_variant_report_rejects_mode_conflict() -> None:
    """
    Verify variant-report endpoint reuses deterministic mode-conflict validation contract.

    Args:
        None.
    Returns:
        None.
    Assumptions:
        Mode selection remains `strategy_id xor template`.
    Raises:
        AssertionError: If endpoint does not return deterministic validation_error payload.
    Side Effects:
        None.
    """
    payload = _variant_report_payload()
    payload["strategy_id"] = "00000000-0000-0000-0000-000000000123"

    client = _build_client(
        use_case=_FakeRunBacktestUseCase(variant_report_result=_variant_report_response())
    )
    response = client.post(
        "/backtests/variant-report",
        json=payload,
        headers={"x-user-id": "00000000-0000-0000-0000-000000000777"},
    )

    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "code": "validation_error",
            "message": "POST /backtests requires exactly one mode: strategy_id xor template",
            "details": {
                "errors": [
                    {
                        "path": "body.strategy_id",
                        "code": "mode_conflict",
                        "message": "Provide exactly one of strategy_id or template",
                    },
                    {
                        "path": "body.template",
                        "code": "mode_conflict",
                        "message": "Provide exactly one of strategy_id or template",
                    },
                ]
            },
        }
    }


def _template_payload() -> dict[str, Any]:
    """
    Build minimal valid ad-hoc template payload for API route tests.

    Args:
        None.
    Returns:
        dict[str, Any]: Template request JSON payload.
    Assumptions:
        One indicator grid is sufficient for endpoint contract tests.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "instrument_id": {
            "market_id": 1,
            "symbol": "BTCUSDT",
        },
        "timeframe": "1m",
        "indicator_grids": [
            {
                "indicator_id": "ma.sma",
                "params": {
                    "window": {"mode": "explicit", "values": [20]},
                },
            }
        ],
        "signal_grids": {
            "ma.sma": {
                "cross_up": {"mode": "explicit", "values": [0.5]},
            }
        },
        "risk_grid": {
            "sl_enabled": True,
            "tp_enabled": True,
            "sl": {"mode": "explicit", "values": [2.0]},
            "tp": {"mode": "explicit", "values": [4.0]},
        },
        "execution": {
            "init_cash_quote": 10000,
            "fee_pct": 0.075,
            "slippage_pct": 0.01,
            "fixed_quote": 100,
            "safe_profit_percent": 30,
        },
    }


def _saved_mode_payload() -> dict[str, Any]:
    """
    Build minimal valid saved-mode payload for route tests.

    Args:
        None.
    Returns:
        dict[str, Any]: Saved-mode request JSON payload.
    Assumptions:
        `strategy_id` mode is enough for mapping/error tests.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "time_range": {
            "start": "2026-02-16T00:00:00Z",
            "end": "2026-02-16T01:00:00Z",
        },
        "strategy_id": "00000000-0000-0000-0000-000000000123",
    }


def _variant_report_payload() -> dict[str, Any]:
    """
    Build deterministic valid payload for `POST /backtests/variant-report` endpoint tests.

    Args:
        None.
    Returns:
        dict[str, Any]: Variant-report request JSON payload.
    Assumptions:
        Payload uses template mode and explicit selected variant from top list.
    Raises:
        None.
    Side Effects:
        None.
    """
    return {
        "time_range": {
            "start": "2026-02-16T00:00:00Z",
            "end": "2026-02-16T01:00:00Z",
        },
        "template": _template_payload(),
        "include_trades": True,
        "variant": {
            "indicator_selections": [
                {
                    "indicator_id": "ma.sma",
                    "inputs": {"source": "close"},
                    "params": {"window": 20},
                }
            ],
            "signal_params": {"ma.sma": {"cross_up": 0.5}},
            "risk_params": {
                "sl_enabled": True,
                "sl_pct": 2.0,
                "tp_enabled": True,
                "tp_pct": 4.0,
            },
            "execution_params": {
                "init_cash_quote": 10000.0,
                "fee_pct": 0.075,
                "slippage_pct": 0.01,
                "fixed_quote": 100.0,
                "safe_profit_percent": 30.0,
            },
            "direction_mode": "long-short",
            "sizing_mode": "all_in",
        },
    }


def _variant_report_response() -> BacktestReportV1:
    """
    Build deterministic report payload fixture for variant-report endpoint tests.

    Args:
        None.
    Returns:
        BacktestReportV1: Report fixture with rows, markdown table, and one trade.
    Assumptions:
        One trade item is enough to validate strict response serialization.
    Raises:
        ValueError: If fixture violates domain/entity invariants.
    Side Effects:
        None.
    """
    return BacktestReportV1(
        rows=(BacktestMetricRowV1(metric="Total Return [%]", value="12.00"),),
        table_md="|Metric|Value|\n|---|---|\n|Total Return [%]|12.00|",
        trades=(
            TradeV1(
                trade_id=1,
                direction="long",
                entry_bar_index=0,
                exit_bar_index=1,
                entry_fill_price=100.0,
                exit_fill_price=101.0,
                qty_base=1.0,
                entry_quote_amount=100.0,
                exit_quote_amount=101.0,
                entry_fee_quote=0.0,
                exit_fee_quote=0.0,
                gross_pnl_quote=1.0,
                net_pnl_quote=1.0,
                locked_profit_quote=0.0,
                exit_reason="signal_exit",
            ),
        ),
    )


def _variant(
    *,
    variant_index: int,
    variant_key: str,
    total_return_pct: float,
) -> BacktestVariantPreview:
    """
    Build deterministic variant preview fixture for route mapping tests.

    Args:
        variant_index: Deterministic variant index.
        variant_key: Deterministic variant key (64 hex characters).
        total_return_pct: Ranking metric value.
    Returns:
        BacktestVariantPreview: Variant fixture.
    Assumptions:
        Payload contains explicit saveable indicator/signal/risk/execution values.
    Raises:
        ValueError: If fixture violates DTO invariants.
    Side Effects:
        None.
    """
    return BacktestVariantPreview(
        variant_index=variant_index,
        variant_key=variant_key,
        indicator_variant_key="1" * 64,
        total_return_pct=total_return_pct,
        payload=BacktestVariantPayloadV1(
            indicator_selections=(
                IndicatorVariantSelection(
                    indicator_id="ma.sma",
                    inputs={"source": "close"},
                    params={"window": 20},
                ),
            ),
            signal_params={"ma.sma": {"cross_up": 0.5}},
            risk_params={
                "sl_enabled": True,
                "sl_pct": 2.0,
                "tp_enabled": True,
                "tp_pct": 4.0,
            },
            execution_params={
                "init_cash_quote": 10000.0,
                "fee_pct": 0.075,
                "slippage_pct": 0.01,
                "fixed_quote": 100.0,
                "safe_profit_percent": 30.0,
            },
            direction_mode="long-short",
            sizing_mode="all_in",
        ),
        report=BacktestReportV1(
            rows=(BacktestMetricRowV1(metric="Total Return [%]", value=f"{total_return_pct:.2f}"),),
            table_md="|Metric|Value|\n|---|---|\n|Total Return [%]|1.00|",
        ),
    )


def _saved_mode_response() -> RunBacktestResponse:
    """
    Build deterministic saved-mode response fixture.

    Args:
        None.
    Returns:
        RunBacktestResponse: Saved-mode use-case response fixture.
    Assumptions:
        Response is already sorted by ranking contract.
    Raises:
        ValueError: If fixture violates DTO invariants.
    Side Effects:
        None.
    """
    return RunBacktestResponse(
        mode="saved",
        strategy_id=UUID("00000000-0000-0000-0000-000000000123"),
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        top_k=2,
        preselect=100,
        variants=(
            _variant(variant_index=0, variant_key="a" * 64, total_return_pct=12.0),
            _variant(variant_index=1, variant_key="b" * 64, total_return_pct=10.0),
        ),
        total_indicator_compute_calls=1,
        run_id=UUID("00000000-0000-0000-0000-000000000910"),
        state="succeeded",
        execution_mode="sync_inline",
        engine_version="signal_tf + 1m_risk",
        artifact_slot="slot_b",
        artifact_slot_generation=11,
        artifact_asof_date="2026-03-28",
        artifact_manifest_hash="c" * 64,
        engine_params_hash="e" * 64,
    )


def _template_mode_response() -> RunBacktestResponse:
    """
    Build deterministic template-mode response fixture.

    Args:
        None.
    Returns:
        RunBacktestResponse: Template-mode use-case response fixture.
    Assumptions:
        Response is already sorted by ranking contract.
    Raises:
        ValueError: If fixture violates DTO invariants.
    Side Effects:
        None.
    """
    return RunBacktestResponse(
        mode="template",
        strategy_id=None,
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        top_k=2,
        preselect=100,
        variants=(
            _variant(variant_index=0, variant_key="a" * 64, total_return_pct=12.0),
        ),
        total_indicator_compute_calls=1,
        run_id=UUID("00000000-0000-0000-0000-000000000910"),
        state="succeeded",
        execution_mode="sync_inline",
        engine_version="signal_tf + 1m_risk",
        artifact_slot="slot_b",
        artifact_slot_generation=11,
        artifact_asof_date="2026-03-28",
        artifact_manifest_hash="c" * 64,
        engine_params_hash="e" * 64,
    )


def _template_mode_two_variant_response() -> RunBacktestResponse:
    """
    Build deterministic template-mode response fixture with two ranked variants.

    Args:
        None.
    Returns:
        RunBacktestResponse: Template-mode use-case response fixture.
    Assumptions:
        Response is already sorted by `total_return_pct DESC, variant_key ASC`.
    Raises:
        ValueError: If fixture violates DTO invariants.
    Side Effects:
        None.
    """
    return RunBacktestResponse(
        mode="template",
        strategy_id=None,
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        top_k=2,
        preselect=100,
        variants=(
            _variant(variant_index=0, variant_key="a" * 64, total_return_pct=12.0),
            _variant(variant_index=1, variant_key="b" * 64, total_return_pct=12.0),
        ),
        total_indicator_compute_calls=1,
        run_id=UUID("00000000-0000-0000-0000-000000000910"),
        state="succeeded",
        execution_mode="sync_inline",
        engine_version="signal_tf + 1m_risk",
        artifact_slot="slot_b",
        artifact_slot_generation=11,
        artifact_asof_date="2026-03-28",
        artifact_manifest_hash="c" * 64,
        engine_params_hash="e" * 64,
    )


def _template_background_auto_response() -> RunBacktestResponse:
    """
    Build deterministic queued `background_auto` launch response fixture.

    Args:
        None.
    Returns:
        RunBacktestResponse: Template-mode queued background launch fixture.
    Assumptions:
        Fallback launch is explicit and does not materialize ranked variants inline.
    Raises:
        ValueError: If fixture violates DTO invariants.
    Side Effects:
        None.
    """
    return RunBacktestResponse(
        mode="template",
        strategy_id=None,
        instrument_id=InstrumentId(market_id=MarketId(1), symbol=Symbol("BTCUSDT")),
        timeframe=Timeframe("1m"),
        top_k=2,
        preselect=100,
        variants=tuple(),
        total_indicator_compute_calls=0,
        run_id=UUID("00000000-0000-0000-0000-000000000911"),
        state="queued",
        execution_mode="background_auto",
        engine_version="signal_tf + 1m_risk",
        artifact_slot="slot_b",
        artifact_slot_generation=11,
        artifact_asof_date="2026-03-28",
        artifact_manifest_hash="c" * 64,
        engine_params_hash="e" * 64,
    )
