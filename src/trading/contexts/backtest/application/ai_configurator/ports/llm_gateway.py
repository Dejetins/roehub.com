from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from trading.contexts.backtest.application.ai_configurator.dto import BacktestAiConfigJob
from trading.contexts.backtest.application.ai_configurator.services.catalog import (
    BacktestAiAllowedCatalog,
)
from trading.contexts.backtest.application.ai_configurator.services.prompt_profiles import (
    BacktestAiPromptProfile,
)


@dataclass(frozen=True, slots=True)
class BacktestConfigLLMRequest:
    job: BacktestAiConfigJob
    catalog: BacktestAiAllowedCatalog
    prompt_profile: BacktestAiPromptProfile
    prompt_text: str
    catalog_subset_json: Mapping[str, Any]
    output_schema_json: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class BacktestConfigLLMRepairRequest(BacktestConfigLLMRequest):
    failed_raw_output: str | None
    parsed_draft_json: Mapping[str, Any] | None
    validation_errors_json: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class BacktestConfigLLMResponse:
    raw_output: str
    model_id: str
    model_path_hash: str | None = None
    input_tokens_estimate: int | None = None
    output_tokens_estimate: int | None = None
    latency_ms: int | None = None
    finish_reason: str | None = None


class BacktestConfigLLMGateway(Protocol):
    def generate_config(
        self,
        request: BacktestConfigLLMRequest,
    ) -> BacktestConfigLLMResponse:
        """
        Produce one untrusted JSON draft. Implementations must not execute tools.
        """
        ...

    def repair_config(
        self,
        request: BacktestConfigLLMRepairRequest,
    ) -> BacktestConfigLLMResponse:
        """
        Produce one repair draft from untrusted failed output and deterministic errors.
        """
        ...


__all__ = [
    "BacktestConfigLLMGateway",
    "BacktestConfigLLMRepairRequest",
    "BacktestConfigLLMRequest",
    "BacktestConfigLLMResponse",
]
