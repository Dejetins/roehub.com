from __future__ import annotations

from dataclasses import dataclass

from trading.integration import JobCapability


@dataclass(frozen=True, slots=True)
class JobCapabilityPolicy:
    capability: JobCapability
    domain_owner: str
    durable_output_required: bool = True
    exchange_access: bool = False
    strategy_decisions_required: bool = False


CAPABILITY_POLICIES: dict[JobCapability, JobCapabilityPolicy] = {
    "backtest": JobCapabilityPolicy("backtest", "backtest"),
    "optimize": JobCapabilityPolicy("optimize", "optimize"),
    "history_import": JobCapabilityPolicy("history_import", "market_data"),
    "report": JobCapabilityPolicy("report", "report"),
    "artifact_transform": JobCapabilityPolicy("artifact_transform", "backtest_artifacts"),
    "ml_training": JobCapabilityPolicy("ml_training", "ml"),
    "ml_inference": JobCapabilityPolicy("ml_inference", "ml"),
    "rl_training": JobCapabilityPolicy("rl_training", "rl_trading"),
    "rl_inference": JobCapabilityPolicy("rl_inference", "rl_trading"),
    "custom_strategy": JobCapabilityPolicy(
        "custom_strategy",
        "strategy",
        strategy_decisions_required=True,
    ),
}


def capability_policy(capability: JobCapability) -> JobCapabilityPolicy:
    return CAPABILITY_POLICIES[capability]


__all__ = ["CAPABILITY_POLICIES", "JobCapabilityPolicy", "capability_policy"]
