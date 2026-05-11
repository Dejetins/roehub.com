from __future__ import annotations

from dataclasses import dataclass

from .validator import BacktestAiConfigValidationOutcome


@dataclass(frozen=True, slots=True)
class BacktestAiRepairController:
    repair_attempts: int = 1

    def should_repair(
        self,
        *,
        outcome: BacktestAiConfigValidationOutcome,
        repairs_used: int,
    ) -> bool:
        if repairs_used >= self.repair_attempts:
            return False
        return outcome.status == "needs_clarification" and bool(outcome.validation_errors)


__all__ = ["BacktestAiRepairController"]
