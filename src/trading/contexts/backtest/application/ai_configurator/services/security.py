from __future__ import annotations

import json
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping

from .catalog import BacktestAiAllowedCatalog

BacktestAiSecurityDecision = Literal[
    "allow",
    "allow_with_audit",
    "block",
    "security_review",
]
BacktestAiSecurityTerminalStatus = Literal[
    "blocked_by_policy",
    "input_too_large",
    "security_review",
]

_MAX_INPUT_BYTES = 12_000
_MAX_INPUT_CHARS = 8_000
_SECURITY_GATES_PATH_ENV = "ROEHUB_BACKTEST_AI_SECURITY_GATES_PATH"

_DOMAIN_TERMS = (
    "backtest",
    "back test",
    "config",
    "configuration",
    "indicator",
    "rsi",
    "ema",
    "sma",
    "dema",
    "atr",
    "btc",
    "eth",
    "usdt",
    "бэктест",
    "бек тест",
    "конфиг",
    "конфигурац",
    "индикатор",
    "собери",
    "создай",
    "параметр",
    "битк",
    "биток",
    "эфир",
)
_JAILBREAK_PATTERNS = (
    re.compile(r"\bignore\s+(all\s+)?(previous|prior)\s+instructions?\b", re.I),
    re.compile(r"\bdeveloper\s+mode\b", re.I),
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"\bsystem\s+prompt\b", re.I),
    re.compile(r"\breveal\s+.*\b(prompt|policy|instructions?)\b", re.I),
    re.compile(r"\bact\s+as\s+(dan|unrestricted|uncensored)\b", re.I),
)
_ENCODED_PATTERNS = (
    re.compile(r"\b(base64|rot13|url\s*decode|decode\s+and\s+(follow|execute))\b", re.I),
    re.compile(r"(?:%[0-9a-fA-F]{2}){6,}"),
    re.compile(r"\b[A-Za-z0-9+/]{80,}={0,2}\b"),
)
_SECRET_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\b(sk|pk)-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"\b(AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\b(password|passwd|secret|token|api[_-]?key|dsn)\s*[:=]\s*\S+", re.I),
)
_SECRET_EXFILTRATION_REQUEST_PATTERNS = (
    re.compile(
        r"\b(include|show|list|print|dump|expose|reveal|return|send)\b"
        r".*\b(env(?:ironment)?\s*vars?|dsn|api\s*tokens?|tokens?|secrets?|"
        r"credentials?|tailscale\s+urls?)\b",
        re.I,
    ),
    re.compile(
        r"\b(env(?:ironment)?\s*vars?|dsn|api\s*tokens?|tokens?|secrets?|"
        r"credentials?|tailscale\s+urls?)\b"
        r".*\b(include|show|list|print|dump|expose|reveal|return|send)\b",
        re.I,
    ),
)
_HTML_OR_LINK_PATTERNS = (
    re.compile(r"<\s*/?\s*[a-z][^>]*>", re.I),
    re.compile(r"\bon[a-z]+\s*=", re.I),
    re.compile(r"\[[^\]]+\]\([^)]+\)"),
    re.compile(r"\b(?:javascript|data):", re.I),
    re.compile(r"https?://", re.I),
)
_OUTPUT_INJECTION_REQUEST_PATTERNS = (
    re.compile(
        r"\b(put|include|return|show|render|add)\b.*"
        r"(<\s*script\b|</\s*script\s*>|javascript:|data:|on[a-z]+\s*=)",
        re.I,
    ),
    re.compile(
        r"(<\s*script\b|</\s*script\s*>|javascript:|data:|on[a-z]+\s*=).*"
        r"\b(answer|assistant|response|message|output)\b",
        re.I,
    ),
)
_PRIVATE_LEAK_PATTERNS = (
    re.compile(r"/Users/[^\s]+"),
    re.compile(r"/opt/roehub/[^\s]+"),
    re.compile(r"\b(?:127\.0\.0\.1|localhost)\b", re.I),
    re.compile(r"\btailscale\b", re.I),
    re.compile(r"\b(?:model_path|mlx_lm\.server|STRATEGY_PG_DSN)\b", re.I),
    re.compile(r"\b[A-Z][A-Z0-9_]*(?:TOKEN|SECRET|PASSWORD|DSN|API_KEY)\b"),
)
_AUTO_ACTION_PATTERNS = (
    re.compile(r"\b(run|launch|created|started)\s+(the\s+)?backtest\b", re.I),
    re.compile(r"\bзапуст(и|ил|ила)\s+б[эе]ктест\b", re.I),
)
_AUTO_ACTION_REQUEST_PATTERNS = (
    re.compile(
        r"\b(run|launch|start|create|submit)\b.*\b(backtest|job)\b.*\b(auto(?:matically)?|now)\b",
        re.I,
    ),
    re.compile(
        r"\b(run|launch|start|create|submit)\b.*\bbacktest\b.*\b(delete|cancel|remove)\b",
        re.I,
    ),
    re.compile(r"\b(delete|cancel|remove)\b.*\b(failed\s+)?jobs?\b", re.I),
    re.compile(r"\bзапуст(и|ить)\b.*\bб[эе]ктест\b.*\bавтомат", re.I),
)


@dataclass(frozen=True, slots=True)
class BacktestAiSecurityIssue:
    path: str
    code: str
    message: str

    def as_mapping(self) -> dict[str, str]:
        return {
            "path": self.path,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class BacktestAiInputGateResult:
    decision: BacktestAiSecurityDecision
    flags: tuple[str, ...] = ()
    risk_score: int = 0
    terminal_status: BacktestAiSecurityTerminalStatus | None = None
    user_message: str | None = None
    normalized_message: str | None = None

    @property
    def allowed(self) -> bool:
        return self.decision in {"allow", "allow_with_audit"}


@dataclass(frozen=True, slots=True)
class BacktestAiOutputGateResult:
    decision: BacktestAiSecurityDecision
    issues: tuple[BacktestAiSecurityIssue, ...] = ()
    flags: tuple[str, ...] = ()

    @property
    def allowed(self) -> bool:
        return self.decision in {"allow", "allow_with_audit"}


@dataclass(frozen=True, slots=True)
class BacktestAiInputGate:
    max_input_bytes: int = _MAX_INPUT_BYTES
    max_input_chars: int = _MAX_INPUT_CHARS

    def evaluate(
        self,
        *,
        message: str,
        locale: str,
        mode: str,
    ) -> BacktestAiInputGateResult:
        normalized = _normalize_text(message)
        encoded = normalized.encode("utf-8")
        if len(encoded) > self.max_input_bytes or len(normalized) > self.max_input_chars:
            return BacktestAiInputGateResult(
                decision="block",
                flags=("input_too_large",),
                risk_score=100,
                terminal_status="input_too_large",
                user_message=_message("input_too_large", locale=locale),
                normalized_message=normalized,
            )

        flags: list[str] = []
        if _contains_blocked_control(normalized):
            flags.append("control_characters")
        if _matches_any(_SECRET_PATTERNS, normalized):
            flags.append("secret_or_credential")
        if _matches_any(_SECRET_EXFILTRATION_REQUEST_PATTERNS, normalized):
            flags.append("secret_exfiltration_request")
        if _matches_any(_OUTPUT_INJECTION_REQUEST_PATTERNS, normalized):
            flags.append("output_injection_request")
        if _matches_any(_AUTO_ACTION_REQUEST_PATTERNS, normalized):
            flags.append("auto_run_backtest_attempt")
        if _matches_any(_JAILBREAK_PATTERNS, normalized):
            flags.append("prompt_injection")
        if _matches_any(_ENCODED_PATTERNS, normalized):
            flags.append("encoded_instruction")
        flags.extend(_external_security_flags(normalized))
        if not _is_domain_prompt(normalized=normalized, mode=mode):
            flags.append("off_topic")

        if (
            "secret_or_credential" in flags
            or "secret_exfiltration_request" in flags
            or "output_injection_request" in flags
            or "auto_run_backtest_attempt" in flags
            or "prompt_injection" in flags
            or any(flag.startswith("external_") for flag in flags)
        ):
            return BacktestAiInputGateResult(
                decision="block",
                flags=tuple(flags),
                risk_score=90,
                terminal_status="blocked_by_policy",
                user_message=_message("blocked_by_policy", locale=locale),
                normalized_message=normalized,
            )
        if "encoded_instruction" in flags or "control_characters" in flags:
            return BacktestAiInputGateResult(
                decision="security_review",
                flags=tuple(flags),
                risk_score=70,
                terminal_status="security_review",
                user_message=_message("security_review", locale=locale),
                normalized_message=normalized,
            )
        if "off_topic" in flags:
            return BacktestAiInputGateResult(
                decision="block",
                flags=tuple(flags),
                risk_score=50,
                terminal_status="blocked_by_policy",
                user_message=_message("blocked_by_policy", locale=locale),
                normalized_message=normalized,
            )
        return BacktestAiInputGateResult(
            decision="allow",
            normalized_message=normalized,
        )


@dataclass(frozen=True, slots=True)
class BacktestAiOutputGate:
    def evaluate(
        self,
        *,
        raw_output: str,
        parsed: Mapping[str, Any] | None,
        catalog: BacktestAiAllowedCatalog,
    ) -> BacktestAiOutputGateResult:
        issues: list[BacktestAiSecurityIssue] = []
        stripped = raw_output.strip()
        if not stripped.startswith("{") or not stripped.endswith("}"):
            issues.append(
                BacktestAiSecurityIssue(
                    path="body",
                    code="not_plain_json_object",
                    message="Model output must be one JSON object without Markdown wrapper",
                )
            )
        if parsed is not None:
            _check_text_fields(payload=parsed, path="", issues=issues)
            config = parsed.get("config")
            if isinstance(config, Mapping):
                _check_output_values(config=config, catalog=catalog, issues=issues)
        if issues:
            return BacktestAiOutputGateResult(
                decision="block",
                issues=tuple(issues),
                flags=tuple(sorted({issue.code for issue in issues})),
            )
        return BacktestAiOutputGateResult(decision="allow")


def _check_text_fields(
    *,
    payload: Mapping[str, Any],
    path: str,
    issues: list[BacktestAiSecurityIssue],
) -> None:
    for key, value in payload.items():
        item_path = f"{path}.{key}" if path else str(key)
        if isinstance(value, str):
            _check_plain_text(value=value, path=item_path, issues=issues)
        elif isinstance(value, Mapping):
            _check_text_fields(payload=value, path=item_path, issues=issues)
        elif isinstance(value, list | tuple):
            for index, item in enumerate(value):
                indexed_path = f"{item_path}.{index}"
                if isinstance(item, str):
                    _check_plain_text(value=item, path=indexed_path, issues=issues)
                elif isinstance(item, Mapping):
                    _check_text_fields(payload=item, path=indexed_path, issues=issues)


def _check_plain_text(
    *,
    value: str,
    path: str,
    issues: list[BacktestAiSecurityIssue],
) -> None:
    if _contains_blocked_control(value):
        issues.append(_issue(path=path, code="hidden_control_text"))
    if _matches_any(_HTML_OR_LINK_PATTERNS, value):
        issues.append(_issue(path=path, code="unsafe_markup_or_link"))
    if _matches_any(_PRIVATE_LEAK_PATTERNS, value) or _matches_any(_SECRET_PATTERNS, value):
        issues.append(_issue(path=path, code="private_or_secret_leakage"))
    if _matches_any(_AUTO_ACTION_PATTERNS, value):
        issues.append(_issue(path=path, code="automatic_backtest_action"))


def _check_output_values(
    *,
    config: Mapping[str, Any],
    catalog: BacktestAiAllowedCatalog,
    issues: list[BacktestAiSecurityIssue],
) -> None:
    if "strategy" in config:
        issues.append(_issue(path="config.strategy", code="unsupported_config_field"))
    if "symbols" in config:
        issues.append(_issue(path="config.symbols", code="multi_symbol_field_not_allowed"))
    coordinates = config.get("coordinates")
    if isinstance(coordinates, Mapping):
        _choice_issue(
            value=coordinates.get("exchange"),
            allowed=catalog.exchanges,
            path="config.coordinates.exchange",
            code="unsupported_exchange",
            issues=issues,
        )
        _choice_issue(
            value=coordinates.get("market_type"),
            allowed=catalog.market_types,
            path="config.coordinates.market_type",
            code="unsupported_market_type",
            issues=issues,
        )
        _choice_issue(
            value=coordinates.get("symbol"),
            allowed=catalog.symbols,
            path="config.coordinates.symbol",
            code="unsupported_symbol",
            issues=issues,
            upper=True,
        )
    _choice_issue(
        value=config.get("timeframe"),
        allowed=catalog.timeframes,
        path="config.timeframe",
        code="unsupported_timeframe",
        issues=issues,
    )
    for index, indicator in enumerate(config.get("indicators") or []):
        if not isinstance(indicator, Mapping):
            continue
        _choice_issue(
            value=indicator.get("indicator_id"),
            allowed=catalog.indicator_ids,
            path=f"config.indicators.{index}.indicator_id",
            code="unsupported_indicator",
            issues=issues,
        )


def _choice_issue(
    *,
    value: Any,
    allowed: tuple[str, ...],
    path: str,
    code: str,
    issues: list[BacktestAiSecurityIssue],
    upper: bool = False,
) -> None:
    if not isinstance(value, str):
        return
    normalized = value.strip().upper() if upper else value.strip().lower()
    allowed_set = {item.upper() if upper else item.lower() for item in allowed}
    if normalized not in allowed_set:
        issues.append(_issue(path=path, code=code))


def _issue(*, path: str, code: str) -> BacktestAiSecurityIssue:
    return BacktestAiSecurityIssue(
        path=path,
        code=code,
        message="Assistant output did not pass deterministic safety checks",
    )


def _normalize_text(value: str) -> str:
    return unicodedata.normalize("NFKC", value).strip()


def _contains_blocked_control(value: str) -> bool:
    for char in value:
        category = unicodedata.category(char)
        if category in {"Cc", "Cf"} and char not in {"\n", "\r", "\t"}:
            return True
    return False


def _matches_any(patterns: tuple[re.Pattern[str], ...], value: str) -> bool:
    return any(pattern.search(value) is not None for pattern in patterns)


def _is_domain_prompt(*, normalized: str, mode: str) -> bool:
    text = normalized.casefold()
    if mode in {"explain", "repair", "suggest_safer"} and len(text) < 160:
        return True
    return any(term in text for term in _DOMAIN_TERMS)


def _message(status: str, *, locale: str) -> str:
    is_ru = locale == "ru"
    if status == "input_too_large":
        return (
            "Запрос слишком большой. Сократите его до одной конфигурации backtest."
            if is_ru
            else "The request is too large. Please shorten it to one backtest configuration."
        )
    if status == "security_review":
        return (
            "Я не могу обработать этот запрос автоматически. "
            "Уберите кодированные или скрытые инструкции и повторите."
            if is_ru
            else (
                "I cannot process this request automatically. "
                "Remove encoded or hidden instructions and try again."
            )
        )
    return (
        "Я могу помогать только с безопасной настройкой конфигурации /backtests. "
        "Сформулируйте запрос про символ, индикатор, период и риск."
        if is_ru
        else (
            "I can only help with safe /backtests configuration. "
            "Ask for a symbol, indicator, period and risk settings."
        )
    )


def _external_security_flags(normalized: str) -> list[str]:
    raw_path = os.environ.get(_SECURITY_GATES_PATH_ENV, "").strip()
    if not raw_path:
        return []
    path = Path(raw_path)
    if not path.is_absolute():
        return ["external_security_policy_invalid"]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ["external_security_policy_invalid"]
    if not isinstance(payload, Mapping) or payload.get("schema_version") != 1:
        return ["external_security_policy_invalid"]
    patterns = payload.get("block_patterns")
    if not isinstance(patterns, list):
        return []
    flags: list[str] = []
    for item in patterns:
        if not isinstance(item, Mapping):
            continue
        flag = item.get("flag")
        pattern = item.get("pattern")
        if not isinstance(flag, str) or not flag.strip().startswith("external_"):
            continue
        if not isinstance(pattern, str) or not pattern.strip():
            continue
        try:
            compiled = re.compile(pattern, re.I)
        except re.error:
            return ["external_security_policy_invalid"]
        if compiled.search(normalized) is not None:
            flags.append(flag.strip())
    return flags


__all__ = [
    "BacktestAiInputGate",
    "BacktestAiInputGateResult",
    "BacktestAiOutputGate",
    "BacktestAiOutputGateResult",
    "BacktestAiSecurityDecision",
    "BacktestAiSecurityIssue",
]
