from __future__ import annotations

import os
from collections.abc import Mapping

from fastapi import FastAPI

from trading.contexts.exchange_control.adapters.inbound.http import (
    create_exchange_control_app,
)
from trading.contexts.exchange_control.adapters.inbound.http.app import (
    ExchangeControlRuntimeConfig,
)


def create_app(*, environ: Mapping[str, str] | None = None) -> FastAPI:
    config = ExchangeControlRuntimeConfig.from_environ(environ=environ or os.environ)
    return create_exchange_control_app(config=config)


app = create_app()

__all__ = ["app", "create_app"]
