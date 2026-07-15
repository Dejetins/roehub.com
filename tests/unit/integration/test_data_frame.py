from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import pytest
import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from trading.integration import (
    AppContributionSection,
    DataFrameColumn,
    DataFrameFreshness,
    DataFrameMetadata,
    DataFrameUnit,
    DataSourceQueryRequest,
    PanelDataSource,
    PanelPresentation,
    RoehubAppContribution,
    RoehubDataFrame,
    RoehubPanelContribution,
)

_SCHEMA_ROOT = Path(__file__).resolve().parents[3] / "schemas" / "plugins"


def test_roehub_data_frame_v1_is_typed_and_rejects_sensitive_fields() -> None:
    frame = RoehubDataFrame(
        frame_id="fixture.frame",
        title="Fixture series",
        columns=[
            DataFrameColumn(
                key="timestamp",
                label="Time",
                data_type="timestamp",
                role="dimension",
                unit=DataFrameUnit(kind="timestamp", symbol="UTC"),
                nullable=False,
            ),
            DataFrameColumn(
                key="pnl",
                label="PnL",
                data_type="number",
                role="measure",
                unit=DataFrameUnit(kind="currency", symbol="USD"),
                nullable=False,
            ),
        ],
        rows=[{"timestamp": "2026-07-13T10:00:00Z", "pnl": 12.5}],
        metadata=DataFrameMetadata(
            source_label="External fixture",
            query_label="PnL by minute",
            generated_at=datetime(2026, 7, 13, tzinfo=UTC),
        ),
        freshness=DataFrameFreshness(
            status="fresh",
            observed_at=datetime(2026, 7, 13, tzinfo=UTC),
            age_seconds=0,
            max_age_seconds=60,
        ),
    )

    assert frame.contract == "RoehubDataFrame/v1"
    assert frame.rows[0]["pnl"] == 12.5

    with pytest.raises(ValidationError, match="secret-shaped columns"):
        DataFrameColumn(
            key="_".join(("api", "token")),
            label="Sensitive",
            data_type="string",
            role="dimension",
        )


def test_declarative_contributions_cannot_represent_plugin_code_or_markup() -> None:
    query = DataSourceQueryRequest(
        dataset="portfolio.pnl",
        dimensions=["timestamp"],
        measures=["pnl"],
    )
    panel = RoehubPanelContribution(
        contribution_id="fixture.pnl",
        title="PnL",
        description="Bounded portfolio PnL",
        renderer="trading-time-series",
        source=PanelDataSource(instance_id=uuid4(), query=query),
        presentation=PanelPresentation(
            x_column="timestamp",
            y_columns=["pnl"],
            table_columns=["timestamp", "pnl"],
        ),
    )
    app = RoehubAppContribution(
        contribution_id="fixture.research",
        title="Research",
        description="Declarative research workspace",
        sections=[
            AppContributionSection(
                section_id="overview",
                title="Overview",
                panel_contribution_ids=[panel.contribution_id],
            )
        ],
    )

    assert "javascript" not in panel.model_dump_json().lower()
    assert "html" not in panel.model_json_schema()["properties"]
    assert "script" not in app.model_json_schema()["properties"]

    with pytest.raises(ValidationError):
        RoehubPanelContribution.model_validate(
            {**panel.model_dump(mode="json"), "script_url": "https://plugin.invalid/ui.js"}
        )

    for schema_name, payload in (
        ("roehub-panel-contribution-v1.schema.json", panel.model_dump(mode="json")),
        ("roehub-app-contribution-v1.schema.json", app.model_dump(mode="json")),
    ):
        schema = json.loads((_SCHEMA_ROOT / schema_name).read_text(encoding="utf-8"))
        Draft202012Validator(schema).validate(payload)


def test_data_frame_schema_rejects_structural_duplicate_columns() -> None:
    schema = json.loads(
        (_SCHEMA_ROOT / "roehub-data-frame-v1.schema.json").read_text(encoding="utf-8")
    )
    column = {
        "key": "value",
        "label": "Value",
        "data_type": "number",
        "role": "measure",
        "unit": None,
        "nullable": False,
    }
    payload = {
        "contract": "RoehubDataFrame/v1",
        "frame_id": "fixture.duplicate",
        "title": "Duplicate columns",
        "columns": [column, column],
        "rows": [{"value": 1.0}],
        "metadata": {
            "source_label": "Fixture",
            "query_label": "Duplicate",
            "generated_at": "2026-07-13T10:00:00Z",
            "attributes": {},
        },
        "freshness": {
            "status": "unknown",
            "observed_at": None,
            "age_seconds": None,
            "max_age_seconds": None,
        },
        "notices": [],
        "partial": False,
        "errors": [],
    }

    errors = list(Draft202012Validator(schema).iter_errors(payload))

    assert any(error.validator == "uniqueItems" for error in errors)


def test_panel_schema_and_rpc_openapi_enforce_structural_query_bounds() -> None:
    panel_schema = json.loads(
        (_SCHEMA_ROOT / "roehub-panel-contribution-v1.schema.json").read_text(
            encoding="utf-8"
        )
    )
    invalid_panel = {
        "contract": "RoehubPanelContribution/v1",
        "contribution_id": "fixture.invalid-series",
        "title": "Invalid series",
        "description": "Missing visual axes",
        "renderer": "analytics-series",
        "source": {
            "instance_id": str(uuid4()),
            "query": {
                "contract": "DataSourceQuery/v1",
                "dataset": "portfolio.pnl",
                "dimensions": ["timestamp"],
                "measures": ["pnl"],
                "filters": [],
                "row_limit": 10,
                "byte_limit": 4096,
                "point_limit": 10,
                "timeout_ms": 100,
                "read_only": True,
            },
        },
        "presentation": {
            "x_column": None,
            "y_columns": [],
            "table_columns": ["timestamp", "pnl"],
            "default_view": "visual",
        },
    }
    assert list(Draft202012Validator(panel_schema).iter_errors(invalid_panel))

    openapi = yaml.safe_load(
        (_SCHEMA_ROOT / "plugin-rpc-v1alpha1.openapi.yaml").read_text(
            encoding="utf-8"
        )
    )
    query_schema = openapi["components"]["schemas"]["DataSourceQuery"]
    invalid_query = {
        "contract": "DataSourceQuery/v1",
        "dataset": "portfolio.pnl",
        "dimensions": ["timestamp", "timestamp"],
        "measures": ["pnl"],
        "filters": [],
        "limits": {"rows": 10, "bytes": 4096, "points": 10, "timeout_ms": 100},
        "read_only": True,
    }
    errors = list(Draft202012Validator(query_schema).iter_errors(invalid_query))
    assert any(error.validator == "uniqueItems" for error in errors)


def test_query_contract_is_read_only_and_has_hard_client_budgets() -> None:
    with pytest.raises(ValidationError):
        DataSourceQueryRequest(
            dataset="portfolio.pnl",
            dimensions=["timestamp"],
            measures=["pnl"],
            read_only=False,  # type: ignore[arg-type]
        )
    with pytest.raises(ValidationError):
        DataSourceQueryRequest(
            dataset="portfolio.pnl",
            dimensions=["timestamp"],
            measures=["pnl"],
            row_limit=1001,
        )


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (float("nan"), "cell number must be finite"),
        (float("inf"), "cell number must be finite"),
        ("2026-07-13T10:00:00", "cell type does not match"),
        ("not-a-time", "cell type does not match"),
    ],
)
def test_data_frame_rejects_non_finite_numbers_and_unscoped_timestamps(
    value: object,
    message: str,
) -> None:
    data_type = "number" if isinstance(value, float) else "timestamp"
    unit = (
        DataFrameUnit(kind="currency", symbol="USD")
        if data_type == "number"
        else DataFrameUnit(kind="timestamp", symbol="UTC")
    )
    with pytest.raises(ValidationError, match=message):
        RoehubDataFrame(
            frame_id="fixture.invalid",
            title="Invalid fixture",
            columns=[
                DataFrameColumn(
                    key="value",
                    label="Value",
                    data_type=data_type,  # type: ignore[arg-type]
                    role="measure",
                    unit=unit,
                    nullable=False,
                )
            ],
            rows=[{"value": value}],  # type: ignore[list-item]
            metadata=DataFrameMetadata(
                source_label="External fixture",
                query_label="Invalid value",
                generated_at=datetime(2026, 7, 13, tzinfo=UTC),
            ),
            freshness=DataFrameFreshness(status="unknown"),
        )


def test_declarative_contributions_reject_ambiguous_composition() -> None:
    with pytest.raises(ValidationError, match="series renderers require"):
        RoehubPanelContribution(
            contribution_id="fixture.missing-series",
            title="Missing series",
            description="Invalid visual declaration",
            renderer="analytics-series",
            source=PanelDataSource(
                instance_id=uuid4(),
                query=DataSourceQueryRequest(
                    dataset="portfolio.pnl",
                    dimensions=["timestamp"],
                    measures=["pnl"],
                ),
            ),
            presentation=PanelPresentation(table_columns=["timestamp", "pnl"]),
        )

    with pytest.raises(ValidationError, match="section ids must be unique"):
        RoehubAppContribution(
            contribution_id="fixture.duplicate-sections",
            title="Duplicate sections",
            description="Invalid app declaration",
            sections=[
                AppContributionSection(
                    section_id="overview",
                    title="Overview",
                    panel_contribution_ids=["fixture.pnl"],
                ),
                AppContributionSection(
                    section_id="overview",
                    title="Overview again",
                    panel_contribution_ids=["fixture.other"],
                ),
            ],
        )

    with pytest.raises(ValidationError, match="selected query measure"):
        RoehubPanelContribution(
            contribution_id="fixture.unselected-series",
            title="Unselected series",
            description="Invalid panel field reference",
            renderer="analytics-series",
            source=PanelDataSource(
                instance_id=uuid4(),
                query=DataSourceQueryRequest(
                    dataset="portfolio.pnl",
                    dimensions=["timestamp"],
                    measures=["pnl"],
                ),
            ),
            presentation=PanelPresentation(
                x_column="timestamp",
                y_columns=["drawdown"],
                table_columns=["timestamp", "pnl"],
            ),
        )
