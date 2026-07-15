from __future__ import annotations

from pathlib import Path


def test_research_schema_is_greenfield_and_organization_scoped() -> None:
    repo_root = Path(__file__).resolve().parents[4]
    sql = (
        repo_root / "migrations/postgres/0014_research_organization_isolation_v1.sql"
    ).read_text(encoding="utf-8")

    assert "research organization schema requires empty greenfield table" in sql
    assert "backtest_jobs_org_user_id_unique" in sql
    assert "backtest_top_variants_org_job_fk" in sql
    assert "backtest_stage_a_shortlist_org_job_fk" in sql
    assert "backtest_lazy_materializations_org_member_fk" in sql
    assert "backtest_lazy_materializations_org_job_fk" in sql
    assert "backtest_lazy_materializations_org_identity_unique" in sql
    assert "backtest_ai_" not in sql
    assert "UPDATE backtest_jobs" not in sql
