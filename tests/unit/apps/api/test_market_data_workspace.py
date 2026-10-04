from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from apps.api.routes.market_data_workspace import (
    WorkRequestResponse,
    build_market_data_workspace_router,
)
from trading.contexts.identity.application.ports.current_user import CurrentUserPrincipal
from trading.shared_kernel.primitives import (
    InstrumentId,
    MarketId,
    OrganizationId,
    PaidLevel,
    Symbol,
    UserId,
    UtcTimestamp,
)


def stub(**values: Any) -> Any:
    """Partial dependencies expose only the capabilities exercised by each route test."""
    return SimpleNamespace(**values)


@pytest.fixture
def surface():
    org, actor = uuid4(), uuid4()
    seen = []

    def current(request: Request):
        if request.headers.get("x-user") != "allowed":
            raise HTTPException(401)
        return CurrentUserPrincipal(UserId(actor), PaidLevel.free())

    def resolve(*, user_id):
        return stub(organization_id=OrganizationId(org))

    def get(**kwargs):
        seen.append(kwargs)
        return None

    instrument = InstrumentId(MarketId(2), Symbol("BTCUSDT"))
    end = datetime(2026, 9, 1, tzinfo=UTC)

    def submit(**kwargs):
        seen.append(kwargs)
        command = kwargs["request"]
        now = kwargs["now"]
        return dict(
            job_id=uuid4(),
            kind=command.kind,
            market_id=command.market_id,
            symbols=command.symbols,
            timeframe=command.timeframe,
            start_at=command.start_at,
            end_at=command.end_at,
            state="queued",
            attempt=1,
            completed_units=0,
            total_units=command.validate(now=now),
            rows_read=0,
            rows_written=0,
            error_code=None,
            created_at=now,
            updated_at=now,
            started_at=None,
            finished_at=None,
            attempts=[],
        )

    jobs = stub(get=get, submit=submit,
                pause=lambda **kwargs: (seen.append(kwargs) or None),
                resume=lambda **kwargs: (seen.append(kwargs) or None),
                backlog=lambda **kwargs: (seen.append(kwargs) or dict(
                    queued=1, running=0, cancel_requested=0, oldest_queued_at=end)))
    app = FastAPI()
    app.state.jobs = jobs
    app.include_router(
        build_market_data_workspace_router(
            current_user_dependency=current,
            scope_resolver=stub(resolve=resolve),
            selections=stub(),
            snapshots=stub(),
            jobs=jobs,
            markets=stub(execute=lambda: [stub(market_id=MarketId(2))]),
            search=stub(execute=lambda **kwargs: [instrument]),
            index=stub(
                distinct_ts_opens=lambda **kwargs: [UtcTimestamp(end - timedelta(minutes=2))]
            ),
        )
    )
    return TestClient(app), seen, org, end


def test_work_request_auth_origin_scope_and_admission_before_write(surface):
    client, seen, org, end = surface
    body = dict(idempotency_key=str(uuid4()), kind="catalog_refresh", market_id=2)
    assert client.post("/market-data/work-requests", json=body).status_code == 401
    for origin in (None, "null", "http://evil.invalid"):
        headers = {"x-user": "allowed", **({"origin": origin} if origin else {})}
        assert (
            client.post("/market-data/work-requests", json=body, headers=headers).status_code == 403
        )
    headers = {"x-user": "allowed", "origin": "http://testserver"}
    invalid = {
        **body,
        "kind": "candle_ingestion",
        "symbols": ["BTCUSDT"],
        "start_at": datetime(2016, 1, 1, tzinfo=UTC).isoformat(),
        "end_at": end.isoformat(),
    }
    assert (
        client.post("/market-data/work-requests", json=invalid, headers=headers).status_code == 422
    )
    assert seen == []
    assert (
        client.get("/market-data/work-requests/" + str(uuid4()), headers=headers).status_code == 404
    )
    assert seen[0]["organization_id"] == org
    assert (
        client.post(
            "/market-data/work-requests",
            json={**body, "organization_id": str(uuid4())},
            headers=headers,
        ).status_code
        == 422
    )


def test_coverage_uses_closed_canonical_minutes_and_reports_exact_gaps(surface):
    client, _, _, end = surface
    response = client.get(
        "/market-data/workspace/coverage",
        headers={"x-user": "allowed"},
        params={
            "market_id": 2,
            "symbol": "BTCUSDT",
            "start_at": (end - timedelta(minutes=3)).isoformat(),
            "end_at": end.isoformat(),
            "timeframe": "1m",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["actual_candles"] == 1 and body["expected_candles"] == 3
    assert body["coverage_percent"] == 33.33 and body["gap_count"] == 2
    assert [g["missing_candles"] for g in body["gaps"]] == [1, 1]


def test_work_request_projection_never_exposes_worker_token_or_actor():
    now = datetime.now(UTC)
    body = dict(
        job_id=uuid4(),
        kind="catalog_refresh",
        market_id=2,
        symbols=[],
        timeframe="1m",
        start_at=None,
        end_at=None,
        state="succeeded",
        attempt=1,
        completed_units=1,
        total_units=1,
        rows_read=0,
        rows_written=0,
        error_code=None,
        created_at=now,
        updated_at=now,
        started_at=now,
        finished_at=now,
        can_cancel=False,
        can_retry=False,
        worker_token="sensitive-internal-token",
        actor_user_id=uuid4(),
        attempts=[
            dict(
                attempt=1,
                state="succeeded",
                error_code=None,
                finished_at=now,
                provider_payload="private-provider-data",
            )
        ],
    )
    public = WorkRequestResponse.model_validate(body).model_dump_json()
    assert "sensitive-internal-token" not in public and "actor_user_id" not in public
    assert "private-provider-data" not in public


def test_backlog_read_resolves_current_organization(surface):
    client, seen, org, _ = surface
    assert client.get("/market-data/workspace/backlog").status_code == 401
    assert not seen
    response = client.get("/market-data/workspace/backlog", headers={"x-user": "allowed"})
    assert response.status_code == 200
    assert seen == [{"organization_id": org}]
    assert response.json()["queued"] == 1
    assert "organization_id" not in response.json()


def test_coverage_still_bounds_read_cost_after_full_history_downloads(surface):
    client, _, _, end = surface
    assert (
        client.get(
            "/market-data/workspace/coverage",
            headers={"x-user": "allowed"},
            params=dict(
                market_id=2,
                symbol="BTCUSDT",
                timeframe="1m",
                start_at=(end - timedelta(days=8)).isoformat(),
                end_at=end.isoformat(),
            ),
        ).status_code
        == 422
    )


def test_submit_accepts_full_history_through_existing_authenticated_command(surface):
    client, seen, org, end = surface
    response = client.post(
        "/market-data/work-requests",
        headers={"x-user": "allowed", "origin": "http://testserver"},
        json=dict(
            idempotency_key=str(uuid4()),
            kind="candle_ingestion",
            market_id=2,
            symbols=["BTCUSDT"],
            start_at="2017-08-17T04:00:00Z",
            end_at=end.isoformat(),
        ),
    )
    assert response.status_code == 200
    assert response.json()["total_units"] > 10080
    assert response.json()["state"] == "queued"
    assert seen[-1]["organization_id"] == org


@pytest.mark.parametrize("action", ["pause", "resume"])
def test_control_auth_origin_scope_and_version_fence(surface, action):
    client, seen, org, _ = surface
    route = f"/market-data/work-requests/{uuid4()}/{action}"
    assert client.post(route, json={"control_version": 0}).status_code == 401
    assert client.post(route, json={"control_version": 0},
                       headers={"x-user": "allowed"}).status_code == 403
    headers = {"x-user": "allowed", "origin": "http://testserver"}
    assert client.post(route, json={"control_version": -1}, headers=headers).status_code == 422
    assert client.post(route, json={"control_version": 0, "organization_id": str(uuid4())},
                       headers=headers).status_code == 422
    assert seen == []
    assert client.post(route, json={"control_version": 0}, headers=headers).status_code == 404
    assert all(call["organization_id"] == org for call in seen)
    assert seen[0]["version"] == 0


def test_duplicate_resume_acknowledges_saved_state_but_stale_cycle_is_conflict(surface):
    client, _, _, _ = surface
    headers = {"x-user": "allowed", "origin": "http://testserver"}
    saved = client.post("/market-data/work-requests", headers=headers,
                        json=dict(idempotency_key=str(uuid4()), kind="catalog_refresh",
                                  market_id=2)).json()
    saved.update(state="queued", control_version=2)
    client.app.state.jobs.get = lambda **kwargs: saved
    route = f"/market-data/work-requests/{saved['job_id']}/resume"
    reply = client.post(route, json={"control_version":1}, headers=headers)
    assert reply.status_code == 200 and reply.json()["can_pause"]
    saved.update(state="paused", control_version=3)
    assert client.post(route, json={"control_version":1}, headers=headers).status_code == 409


def test_batch_and_events_origin_scope_projection_and_conflicts(surface):
    client, seen, org, end = surface
    headers = {'x-user':'allowed', 'origin':'http://testserver'}
    jobs = client.app.state.jobs
    def batch(**kwargs):
        seen.append(kwargs)
        return [jobs.submit(request=c, now=kwargs['now']) for c in kwargs['requests']]
    jobs.submit_batch = batch
    body = dict(idempotency_key=str(uuid4()), market_id=2, symbols=['BTCUSDT'],
                start_at=(end-timedelta(days=30)).isoformat(), end_at=end.isoformat())
    route = '/market-data/work-batches'
    assert client.post(route, json=body).status_code == 401
    assert client.post(route, json=body, headers={'x-user':'allowed'}).status_code == 403
    assert not seen
    reply = client.post(route, json=body, headers=headers)
    assert reply.status_code == 200 and len(reply.json()['items']) == 1
    assert seen[0]['organization_id'] == org
    assert client.post(route, json={**body,'symbols':['BTCUSDT']*51},
                       headers=headers).status_code == 422
    job = reply.json()['items'][0]
    events_route = f"/market-data/work-requests/{job['job_id']}/events"
    assert client.get(events_route, headers=headers).status_code == 404
    jobs.get = lambda **kwargs: job
    def events(**kwargs):
        seen.append(kwargs)
        return [dict(event_id=2,event_type='queued',occurred_at=end,state='queued',attempt=1,
                     completed_units=0,total_units=10,rows_written=0,error_code=None,
                     error_phase=None,next_retry_at=None,worker_token='must-not-leak')]
    jobs.events = events
    result = client.get(events_route, headers=headers)
    assert result.status_code == 200 and seen[-1]['organization_id'] == org
    assert 'worker_token' not in result.text
    assert client.get(events_route+'?limit=101', headers=headers).status_code == 422
