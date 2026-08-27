"""Phase 2: `log`, `show` and the embedded dashboard."""

from __future__ import annotations

import httpx
import pytest
from typer.testing import CliRunner

from localdoctor.cli import app
from localdoctor.dashboard import create_dashboard_app
from localdoctor.store import AmbiguousId
from tests.conftest import CHAT, chat_body

runner = CliRunner()

TRUNCATED = {
    "json": chat_body(
        messages=[{"role": "user", "content": "lorem ipsum dolor sit amet " * 1600}],
        options={"num_ctx": 512},
    ),
    "headers": {"x-fake-scenario": "context_shift"},
}
HEALTHY = {"json": chat_body(options={"num_ctx": 131_072}), "headers": {}}


async def populate(stack, calls=(TRUNCATED, HEALTHY)):
    async with httpx.AsyncClient(timeout=30) as client:
        for call in calls:
            response = await client.post(stack.base_url + CHAT, **call)
            assert response.status_code == 200
            await stack.settle()


# --- store reads --------------------------------------------------------


@pytest.mark.asyncio
async def test_recent_defaults_to_requests_with_findings(stack):
    await populate(stack)
    assert len(stack.store.query("SELECT * FROM requests")) == 2
    assert len(stack.store.recent()) == 1
    assert len(stack.store.recent(only_findings=False)) == 2


@pytest.mark.asyncio
async def test_recent_filters(stack):
    await populate(stack)
    assert len(stack.store.recent(rule="R001")) == 1
    assert len(stack.store.recent(rule="R003")) == 0
    assert len(stack.store.recent(model="fake")) == 1
    assert len(stack.store.recent(model="nope")) == 0
    assert len(stack.store.recent(endpoint="/api/embed", only_findings=False)) == 0


@pytest.mark.asyncio
async def test_find_request_by_tail_and_full_id(stack):
    await populate(stack)
    full = stack.store.recent()[0]["id"]
    assert stack.store.find_request(full)["id"] == full
    assert stack.store.find_request(full[-8:])["id"] == full
    assert stack.store.find_request(full.lower())["id"] == full
    assert stack.store.find_request("ZZZZZZZZ") is None


@pytest.mark.asyncio
async def test_ambiguous_id_is_reported_not_guessed(stack):
    await populate(stack)
    rows = stack.store.query("SELECT id FROM requests")
    shared = rows[0]["id"][:4]
    with pytest.raises(AmbiguousId) as caught:
        stack.store.find_request(shared)
    assert len(caught.value.matches) == 2


# --- cli ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_log_lists_findings(stack, tmp_path):
    await populate(stack)
    result = runner.invoke(app, ["log", "--db", str(tmp_path / "test.db")])
    assert result.exit_code == 0
    assert "R001" in result.stdout
    assert "/api/chat" in result.stdout


@pytest.mark.asyncio
async def test_log_all_includes_healthy_requests(stack, tmp_path):
    await populate(stack)
    findings_only = runner.invoke(app, ["log", "--db", str(tmp_path / "test.db")])
    with_healthy = runner.invoke(app, ["log", "--all", "--db", str(tmp_path / "test.db")])
    assert with_healthy.stdout.count("/api/chat") > findings_only.stdout.count("/api/chat")


@pytest.mark.asyncio
async def test_log_surfaces_low_confidence_records(stack, tmp_path):
    """Spec 5: `low` never reaches the live terminal, but must be inspectable.

    A cache hit makes the server report fewer tokens than were sent, which trips
    S3 alone. Truncation is one explanation; a warm cache is another. Low it is.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            stack.base_url + CHAT,
            json=chat_body(
                messages=[{"role": "user", "content": "x" * 40_000}],
                options={"num_ctx": 131_072},
            ),
            headers={"x-fake-scenario": "cached_prompt"},
        )
        assert response.status_code == 200
    await stack.settle()

    low = stack.store.query("SELECT * FROM diagnoses WHERE confidence = 'low'")
    assert low, "expected a low-confidence record for this shape"
    assert stack.printed() == "", "low must never be printed live"

    result = runner.invoke(app, ["log", "--db", str(tmp_path / "test.db")])
    assert low[0]["rule_id"] in result.stdout


@pytest.mark.asyncio
async def test_show_prints_evidence_and_fix(stack, tmp_path):
    await populate(stack)
    request_id = stack.store.recent()[0]["id"]
    result = runner.invoke(app, ["show", request_id[-8:], "--db", str(tmp_path / "test.db")])
    assert result.exit_code == 0
    assert "CONTEXT LIMIT" in result.stdout or "R001" in result.stdout
    assert "Model read" in result.stdout
    assert "num_ctx" in result.stdout
    assert "request body" in result.stdout


@pytest.mark.asyncio
async def test_show_reports_a_missing_id(stack, tmp_path):
    await populate(stack)
    result = runner.invoke(app, ["show", "ZZZZZZZZ", "--db", str(tmp_path / "test.db")])
    assert result.exit_code == 1
    assert "no request matching" in result.stdout


def test_commands_explain_a_missing_database(tmp_path):
    for command in (["log"], ["show", "abc"], ["replay", "abc"]):
        result = runner.invoke(app, command + ["--db", str(tmp_path / "absent.db")])
        assert result.exit_code == 1
        assert "no database at" in result.stdout


# --- dashboard ----------------------------------------------------------


@pytest.mark.asyncio
async def test_dashboard_api_serves_records(stack):
    await populate(stack)
    dashboard = create_dashboard_app(stack.store)
    transport = httpx.ASGITransport(app=dashboard)
    async with httpx.AsyncClient(transport=transport, base_url="http://dash") as client:
        page = await client.get("/")
        assert page.status_code == 200
        assert "LocalDoctor" in page.text

        stats = (await client.get("/api/stats")).json()
        assert stats["total_requests"] == 2
        assert stats["flagged_requests"] >= 1

        listing = (await client.get("/api/requests")).json()
        assert len(listing) == 1
        assert listing[0]["diagnoses"][0]["rule_id"] == "R001"

        everything = (await client.get("/api/requests?all=true")).json()
        assert len(everything) == 2

        detail = (await client.get(f"/api/requests/{listing[0]['id']}")).json()
        assert detail["request_body"].startswith("{")
        assert detail["diagnoses"][0]["evidence"]["signals"] == ["S2", "S3"]

        missing = await client.get("/api/requests/ZZZZZZZZ")
        assert missing.status_code == 404


@pytest.mark.asyncio
async def test_dashboard_page_loads_nothing_external(stack):
    """"Fully local" is a product principle, not a preference."""
    import re

    from localdoctor.dashboard import PAGE

    assert not re.findall(r"https?://", PAGE)
    assert "cdn" not in PAGE.lower()
