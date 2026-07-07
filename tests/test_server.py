"""End-to-end MCP wire tests over the mounted lipsync endpoint."""

import pytest
from starlette.testclient import TestClient

from papers_mcp import server
from tests.conftest import CACHE_DIR, LIPSYNC_REPO

MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


def rpc(method: str, params: dict) -> dict:
    return {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}


@pytest.fixture(scope="module")
def client(monkeypatch_module, lipsync_corpus) -> TestClient:
    monkeypatch_module.setattr(server, "CORPORA", {"lipsync": LIPSYNC_REPO})
    monkeypatch_module.setattr(server, "DATA_DIR", CACHE_DIR)
    app = server.create_app()
    with TestClient(app) as test_client:  # runs lifespan: sync + load + index
        yield test_client


def test_tools_are_listed(client: TestClient) -> None:
    resp = client.post("/lipsync/mcp", json=rpc("tools/list", {}), headers=MCP_HEADERS)
    assert resp.status_code == 200
    tools = {t["name"] for t in resp.json()["result"]["tools"]}
    assert tools == {"search_papers", "get_paper", "get_citations", "list_recent"}


def test_search_papers_tool(client: TestClient) -> None:
    resp = client.post(
        "/lipsync/mcp",
        json=rpc(
            "tools/call",
            {"name": "search_papers", "arguments": {"query": "latent diffusion lip sync SyncNet"}},
        ),
        headers=MCP_HEADERS,
    )
    text = resp.json()["result"]["content"][0]["text"]
    assert "2412.09262" in text


def test_get_paper_tool(client: TestClient) -> None:
    resp = client.post(
        "/lipsync/mcp",
        json=rpc("tools/call", {"name": "get_paper", "arguments": {"paper_id": "2412.09262"}}),
        headers=MCP_HEADERS,
    )
    text = resp.json()["result"]["content"][0]["text"]
    assert "LatentSync" in text and len(text) > 5000


def test_unknown_paper_id_is_an_error(client: TestClient) -> None:
    resp = client.post(
        "/lipsync/mcp",
        json=rpc("tools/call", {"name": "get_paper", "arguments": {"paper_id": "0000.00000"}}),
        headers=MCP_HEADERS,
    )
    assert resp.json()["result"]["isError"] is True


def test_blank_query_is_an_error(client: TestClient) -> None:
    resp = client.post(
        "/lipsync/mcp",
        json=rpc("tools/call", {"name": "search_papers", "arguments": {"query": "  "}}),
        headers=MCP_HEADERS,
    )
    assert resp.json()["result"]["isError"] is True


def test_get_citations_tool(client: TestClient) -> None:
    resp = client.post(
        "/lipsync/mcp",
        json=rpc("tools/call", {"name": "get_citations", "arguments": {"paper_id": "2412.09262"}}),
        headers=MCP_HEADERS,
    )
    text = resp.json()["result"]["content"][0]["text"]
    assert "Cites" in text and "Cited by" in text


def test_list_recent_tool(client: TestClient) -> None:
    import re
    from datetime import date, timedelta

    resp = client.post(
        "/lipsync/mcp",
        json=rpc("tools/call", {"name": "list_recent", "arguments": {"days": 365}}),
        headers=MCP_HEADERS,
    )
    text = resp.json()["result"]["content"][0]["text"]
    dates = re.findall(r", (\d{4}-\d{2}-\d{2})\)", text)
    assert len(dates) > 5
    assert dates == sorted(dates, reverse=True)  # newest first
    assert min(dates) >= (date.today() - timedelta(days=365)).isoformat()
