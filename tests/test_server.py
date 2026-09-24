"""End-to-end MCP wire tests over the mounted lipsync endpoint."""

import re

import pytest
from starlette.testclient import TestClient

from papers_mcp import server
from tests.conftest import DIFFTALK_ID, LATENTSYNC_ID, PAPERS, WAV2LIP_ID

MCP_HEADERS = {
    "Accept": "application/json, text/event-stream",
    "Content-Type": "application/json",
}


def rpc(method: str, params: dict) -> dict:
    return {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}


def call_tool(client: TestClient, name: str, arguments: dict) -> dict:
    resp = client.post(
        "/lipsync/mcp",
        json=rpc("tools/call", {"name": name, "arguments": arguments}),
        headers=MCP_HEADERS,
    )
    return resp.json()["result"]


@pytest.fixture(scope="module")
def client(monkeypatch_module, corpus_repo, tmp_path_factory) -> TestClient:
    monkeypatch_module.setattr(server, "CORPORA", {"lipsync": corpus_repo})
    monkeypatch_module.setattr(server, "DATA_DIR", tmp_path_factory.mktemp("data"))
    app = server.create_app()
    with TestClient(app) as test_client:  # runs lifespan: sync + load + index
        yield test_client


def test_tools_are_listed(client: TestClient) -> None:
    resp = client.post("/lipsync/mcp", json=rpc("tools/list", {}), headers=MCP_HEADERS)
    assert resp.status_code == 200
    tools = {t["name"] for t in resp.json()["result"]["tools"]}
    assert tools == {"search_papers", "get_paper", "get_citations", "list_recent"}


def test_tool_descriptions_document_the_id_format(client: TestClient) -> None:
    resp = client.post("/lipsync/mcp", json=rpc("tools/list", {}), headers=MCP_HEADERS)
    descriptions = {t["name"]: t["description"] for t in resp.json()["result"]["tools"]}
    for name in ("search_papers", "get_paper", "get_citations"):
        assert "arxiv:2412.09262v1" in descriptions[name]


def test_search_papers_tool(client: TestClient) -> None:
    result = call_tool(client, "search_papers", {"query": "latent diffusion lip sync SyncNet"})
    text = result["content"][0]["text"]
    assert text.startswith(f"**{PAPERS[0][1]}** ({LATENTSYNC_ID}, ")
    assert "Chunyu Li, Chao Zhang" in text


def test_get_paper_tool(client: TestClient) -> None:
    result = call_tool(client, "get_paper", {"paper_id": LATENTSYNC_ID})
    text = result["content"][0]["text"]
    assert "LatentSync" in text and len(text) > 5000


def test_get_paper_without_markdown_is_an_error(client: TestClient) -> None:
    result = call_tool(client, "get_paper", {"paper_id": DIFFTALK_ID})
    assert result["isError"] is True
    assert "no converted markdown" in result["content"][0]["text"]


def test_bare_arxiv_id_is_an_error(client: TestClient) -> None:
    result = call_tool(client, "get_paper", {"paper_id": "2412.09262"})
    assert result["isError"] is True
    assert "arxiv:2412.09262v1" in result["content"][0]["text"]


def test_blank_query_is_an_error(client: TestClient) -> None:
    assert call_tool(client, "search_papers", {"query": "  "})["isError"] is True


def test_get_citations_tool(client: TestClient) -> None:
    result = call_tool(client, "get_citations", {"paper_id": LATENTSYNC_ID})
    text = result["content"][0]["text"]
    assert f"## Cites (2)\n- {WAV2LIP_ID}: " in text
    assert f"- {DIFFTALK_ID}: " in text
    assert f"## Cited by (1)\n- {WAV2LIP_ID}: " in text


def test_list_recent_tool(client: TestClient) -> None:
    result = call_tool(client, "list_recent", {"days": 365})
    text = result["content"][0]["text"]
    ids = re.findall(r"\*\* \(([^,]+), \d{4}-\d{2}-\d{2}\)", text)
    # Newest first; the 400-day-old Wav2Lip paper is outside the window.
    assert ids == [DIFFTALK_ID, LATENTSYNC_ID, "arxiv:2301.00001v2", "arxiv:2302.00002v1"]


def test_list_recent_days_over_cap_is_an_error(client: TestClient) -> None:
    assert call_tool(client, "list_recent", {"days": 366})["isError"] is True
