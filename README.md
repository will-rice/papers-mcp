---
title: papers-mcp
emoji: 📚
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# papers-mcp

MCP servers over the [lipsync-papers](https://github.com/will-rice/lipsync-papers)
and [tts-papers](https://github.com/will-rice/tts-papers) research corpora.
Public, read-only, streamable HTTP.

| Endpoint | Corpus |
|---|---|
| `https://wrice-papers-mcp.hf.space/lipsync/mcp` | lipsync-papers |
| `https://wrice-papers-mcp.hf.space/tts/mcp` | tts-papers |

Tools per server: `search_papers` (hybrid BM25 + embedding search),
`get_paper` (full markdown), `get_citations` (in-corpus citation graph),
`list_recent`. Corpora re-sync from GitHub every 6 hours.

## Connect

```bash
claude mcp add --transport http lipsync-papers https://wrice-papers-mcp.hf.space/lipsync/mcp
claude mcp add --transport http tts-papers https://wrice-papers-mcp.hf.space/tts/mcp
```

## Develop

```bash
uv sync
uv run pytest
uv run uvicorn --factory papers_mcp.server:create_app --port 7860
```
