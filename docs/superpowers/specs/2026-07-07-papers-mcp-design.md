# papers-mcp — MCP servers over the lipsync-papers and tts-papers corpora

**Date:** 2026-07-07
**Status:** Approved

## Purpose

Give any MCP-capable agent remote, read-only query access to the research
corpora maintained in [will-rice/lipsync-papers](https://github.com/will-rice/lipsync-papers)
and [will-rice/tts-papers](https://github.com/will-rice/tts-papers): search the
literature, read full papers as markdown, walk the in-corpus citation graph,
and list recent drops — without cloning either repo.

## Topology

One Hugging Face Docker Space runs a single Python process that mounts **two
independent MCP servers** at separate paths:

| Endpoint | MCP server name | Corpus |
|---|---|---|
| `https://<space>.hf.space/lipsync/mcp` | `lipsync-papers` | will-rice/lipsync-papers |
| `https://<space>.hf.space/tts/mcp` | `tts-papers` | will-rice/tts-papers |

Each endpoint is a distinct MCP server with its own name and instructions;
tools take no corpus parameter. The servers share one process, one embedding
model in RAM, and one refresh loop. Transport is streamable HTTP. Public,
read-only, no auth — the corpora are public repos.

The corpus list is a module-level constant mapping mount path → GitHub repo.
Adding a corpus is one entry, not a new deployment.

## Architecture

- **Framework:** official MCP Python SDK (FastMCP), each server mounted as an
  ASGI sub-app under one Starlette/uvicorn app.
- **Startup:** shallow-clone both corpus repos, load `papers.csv` plus the
  per-paper markdown frontmatter, build per-corpus indexes:
  - BM25 over title + abstract + authors.
  - Embeddings of title + abstract with `sentence-transformers/all-MiniLM-L6-v2`
    (~2,700 papers total; seconds on the free CPU tier).
  - Citation graph: forward links parsed from resolved sibling links
    (`papers/<year>/<id>.md`) in each paper's markdown body; reverse index
    derived from the forward links.
- **Search:** hybrid retrieval — BM25 rank and embedding-cosine rank merged
  with reciprocal rank fusion.
- **Refresh:** a background thread runs `git pull` + re-index every 6 hours,
  so the corpora's daily CI drops flow through without a deploy. If a pull or
  re-index fails, the server keeps serving the last good index and logs the
  error; refresh failures never take queries down.

## Tool surface (per server, identical shape)

1. `search_papers(query: str, limit: int = 10)` — hybrid search; returns
   arXiv/s2 id, title, authors, submitted date, and abstract per hit,
   formatted as concise markdown.
2. `get_paper(paper_id: str)` — the paper's full converted markdown body
   (frontmatter included). This is how an agent reads method sections.
3. `get_citations(paper_id: str)` — in-corpus papers this paper cites and
   in-corpus papers citing it, as two lists of id + title.
4. `list_recent(days: int = 30)` — papers submitted in the last N days,
   newest first.

## Error handling

- Unknown `paper_id` → MCP tool error stating the id was not found in this
  corpus.
- Invalid arguments (e.g. `limit < 1`) → concise tool error naming the valid
  range. No fallback logic; unsupported cases raise.

## Deployment

- This repo (`papers-mcp`) **is** the Space repo: `git push` to the HF Space
  remote deploys. No separate sync pipeline.
- `Dockerfile` (HF Docker Space, port 7860), dependencies managed with `uv`.
- The corpus repos are untouched except for a README link to the endpoints.

## Testing

pytest, functional style, against the real tool functions with the real
cloned corpora (no mocks):

- `search_papers("latent diffusion lip sync")` surfaces LatentSync
  (2412.09262) on the lipsync server.
- `get_paper` round-trips a known id; unknown id raises.
- Citation forward/reverse indexes are mutually consistent.
- `list_recent` respects the date window and ordering.

## Non-goals

- No auth or rate limiting (revisit if abuse appears).
- No full-text/chunked embedding of paper bodies — abstracts for retrieval,
  `get_paper` for depth.
- No write path of any kind; the corpora's GitHub CI remains the single
  source of truth.
