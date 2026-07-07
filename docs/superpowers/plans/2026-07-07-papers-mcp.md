# papers-mcp Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** One HF Docker Space process serving two independent MCP servers — `/lipsync/mcp` and `/tts/mcp` — with hybrid search, full-paper reads, citation-graph lookups, and recent listings over the lipsync-papers and tts-papers corpora.

**Architecture:** A Starlette app mounts one FastMCP server per corpus (official v1.x multi-server pattern: `Mount(f"/{name}", server.streamable_http_app())` + combined lifespan running each `session_manager`). At startup the lifespan clones both corpus repos, loads `papers.csv` + markdown, and builds per-corpus BM25 + embedding indexes sharing one `all-MiniLM-L6-v2` model; a daemon thread re-pulls and re-indexes every 6 hours, keeping the old index on failure.

**Tech Stack:** Python ≥3.11, `mcp>=1.27,<2` (v1.x stable — v2 is pre-release, do NOT use), sentence-transformers, rank-bm25, uvicorn, uv, pytest, HF Docker Space.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-07-papers-mcp-design.md`. Read it before starting.
- Pin `mcp>=1.27,<2`. The v2 pre-releases are breaking; v1.x is the stable line.
- Endpoints must be exactly `/lipsync/mcp` and `/tts/mcp`. Tools take NO corpus parameter.
- Public, read-only, no auth. No fallback logic — unsupported cases raise `ValueError` with a concise message.
- Corpus repos: `https://github.com/will-rice/lipsync-papers`, `https://github.com/will-rice/tts-papers`. Metadata source of truth is each repo's `papers.csv` (columns: `arxiv_id,title,authors,submitted,categories,url,abstract`). Paper ids are arXiv-style (`2412.09262`) or `s2:`-prefixed (`s2:0a429a0c…`).
- Corpus markdown lives at `papers/<year>/<id>.md`; year dirs also contain a `README.md` that is NOT a paper. In-corpus citation links appear in two forms: `](../<year>/<id>.md)` and same-directory `](<id>.md)`.
- User conventions: `uv run` for everything; `uv run pre-commit run -a` must pass before every commit; `logging.info` not `print`; Google docstrings; src layout; absolute imports; pytest functional style, no mocks — tests run against the real cloned corpus; PyTorch with batch dims; vectorize, don't loop over scores.
- Embedding model: `sentence-transformers/all-MiniLM-L6-v2`, CPU. RRF constant k=60.

---

### Task 1: Project scaffold + `Corpus.sync`

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `.pre-commit-config.yaml`
- Create: `src/papers_mcp/__init__.py`, `src/papers_mcp/corpus.py`
- Test: `tests/conftest.py`, `tests/test_corpus.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `Corpus` dataclass with fields `name: str`, `repo_url: str`, `clone_dir: Path`, `papers: dict[str, Paper]` and method `sync() -> None`. `Paper` dataclass with fields `paper_id: str`, `title: str`, `authors: str`, `submitted: str`, `url: str`, `abstract: str`, `md_path: Path | None`, `cites: list[str]`, `cited_by: list[str]`. Session fixture `lipsync_corpus` (synced + loaded) and constant `CACHE_DIR` in `tests/conftest.py`.

- [ ] **Step 1: Scaffold the project**

```bash
cd ~/Documents/projects/papers-mcp
```

Write `pyproject.toml`:

```toml
[project]
name = "papers-mcp"
version = "0.1.0"
description = "MCP servers over the lipsync-papers and tts-papers research corpora"
requires-python = ">=3.11"
dependencies = [
    "mcp>=1.27,<2",
    "rank-bm25>=0.2.2",
    "sentence-transformers>=3.0",
    "uvicorn>=0.30",
]

[dependency-groups]
dev = [
    "httpx>=0.27",
    "pre-commit>=4.0",
    "pytest>=8.0",
]

[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/papers_mcp"]

[tool.ruff]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
```

Write `.gitignore`:

```
__pycache__/
*.egg-info/
.venv/
.pytest_cache/
data/
tests/.cache/
```

Write `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v5.0.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.12.0
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format
```

Write empty `src/papers_mcp/__init__.py`. Then:

```bash
uv sync
```

Expected: creates `.venv` and `uv.lock` without errors. The pytorch-cpu marker only affects Linux; on macOS torch resolves from PyPI (already CPU).

- [ ] **Step 2: Write the failing test for `Corpus.sync`**

`tests/conftest.py`:

```python
"""Shared fixtures: a real lipsync-papers clone, cached across test runs."""

from pathlib import Path

import pytest

from papers_mcp.corpus import Corpus

CACHE_DIR = Path(__file__).parent / ".cache"
LIPSYNC_REPO = "https://github.com/will-rice/lipsync-papers"


@pytest.fixture(scope="session")
def lipsync_corpus() -> Corpus:
    corpus = Corpus(
        name="lipsync", repo_url=LIPSYNC_REPO, clone_dir=CACHE_DIR / "lipsync-papers"
    )
    corpus.sync()
    corpus.load()
    return corpus
```

`tests/test_corpus.py`:

```python
"""Tests for corpus syncing and loading against the real lipsync-papers repo."""

from papers_mcp.corpus import Corpus


def test_sync_clones_then_pulls(lipsync_corpus: Corpus) -> None:
    assert (lipsync_corpus.clone_dir / "papers.csv").exists()
    # Second sync takes the pull path and must not raise.
    lipsync_corpus.sync()
    assert (lipsync_corpus.clone_dir / "papers.csv").exists()
```

Note: `corpus.load()` in the fixture will fail with `AttributeError` until Task 2; for this task, temporarily comment the `corpus.load()` line out, and uncomment it in Task 2 Step 1.

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/test_corpus.py -v`
Expected: FAIL (ImportError: cannot import name `Corpus`).

- [ ] **Step 4: Implement `Corpus.sync`**

`src/papers_mcp/corpus.py`:

```python
"""Load and sync a research-papers corpus (papers.csv + markdown) from GitHub."""

import csv
import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

# In-corpus citation links: `](../<year>/<id>.md)` or same-directory `](<id>.md)`.
CITATION_LINK_RE = re.compile(r"\]\((?:\.\./\d{4}/)?([^/()\s]+)\.md\)")


@dataclass
class Paper:
    """One paper's metadata, markdown location, and in-corpus citation edges."""

    paper_id: str
    title: str
    authors: str
    submitted: str
    url: str
    abstract: str
    md_path: Path | None = None
    cites: list[str] = field(default_factory=list)
    cited_by: list[str] = field(default_factory=list)


@dataclass
class Corpus:
    """A cloned corpus repo and its loaded papers, keyed by paper id."""

    name: str
    repo_url: str
    clone_dir: Path
    papers: dict[str, Paper] = field(default_factory=dict)

    def sync(self) -> None:
        """Clone the corpus repo if absent, otherwise fast-forward pull."""
        if (self.clone_dir / ".git").exists():
            subprocess.run(
                ["git", "-C", str(self.clone_dir), "pull", "--ff-only"],
                check=True,
                capture_output=True,
            )
        else:
            self.clone_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["git", "clone", "--depth", "1", self.repo_url, str(self.clone_dir)],
                check=True,
                capture_output=True,
            )
        logging.info("synced %s corpus at %s", self.name, self.clone_dir)
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/test_corpus.py -v`
Expected: PASS (first run clones ~30MB; later runs pull).

- [ ] **Step 6: Commit**

```bash
uv run pre-commit run -a
git add -A && git commit -m "feat: project scaffold and corpus git sync"
```

---

### Task 2: `Corpus.load` — papers, markdown paths, citation graph

**Files:**
- Modify: `src/papers_mcp/corpus.py` (add `load` method)
- Test: `tests/test_corpus.py`

**Interfaces:**
- Consumes: `Corpus`, `Paper`, `CITATION_LINK_RE` from Task 1.
- Produces: `Corpus.load() -> None` populating `self.papers: dict[str, Paper]` with `md_path` set for papers that have a corpus markdown file and `cites`/`cited_by` filled with in-corpus paper ids.

- [ ] **Step 1: Write the failing tests**

Uncomment `corpus.load()` in `tests/conftest.py` (see Task 1 Step 2). Append to `tests/test_corpus.py`:

```python
def test_load_populates_papers(lipsync_corpus: Corpus) -> None:
    assert len(lipsync_corpus.papers) > 500
    latentsync = lipsync_corpus.papers["2412.09262"]
    assert "LatentSync" in latentsync.title
    assert latentsync.md_path is not None and latentsync.md_path.exists()
    assert latentsync.submitted == "2024-12-12"


def test_readme_files_are_not_papers(lipsync_corpus: Corpus) -> None:
    assert "README" not in lipsync_corpus.papers


def test_citation_graph_is_consistent(lipsync_corpus: Corpus) -> None:
    papers = lipsync_corpus.papers
    assert any(p.cites for p in papers.values())
    for paper in papers.values():
        for cited_id in paper.cites:
            assert cited_id in papers
            assert paper.paper_id in papers[cited_id].cited_by
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_corpus.py -v`
Expected: ERROR at fixture setup (`Corpus` has no attribute `load`).

- [ ] **Step 3: Implement `Corpus.load`**

Add to the `Corpus` class in `src/papers_mcp/corpus.py`:

```python
    def load(self) -> None:
        """Load papers.csv, locate corpus markdown files, and build the citation graph."""
        papers: dict[str, Paper] = {}
        with (self.clone_dir / "papers.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                papers[row["arxiv_id"]] = Paper(
                    paper_id=row["arxiv_id"],
                    title=row["title"],
                    authors=row["authors"],
                    submitted=row["submitted"],
                    url=row["url"],
                    abstract=row["abstract"],
                )

        for md_path in sorted(self.clone_dir.glob("papers/*/*.md")):
            paper = papers.get(md_path.stem)  # skips per-year README.md files
            if paper:
                paper.md_path = md_path

        for paper in papers.values():
            if paper.md_path is None:
                continue
            body = paper.md_path.read_text(encoding="utf-8")
            for cited_id in CITATION_LINK_RE.findall(body):
                if cited_id != paper.paper_id and cited_id in papers and cited_id not in paper.cites:
                    paper.cites.append(cited_id)
        for paper in papers.values():
            for cited_id in paper.cites:
                papers[cited_id].cited_by.append(paper.paper_id)

        self.papers = papers
        logging.info("loaded %d papers for %s corpus", len(papers), self.name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_corpus.py -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
uv run pre-commit run -a
git add -A && git commit -m "feat: load corpus metadata, markdown paths, and citation graph"
```

---

### Task 3: Hybrid search index

**Files:**
- Create: `src/papers_mcp/search.py`
- Test: `tests/test_search.py`

**Interfaces:**
- Consumes: `Paper` from `papers_mcp.corpus`; `lipsync_corpus` fixture.
- Produces: `EMBEDDING_MODEL: str` constant; `tokenize(text: str) -> list[str]`; `SearchIndex(papers: list[Paper], model: SentenceTransformer)` with method `search(query: str, limit: int) -> list[Paper]`.

- [ ] **Step 1: Write the failing tests**

`tests/test_search.py`:

```python
"""Search tests against the real lipsync corpus."""

import pytest
from sentence_transformers import SentenceTransformer

from papers_mcp.corpus import Corpus
from papers_mcp.search import EMBEDDING_MODEL, SearchIndex


@pytest.fixture(scope="session")
def lipsync_index(lipsync_corpus: Corpus) -> SearchIndex:
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return SearchIndex(list(lipsync_corpus.papers.values()), model)


def test_keyword_query_finds_latentsync(lipsync_index: SearchIndex) -> None:
    results = lipsync_index.search("latent diffusion lip sync SyncNet", limit=10)
    assert "2412.09262" in [p.paper_id for p in results]


def test_semantic_query_returns_relevant_papers(lipsync_index: SearchIndex) -> None:
    results = lipsync_index.search("make the mouth match new audio in a video", limit=5)
    assert len(results) == 5
    haystack = " ".join(f"{p.title} {p.abstract}".lower() for p in results)
    assert "lip" in haystack


def test_limit_is_respected(lipsync_index: SearchIndex) -> None:
    assert len(lipsync_index.search("talking head", limit=3)) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_search.py -v`
Expected: FAIL (ModuleNotFoundError: `papers_mcp.search`). First run downloads the ~80MB model into the HF cache.

- [ ] **Step 3: Implement `SearchIndex`**

`src/papers_mcp/search.py`:

```python
"""Hybrid BM25 + embedding search over a corpus's papers."""

import re

import torch
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from papers_mcp.corpus import Paper

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RRF_K = 60

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens for BM25."""
    return TOKEN_RE.findall(text.lower())


class SearchIndex:
    """Ranks papers by reciprocal-rank fusion of BM25 and embedding-cosine ranks."""

    def __init__(self, papers: list[Paper], model: SentenceTransformer) -> None:
        self.papers = papers
        self.model = model
        texts = [f"{p.title} {p.abstract}" for p in papers]
        self.bm25 = BM25Okapi(
            [tokenize(f"{text} {p.authors}") for text, p in zip(texts, papers)]
        )
        self.embeddings = model.encode(
            texts, convert_to_tensor=True, normalize_embeddings=True
        )

    def search(self, query: str, limit: int) -> list[Paper]:
        """Return the top *limit* papers for *query* by fused BM25 + cosine rank."""
        bm25_scores = torch.tensor(self.bm25.get_scores(tokenize(query)))
        query_embedding = self.model.encode(
            [query], convert_to_tensor=True, normalize_embeddings=True
        )
        cosine_scores = (self.embeddings @ query_embedding.T).flatten()

        fused = torch.zeros(len(self.papers))
        for scores in (bm25_scores, cosine_scores):
            ranks = scores.argsort(descending=True).argsort()
            fused += 1.0 / (RRF_K + 1 + ranks)

        top = fused.argsort(descending=True)[:limit]
        return [self.papers[i] for i in top.tolist()]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_search.py -v`
Expected: PASS (3 tests; ~540 abstracts embed in well under a minute on CPU).

- [ ] **Step 5: Commit**

```bash
uv run pre-commit run -a
git add -A && git commit -m "feat: hybrid BM25 + embedding search with reciprocal rank fusion"
```

---

### Task 4: MCP servers, tools, app factory, refresh loop

**Files:**
- Create: `src/papers_mcp/server.py`
- Test: `tests/test_server.py`

**Interfaces:**
- Consumes: `Corpus`, `Paper` from `papers_mcp.corpus`; `EMBEDDING_MODEL`, `SearchIndex` from `papers_mcp.search`.
- Produces: module globals `CORPORA: dict[str, str]`, `DATA_DIR: Path`, `REFRESH_INTERVAL_SECONDS: int`, `corpora: dict[str, Corpus]`, `indexes: dict[str, SearchIndex]`; functions `create_app() -> Starlette` (uvicorn factory), `build_corpus(name: str, model: SentenceTransformer) -> None`, `lookup(name: str, paper_id: str) -> Paper`, `format_paper(paper: Paper) -> str`. Four MCP tools per server: `search_papers(query, limit=10)`, `get_paper(paper_id)`, `get_citations(paper_id)`, `list_recent(days=30)`, all returning markdown strings.

- [ ] **Step 1: Write the failing tests**

`tests/test_server.py`:

```python
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
```

Add the module-scoped monkeypatch helper to `tests/conftest.py`:

```python
@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as mp:
        yield mp
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_server.py -v`
Expected: FAIL (ModuleNotFoundError: `papers_mcp.server`).

- [ ] **Step 3: Implement the server**

`src/papers_mcp/server.py`:

```python
"""MCP servers over research paper corpora — one streamable-HTTP endpoint per corpus."""

import contextlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from sentence_transformers import SentenceTransformer
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Mount, Route

from papers_mcp.corpus import Corpus, Paper
from papers_mcp.search import EMBEDDING_MODEL, SearchIndex

CORPORA = {
    "lipsync": "https://github.com/will-rice/lipsync-papers",
    "tts": "https://github.com/will-rice/tts-papers",
}
DATA_DIR = Path("data")
REFRESH_INTERVAL_SECONDS = 6 * 60 * 60
MAX_SEARCH_LIMIT = 50

corpora: dict[str, Corpus] = {}
indexes: dict[str, SearchIndex] = {}


def create_app() -> Starlette:
    """Build the Starlette app mounting one MCP server per corpus (uvicorn factory)."""
    logging.basicConfig(level=logging.INFO)
    servers = {name: make_server(name) for name in CORPORA}

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette):
        model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
        with ThreadPoolExecutor() as pool:
            list(pool.map(lambda name: build_corpus(name, model), CORPORA))
        threading.Thread(target=refresh_loop, args=(model,), daemon=True).start()
        async with contextlib.AsyncExitStack() as stack:
            for server in servers.values():
                await stack.enter_async_context(server.session_manager.run())
            yield

    async def index_page(request: Request) -> PlainTextResponse:
        lines = ["papers-mcp — MCP servers over research paper corpora", ""]
        lines += [f"  {CORPORA[name]}  →  /{name}/mcp" for name in CORPORA]
        return PlainTextResponse("\n".join(lines))

    routes: list[Mount | Route] = [Route("/", index_page)]
    routes += [
        Mount(f"/{name}", server.streamable_http_app()) for name, server in servers.items()
    ]
    return Starlette(routes=routes, lifespan=lifespan)


def make_server(name: str) -> FastMCP:
    """Create the FastMCP server (and its four tools) for one corpus."""
    mcp = FastMCP(
        name=f"{name}-papers",
        instructions=(
            f"Query the {name}-papers research corpus ({CORPORA[name]}): "
            "search titles/abstracts, read full papers as markdown, follow the "
            "in-corpus citation graph, and list recent papers."
        ),
        stateless_http=True,
        json_response=True,
    )

    @mcp.tool()
    def search_papers(query: str, limit: int = 10) -> str:
        """Hybrid keyword + semantic search over paper titles, abstracts, and authors.

        Returns the top matches with paper id, title, authors, submission date,
        and abstract. Use the paper id with get_paper or get_citations.
        """
        if not 1 <= limit <= MAX_SEARCH_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_SEARCH_LIMIT}")
        results = indexes[name].search(query, limit)
        return "\n\n".join(format_paper(paper) for paper in results)

    @mcp.tool()
    def get_paper(paper_id: str) -> str:
        """Return the paper's full converted markdown (methods, figures, references)."""
        paper = lookup(name, paper_id)
        if paper.md_path is None:
            raise ValueError(
                f"{paper_id} has no converted markdown; its metadata and abstract "
                "are available via search_papers"
            )
        return paper.md_path.read_text(encoding="utf-8")

    @mcp.tool()
    def get_citations(paper_id: str) -> str:
        """List in-corpus papers this paper cites, and in-corpus papers citing it."""
        paper = lookup(name, paper_id)
        papers = corpora[name].papers

        def title_list(ids: list[str]) -> str:
            if not ids:
                return "(none in corpus)"
            return "\n".join(f"- {pid}: {papers[pid].title}" for pid in ids)

        return (
            f"## Cites ({len(paper.cites)})\n{title_list(paper.cites)}\n\n"
            f"## Cited by ({len(paper.cited_by)})\n{title_list(paper.cited_by)}"
        )

    @mcp.tool()
    def list_recent(days: int = 30) -> str:
        """List papers submitted in the last N days, newest first."""
        if days < 1:
            raise ValueError("days must be at least 1")
        cutoff = (date.today() - timedelta(days=days)).isoformat()
        recent = sorted(
            (p for p in corpora[name].papers.values() if p.submitted >= cutoff),
            key=lambda p: p.submitted,
            reverse=True,
        )
        if not recent:
            return f"No papers submitted in the last {days} days."
        return "\n\n".join(format_paper(paper) for paper in recent)

    return mcp


def build_corpus(name: str, model: SentenceTransformer) -> None:
    """Sync, load, and index one corpus, then atomically swap it into the registry."""
    corpus = Corpus(name=name, repo_url=CORPORA[name], clone_dir=DATA_DIR / f"{name}-papers")
    corpus.sync()
    corpus.load()
    index = SearchIndex(list(corpus.papers.values()), model)
    corpora[name] = corpus
    indexes[name] = index


def refresh_loop(model: SentenceTransformer) -> None:
    """Re-sync and re-index every corpus on an interval; keep the old index on failure."""
    while True:
        time.sleep(REFRESH_INTERVAL_SECONDS)
        for name in CORPORA:
            try:
                build_corpus(name, model)
            except Exception:
                logging.exception("refresh failed for %s; serving previous index", name)


def lookup(name: str, paper_id: str) -> Paper:
    """Return the paper for *paper_id*, raising a concise error when unknown."""
    paper = corpora[name].papers.get(paper_id)
    if paper is None:
        raise ValueError(f"paper id {paper_id!r} not found in the {name} corpus")
    return paper


def format_paper(paper: Paper) -> str:
    """One search/listing hit as compact markdown."""
    return (
        f"**{paper.title}** ({paper.paper_id}, {paper.submitted})\n"
        f"{paper.authors}\n"
        f"{paper.abstract}"
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_server.py -v`
Expected: PASS (6 tests). The lifespan clones/pulls only the monkeypatched lipsync corpus from the test cache and embeds ~540 abstracts once for the module.

- [ ] **Step 5: Run the whole suite**

Run: `uv run pytest -v`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
uv run pre-commit run -a
git add -A && git commit -m "feat: mount per-corpus MCP servers with search, read, citation, and recency tools"
```

---

### Task 5: Dockerfile + Space README + local run verification

**Files:**
- Create: `Dockerfile`, `README.md`

**Interfaces:**
- Consumes: `create_app` factory from Task 4; `pyproject.toml`/`uv.lock` from Task 1.
- Produces: a container serving on port 7860; README with HF Space frontmatter (`sdk: docker`, `app_port: 7860`).

- [ ] **Step 1: Write the Dockerfile**

```dockerfile
FROM python:3.13-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    HF_HOME=/home/user/.cache/huggingface \
    UV_PROJECT_ENVIRONMENT=/home/user/.venv
WORKDIR /app

COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY --chown=user . .
RUN uv sync --frozen --no-dev

EXPOSE 7860
CMD ["uv", "run", "--no-sync", "uvicorn", "--factory", "papers_mcp.server:create_app", \
     "--host", "0.0.0.0", "--port", "7860"]
```

- [ ] **Step 2: Write README.md with HF Space frontmatter**

```markdown
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
| `https://<space-host>/lipsync/mcp` | lipsync-papers |
| `https://<space-host>/tts/mcp` | tts-papers |

Tools per server: `search_papers` (hybrid BM25 + embedding search),
`get_paper` (full markdown), `get_citations` (in-corpus citation graph),
`list_recent`. Corpora re-sync from GitHub every 6 hours.

## Connect

```bash
claude mcp add --transport http lipsync-papers https://<space-host>/lipsync/mcp
claude mcp add --transport http tts-papers https://<space-host>/tts/mcp
```

## Develop

```bash
uv sync
uv run pytest
uv run uvicorn --factory papers_mcp.server:create_app --port 7860
```
```

- [ ] **Step 3: Verify the server runs locally end-to-end**

```bash
uv run uvicorn --factory papers_mcp.server:create_app --port 7860 &
sleep 240  # first start clones both corpora and embeds ~2,700 abstracts
curl -s http://localhost:7860/
curl -s -X POST http://localhost:7860/tts/mcp \
  -H "Accept: application/json, text/event-stream" -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"search_papers","arguments":{"query":"zero-shot voice cloning"}}}'
kill %1
```

Expected: index page lists both endpoints; the tools/call response contains TTS paper titles. This exercises the real `/tts` corpus for the first time.

- [ ] **Step 4: Verify the Docker build (optional if Docker is unavailable locally)**

Run: `docker build -t papers-mcp . && docker run --rm -p 7860:7860 papers-mcp`
Expected: same curl checks pass against the container. If Docker is not installed, note it and rely on the Space build in Task 6.

- [ ] **Step 5: Commit**

```bash
uv run pre-commit run -a
git add -A && git commit -m "feat: Dockerfile and HF Space README"
```

---

### Task 6: Deploy to the Hugging Face Space and smoke-test

**Files:**
- Modify: `README.md` (fill in the real Space host)

**Interfaces:**
- Consumes: the complete repo from Tasks 1–5.
- Produces: live endpoints `https://<user>-papers-mcp.hf.space/lipsync/mcp` and `.../tts/mcp`.

- [ ] **Step 1: Create the Space and push**

```bash
hf auth whoami   # if not logged in: hf auth login
hf repo create papers-mcp --repo-type space --space-sdk docker
git remote add space https://huggingface.co/spaces/<user>/papers-mcp
git push space main
```

Expected: Space build starts; watch it at the Space page. (`<user>` is the whoami result.)

- [ ] **Step 2: Smoke-test the live endpoints**

Wait for the Space to show Running (first boot clones + embeds; a few minutes), then:

```bash
curl -s https://<user>-papers-mcp.hf.space/
curl -s -X POST https://<user>-papers-mcp.hf.space/lipsync/mcp \
  -H "Accept: application/json, text/event-stream" -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/list","params":{}}'
```

Expected: index page text; a JSON result listing the four tools. Repeat tools/list against `/tts/mcp`.

- [ ] **Step 3: Update README with the real host and commit**

Replace `<space-host>` placeholders in `README.md` with the real host, then:

```bash
uv run pre-commit run -a
git add README.md && git commit -m "docs: link live Space endpoints"
git push space main
```

- [ ] **Step 4: Link the endpoints from the corpus repos**

In `lipsync-papers` and `tts-papers`, add one line to each README's "How it works" section:

```markdown
* Query this corpus over MCP: `https://<user>-papers-mcp.hf.space/<corpus>/mcp` ([server code](https://huggingface.co/spaces/<user>/papers-mcp)).
```

Open a small PR in each repo (conventional commit, e.g. `docs: link the papers-mcp query endpoint`).
