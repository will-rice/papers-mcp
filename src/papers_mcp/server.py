"""MCP servers over research paper corpora — one streamable-HTTP endpoint per corpus."""

import contextlib
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
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
    routes += [Mount(f"/{name}", server.streamable_http_app()) for name, server in servers.items()]
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
        # This server is mounted into a Starlette app (not run standalone via
        # mcp.run()), so FastMCP's own Host-header DNS-rebinding heuristic --
        # which only ever allowlists 127.0.0.1/localhost -- would 421 every
        # request once deployed under a real hostname. Access control belongs
        # at the reverse-proxy/deployment layer instead.
        transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
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
