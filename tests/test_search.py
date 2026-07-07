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
