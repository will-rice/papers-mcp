"""Search tests against the fixture lipsync corpus."""

import pytest
from sentence_transformers import SentenceTransformer

from papers_mcp.corpus import Corpus
from papers_mcp.search import EMBEDDING_MODEL, SearchIndex
from tests.conftest import LATENTSYNC_ID


@pytest.fixture(scope="session")
def lipsync_index(lipsync_corpus: Corpus) -> SearchIndex:
    model = SentenceTransformer(EMBEDDING_MODEL, device="cpu")
    return SearchIndex(list(lipsync_corpus.papers.values()), model)


def test_keyword_query_finds_latentsync(lipsync_index: SearchIndex) -> None:
    results = lipsync_index.search("latent diffusion lip sync SyncNet", limit=10)
    assert results[0].paper_id == LATENTSYNC_ID


def test_semantic_query_returns_relevant_papers(lipsync_index: SearchIndex) -> None:
    results = lipsync_index.search("make the mouth match new audio in a video", limit=2)
    assert len(results) == 2
    assert all("lip" in f"{p.title} {p.abstract}".lower() or "mouth" in p.abstract for p in results)


def test_authors_are_searchable(lipsync_index: SearchIndex) -> None:
    assert lipsync_index.search("Prajwal Mukhopadhyay", limit=1)[0].title.startswith("A Lip Sync")


def test_limit_is_respected(lipsync_index: SearchIndex) -> None:
    assert len(lipsync_index.search("talking head", limit=3)) == 3
