"""Shared fixtures: a real lipsync-papers clone, cached across test runs."""

from pathlib import Path

import pytest

from papers_mcp.corpus import Corpus

CACHE_DIR = Path(__file__).parent / ".cache"
LIPSYNC_REPO = "https://github.com/will-rice/lipsync-papers"


@pytest.fixture(scope="session")
def lipsync_corpus() -> Corpus:
    corpus = Corpus(name="lipsync", repo_url=LIPSYNC_REPO, clone_dir=CACHE_DIR / "lipsync-papers")
    corpus.sync()
    corpus.load()
    return corpus
