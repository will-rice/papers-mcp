"""Tests for corpus syncing and loading against the real lipsync-papers repo."""

from papers_mcp.corpus import Corpus


def test_sync_clones_then_pulls(lipsync_corpus: Corpus) -> None:
    assert (lipsync_corpus.clone_dir / "papers.csv").exists()
    # Second sync takes the pull path and must not raise.
    lipsync_corpus.sync()
    assert (lipsync_corpus.clone_dir / "papers.csv").exists()
