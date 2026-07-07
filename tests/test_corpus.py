"""Tests for corpus syncing and loading against the real lipsync-papers repo."""

from papers_mcp.corpus import Corpus


def test_sync_clones_then_pulls(lipsync_corpus: Corpus) -> None:
    assert (lipsync_corpus.clone_dir / "papers.csv").exists()
    # Second sync takes the pull path and must not raise.
    lipsync_corpus.sync()
    assert (lipsync_corpus.clone_dir / "papers.csv").exists()


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
