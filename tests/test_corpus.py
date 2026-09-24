"""Tests for corpus syncing and loading against a papers-template layout corpus."""

from datetime import datetime

import pytest

from papers_mcp.corpus import Corpus, markdown_stem
from tests.conftest import DIFFTALK_ID, LATENTSYNC_ID, PAPERS, WAV2LIP_ID


@pytest.mark.parametrize(
    ("identifier", "stem"),
    [
        # Computed with papers-template v0.2.0 `expected_markdown`; the first pair is a
        # published file in will-rice/speech-enhancement-papers.
        ("arxiv:2608.26403v1", "arxiv-2608-26403v1--3abf19d76b17"),
        ("arxiv:2412.09262", "arxiv-2412-09262--dc01b6154080"),
        (
            "doi:10.1109/ICASSP.2024.10447000",
            "doi-10-1109-icassp-2024-10447000--d398f449d9f2",
        ),
        (
            "doi:10.1000/" + "X" * 100,
            "doi-10-1000-" + "x" * 68 + "--9804214c729d",
        ),
        (":::", "paper--f1ae2a75ed1f"),
    ],
)
def test_markdown_stem_matches_papers_template(identifier: str, stem: str) -> None:
    assert markdown_stem(identifier) == stem


def test_sync_clones_then_resets(lipsync_corpus: Corpus) -> None:
    assert (lipsync_corpus.clone_dir / "papers.csv").exists()
    # Second sync takes the fetch + reset path and must not raise.
    lipsync_corpus.sync()
    assert (lipsync_corpus.clone_dir / "papers.csv").exists()


def test_load_populates_papers(lipsync_corpus: Corpus) -> None:
    assert set(lipsync_corpus.papers) == {paper[0] for paper in PAPERS}
    latentsync = lipsync_corpus.papers[LATENTSYNC_ID]
    assert "LatentSync" in latentsync.title
    assert latentsync.authors == ("Chunyu Li", "Chao Zhang")
    assert isinstance(latentsync.published, datetime)
    assert latentsync.published.tzinfo is not None
    assert latentsync.md_path == (
        lipsync_corpus.clone_dir / "papers" / f"{markdown_stem(LATENTSYNC_ID)}.md"
    )
    assert "Latent diffusion lip sync" in latentsync.markdown


def test_papers_without_markdown_are_listed(lipsync_corpus: Corpus) -> None:
    difftalk = lipsync_corpus.papers[DIFFTALK_ID]
    assert difftalk.md_path is None
    assert difftalk.markdown == ""


def test_citations_map_filenames_to_identifiers(lipsync_corpus: Corpus) -> None:
    papers = lipsync_corpus.papers
    # Duplicate, self, unknown, and external links are dropped.
    assert papers[LATENTSYNC_ID].cites == [WAV2LIP_ID, DIFFTALK_ID]
    assert papers[WAV2LIP_ID].cites == [LATENTSYNC_ID]
    assert papers[LATENTSYNC_ID].cited_by == [WAV2LIP_ID]
    assert papers[DIFFTALK_ID].cited_by == [LATENTSYNC_ID]
    for paper in papers.values():
        for cited_id in paper.cites:
            assert paper.paper_id in papers[cited_id].cited_by
