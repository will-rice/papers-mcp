"""Shared fixtures: a local git corpus in the papers-template layout."""

import csv
import subprocess
from datetime import UTC, datetime, timedelta

import pytest

from papers_mcp.corpus import Corpus, markdown_stem

NOW = datetime.now(UTC).replace(microsecond=0)
LATENTSYNC_ID = "arxiv:2412.09262v1"
WAV2LIP_ID = "doi:10.1145/3394171.3413532"
DIFFTALK_ID = "semantic_scholar:4f2d1c7a9b0e3f5d6c8a1b2e4f6a8c0d2e4f6a8b"
PAPERS_CSV_FIELDS = (
    "identifier",
    "title",
    "abstract",
    "authors",
    "published",
    "url",
    "source",
    "input_format",
    "input_url",
    "categories",
    "doi",
    "arxiv_id",
)
# (identifier, title, abstract, authors, days since published, markdown body or None)
PAPERS = (
    (
        LATENTSYNC_ID,
        "LatentSync: Taming Audio-Conditioned Latent Diffusion Models for Lip Sync",
        "We present LatentSync, an end-to-end lip sync framework based on audio-conditioned "
        "latent diffusion models, with SyncNet supervision for lip-sync accuracy.",
        ("Chunyu Li", "Chao Zhang"),
        10,
        f"# LatentSync\n\n{'Latent diffusion lip sync. ' * 300}\n\n"
        f"Builds on [Wav2Lip]({markdown_stem(WAV2LIP_ID)}.md) and "
        f"[DiffTalk]({markdown_stem(DIFFTALK_ID)}.md); see also "
        f"[Wav2Lip again]({markdown_stem(WAV2LIP_ID)}.md), "
        f"[this paper]({markdown_stem(LATENTSYNC_ID)}.md), "
        "[an unknown paper](unknown--000000000000.md), and "
        "[arXiv](https://arxiv.org/abs/2412.09262v1).\n",
    ),
    (
        WAV2LIP_ID,
        "A Lip Sync Expert Is All You Need for Speech to Lip Generation In the Wild",
        "We lip-sync talking face videos of arbitrary identities to match a target speech "
        "segment using a pre-trained lip-sync expert discriminator.",
        ("K R Prajwal", "Rudrabha Mukhopadhyay"),
        400,
        f"# Wav2Lip\n\nCited by [LatentSync]({markdown_stem(LATENTSYNC_ID)}.md).\n",
    ),
    (
        DIFFTALK_ID,
        "DiffTalk: Crafting Diffusion Models for Generalized Audio-Driven Portraits Animation",
        "DiffTalk models talking head generation as an audio-driven temporally coherent "
        "denoising process that animates the mouth to match speech.",
        ("Shuai Shen",),
        3,
        None,
    ),
    (
        "arxiv:2301.00001v2",
        "Scaling Vision Transformers for Image Classification",
        "We scale vision transformers on ImageNet classification benchmarks.",
        ("Ada Vision",),
        30,
        None,
    ),
    (
        "arxiv:2302.00002v1",
        "Neural Codec Language Models for Zero-Shot Text to Speech",
        "A neural codec language model synthesizes speech for unseen speakers from text.",
        ("Tom Speech",),
        60,
        None,
    ),
)


@pytest.fixture(scope="session")
def corpus_repo(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Commit a small corpus in the papers-template layout and return its clone URL."""
    root = tmp_path_factory.mktemp("lipsync-papers")
    (root / "papers").mkdir()
    with (root / "papers.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PAPERS_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for identifier, title, abstract, authors, days, markdown in PAPERS:
            source, _, source_id = identifier.partition(":")
            writer.writerow(
                {
                    "identifier": identifier,
                    "title": title,
                    "abstract": abstract,
                    "authors": "|".join(authors),
                    "published": (NOW - timedelta(days=days)).isoformat(),
                    "url": f"https://example.org/{source_id}",
                    "source": source,
                    "input_format": "pdf",
                    "input_url": f"https://example.org/{source_id}.pdf",
                    "categories": "cs.CV|cs.SD",
                    "doi": source_id if source == "doi" else "",
                    "arxiv_id": source_id if source == "arxiv" else "",
                }
            )
            if markdown is not None:
                (root / "papers" / f"{markdown_stem(identifier)}.md").write_text(markdown)
    (root / "README.md").write_text("# Lipsync Papers\n")
    for args in (
        ["init", "-q"],
        ["add", "."],
        ["-c", "user.name=test", "-c", "user.email=test@example.org", "commit", "-qm", "corpus"],
    ):
        subprocess.run(["git", "-C", str(root), *args], check=True)
    return root.as_uri()


@pytest.fixture(scope="session")
def lipsync_corpus(corpus_repo: str, tmp_path_factory: pytest.TempPathFactory) -> Corpus:
    clone_dir = tmp_path_factory.mktemp("data") / "lipsync-papers"
    corpus = Corpus(name="lipsync", repo_url=corpus_repo, clone_dir=clone_dir)
    corpus.sync()
    corpus.load()
    return corpus


@pytest.fixture(scope="module")
def monkeypatch_module():
    with pytest.MonkeyPatch.context() as mp:
        yield mp
