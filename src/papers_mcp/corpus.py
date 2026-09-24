"""Load and sync a research-papers corpus (papers.csv + markdown) from GitHub."""

import csv
import hashlib
import logging
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# Markdown filename rule, mirrored from papers-template's `expected_markdown`
# (template/src/papers_pipeline/batching.py): `papers/<slug>--<digest>.md`.
SLUG_LIMIT = 80
DIGEST_LENGTH = 12
SLUG_RE = re.compile(r"[^a-z0-9]+")

# In-corpus citation links point at a sibling markdown file: `](<slug>--<digest>.md)`.
CITATION_LINK_RE = re.compile(rf"\]\(([a-z0-9-]+--[0-9a-f]{{{DIGEST_LENGTH}}})\.md\)")


def markdown_stem(identifier: str) -> str:
    """Return the markdown filename stem papers-template assigns to *identifier*.

    Mirrors `_identifier_slug` and `_identifier_digest` in papers-template's
    `papers_pipeline.batching`; keep in sync with its `expected_markdown`.
    """
    slug = SLUG_RE.sub("-", identifier.casefold()).strip("-")[:SLUG_LIMIT].rstrip("-") or "paper"
    digest = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:DIGEST_LENGTH]
    return f"{slug}--{digest}"


@dataclass
class Paper:
    """One paper's metadata, markdown location, and in-corpus citation edges."""

    paper_id: str
    title: str
    authors: tuple[str, ...]
    published: datetime
    url: str
    abstract: str
    md_path: Path | None = None
    markdown: str = ""
    cites: list[str] = field(default_factory=list)
    cited_by: list[str] = field(default_factory=list)


@dataclass
class Corpus:
    """A cloned corpus repo and its loaded papers, keyed by papers.csv `identifier`."""

    name: str
    repo_url: str
    clone_dir: Path
    papers: dict[str, Paper] = field(default_factory=dict)

    def sync(self) -> None:
        """Clone the corpus repo if absent, otherwise hard-reset to the latest origin HEAD.

        A reset-to-fetched-ref mirror (rather than `pull --ff-only`) is immune to the
        upstream repo ever force-pushing, which would otherwise fail every refresh forever.
        """
        if (self.clone_dir / ".git").exists():
            self._git(["-C", str(self.clone_dir), "fetch", "--depth", "1", "origin", "HEAD"])
            self._git(["-C", str(self.clone_dir), "reset", "--hard", "FETCH_HEAD"])
        else:
            self.clone_dir.parent.mkdir(parents=True, exist_ok=True)
            self._git(["clone", "--depth", "1", self.repo_url, str(self.clone_dir)])
        logging.info("synced %s corpus at %s", self.name, self.clone_dir)

    def _git(self, args: list[str]) -> None:
        """Run a git command, surfacing its stderr on failure instead of swallowing it."""
        try:
            subprocess.run(["git", *args], check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(f"git {args} failed for {self.name}: {exc.stderr.strip()}") from exc

    def load(self) -> None:
        """Load papers.csv, attach generated markdown, and build the citation graph.

        Papers listed in papers.csv whose markdown has not been generated yet keep
        `md_path=None`.
        """
        papers: dict[str, Paper] = {}
        with (self.clone_dir / "papers.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                published = datetime.fromisoformat(row["published"])
                if published.tzinfo is None:
                    raise ValueError(f"{row['identifier']} has a timezone-naive published date")
                papers[row["identifier"]] = Paper(
                    paper_id=row["identifier"],
                    title=row["title"],
                    authors=tuple(filter(None, row["authors"].split("|"))),
                    published=published,
                    url=row["url"],
                    abstract=row["abstract"],
                )

        ids_by_stem = {markdown_stem(paper_id): paper_id for paper_id in papers}
        for stem, paper_id in ids_by_stem.items():
            md_path = self.clone_dir / "papers" / f"{stem}.md"
            if md_path.exists():
                papers[paper_id].md_path = md_path

        for paper in papers.values():
            if paper.md_path is None:
                continue
            paper.markdown = paper.md_path.read_text(encoding="utf-8")
            for stem in CITATION_LINK_RE.findall(paper.markdown):
                cited_id = ids_by_stem.get(stem)
                if (
                    cited_id is not None
                    and cited_id != paper.paper_id
                    and cited_id not in paper.cites
                ):
                    paper.cites.append(cited_id)
        for paper in papers.values():
            for cited_id in paper.cites:
                papers[cited_id].cited_by.append(paper.paper_id)

        self.papers = papers
        logging.info("loaded %d papers for %s corpus", len(papers), self.name)
